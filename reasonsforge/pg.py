"""PostgreSQL-native storage backend for the dependency network.

Each operation is a SQL transaction — no full-network load/save.
Enables concurrent writers and multi-tenant deployment.

Requires psycopg v3: pip install 'psycopg[binary]>=3.1'
"""

import json
import re
from collections import deque
from datetime import datetime, timezone

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None  # type: ignore[assignment]


SCHEMA = """
CREATE TABLE IF NOT EXISTS rms_nodes (
    id TEXT NOT NULL,
    project_id UUID NOT NULL,
    text TEXT NOT NULL,
    truth_value TEXT NOT NULL DEFAULT 'IN' CHECK (truth_value IN ('IN', 'OUT')),
    source TEXT DEFAULT '',
    source_url TEXT DEFAULT '',
    source_hash TEXT DEFAULT '',
    date TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ,
    reviewed_at TIMESTAMPTZ,
    verified_at TIMESTAMPTZ,
    retracted_at TIMESTAMPTZ,
    PRIMARY KEY (id, project_id)
);

CREATE TABLE IF NOT EXISTS rms_justifications (
    id SERIAL PRIMARY KEY,
    node_id TEXT NOT NULL,
    project_id UUID NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('SL', 'CP')),
    antecedents JSONB NOT NULL DEFAULT '[]',
    outlist JSONB NOT NULL DEFAULT '[]',
    label TEXT DEFAULT '',
    FOREIGN KEY (node_id, project_id) REFERENCES rms_nodes(id, project_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rms_nogoods (
    id TEXT NOT NULL,
    project_id UUID NOT NULL,
    nodes JSONB NOT NULL DEFAULT '[]',
    discovered TEXT DEFAULT '',
    resolution TEXT DEFAULT '',
    PRIMARY KEY (id, project_id)
);

CREATE TABLE IF NOT EXISTS rms_propagation_log (
    id SERIAL PRIMARY KEY,
    project_id UUID NOT NULL,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rms_network_meta (
    key TEXT NOT NULL,
    project_id UUID NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (key, project_id)
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_rms_nodes_project ON rms_nodes(project_id);
CREATE INDEX IF NOT EXISTS idx_rms_nodes_status ON rms_nodes(project_id, truth_value);
CREATE INDEX IF NOT EXISTS idx_rms_justifications_node ON rms_justifications(node_id, project_id);
CREATE INDEX IF NOT EXISTS idx_rms_nogoods_project ON rms_nogoods(project_id);
CREATE INDEX IF NOT EXISTS idx_rms_log_project ON rms_propagation_log(project_id);
CREATE INDEX IF NOT EXISTS idx_rms_nodes_fts ON rms_nodes USING gin(to_tsvector('english', text));
CREATE INDEX IF NOT EXISTS idx_rms_justifications_antecedents ON rms_justifications USING gin(antecedents);
CREATE INDEX IF NOT EXISTS idx_rms_justifications_outlist ON rms_justifications USING gin(outlist);
"""


def _require_psycopg():
    if psycopg is None:
        raise ImportError(
            "psycopg is required for PostgreSQL support. "
            "Install it with: pip install 'psycopg[binary]>=3.1'"
        )


class PgApi:
    """PostgreSQL-native API for the dependency network.

    Each method executes as a SQL transaction. No in-memory Network object.
    """

    def __init__(self, conninfo, project_id):
        _require_psycopg()
        if isinstance(conninfo, str):
            self.conn = psycopg.connect(conninfo, autocommit=False)
            self._owns_conn = True
        else:
            self.conn = conninfo
            self._owns_conn = False
        self.project_id = str(project_id)

    def close(self):
        if self._owns_conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is not None:
            self.conn.rollback()
        self.close()

    # ── Schema ──────────────────────────────────────────────────

    def init_db(self, project_name: str = ""):
        from .metadata import SCHEMA_VERSION
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute(INDEXES)
            # Migrate existing databases: add source_url if missing
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'rms_nodes' AND column_name = 'source_url'"
            )
            if not cur.fetchone():
                cur.execute("ALTER TABLE rms_nodes ADD COLUMN source_url TEXT DEFAULT ''")
            for col in ("updated_at", "reviewed_at", "verified_at", "retracted_at"):
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'rms_nodes' AND column_name = %s",
                    (col,),
                )
                if not cur.fetchone():
                    cur.execute(f"ALTER TABLE rms_nodes ADD COLUMN {col} TIMESTAMPTZ")
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            pname = project_name or self.project_id
            for key, val in [
                ("schema_version", SCHEMA_VERSION),
                ("project_name", pname),
                ("created_at", now),
                ("updated_at", now),
            ]:
                cur.execute(
                    "INSERT INTO rms_network_meta (key, project_id, value) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (key, project_id) DO NOTHING",
                    (key, self.project_id, val),
                )
        self.conn.commit()
        return {"project_id": self.project_id, "created": True}

    # ── Core mutations ──────────────────────────────────────────

    def _add_node_raw(self, cur, node_id, text, justifications=None,
                      source="", source_url="", source_hash="", date="",
                      metadata=None, created_at="", updated_at=""):
        """Insert a node with pre-parsed justification dicts.

        Does NOT commit, validate refs, or inherit access tags.
        Caller is responsible for transaction management.
        """
        pid = self.project_id
        now_ts = datetime.now(timezone.utc)
        now = date or now_ts.isoformat(timespec="seconds")
        meta = metadata or {}
        ts_created = created_at or now_ts.isoformat(timespec="seconds")
        ts_updated = updated_at or ts_created

        cur.execute(
            "INSERT INTO rms_nodes (id, project_id, text, source, source_url, "
            "source_hash, date, metadata, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (node_id, pid, text, source, source_url, source_hash, now,
             json.dumps(meta), ts_created, ts_updated),
        )

        if justifications:
            for j in justifications:
                cur.execute(
                    "INSERT INTO rms_justifications (node_id, project_id, type, "
                    "antecedents, outlist, label) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (node_id, pid, j["type"],
                     json.dumps(j.get("antecedents", [])),
                     json.dumps(j.get("outlist", [])),
                     j.get("label", "")),
                )

        if justifications:
            truth = self._compute_truth(cur, node_id)
        else:
            truth = "IN"

        cur.execute(
            "UPDATE rms_nodes SET truth_value = %s WHERE id = %s AND project_id = %s",
            (truth, node_id, pid),
        )

        self._log(cur, "add", node_id, truth)
        return truth

    def add_node(self, node_id, text, sl="", cp="", unless="", label="",
                 source="", source_url="", access_tags=None,
                 namespace=None, example=None):
        pid = self.project_id
        metadata = {}
        if access_tags:
            metadata["access_tags"] = sorted(access_tags)
        if example is not None:
            metadata["example"] = example

        if namespace:
            if ":" not in node_id:
                node_id = f"{namespace}:{node_id}"
            self.ensure_namespace(namespace)

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM rms_nodes WHERE id = %s AND project_id = %s",
                (node_id, pid),
            )
            if cur.fetchone():
                raise ValueError(f"Node '{node_id}' already exists")

            justifications = self._parse_justifications(sl, cp, unless, label)

            if namespace:
                active_id = f"{namespace}:active"
                for j in justifications:
                    j["antecedents"] = [
                        f"{namespace}:{a}" if ":" not in a else a
                        for a in j["antecedents"]
                    ]
                    j["outlist"] = [
                        f"{namespace}:{o}" if ":" not in o else o
                        for o in j["outlist"]
                    ]
                if not justifications:
                    justifications.append({
                        "type": "SL",
                        "antecedents": [active_id],
                        "outlist": [],
                        "label": "",
                    })
                else:
                    for j in justifications:
                        if active_id not in j["antecedents"]:
                            j["antecedents"].append(active_id)

            if justifications:
                self._validate_refs(cur, justifications)

            truth = self._add_node_raw(
                cur, node_id, text, justifications=justifications or None,
                source=source, source_url=source_url, metadata=metadata,
            )

            if justifications:
                self._inherit_access_tags(cur, node_id, justifications)

            cur.execute(
                "SELECT COUNT(*) FROM rms_nodes WHERE project_id = %s "
                "AND NOT EXISTS (SELECT 1 FROM rms_justifications j "
                "WHERE j.node_id = rms_nodes.id AND j.project_id = rms_nodes.project_id)",
                (pid,),
            )
            premise_count = cur.fetchone()[0]

        self.conn.commit()
        return {
            "node_id": node_id,
            "truth_value": truth,
            "type": "premise" if not justifications else "derived",
            "premise_count": premise_count,
        }

    def add_justification(self, node_id, sl="", cp="", unless="", label="",
                          namespace=None):
        pid = self.project_id

        if namespace and ":" not in node_id:
            node_id = f"{namespace}:{node_id}"

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT truth_value FROM rms_nodes WHERE id = %s AND project_id = %s",
                (node_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{node_id}' not found")
            old_value = row[0]

            justifications = self._parse_justifications(sl, cp, unless, label)
            if not justifications:
                raise ValueError("No justification specified (use --sl or --cp)")

            if namespace:
                for j in justifications:
                    j["antecedents"] = [
                        f"{namespace}:{a}" if ":" not in a else a
                        for a in j["antecedents"]
                    ]
                    j["outlist"] = [
                        f"{namespace}:{o}" if ":" not in o else o
                        for o in j["outlist"]
                    ]

            self._validate_refs(cur, justifications)

            for j in justifications:
                cur.execute(
                    "INSERT INTO rms_justifications (node_id, project_id, type, antecedents, outlist, label) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (node_id, pid, j["type"],
                     json.dumps(j["antecedents"]), json.dumps(j["outlist"]), j["label"]),
                )

            self._inherit_access_tags(cur, node_id, justifications)

            new_value = self._compute_truth(cur, node_id)
            changed = []

            if old_value != new_value:
                cur.execute(
                    "UPDATE rms_nodes SET truth_value = %s WHERE id = %s AND project_id = %s",
                    (new_value, node_id, pid),
                )
                changed.append(node_id)
                went_out, went_in = self._propagate(cur, node_id)
                changed.extend(went_out)
                changed.extend(went_in)

            self._log(cur, "add-justification", node_id, new_value)

        self.conn.commit()
        return {
            "node_id": node_id,
            "old_truth_value": old_value,
            "new_truth_value": new_value,
            "changed": changed,
        }

    def retract_node(self, node_id, reason=""):
        pid = self.project_id

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT truth_value, metadata FROM rms_nodes WHERE id = %s AND project_id = %s",
                (node_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{node_id}' not found")

            old_value, metadata = row[0], row[1]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            metadata["_retracted"] = True
            if reason:
                metadata["retract_reason"] = reason

            now = datetime.now(timezone.utc).isoformat(timespec="seconds")

            if old_value == "OUT":
                cur.execute(
                    "UPDATE rms_nodes SET metadata = %s, retracted_at = %s, updated_at = %s "
                    "WHERE id = %s AND project_id = %s",
                    (json.dumps(metadata), now, now, node_id, pid),
                )
                self.conn.commit()
                return {"changed": [], "went_out": [], "went_in": []}

            cur.execute(
                "UPDATE rms_nodes SET truth_value = 'OUT', metadata = %s, "
                "retracted_at = %s, updated_at = %s "
                "WHERE id = %s AND project_id = %s",
                (json.dumps(metadata), now, now, node_id, pid),
            )
            self._log(cur, "retract", node_id, reason or "OUT")

            went_out, went_in = self._propagate(cur, node_id)

        self.conn.commit()
        all_changed = [node_id] + went_out + went_in
        return {"changed": all_changed, "went_out": [node_id] + went_out, "went_in": went_in}

    def assert_node(self, node_id):
        pid = self.project_id

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT truth_value, metadata FROM rms_nodes WHERE id = %s AND project_id = %s",
                (node_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{node_id}' not found")

            old_value, metadata = row[0], row[1]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            if old_value == "IN":
                return {"changed": [], "went_out": [], "went_in": []}

            metadata.pop("_retracted", None)
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            cur.execute(
                "UPDATE rms_nodes SET truth_value = 'IN', metadata = %s, "
                "retracted_at = NULL, updated_at = %s "
                "WHERE id = %s AND project_id = %s",
                (json.dumps(metadata), now, node_id, pid),
            )
            self._log(cur, "assert", node_id, "IN")

            went_out, went_in = self._propagate(cur, node_id)

        self.conn.commit()
        all_changed = [node_id] + went_out + went_in
        return {"changed": all_changed, "went_out": went_out, "went_in": [node_id] + went_in}

    # ── What-if simulation ──────────────────────────────────────

    def what_if_retract(self, node_id):
        pid = self.project_id
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT truth_value FROM rms_nodes WHERE id = %s AND project_id = %s",
                    (node_id, pid),
                )
                row = cur.fetchone()
                if not row:
                    raise KeyError(f"Node '{node_id}' not found")
                if row[0] == "OUT":
                    return {
                        "node_id": node_id, "already_out": True,
                        "retracted": [], "restored": [], "total_affected": 0,
                    }

                cur.execute(
                    "SELECT id, truth_value FROM rms_nodes WHERE project_id = %s",
                    (pid,),
                )
                before = {r[0]: r[1] for r in cur.fetchall()}

                cur.execute(
                    "UPDATE rms_nodes SET truth_value = 'OUT' "
                    "WHERE id = %s AND project_id = %s",
                    (node_id, pid),
                )
                went_out, went_in = self._propagate(cur, node_id)

                changed = went_out + went_in
                retracted, restored = self._collect_what_if_results(
                    cur, changed, before, node_id,
                )
        finally:
            self.conn.rollback()

        retracted.sort(key=lambda c: (c["depth"], c["id"]))
        restored.sort(key=lambda c: (c["depth"], c["id"]))
        return {
            "node_id": node_id, "already_out": False,
            "retracted": retracted, "restored": restored,
            "total_affected": len(retracted) + len(restored),
        }

    def what_if_assert(self, node_id):
        pid = self.project_id
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT truth_value FROM rms_nodes WHERE id = %s AND project_id = %s",
                    (node_id, pid),
                )
                row = cur.fetchone()
                if not row:
                    raise KeyError(f"Node '{node_id}' not found")
                if row[0] == "IN":
                    return {
                        "node_id": node_id, "already_in": True,
                        "retracted": [], "restored": [], "total_affected": 0,
                    }

                cur.execute(
                    "SELECT id, truth_value FROM rms_nodes WHERE project_id = %s",
                    (pid,),
                )
                before = {r[0]: r[1] for r in cur.fetchall()}

                cur.execute(
                    "UPDATE rms_nodes SET truth_value = 'IN' "
                    "WHERE id = %s AND project_id = %s",
                    (node_id, pid),
                )
                went_out, went_in = self._propagate(cur, node_id)

                changed = went_out + went_in
                retracted, restored = self._collect_what_if_results(
                    cur, changed, before, node_id,
                )
        finally:
            self.conn.rollback()

        retracted.sort(key=lambda c: (c["depth"], c["id"]))
        restored.sort(key=lambda c: (c["depth"], c["id"]))
        return {
            "node_id": node_id, "already_in": False,
            "retracted": retracted, "restored": restored,
            "total_affected": len(retracted) + len(restored),
        }

    def _collect_what_if_results(self, cur, changed, before, source_id):
        pid = self.project_id
        retracted = []
        restored = []
        if not changed:
            return retracted, restored

        cur.execute(
            "SELECT id, text, truth_value FROM rms_nodes "
            "WHERE project_id = %s AND id = ANY(%s)",
            (pid, list(changed)),
        )
        after_states = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

        for nid in changed:
            if nid not in after_states:
                continue
            text, new_tv = after_states[nid]
            old_tv = before.get(nid)
            if old_tv == new_tv:
                continue
            depth = self._cascade_depth_pg(cur, nid, source_id)
            dep_count = len(self._find_dependents(cur, [nid]))
            info = {"id": nid, "text": text, "depth": depth, "dependents": dep_count}
            if old_tv == "IN" and new_tv == "OUT":
                retracted.append(info)
            elif old_tv == "OUT" and new_tv == "IN":
                restored.append(info)
        return retracted, restored

    def _cascade_depth_pg(self, cur, target_id, source_id):
        visited = {source_id}
        queue = deque([(source_id, 0)])
        while queue:
            current_id, depth = queue.popleft()
            deps = self._find_dependents(cur, [current_id])
            for dep_id in deps:
                if dep_id in visited:
                    continue
                if dep_id == target_id:
                    return depth + 1
                visited.add(dep_id)
                queue.append((dep_id, depth + 1))
        return 0

    # ── Dialectical operations ─────────────────────────────────

    def challenge(self, target_id, reason, challenge_id=None):
        with self.conn.cursor() as cur:
            result = self._challenge_internal(cur, target_id, reason, challenge_id)
        self.conn.commit()
        return result

    def defend(self, target_id, challenge_id, reason, defense_id=None):
        pid = self.project_id

        with self.conn.cursor() as cur:
            # Verify target and challenge exist
            cur.execute(
                "SELECT 1 FROM rms_nodes WHERE id = %s AND project_id = %s",
                (target_id, pid),
            )
            if not cur.fetchone():
                raise KeyError(f"Node '{target_id}' not found")
            cur.execute(
                "SELECT 1 FROM rms_nodes WHERE id = %s AND project_id = %s",
                (challenge_id, pid),
            )
            if not cur.fetchone():
                raise KeyError(f"Challenge '{challenge_id}' not found")

            # Generate defense ID
            if defense_id is None:
                defense_id = f"defense-{challenge_id}"
                suffix = 1
                while True:
                    cur.execute(
                        "SELECT 1 FROM rms_nodes WHERE id = %s AND project_id = %s",
                        (defense_id, pid),
                    )
                    if not cur.fetchone():
                        break
                    suffix += 1
                    defense_id = f"defense-{challenge_id}-{suffix}"
            else:
                cur.execute(
                    "SELECT 1 FROM rms_nodes WHERE id = %s AND project_id = %s",
                    (defense_id, pid),
                )
                if cur.fetchone():
                    raise ValueError(f"Defense node '{defense_id}' already exists")

            # Challenge the challenge (defense = challenge against the challenge)
            result = self._challenge_internal(cur, challenge_id, reason, defense_id)

            # Update defense node metadata
            cur.execute(
                "SELECT metadata FROM rms_nodes WHERE id = %s AND project_id = %s",
                (defense_id, pid),
            )
            meta = cur.fetchone()[0]
            if isinstance(meta, str):
                meta = json.loads(meta)
            meta["defense_target"] = challenge_id
            meta["defends"] = target_id
            cur.execute(
                "UPDATE rms_nodes SET metadata = %s WHERE id = %s AND project_id = %s",
                (json.dumps(meta), defense_id, pid),
            )

        self.conn.commit()
        return {
            "defense_id": defense_id,
            "challenge_id": challenge_id,
            "target_id": target_id,
            "changed": result["changed"],
        }

    # ── Read operations ─────────────────────────────────────────

    def get_status(self, visible_to=None):
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT n.id, n.text, n.truth_value, n.metadata, "
                "(SELECT COUNT(*) FROM rms_justifications j "
                " WHERE j.node_id = n.id AND j.project_id = n.project_id) AS jcount "
                "FROM rms_nodes n WHERE n.project_id = %s ORDER BY n.id",
                (pid,),
            )
            rows = cur.fetchall()

        nodes = []
        for row in rows:
            nid, text, tv, meta, jcount = row
            if isinstance(meta, str):
                meta = json.loads(meta)
            if visible_to is not None and not self._is_visible(meta, visible_to):
                continue
            nodes.append({
                "id": nid,
                "text": text,
                "truth_value": tv,
                "justification_count": jcount,
            })

        in_count = sum(1 for n in nodes if n["truth_value"] == "IN")
        return {"nodes": nodes, "in_count": in_count, "total": len(nodes)}

    def show_node(self, node_id, visible_to=None):
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, text, truth_value, source, source_url, source_hash, metadata, "
                "created_at, updated_at, reviewed_at, verified_at, retracted_at "
                "FROM rms_nodes WHERE id = %s AND project_id = %s",
                (node_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{node_id}' not found")

            nid, text, tv, source, source_url, source_hash, meta, \
                created_at, updated_at, reviewed_at, verified_at, retracted_at = row
            if isinstance(meta, str):
                meta = json.loads(meta)

            if visible_to is not None and not self._is_visible(meta, visible_to):
                raise PermissionError(f"Access denied for node '{node_id}'")

            cur.execute(
                "SELECT type, antecedents, outlist, label FROM rms_justifications "
                "WHERE node_id = %s AND project_id = %s ORDER BY id",
                (node_id, pid),
            )
            justifications = []
            for jrow in cur.fetchall():
                jtype, ants, outs, jlabel = jrow
                if isinstance(ants, str):
                    ants = json.loads(ants)
                if isinstance(outs, str):
                    outs = json.loads(outs)
                j = {"type": jtype, "antecedents": ants, "outlist": outs, "label": jlabel}
                justifications.append(j)

            dependents = sorted(self._find_dependents(cur, [node_id]))

        def _fmt_ts(val):
            return val.isoformat(timespec="seconds") if val else ""

        return {
            "id": nid,
            "text": text,
            "truth_value": tv,
            "source": source,
            "source_url": source_url or "",
            "source_hash": source_hash,
            "justifications": justifications,
            "dependents": dependents,
            "metadata": meta,
            "created_at": _fmt_ts(created_at),
            "updated_at": _fmt_ts(updated_at),
            "reviewed_at": _fmt_ts(reviewed_at),
            "verified_at": _fmt_ts(verified_at),
            "retracted_at": _fmt_ts(retracted_at),
        }

    def search(self, query, visible_to=None, format="markdown"):
        pid = self.project_id

        with self.conn.cursor() as cur:
            # plainto_tsquery handles arbitrary user input safely
            if query.strip():
                cur.execute(
                    "SELECT id, text, truth_value, source, metadata "
                    "FROM rms_nodes "
                    "WHERE project_id = %s "
                    "AND (to_tsvector('english', text) @@ plainto_tsquery('english', %s) "
                    "     OR id ILIKE %s) "
                    "ORDER BY ts_rank(to_tsvector('english', text), "
                    "         plainto_tsquery('english', %s)) DESC "
                    "LIMIT 20",
                    (pid, query, f"%{query}%", query),
                )
            else:
                cur.execute(
                    "SELECT id, text, truth_value, source, metadata "
                    "FROM rms_nodes WHERE project_id = %s AND id ILIKE %s LIMIT 20",
                    (pid, f"%{query}%"),
                )

            matched_rows = cur.fetchall()

        if not matched_rows:
            if format == "dict":
                return {"results": [], "count": 0}
            return "No results found."

        # Apply visibility filter
        matched = []
        for row in matched_rows:
            nid, text, tv, source, meta = row
            if isinstance(meta, str):
                meta = json.loads(meta)
            if visible_to is not None and not self._is_visible(meta, visible_to):
                continue
            matched.append({
                "id": nid, "text": text, "truth_value": tv,
                "source": source, "metadata": meta,
            })

        if not matched:
            if format == "dict":
                return {"results": [], "count": 0}
            return "No results found."

        matched_ids = [m["id"] for m in matched]

        # Neighbor expansion
        with self.conn.cursor() as cur:
            neighbors = self._expand_neighbors(cur, matched_ids, visible_to)

        if format == "dict":
            return {"results": matched, "neighbors": neighbors, "count": len(matched)}

        return self._format_results(matched, neighbors, format)

    def hash_sources(self, force=False, repos=None):
        """Backfill source hashes for nodes with source paths but no stored hash."""
        from pathlib import Path
        from .check_stale import hash_file, resolve_source_path

        pid = self.project_id
        repo_paths = None
        if repos:
            repo_paths = {k: Path(v) for k, v in repos.items()}
        else:
            repo_paths = self._load_repos()

        with self.conn.cursor() as cur:
            if force:
                cur.execute(
                    "SELECT id, source, source_hash, metadata FROM rms_nodes "
                    "WHERE project_id = %s AND source != '' AND source IS NOT NULL",
                    (pid,),
                )
            else:
                cur.execute(
                    "SELECT id, source, source_hash, metadata FROM rms_nodes "
                    "WHERE project_id = %s AND source != '' AND source IS NOT NULL "
                    "AND (source_hash = '' OR source_hash IS NULL)",
                    (pid,),
                )
            rows = cur.fetchall()

        results = []
        for nid, source, source_hash, meta in sorted(rows, key=lambda r: r[0]):
            if isinstance(meta, str):
                meta = json.loads(meta) if meta else {}
            agent = meta.get("agent") if meta else None
            path = resolve_source_path(source, repo_paths, agent=agent)
            if path is None:
                continue
            new_hash = hash_file(path)
            was_empty = not source_hash
            results.append({
                "node_id": nid,
                "source": source,
                "hash": new_hash,
                "was_empty": was_empty,
            })

        if results:
            with self.conn.cursor() as cur:
                for item in results:
                    cur.execute(
                        "UPDATE rms_nodes SET source_hash = %s "
                        "WHERE id = %s AND project_id = %s",
                        (item["hash"], item["node_id"], pid),
                    )
            self.conn.commit()

        return {"hashed": results, "count": len(results)}

    def check_stale(self, repos=None, upgrade_hashes=False, git_aware=False):
        """Check all IN nodes for source file staleness."""
        from pathlib import Path
        from .check_stale import hash_file, resolve_source_path

        pid = self.project_id
        repo_paths = None
        if repos:
            repo_paths = {k: Path(v) for k, v in repos.items()}
        else:
            repo_paths = self._load_repos()

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, source_hash, metadata FROM rms_nodes "
                "WHERE project_id = %s AND truth_value = 'IN' "
                "AND source != '' AND source IS NOT NULL "
                "AND source_hash != '' AND source_hash IS NOT NULL",
                (pid,),
            )
            rows = cur.fetchall()

        results = []
        upgraded = 0
        upgrades_to_commit = []

        for nid, source, source_hash, meta in sorted(rows, key=lambda r: r[0]):
            if isinstance(meta, str):
                meta = json.loads(meta) if meta else {}
            agent = meta.get("agent") if meta else None
            path = resolve_source_path(source, repo_paths, agent=agent)
            if path is None:
                results.append({
                    "node_id": nid,
                    "old_hash": source_hash,
                    "new_hash": None,
                    "source": source,
                    "source_path": None,
                    "reason": "source_deleted",
                })
                continue

            current_hash = hash_file(path)
            if current_hash != source_hash:
                if len(source_hash) == 16 and current_hash.startswith(source_hash):
                    if upgrade_hashes:
                        upgrades_to_commit.append((current_hash, nid))
                        upgraded += 1
                        continue
                    results.append({
                        "node_id": nid,
                        "old_hash": source_hash,
                        "new_hash": current_hash,
                        "source": source,
                        "source_path": str(path),
                        "reason": "truncated_hash",
                    })
                    continue
                results.append({
                    "node_id": nid,
                    "old_hash": source_hash,
                    "new_hash": current_hash,
                    "source": source,
                    "source_path": str(path),
                    "reason": "content_changed",
                })

        if upgrades_to_commit:
            with self.conn.cursor() as cur:
                for new_hash, nid in upgrades_to_commit:
                    cur.execute(
                        "UPDATE rms_nodes SET source_hash = %s "
                        "WHERE id = %s AND project_id = %s",
                        (new_hash, nid, pid),
                    )
            self.conn.commit()

        return {
            "stale": results,
            "checked": len(rows),
            "stale_count": len(results),
            "upgraded": upgraded,
            "sha_bumped": 0,
        }

    def lookup(self, query, visible_to=None):
        """Simple all-terms substring search over full belief blocks."""
        pid = self.project_id
        query_terms = query.lower().split()

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT n.id, n.text, n.truth_value, n.source, n.source_hash, "
                "n.date, n.metadata FROM rms_nodes n "
                "WHERE n.project_id = %s ORDER BY n.id",
                (pid,),
            )
            node_rows = cur.fetchall()

            cur.execute(
                "SELECT node_id, antecedents, outlist FROM rms_justifications "
                "WHERE project_id = %s",
                (pid,),
            )
            just_rows = cur.fetchall()

        refs_by_node = {}
        dependents_by_node = {}
        for node_id, ants, outs in just_rows:
            if isinstance(ants, str):
                ants = json.loads(ants)
            if isinstance(outs, str):
                outs = json.loads(outs)
            all_refs = list(ants) + list(outs or [])
            refs_by_node.setdefault(node_id, []).extend(all_refs)
            for ref in all_refs:
                dependents_by_node.setdefault(ref, []).append(node_id)

        matches = []
        for nid, text, tv, source, source_hash, date, meta in node_rows:
            if isinstance(meta, str):
                meta = json.loads(meta) if meta else {}
            if visible_to is not None and not self._is_visible(meta, visible_to):
                continue

            block_parts = [nid, text]
            if source:
                block_parts.append(source)
            if source_hash:
                block_parts.append(source_hash)
            if date:
                block_parts.append(date)
            for ref in refs_by_node.get(nid, []):
                block_parts.append(ref)
            for dep in dependents_by_node.get(nid, []):
                block_parts.append(dep)
            block_lower = " ".join(block_parts).lower()

            if all(term in block_lower for term in query_terms):
                matches.append({
                    "id": nid, "text": text, "truth_value": tv,
                    "source": source, "source_hash": source_hash,
                    "date": date,
                    "depends_on": refs_by_node.get(nid, []),
                })

        if not matches:
            return f"No beliefs found matching '{query}'"

        parts = [f"Found {len(matches)} matching belief(s):", ""]
        for m in matches[:20]:
            parts.append(f"### {m['id']} [{m['truth_value']}]")
            parts.append(m["text"])
            if m["source"]:
                parts.append(f"- Source: {m['source']}")
            if m["source_hash"]:
                parts.append(f"- Source hash: {m['source_hash']}")
            if m["date"]:
                parts.append(f"- Date: {m['date']}")
            if m["depends_on"]:
                parts.append(f"- Depends on: {', '.join(m['depends_on'])}")
            parts.append("")

        return "\n".join(parts)

    def add_repo(self, name, path):
        """Register a repo name/path for source tracking."""
        pid = self.project_id
        key = f"repo:{name}"
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rms_network_meta (key, project_id, value) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (key, project_id) DO UPDATE SET value = EXCLUDED.value",
                (key, pid, path),
            )
        self.conn.commit()
        return {"name": name, "path": path}

    def list_repos(self):
        """List all registered repos."""
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT key, value FROM rms_network_meta "
                "WHERE project_id = %s AND key LIKE 'repo:%%'",
                (pid,),
            )
            rows = cur.fetchall()
        repos = {key[5:]: value for key, value in rows}
        return {"repos": repos}

    def _load_repos(self):
        """Load stored repos as a Path dict for source resolution."""
        from pathlib import Path
        result = self.list_repos()
        if result["repos"]:
            return {k: Path(v) for k, v in result["repos"].items()}
        return None

    def list_negative(self, visible_to=None, model="claude", skip_llm=False):
        """Find IN beliefs describing problems/defects/risks."""
        import sys
        from .api import NEGATIVE_TERMS, NEGATIVE_CLASSIFY_PROMPT, NEGATIVE_BATCH_SIZE

        pid = self.project_id

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, text, metadata FROM rms_nodes "
                "WHERE project_id = %s AND truth_value = 'IN' "
                "ORDER BY id",
                (pid,),
            )
            rows = cur.fetchall()

        in_nodes = []
        for nid, text, meta in rows:
            if isinstance(meta, str):
                meta = json.loads(meta) if meta else {}
            if visible_to is not None and not self._is_visible(meta, visible_to):
                continue
            in_nodes.append((nid, text))

        total = len(in_nodes)
        empty = {"negative": [], "count": 0, "candidates": 0, "total": total}

        if not in_nodes:
            return empty

        candidates = []
        for nid, text in in_nodes:
            text_lower = text.lower()
            if any(term in text_lower for term in NEGATIVE_TERMS):
                candidates.append((nid, text))

        if not candidates:
            return empty

        if skip_llm:
            return {
                "negative": [{"id": nid, "text": text} for nid, text in candidates],
                "count": len(candidates),
                "candidates": len(candidates),
                "total": total,
            }

        from .llm import invoke_model

        negative_ids = set()
        total_batches = (len(candidates) + NEGATIVE_BATCH_SIZE - 1) // NEGATIVE_BATCH_SIZE
        for i in range(0, len(candidates), NEGATIVE_BATCH_SIZE):
            batch = candidates[i:i + NEGATIVE_BATCH_SIZE]
            lines = [f"- [{nid}] `{text}`" for nid, text in batch]
            prompt = NEGATIVE_CLASSIFY_PROMPT.format(candidates="\n".join(lines))

            batch_num = i // NEGATIVE_BATCH_SIZE + 1
            print(f"  Classifying batch {batch_num}/{total_batches} "
                  f"({len(batch)} candidates)...", file=sys.stderr)

            response = invoke_model(prompt, model=model)

            for match in re.finditer(r"\[.*?\]", response, re.DOTALL):
                try:
                    ids = json.loads(match.group())
                    if isinstance(ids, list):
                        negative_ids.update(ids)
                        break
                except json.JSONDecodeError:
                    continue

        candidate_map = {nid: text for nid, text in candidates}
        negative = [
            {"id": nid, "text": candidate_map[nid]}
            for nid in negative_ids
            if nid in candidate_map
        ]
        negative.sort(key=lambda x: x["id"])

        return {
            "negative": negative,
            "count": len(negative),
            "candidates": len(candidates),
            "total": total,
        }

    def list_nodes(self, status=None, premises_only=False, has_dependents=False,
                   namespace=None, visible_to=None, label=None):
        pid = self.project_id
        conditions = ["n.project_id = %s"]
        params = [pid]

        if status:
            conditions.append("n.truth_value = %s")
            params.append(status)

        if premises_only:
            conditions.append(
                "NOT EXISTS (SELECT 1 FROM rms_justifications j "
                "WHERE j.node_id = n.id AND j.project_id = n.project_id)"
            )

        if namespace:
            conditions.append("n.id LIKE %s")
            params.append(f"{namespace}:%")

        if label:
            conditions.append(
                "EXISTS (SELECT 1 FROM rms_justifications j "
                "WHERE j.node_id = n.id AND j.project_id = n.project_id "
                "AND j.label = %s)"
            )
            params.append(label)

        where = " AND ".join(conditions)

        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT n.id, n.text, n.truth_value, n.metadata, "
                f"(SELECT COUNT(*) FROM rms_justifications j "
                f" WHERE j.node_id = n.id AND j.project_id = n.project_id) AS jcount "
                f"FROM rms_nodes n WHERE {where} ORDER BY n.id",
                params,
            )
            rows = cur.fetchall()

            # For has_dependents filter, we need the reverse lookup
            if has_dependents:
                all_ids = [r[0] for r in rows]
                dep_set = self._find_dependents(cur, all_ids) if all_ids else set()
                # dep_set contains nodes that ARE dependents, not nodes that HAVE dependents
                # We need nodes that appear in others' justifications
                ids_with_deps = set()
                if all_ids:
                    cur.execute(
                        "SELECT DISTINCT je.value FROM rms_justifications j, "
                        "jsonb_array_elements_text(j.antecedents) je(value) "
                        "WHERE j.project_id = %s "
                        "UNION "
                        "SELECT DISTINCT je.value FROM rms_justifications j, "
                        "jsonb_array_elements_text(j.outlist) je(value) "
                        "WHERE j.project_id = %s",
                        (pid, pid),
                    )
                    ids_with_deps = {r[0] for r in cur.fetchall()}

        nodes = []
        for row in rows:
            nid, text, tv, meta, jcount = row
            if isinstance(meta, str):
                meta = json.loads(meta)
            if visible_to is not None and not self._is_visible(meta, visible_to):
                continue
            if has_dependents and nid not in ids_with_deps:
                continue
            nodes.append({
                "id": nid,
                "text": text,
                "truth_value": tv,
                "justification_count": jcount,
            })

        return {"nodes": nodes, "count": len(nodes)}

    def list_gated(self, visible_to=None):
        pid = self.project_id

        with self.conn.cursor() as cur:
            # Find OUT nodes that have justifications with IN outlist nodes
            cur.execute(
                "SELECT n.id, n.text, n.metadata, j.outlist "
                "FROM rms_nodes n "
                "JOIN rms_justifications j ON j.node_id = n.id AND j.project_id = n.project_id "
                "WHERE n.project_id = %s AND n.truth_value = 'OUT' "
                "AND j.outlist != '[]'::jsonb",
                (pid,),
            )
            candidates = cur.fetchall()

            if not candidates:
                return {"blockers": {}, "gated_count": 0, "blocker_count": 0}

            # Collect all outlist node IDs
            all_outlist_ids = set()
            for row in candidates:
                outlist = row[3]
                if isinstance(outlist, str):
                    outlist = json.loads(outlist)
                all_outlist_ids.update(outlist)

            # Fetch truth values and text for outlist nodes
            if all_outlist_ids:
                cur.execute(
                    "SELECT id, text, truth_value, metadata FROM rms_nodes "
                    "WHERE project_id = %s AND id = ANY(%s)",
                    (pid, list(all_outlist_ids)),
                )
                outlist_info = {}
                for r in cur.fetchall():
                    meta = r[3]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    outlist_info[r[0]] = {"text": r[1], "truth_value": r[2], "metadata": meta}
            else:
                outlist_info = {}

        blockers: dict[str, dict] = {}
        for row in candidates:
            nid, text, meta, outlist = row
            if isinstance(meta, str):
                meta = json.loads(meta)
            if isinstance(outlist, str):
                outlist = json.loads(outlist)
            if meta.get("superseded_by"):
                continue
            if visible_to is not None and not self._is_visible(meta, visible_to):
                continue
            for outlist_id in outlist:
                info = outlist_info.get(outlist_id)
                if not info or info["truth_value"] != "IN":
                    continue
                if outlist_id not in blockers:
                    blockers[outlist_id] = {"text": info["text"], "gated": []}
                if not any(g["id"] == nid for g in blockers[outlist_id]["gated"]):
                    blockers[outlist_id]["gated"].append({"id": nid, "text": text})

        gated_count = sum(len(b["gated"]) for b in blockers.values())
        return {"blockers": blockers, "gated_count": gated_count, "blocker_count": len(blockers)}

    def topics(self, limit=20):
        from reasonsforge.api import _TOPIC_STOP_WORDS
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM rms_nodes WHERE project_id = %s", (pid,),
            )
            node_ids = [row[0] for row in cur.fetchall()]
        word_counts: dict[str, int] = {}
        for nid in node_ids:
            for word in re.split(r'[-._:]', nid):
                if word and len(word) > 2 and word not in _TOPIC_STOP_WORDS:
                    word_counts[word] = word_counts.get(word, 0) + 1
        ranked = sorted(word_counts, key=lambda w: (-word_counts[w], w))[:limit]
        return {
            "topics": [{"topic": t, "count": word_counts[t]} for t in ranked],
            "total_nodes": len(node_ids),
        }

    def export_network(self, visible_to=None):
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, text, truth_value, source, source_url, source_hash, "
                "date, metadata, created_at, updated_at, reviewed_at, "
                "verified_at, retracted_at "
                "FROM rms_nodes WHERE project_id = %s ORDER BY id",
                (pid,),
            )
            def _fmt_ts(val):
                return val.isoformat(timespec="seconds") if val else ""

            nodes = {}
            for row in cur.fetchall():
                nid, text, tv, source, source_url, source_hash, date, meta, \
                    created_at, updated_at, reviewed_at, verified_at, retracted_at = row
                if isinstance(meta, str):
                    meta = json.loads(meta)
                if visible_to is not None and not self._is_visible(meta, visible_to):
                    continue

                nodes[nid] = {
                    "text": text,
                    "truth_value": tv,
                    "justifications": [],
                    "source": source or "",
                    "source_url": source_url or "",
                    "source_hash": source_hash or "",
                    "date": date or "",
                    "metadata": {k: v for k, v in meta.items() if not k.startswith("_")},
                    "created_at": _fmt_ts(created_at),
                    "updated_at": _fmt_ts(updated_at),
                    "reviewed_at": _fmt_ts(reviewed_at),
                    "verified_at": _fmt_ts(verified_at),
                    "retracted_at": _fmt_ts(retracted_at),
                }

            if nodes:
                cur.execute(
                    "SELECT node_id, type, antecedents, outlist, label "
                    "FROM rms_justifications WHERE project_id = %s "
                    "AND node_id = ANY(%s) ORDER BY id",
                    (pid, list(nodes.keys())),
                )
                for row in cur.fetchall():
                    nid, jtype, ants, outs, label = row
                    if isinstance(ants, str):
                        ants = json.loads(ants)
                    if isinstance(outs, str):
                        outs = json.loads(outs)
                    if nid in nodes:
                        nodes[nid]["justifications"].append({
                            "type": jtype,
                            "antecedents": ants,
                            "outlist": outs,
                            "label": label or "",
                        })

            cur.execute(
                "SELECT id, nodes, discovered, resolution FROM rms_nogoods "
                "WHERE project_id = %s ORDER BY id",
                (pid,),
            )
            nogoods = []
            for row in cur.fetchall():
                ng_id, ng_nodes, discovered, resolution = row
                if isinstance(ng_nodes, str):
                    ng_nodes = json.loads(ng_nodes)
                if visible_to is not None:
                    if not all(n in nodes for n in ng_nodes):
                        continue
                nogoods.append({
                    "id": ng_id,
                    "nodes": ng_nodes,
                    "discovered": discovered or "",
                    "resolution": resolution or "",
                })

        repos = self.list_repos()["repos"]

        from .metadata import build_meta
        stored_meta = {}
        with self.conn.cursor() as cur2:
            cur2.execute(
                "SELECT key, value FROM rms_network_meta WHERE project_id = %s",
                (pid,),
            )
            for key, value in cur2.fetchall():
                if key != "next_nogood_id":
                    stored_meta[key] = value
        meta = build_meta(
            project_name=stored_meta.get("project_name", ""),
            node_count=len(nodes),
            created_at=stored_meta.get("created_at", ""),
        )
        return {"meta": meta, "nodes": nodes, "nogoods": nogoods, "repos": repos}

    def remove_justification(self, node_id, index):
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT truth_value FROM rms_nodes WHERE id = %s AND project_id = %s",
                (node_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{node_id}' not found")
            old_value = row[0]

            cur.execute(
                "SELECT id, type, antecedents, outlist, label "
                "FROM rms_justifications WHERE node_id = %s AND project_id = %s "
                "ORDER BY id",
                (node_id, pid),
            )
            justs = cur.fetchall()

            if not justs:
                raise ValueError(f"Node '{node_id}' is a premise (no justifications)")

            if index < 0 or index >= len(justs):
                raise IndexError(
                    f"Justification index {index} out of range "
                    f"(node has {len(justs)})"
                )

            if len(justs) == 1:
                raise ValueError(
                    f"Node '{node_id}' has only one justification; "
                    f"use 'convert-to-premise' or 'retract' instead"
                )

            target = justs[index]
            j_id, jtype, ants, outs, label = target
            if isinstance(ants, str):
                ants = json.loads(ants)
            if isinstance(outs, str):
                outs = json.loads(outs)

            cur.execute(
                "DELETE FROM rms_justifications WHERE id = %s",
                (j_id,),
            )

            new_value = self._compute_truth(cur, node_id)
            changed = []
            if old_value != new_value:
                cur.execute(
                    "UPDATE rms_nodes SET truth_value = %s "
                    "WHERE id = %s AND project_id = %s",
                    (new_value, node_id, pid),
                )
                changed.append(node_id)
                went_out, went_in = self._propagate(cur, node_id)
                changed.extend(went_out)
                changed.extend(went_in)

            self._log(cur, "remove-justification", node_id, new_value)

        self.conn.commit()
        return {
            "node_id": node_id,
            "old_truth_value": old_value,
            "new_truth_value": new_value,
            "removed": {"type": jtype, "antecedents": ants, "outlist": outs, "label": label or ""},
            "remaining": len(justs) - 1,
            "changed": changed,
        }

    def update_node(self, node_id, text=None, source=None, source_url=None,
                    example=None):
        if text is not None:
            raise ValueError(
                f"Text mutation is not allowed — beliefs are immutable propositions. "
                f"Use 'reasons supersede {node_id} --text \"...\"' to create a successor."
            )
        pid = self.project_id
        updates = []
        params = []
        updated_fields = []
        if source is not None:
            updates.append("source = %s")
            params.append(source)
            updated_fields.append("source")
        if source_url is not None:
            updates.append("source_url = %s")
            params.append(source_url)
            updated_fields.append("source_url")

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, metadata FROM rms_nodes WHERE id = %s AND project_id = %s",
                (node_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{node_id}' not found")

            if example is not None:
                meta = json.loads(row[1]) if row[1] else {}
                meta["example"] = example
                updates.append("metadata = %s")
                params.append(json.dumps(meta))
                updated_fields.append("example")

            if not updates:
                return {"node_id": node_id, "updated_fields": []}

            updates.append("updated_at = %s")
            params.append(datetime.now(timezone.utc).isoformat(timespec="seconds"))
            params.extend([node_id, pid])
            cur.execute(
                f"UPDATE rms_nodes SET {', '.join(updates)} "
                f"WHERE id = %s AND project_id = %s",
                params,
            )

        self.conn.commit()
        return {"node_id": node_id, "updated_fields": updated_fields}

    def set_metadata(self, node_id, key, value):
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT metadata FROM rms_nodes WHERE id = %s AND project_id = %s",
                (node_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{node_id}' not found")
            meta = json.loads(row[0]) if row[0] else {}
            meta[key] = value
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            cur.execute(
                "UPDATE rms_nodes SET metadata = %s, updated_at = %s "
                "WHERE id = %s AND project_id = %s",
                (json.dumps(meta), now, node_id, pid),
            )
        self.conn.commit()
        return {"node_id": node_id, "key": key}

    def convert_to_premise(self, node_id):
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT truth_value FROM rms_nodes WHERE id = %s AND project_id = %s",
                (node_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{node_id}' not found")
            old_value = row[0]

            cur.execute(
                "SELECT COUNT(*) FROM rms_justifications "
                "WHERE node_id = %s AND project_id = %s",
                (node_id, pid),
            )
            old_count = cur.fetchone()[0]

            cur.execute(
                "DELETE FROM rms_justifications "
                "WHERE node_id = %s AND project_id = %s",
                (node_id, pid),
            )

            changed = []
            if old_value != "IN":
                cur.execute(
                    "UPDATE rms_nodes SET truth_value = 'IN' "
                    "WHERE id = %s AND project_id = %s",
                    (node_id, pid),
                )
                changed.append(node_id)
                self._log(cur, "convert-to-premise", node_id, "IN")
                went_out, went_in = self._propagate(cur, node_id)
                changed.extend(went_out)
                changed.extend(went_in)
            else:
                self._log(cur, "convert-to-premise", node_id, "IN (unchanged)")

        self.conn.commit()
        return {
            "node_id": node_id,
            "old_justifications": old_count,
            "truth_value": "IN",
            "changed": changed,
        }

    def get_belief_set(self):
        """Return all node IDs currently IN."""
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM rms_nodes "
                "WHERE project_id = %s AND truth_value = 'IN'",
                (pid,),
            )
            return [row[0] for row in cur.fetchall()]

    def propagate(self):
        """Recompute truth values for all derived nodes to a fixpoint."""
        pid = self.project_id
        all_changed = set()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM rms_nodes WHERE project_id = %s",
                (pid,),
            )
            max_iterations = cur.fetchone()[0] + 1

            for _ in range(max_iterations):
                cur.execute(
                    "SELECT DISTINCT n.id, n.truth_value, n.metadata "
                    "FROM rms_nodes n "
                    "JOIN rms_justifications j "
                    "  ON j.node_id = n.id AND j.project_id = n.project_id "
                    "WHERE n.project_id = %s",
                    (pid,),
                )
                rows = cur.fetchall()

                changed_this_pass = []
                for node_id, old_tv, meta in rows:
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    if meta.get("_retracted"):
                        continue
                    new_tv = self._compute_truth(cur, node_id)
                    if old_tv != new_tv:
                        cur.execute(
                            "UPDATE rms_nodes SET truth_value = %s "
                            "WHERE id = %s AND project_id = %s",
                            (new_tv, node_id, pid),
                        )
                        self._log(cur, "recompute", node_id, new_tv)
                        changed_this_pass.append(node_id)

                if not changed_this_pass:
                    break
                all_changed.update(changed_this_pass)

        self.conn.commit()
        return {"changed": list(all_changed)}

    def supersede(self, old_id, new_id):
        """Mark old_id as superseded by new_id using the outlist mechanism."""
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT truth_value, metadata FROM rms_nodes "
                "WHERE id = %s AND project_id = %s",
                (old_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{old_id}' not found")
            old_tv, old_meta = row
            if isinstance(old_meta, str):
                old_meta = json.loads(old_meta)

            cur.execute(
                "SELECT metadata FROM rms_nodes "
                "WHERE id = %s AND project_id = %s",
                (new_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{new_id}' not found")
            new_meta = row[0]
            if isinstance(new_meta, str):
                new_meta = json.loads(new_meta)

            cur.execute(
                "SELECT id, outlist FROM rms_justifications "
                "WHERE node_id = %s AND project_id = %s",
                (old_id, pid),
            )
            just_rows = cur.fetchall()
            if just_rows:
                cur.execute(
                    "UPDATE rms_justifications "
                    "SET outlist = outlist || %s::jsonb "
                    "WHERE node_id = %s AND project_id = %s "
                    "AND NOT outlist @> %s::jsonb",
                    (json.dumps([new_id]), old_id, pid,
                     json.dumps([new_id])),
                )
            else:
                cur.execute(
                    "INSERT INTO rms_justifications "
                    "(node_id, project_id, type, antecedents, outlist, label) "
                    "VALUES (%s, %s, 'SL', '[]', %s, '')",
                    (old_id, pid, json.dumps([new_id])),
                )

            old_meta["superseded_by"] = new_id
            cur.execute(
                "UPDATE rms_nodes SET metadata = %s "
                "WHERE id = %s AND project_id = %s",
                (json.dumps(old_meta), old_id, pid),
            )

            supersedes = new_meta.get("supersedes", [])
            if old_id not in supersedes:
                supersedes.append(old_id)
            new_meta["supersedes"] = supersedes
            cur.execute(
                "UPDATE rms_nodes SET metadata = %s "
                "WHERE id = %s AND project_id = %s",
                (json.dumps(new_meta), new_id, pid),
            )

            new_tv = self._compute_truth(cur, old_id)
            changed = []
            if old_tv != new_tv:
                cur.execute(
                    "UPDATE rms_nodes SET truth_value = %s "
                    "WHERE id = %s AND project_id = %s",
                    (new_tv, old_id, pid),
                )
                self._log(cur, "supersede", old_id,
                          f"superseded by {new_id}")
                changed.append(old_id)
                went_out, went_in = self._propagate(cur, old_id)
                changed.extend(went_out)
                changed.extend(went_in)
            else:
                self._log(cur, "supersede", old_id,
                          f"superseded by {new_id} (unchanged)")

        self.conn.commit()
        return {"old_id": old_id, "new_id": new_id, "changed": changed}

    def summarize(self, summary_id, text, over, source=""):
        """Create a summary node that abstracts over a group of nodes."""
        pid = self.project_id
        now = datetime.now().isoformat(timespec="seconds")

        with self.conn.cursor() as cur:
            if over:
                cur.execute(
                    "SELECT id FROM rms_nodes "
                    "WHERE project_id = %s AND id = ANY(%s)",
                    (pid, list(over)),
                )
                found = {row[0] for row in cur.fetchall()}
                missing = set(over) - found
                if missing:
                    raise KeyError(
                        f"Node(s) not found: {', '.join(sorted(missing))}")

            cur.execute(
                "SELECT 1 FROM rms_nodes "
                "WHERE id = %s AND project_id = %s",
                (summary_id, pid),
            )
            if cur.fetchone():
                raise ValueError(f"Node '{summary_id}' already exists")

            metadata = {"summarizes": list(over)}
            cur.execute(
                "INSERT INTO rms_nodes "
                "(id, project_id, text, source, date, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (summary_id, pid, text, source, now, json.dumps(metadata)),
            )

            justifications = [{"antecedents": list(over), "outlist": []}]
            cur.execute(
                "INSERT INTO rms_justifications "
                "(node_id, project_id, type, antecedents, outlist, label) "
                "VALUES (%s, %s, 'SL', %s, '[]', 'summarizes')",
                (summary_id, pid, json.dumps(list(over))),
            )

            self._inherit_access_tags(cur, summary_id, justifications)

            truth = self._compute_truth(cur, summary_id)
            cur.execute(
                "UPDATE rms_nodes SET truth_value = %s "
                "WHERE id = %s AND project_id = %s",
                (truth, summary_id, pid),
            )
            self._log(cur, "add", summary_id, truth)

            for nid in over:
                cur.execute(
                    "SELECT metadata FROM rms_nodes "
                    "WHERE id = %s AND project_id = %s",
                    (nid, pid),
                )
                row = cur.fetchone()
                meta = row[0] if row else {}
                if isinstance(meta, str):
                    meta = json.loads(meta)
                covered = meta.get("summarized_by", [])
                if summary_id not in covered:
                    covered.append(summary_id)
                meta["summarized_by"] = covered
                cur.execute(
                    "UPDATE rms_nodes SET metadata = %s "
                    "WHERE id = %s AND project_id = %s",
                    (json.dumps(meta), nid, pid),
                )

        self.conn.commit()
        return {
            "summary_id": summary_id,
            "over": list(over),
            "truth_value": truth,
        }

    def get_log(self, last=None):
        pid = self.project_id
        with self.conn.cursor() as cur:
            if last:
                cur.execute(
                    "SELECT timestamp, action, target, value FROM rms_propagation_log "
                    "WHERE project_id = %s ORDER BY id DESC LIMIT %s",
                    (pid, last),
                )
                entries = [
                    {"timestamp": r[0], "action": r[1], "target": r[2], "value": r[3]}
                    for r in reversed(cur.fetchall())
                ]
            else:
                cur.execute(
                    "SELECT timestamp, action, target, value FROM rms_propagation_log "
                    "WHERE project_id = %s ORDER BY id",
                    (pid,),
                )
                entries = [
                    {"timestamp": r[0], "action": r[1], "target": r[2], "value": r[3]}
                    for r in cur.fetchall()
                ]
        return {"entries": entries}

    def compact(self, budget=500, truncate=True, visible_to=None):
        from datetime import date as _date

        pid = self.project_id

        with self.conn.cursor() as cur:
            # Fetch all nodes
            cur.execute(
                "SELECT id, text, truth_value, source, metadata "
                "FROM rms_nodes WHERE project_id = %s ORDER BY id",
                (pid,),
            )
            all_nodes = []
            for row in cur.fetchall():
                nid, text, tv, source, meta = row
                if isinstance(meta, str):
                    meta = json.loads(meta)
                if visible_to is not None and not self._is_visible(meta, visible_to):
                    continue
                all_nodes.append({
                    "id": nid, "text": text, "truth_value": tv,
                    "source": source, "metadata": meta,
                })

            # Fetch nogoods (filter by visible_to — hide nogoods referencing inaccessible nodes)
            cur.execute(
                "SELECT id, nodes, resolution FROM rms_nogoods "
                "WHERE project_id = %s ORDER BY id",
                (pid,),
            )
            visible_ids = {n["id"] for n in all_nodes}
            nogoods = []
            for row in cur.fetchall():
                ng_id, nodes, resolution = row
                if isinstance(nodes, str):
                    nodes = json.loads(nodes)
                if visible_to is not None and not all(n in visible_ids for n in nodes):
                    continue
                nogoods.append({"id": ng_id, "nodes": nodes, "resolution": resolution or ""})

            # Fetch dependent counts per node
            cur.execute(
                "SELECT je.value AS referenced_id, COUNT(DISTINCT j.node_id) AS dep_count "
                "FROM rms_justifications j, "
                "jsonb_array_elements_text(j.antecedents) je(value) "
                "WHERE j.project_id = %s GROUP BY je.value",
                (pid,),
            )
            dep_counts = {r[0]: r[1] for r in cur.fetchall()}
            cur.execute(
                "SELECT je.value AS referenced_id, COUNT(DISTINCT j.node_id) AS dep_count "
                "FROM rms_justifications j, "
                "jsonb_array_elements_text(j.outlist) je(value) "
                "WHERE j.project_id = %s GROUP BY je.value",
                (pid,),
            )
            for row in cur.fetchall():
                dep_counts[row[0]] = dep_counts.get(row[0], 0) + row[1]

            # Fetch first justification antecedents per node (for IN display)
            cur.execute(
                "SELECT DISTINCT ON (node_id) node_id, antecedents, label "
                "FROM rms_justifications "
                "WHERE project_id = %s ORDER BY node_id, id",
                (pid,),
            )
            first_just = {}
            for row in cur.fetchall():
                nid, ants, label = row
                if isinstance(ants, str):
                    ants = json.loads(ants)
                first_just[nid] = {"antecedents": ants, "label": label}

        # Split nodes
        in_nodes = [n for n in all_nodes if n["truth_value"] == "IN"]
        out_nodes = [n for n in all_nodes if n["truth_value"] == "OUT"]

        in_nodes.sort(key=lambda n: dep_counts.get(n["id"], 0), reverse=True)

        today = _date.today().isoformat()
        in_count = len(in_nodes)
        out_count = len(out_nodes)
        total = in_count + out_count
        nogood_count = len(nogoods)

        lines = [
            f"# Belief State Summary ({today})",
            f"# {total} nodes tracked | {nogood_count} nogoods | {in_count} IN | {out_count} OUT",
            "",
        ]

        def _text(text):
            if truncate and len(text) > 80:
                return text[:77] + "..."
            return text

        def _estimate_tokens(text):
            return max(1, len(text) // 4)

        footer_tokens = _estimate_tokens(f"Token count: ~{budget} / {budget} budget")
        _char_count = sum(len(l) for l in lines) + len(lines) - 1

        def _add_line(line):
            nonlocal _char_count
            lines.append(line)
            _char_count += 1 + len(line)

        def _current_tokens():
            return max(1, _char_count // 4)

        def _over_budget(line):
            return _current_tokens() + _estimate_tokens(line) + footer_tokens > budget

        # Section 1: Nogoods
        if nogoods and not _over_budget("## Nogoods"):
            _add_line("## Nogoods")
            added_nogoods = 0
            for ng in nogoods:
                res = f" — {ng['resolution']}" if ng["resolution"] else ""
                line = f"- {ng['id']}: {', '.join(ng['nodes'])}{res}"
                if _over_budget(line):
                    remaining = len(nogoods) - added_nogoods
                    _add_line(f"  ... ({remaining} more nogoods omitted)")
                    break
                _add_line(line)
                added_nogoods += 1
            _add_line("")

        # Section 2: OUT nodes
        if out_nodes and not _over_budget("## OUT (retracted)"):
            _add_line("## OUT (retracted)")
            added_out = 0
            for node in out_nodes:
                reason = ""
                retract_reason = node["metadata"].get("retract_reason") or node["metadata"].get("stale_reason")
                if retract_reason:
                    reason = f" (stale: {retract_reason[:60]})"
                elif node["metadata"].get("superseded_by"):
                    reason = f" (superseded by: {node['metadata']['superseded_by']})"
                line = f"- {node['id']}: {_text(node['text'])}{reason}"
                if _over_budget(line):
                    remaining = len(out_nodes) - added_out
                    _add_line(f"  ... ({remaining} more OUT nodes omitted)")
                    break
                _add_line(line)
                added_out += 1
            _add_line("")

        # Section 3: IN nodes
        if in_nodes and not _over_budget("## IN (active)"):
            covered_by_summary = set()
            summary_nodes = []
            regular_nodes = []
            for node in in_nodes:
                summarizes = node["metadata"].get("summarizes")
                if summarizes:
                    summary_nodes.append(node)
                    for covered_id in summarizes:
                        covered_by_summary.add(covered_id)
                else:
                    regular_nodes.append(node)

            visible_nodes = summary_nodes + [
                n for n in regular_nodes if n["id"] not in covered_by_summary
            ]
            visible_nodes.sort(key=lambda n: dep_counts.get(n["id"], 0), reverse=True)

            hidden_count = len(in_nodes) - len(visible_nodes)

            _add_line("## IN (active)")
            added = 0

            for node in visible_nodes:
                is_summary = bool(node["metadata"].get("summarizes"))
                prefix = "[summary] " if is_summary else ""
                deps = ""
                j = first_just.get(node["id"])
                if j and j["antecedents"] and j["label"] != "summarizes":
                    deps = f" <- {', '.join(j['antecedents'])}"
                dep_count = dep_counts.get(node["id"], 0)
                dep_info = f" ({dep_count} dependents)" if dep_count else ""
                summarizes = node["metadata"].get("summarizes", [])
                sum_info = f" (covers {len(summarizes)} nodes)" if summarizes else ""
                line = f"- {prefix}{node['id']}: {_text(node['text'])}{deps}{dep_info}{sum_info}"

                if _over_budget(line):
                    remaining = len(visible_nodes) - added
                    _add_line(f"  ... ({remaining} more IN nodes omitted)")
                    break

                _add_line(line)
                added += 1

            if hidden_count:
                _add_line(f"  ({hidden_count} nodes hidden by summaries)")
            _add_line("")

        lines.append(f"Token count: ~{_current_tokens()} / {budget} budget")

        return "\n".join(lines)

    # ── Nogoods + explain ───────────────────────────────────────

    def add_nogood(self, node_ids):
        pid = self.project_id

        with self.conn.cursor() as cur:
            # Verify all nodes exist
            cur.execute(
                "SELECT id, truth_value FROM rms_nodes WHERE project_id = %s AND id = ANY(%s)",
                (pid, node_ids),
            )
            found = {r[0]: r[1] for r in cur.fetchall()}
            for nid in node_ids:
                if nid not in found:
                    raise KeyError(f"Node '{nid}' not found")

            # Get next nogood ID
            cur.execute(
                "SELECT value FROM rms_network_meta "
                "WHERE key = 'next_nogood_id' AND project_id = %s",
                (pid,),
            )
            row = cur.fetchone()
            next_id = int(row[0]) if row else 1
            nogood_id = f"nogood-{next_id:03d}"

            cur.execute(
                "INSERT INTO rms_network_meta (key, project_id, value) "
                "VALUES ('next_nogood_id', %s, %s) "
                "ON CONFLICT (key, project_id) DO UPDATE SET value = EXCLUDED.value",
                (pid, str(next_id + 1)),
            )

            cur.execute(
                "INSERT INTO rms_nogoods (id, project_id, nodes, discovered) "
                "VALUES (%s, %s, %s, %s)",
                (nogood_id, pid, json.dumps(node_ids),
                 datetime.now().isoformat(timespec="seconds")),
            )
            self._log(cur, "nogood", nogood_id, str(node_ids))

            # Check if contradiction is active
            all_in = all(found.get(nid) == "IN" for nid in node_ids)
            if not all_in:
                self.conn.commit()
                return {"nogood_id": nogood_id, "changed": [], "backtracked_to": None}

            # Dependency-directed backtracking
            culprits = self._find_culprits_internal(cur, node_ids)

            if culprits:
                victim_id = culprits[0]["premise"]
                self._log(cur, "backtrack", victim_id, f"culprit for {nogood_id}")
            else:
                # Fallback: retract node with fewest dependents
                dep_counts = []
                for nid in node_ids:
                    deps = self._find_dependents(cur, [nid])
                    dep_counts.append((nid, len(deps)))
                dep_counts.sort(key=lambda x: x[1])
                victim_id = dep_counts[0][0]

            # Retract the victim
            cur.execute(
                "SELECT metadata FROM rms_nodes WHERE id = %s AND project_id = %s",
                (victim_id, pid),
            )
            meta = cur.fetchone()[0]
            if isinstance(meta, str):
                meta = json.loads(meta)
            meta["_retracted"] = True
            cur.execute(
                "UPDATE rms_nodes SET truth_value = 'OUT', metadata = %s "
                "WHERE id = %s AND project_id = %s",
                (json.dumps(meta), victim_id, pid),
            )
            self._log(cur, "retract", victim_id, f"backtrack for {nogood_id}")

            went_out, went_in = self._propagate(cur, victim_id)

        self.conn.commit()
        changed = [victim_id] + went_out + went_in
        return {"nogood_id": nogood_id, "changed": changed, "backtracked_to": victim_id}

    def find_culprits(self, node_ids):
        with self.conn.cursor() as cur:
            culprits = self._find_culprits_internal(cur, node_ids)
        return {"culprits": culprits}

    def explain_node(self, node_id, visible_to=None):
        pid = self.project_id

        with self.conn.cursor() as cur:
            steps = self._explain_recursive(cur, node_id, visible_to, set())
        return {"steps": steps}

    def trace_assumptions(self, node_id, visible_to=None):
        pid = self.project_id
        premises = []
        visited = set()

        with self.conn.cursor() as cur:
            self._trace_assumptions_recursive(cur, node_id, premises, visited)

        return {"node_id": node_id, "premises": premises}

    def trace_access_tags(self, node_id, visible_to=None):
        """Trace access_tags union through the dependency chain."""
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT metadata FROM rms_nodes "
                "WHERE id = %s AND project_id = %s",
                (node_id, pid),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Node '{node_id}' not found")
            meta = row[0]
            if isinstance(meta, str):
                meta = json.loads(meta)
            if visible_to is not None and not self._is_visible(meta, visible_to):
                raise PermissionError(
                    f"Node '{node_id}' requires access tags "
                    f"not in {visible_to}")

            all_tags = set()
            visited = set()
            self._trace_access_tags_recursive(
                cur, node_id, all_tags, visited)

        return {"node_id": node_id, "access_tags": sorted(all_tags)}

    def ensure_namespace(self, namespace):
        """Ensure a namespace premise node exists (namespace:active)."""
        pid = self.project_id
        active_id = f"{namespace}:active"
        now = datetime.now().isoformat(timespec="seconds")

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM rms_nodes "
                "WHERE id = %s AND project_id = %s",
                (active_id, pid),
            )
            if cur.fetchone():
                return {"namespace": namespace,
                        "active_node": active_id, "created": False}

            metadata = {"agent": namespace, "role": "agent_premise"}
            cur.execute(
                "INSERT INTO rms_nodes "
                "(id, project_id, text, source, date, metadata, truth_value) "
                "VALUES (%s, %s, %s, '', %s, %s, 'IN')",
                (active_id, pid,
                 f"Agent '{namespace}' beliefs are trusted",
                 now, json.dumps(metadata)),
            )
            self._log(cur, "add", active_id, "IN")

        self.conn.commit()
        return {"namespace": namespace,
                "active_node": active_id, "created": True}

    def list_namespaces(self):
        """List all namespaces with belief counts."""
        pid = self.project_id
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, truth_value FROM rms_nodes "
                "WHERE project_id = %s AND id LIKE '%%:active' "
                "AND metadata->>'role' = 'agent_premise' "
                "ORDER BY id",
                (pid,),
            )
            ns_rows = cur.fetchall()

            namespaces = []
            for active_id, tv in ns_rows:
                ns = active_id[:-len(":active")]
                cur.execute(
                    "SELECT COUNT(*), "
                    "COUNT(*) FILTER (WHERE truth_value = 'IN') "
                    "FROM rms_nodes "
                    "WHERE project_id = %s AND id LIKE %s AND id != %s",
                    (pid, f"{ns}:%", active_id),
                )
                total, in_count = cur.fetchone()
                namespaces.append({
                    "namespace": ns,
                    "active_node": active_id,
                    "active": tv == "IN",
                    "total_beliefs": total,
                    "in_beliefs": in_count,
                })

        return {"namespaces": namespaces}

    # ── Import operations ──────────────────────────────────────

    def import_json(self, data):
        """Import a network from parsed JSON export data.

        Uses topological sort to add nodes in dependency order,
        then propagates truth values and applies overrides.
        """
        pid = self.project_id
        remaining = dict(data.get("nodes", {}))
        nodes_imported = 0
        skipped = 0

        imported_meta = data.get("meta")
        if imported_meta and isinstance(imported_meta, dict):
            with self.conn.cursor() as cur:
                for key in ("schema_version", "project_name", "created_at"):
                    if key in imported_meta and imported_meta[key]:
                        cur.execute(
                            "INSERT INTO rms_network_meta (key, project_id, value) "
                            "VALUES (%s, %s, %s) "
                            "ON CONFLICT (key, project_id) DO UPDATE SET value = EXCLUDED.value",
                            (key, pid, str(imported_meta[key])),
                        )
            self.conn.commit()

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM rms_nodes WHERE project_id = %s", (pid,),
            )
            added = {r[0] for r in cur.fetchall()}

            all_node_ids = set(remaining.keys())
            max_passes = len(remaining) + 1
            for _ in range(max_passes):
                if not remaining:
                    break
                next_remaining = {}
                for nid, ndata in remaining.items():
                    if nid in added:
                        skipped += 1
                        continue
                    all_deps = set()
                    for j in ndata.get("justifications", []):
                        all_deps.update(j.get("antecedents", []))
                        all_deps.update(j.get("outlist", []))
                    deps_in_data = {d for d in all_deps if d in all_node_ids}
                    if all(d in added for d in deps_in_data):
                        justs = ndata.get("justifications", []) or None
                        meta = {k: v for k, v in ndata.get("metadata", {}).items()
                                if not k.startswith("_")}
                        self._add_node_raw(
                            cur, nid, ndata.get("text", ""),
                            justifications=justs,
                            source=ndata.get("source", ""),
                            source_url=ndata.get("source_url", ""),
                            source_hash=ndata.get("source_hash", ""),
                            date=ndata.get("date", ""),
                            metadata=meta,
                            created_at=ndata.get("created_at", ""),
                            updated_at=ndata.get("updated_at", ""),
                        )
                        added.add(nid)
                        nodes_imported += 1
                    else:
                        next_remaining[nid] = ndata
                if len(next_remaining) == len(remaining):
                    for nid, ndata in next_remaining.items():
                        if nid in added:
                            continue
                        justs = ndata.get("justifications", []) or None
                        meta = {k: v for k, v in ndata.get("metadata", {}).items()
                                if not k.startswith("_")}
                        self._add_node_raw(
                            cur, nid, ndata.get("text", ""),
                            justifications=justs,
                            source=ndata.get("source", ""),
                            source_url=ndata.get("source_url", ""),
                            source_hash=ndata.get("source_hash", ""),
                            date=ndata.get("date", ""),
                            metadata=meta,
                            created_at=ndata.get("created_at", ""),
                            updated_at=ndata.get("updated_at", ""),
                        )
                        added.add(nid)
                        nodes_imported += 1
                    break
                remaining = next_remaining

        self.conn.commit()

        self.propagate()

        with self.conn.cursor() as cur:
            for nid, ndata in data.get("nodes", {}).items():
                target_tv = ndata.get("truth_value", "IN")
                cur.execute(
                    "SELECT truth_value FROM rms_nodes "
                    "WHERE id = %s AND project_id = %s",
                    (nid, pid),
                )
                row = cur.fetchone()
                if row and row[0] != target_tv:
                    if target_tv == "OUT":
                        cur.execute(
                            "UPDATE rms_nodes SET truth_value = 'OUT', "
                            "metadata = metadata || %s "
                            "WHERE id = %s AND project_id = %s",
                            (json.dumps({"_retracted": True}), nid, pid),
                        )
                        self._log(cur, "retract", nid, "OUT")
                    else:
                        cur.execute(
                            "UPDATE rms_nodes SET truth_value = 'IN', "
                            "metadata = metadata - '_retracted' "
                            "WHERE id = %s AND project_id = %s",
                            (nid, pid),
                        )
                        self._log(cur, "assert", nid, "IN")

                # Restore exact timestamps AFTER truth-value fixup
                ts_updates = []
                ts_params = []
                for ts_col in ("updated_at", "reviewed_at", "verified_at", "retracted_at"):
                    ts_val = ndata.get(ts_col, "")
                    if ts_val:
                        ts_updates.append(f"{ts_col} = %s")
                        ts_params.append(ts_val)
                if ts_updates:
                    ts_params.extend([nid, pid])
                    cur.execute(
                        f"UPDATE rms_nodes SET {', '.join(ts_updates)} "
                        f"WHERE id = %s AND project_id = %s",
                        ts_params,
                    )

            nogoods_imported = 0
            for ng_data in data.get("nogoods", []):
                cur.execute(
                    "SELECT 1 FROM rms_nogoods WHERE id = %s AND project_id = %s",
                    (ng_data["id"], pid),
                )
                if cur.fetchone():
                    continue
                cur.execute(
                    "INSERT INTO rms_nogoods (id, project_id, nodes, discovered, resolution) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (ng_data["id"], pid,
                     json.dumps(ng_data.get("nodes", [])),
                     ng_data.get("discovered", ""),
                     ng_data.get("resolution", "")),
                )
                nogoods_imported += 1

                m = re.fullmatch(r"nogood-(\d+)", ng_data["id"])
                if m:
                    next_id = int(m.group(1)) + 1
                    cur.execute(
                        "INSERT INTO rms_network_meta (key, project_id, value) "
                        "VALUES ('next_nogood_id', %s, %s) "
                        "ON CONFLICT (key, project_id) DO UPDATE "
                        "SET value = GREATEST(EXCLUDED.value::int, "
                        "rms_network_meta.value::int)::text",
                        (pid, str(next_id)),
                    )

        self.conn.commit()
        return {"nodes_imported": nodes_imported, "nogoods_imported": nogoods_imported}

    def import_beliefs(self, beliefs_text, nogoods_text=None):
        """Import beliefs from markdown text."""
        from .import_beliefs import parse_beliefs, parse_nogoods, strip_frontmatter

        beliefs_text, frontmatter = strip_frontmatter(beliefs_text)
        if frontmatter:
            with self.conn.cursor() as cur:
                for key in ("schema_version", "project_name", "created_at"):
                    if key in frontmatter and frontmatter[key]:
                        cur.execute(
                            "INSERT INTO rms_network_meta (key, project_id, value) "
                            "VALUES (%s, %s, %s) "
                            "ON CONFLICT (key, project_id) DO UPDATE SET value = EXCLUDED.value",
                            (key, self.project_id, frontmatter[key]),
                        )
            self.conn.commit()

        pid = self.project_id
        claims = parse_beliefs(beliefs_text)

        claim_by_id = {c["id"]: c for c in claims}
        ordered = []
        added_ids = set()
        remaining = list(claims)
        max_passes = len(remaining) + 1
        for _ in range(max_passes):
            if not remaining:
                break
            next_remaining = []
            for c in remaining:
                deps_in_registry = [d for d in c["depends_on"] if d in claim_by_id]
                if all(d in added_ids for d in deps_in_registry):
                    ordered.append(c)
                    added_ids.add(c["id"])
                else:
                    next_remaining.append(c)
            if len(next_remaining) == len(remaining):
                ordered.extend(next_remaining)
                break
            remaining = next_remaining

        imported = 0
        skipped = 0
        retracted = 0
        retract_after = []

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM rms_nodes WHERE project_id = %s", (pid,),
            )
            existing = {r[0] for r in cur.fetchall()}

            for claim in ordered:
                if claim["id"] in existing:
                    skipped += 1
                    continue

                justifications = None
                deps_in_network = [d for d in claim["depends_on"] if d in claim_by_id]
                unless_in_network = [u for u in claim.get("unless", [])
                                     if u in claim_by_id]
                if deps_in_network or unless_in_network:
                    justifications = [{
                        "type": "SL",
                        "antecedents": deps_in_network,
                        "outlist": unless_in_network,
                        "label": f"imported from beliefs: {claim['type']}",
                    }]

                metadata = {}
                if claim["type"]:
                    metadata["beliefs_type"] = claim["type"]
                if claim["stale_reason"]:
                    metadata["stale_reason"] = claim["stale_reason"]
                if claim["superseded_by"]:
                    metadata["superseded_by"] = claim["superseded_by"]

                self._add_node_raw(
                    cur, claim["id"], claim["text"],
                    justifications=justifications,
                    source=claim["source"],
                    source_hash=claim["source_hash"],
                    date=claim["date"],
                    metadata=metadata,
                )
                existing.add(claim["id"])
                imported += 1

                if claim["status"] in ("STALE", "OUT"):
                    retract_after.append(claim["id"])

        self.conn.commit()
        self.propagate()

        if retract_after:
            with self.conn.cursor() as cur:
                for nid in retract_after:
                    cur.execute(
                        "UPDATE rms_nodes SET truth_value = 'OUT', "
                        "metadata = metadata || %s "
                        "WHERE id = %s AND project_id = %s",
                        (json.dumps({"_retracted": True}), nid, pid),
                    )
                    self._log(cur, "retract", nid, "OUT")
                    retracted += 1
            self.conn.commit()
            self.propagate()

        nogoods_imported = 0
        if nogoods_text:
            nogoods = parse_nogoods(nogoods_text)
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM rms_nodes WHERE project_id = %s", (pid,),
                )
                all_nodes = {r[0] for r in cur.fetchall()}

                for ng in nogoods:
                    valid_nodes = [a for a in ng["affects"] if a in all_nodes]
                    if len(valid_nodes) >= 2:
                        cur.execute(
                            "SELECT 1 FROM rms_nogoods WHERE id = %s AND project_id = %s",
                            (ng["id"], pid),
                        )
                        if not cur.fetchone():
                            cur.execute(
                                "INSERT INTO rms_nogoods (id, project_id, nodes, "
                                "discovered, resolution) VALUES (%s, %s, %s, %s, %s)",
                                (ng["id"], pid, json.dumps(valid_nodes),
                                 ng["discovered"], ng["resolution"]),
                            )
                            m = re.fullmatch(r"nogood-(\d+)", ng["id"])
                            if m:
                                next_id = int(m.group(1)) + 1
                                cur.execute(
                                    "INSERT INTO rms_network_meta (key, project_id, value) "
                                    "VALUES ('next_nogood_id', %s, %s) "
                                    "ON CONFLICT (key, project_id) DO UPDATE "
                                    "SET value = GREATEST(EXCLUDED.value::int, "
                                    "rms_network_meta.value::int)::text",
                                    (pid, str(next_id)),
                                )
                            nogoods_imported += 1
            self.conn.commit()

        return {
            "claims_imported": imported,
            "claims_skipped": skipped,
            "claims_retracted": retracted,
            "nogoods_imported": nogoods_imported,
        }

    def import_agent(self, agent_name, claims, nogoods, source_path=""):
        """Import another agent's beliefs with namespacing.

        Takes pre-normalized claim dicts from import_agent._normalize_*.
        """
        from .import_agent import _topo_sort_claims

        pid = self.project_id
        prefix = f"{agent_name}:"
        active_id = f"{agent_name}:active"
        inactive_id = f"{agent_name}:inactive"

        self.ensure_namespace(agent_name)
        created_premise = True

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM rms_nodes WHERE id = %s AND project_id = %s",
                (inactive_id, pid),
            )
            if not cur.fetchone():
                self._add_node_raw(
                    cur, inactive_id,
                    f"Agent '{agent_name}' kill switch — IN when active is OUT",
                    justifications=[{
                        "type": "SL", "antecedents": [],
                        "outlist": [active_id], "label": "",
                    }],
                    source=source_path,
                    metadata={"agent": agent_name, "role": "agent_inactive"},
                )
            else:
                created_premise = False
        self.conn.commit()

        ordered = _topo_sort_claims(claims)
        imported = 0
        skipped = 0
        retract_after = []

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM rms_nodes WHERE project_id = %s", (pid,),
            )
            existing = {r[0] for r in cur.fetchall()}

            for claim in ordered:
                node_id = f"{prefix}{claim['id']}"
                if node_id in existing:
                    skipped += 1
                    continue

                justs = []
                for rj in claim["raw_justifications"]:
                    antecedents = [f"{prefix}{a}" for a in rj["antecedents"]]
                    outlist = [inactive_id] + [f"{prefix}{o}" for o in rj["outlist"]]
                    label = rj.get("label") or f"imported from agent: {agent_name}"
                    justs.append({
                        "type": rj["type"], "antecedents": antecedents,
                        "outlist": outlist, "label": label,
                    })
                if not justs:
                    justs = [{
                        "type": "SL", "antecedents": [],
                        "outlist": [inactive_id],
                        "label": f"imported from agent: {agent_name}",
                    }]

                metadata = claim.get("metadata", {}).copy()
                metadata.update({
                    "agent": agent_name,
                    "original_id": claim["id"],
                    "imported_from": source_path,
                })

                self._add_node_raw(
                    cur, node_id, claim["text"],
                    justifications=justs,
                    source=claim.get("source", ""),
                    source_url=claim.get("source_url", ""),
                    source_hash=claim.get("source_hash", ""),
                    date=claim.get("date", ""),
                    metadata=metadata,
                )
                existing.add(node_id)
                imported += 1

                if claim["is_out"] and not claim["raw_justifications"]:
                    retract_after.append(node_id)
                elif claim["is_out"]:
                    cur.execute(
                        "UPDATE rms_nodes SET truth_value = 'OUT', "
                        "metadata = metadata || %s "
                        "WHERE id = %s AND project_id = %s",
                        (json.dumps({"_retracted": True}), node_id, pid),
                    )

        self.conn.commit()
        self.propagate()

        retracted = 0
        if retract_after:
            with self.conn.cursor() as cur:
                for nid in retract_after:
                    cur.execute(
                        "UPDATE rms_nodes SET truth_value = 'OUT', "
                        "metadata = metadata || %s "
                        "WHERE id = %s AND project_id = %s",
                        (json.dumps({"_retracted": True}), nid, pid),
                    )
                    self._log(cur, "retract", nid, "OUT")
                    retracted += 1
            self.conn.commit()
            self.propagate()

        nogoods_imported = self._import_agent_nogoods(prefix=prefix,
                                                      nogoods=nogoods)

        return {
            "agent": agent_name,
            "prefix": prefix,
            "active_node": active_id,
            "created_premise": created_premise,
            "claims_imported": imported,
            "claims_skipped": skipped,
            "claims_retracted": retracted,
            "claims_propagated": 0,
            "nogoods_imported": nogoods_imported,
        }

    def sync_agent(self, agent_name, claims, nogoods, source_path=""):
        """Sync another agent's beliefs (remote wins).

        Takes pre-normalized claim dicts from import_agent._normalize_*.
        """
        from .import_agent import _topo_sort_claims

        pid = self.project_id
        prefix = f"{agent_name}:"
        active_id = f"{agent_name}:active"
        inactive_id = f"{agent_name}:inactive"

        self.ensure_namespace(agent_name)
        created_premise = True

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM rms_nodes WHERE id = %s AND project_id = %s",
                (inactive_id, pid),
            )
            if not cur.fetchone():
                self._add_node_raw(
                    cur, inactive_id,
                    f"Agent '{agent_name}' kill switch — IN when active is OUT",
                    justifications=[{
                        "type": "SL", "antecedents": [],
                        "outlist": [active_id], "label": "",
                    }],
                    source=source_path,
                    metadata={"agent": agent_name, "role": "agent_inactive"},
                )
            else:
                created_premise = False
        self.conn.commit()

        remote_ids = {f"{prefix}{c['id']}" for c in claims}
        infra_ids = {active_id, inactive_id}

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM rms_nodes WHERE project_id = %s AND id LIKE %s",
                (pid, f"{prefix}%"),
            )
            local_agent_ids = {r[0] for r in cur.fetchall()} - infra_ids

        ordered = _topo_sort_claims(claims)

        beliefs_added = 0
        beliefs_updated = 0
        beliefs_unchanged = 0
        retract_after = []
        assert_after = []

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM rms_nodes WHERE project_id = %s", (pid,),
            )
            existing = {r[0] for r in cur.fetchall()}

            for claim in ordered:
                node_id = f"{prefix}{claim['id']}"
                is_out = claim["is_out"]

                justs = []
                for rj in claim["raw_justifications"]:
                    antecedents = [f"{prefix}{a}" for a in rj["antecedents"]]
                    outlist = [inactive_id] + [f"{prefix}{o}" for o in rj["outlist"]]
                    label = rj.get("label") or f"imported from agent: {agent_name}"
                    justs.append({
                        "type": rj["type"], "antecedents": antecedents,
                        "outlist": outlist, "label": label,
                    })
                if not justs:
                    justs = [{
                        "type": "SL", "antecedents": [],
                        "outlist": [inactive_id],
                        "label": f"imported from agent: {agent_name}",
                    }]

                if node_id in existing:
                    cur.execute(
                        "SELECT text, source, source_url, source_hash, date, "
                        "truth_value, metadata FROM rms_nodes "
                        "WHERE id = %s AND project_id = %s",
                        (node_id, pid),
                    )
                    row = cur.fetchone()
                    old_text, old_source, old_source_url, old_source_hash, \
                        old_date, old_tv, old_meta = row
                    if isinstance(old_meta, str):
                        old_meta = json.loads(old_meta)
                    changed = False

                    updates = {}
                    if old_text != claim["text"]:
                        updates["text"] = claim["text"]
                        changed = True
                    if claim.get("source") and old_source != claim["source"]:
                        updates["source"] = claim["source"]
                        changed = True
                    if claim.get("source_url") and old_source_url != claim.get("source_url"):
                        updates["source_url"] = claim["source_url"]
                        changed = True
                    if claim.get("source_hash") and old_source_hash != claim["source_hash"]:
                        updates["source_hash"] = claim["source_hash"]
                        changed = True
                    if claim.get("date") and old_date != claim["date"]:
                        updates["date"] = claim["date"]
                        changed = True

                    if updates:
                        set_parts = [f"{k} = %s" for k in updates]
                        cur.execute(
                            f"UPDATE rms_nodes SET {', '.join(set_parts)} "
                            f"WHERE id = %s AND project_id = %s",
                            (*updates.values(), node_id, pid),
                        )

                    new_meta = old_meta.copy()
                    for k, v in claim.get("metadata", {}).items():
                        if not k.startswith("_"):
                            new_meta[k] = v
                    new_meta["imported_from"] = source_path

                    cur.execute(
                        "SELECT type, antecedents, outlist, label "
                        "FROM rms_justifications WHERE node_id = %s AND project_id = %s "
                        "ORDER BY id",
                        (node_id, pid),
                    )
                    old_justs = []
                    for jrow in cur.fetchall():
                        jtype, ants, outs, jlabel = jrow
                        if isinstance(ants, str):
                            ants = json.loads(ants)
                        if isinstance(outs, str):
                            outs = json.loads(outs)
                        old_justs.append({
                            "type": jtype, "antecedents": ants,
                            "outlist": outs, "label": jlabel or "",
                        })

                    justs_changed = len(old_justs) != len(justs)
                    if not justs_changed:
                        for a, b in zip(old_justs, justs):
                            if (a["type"] != b["type"]
                                    or a["antecedents"] != b["antecedents"]
                                    or a["outlist"] != b["outlist"]
                                    or a.get("label", "") != b.get("label", "")):
                                justs_changed = True
                                break

                    if justs_changed:
                        cur.execute(
                            "DELETE FROM rms_justifications "
                            "WHERE node_id = %s AND project_id = %s",
                            (node_id, pid),
                        )
                        for j in justs:
                            cur.execute(
                                "INSERT INTO rms_justifications "
                                "(node_id, project_id, type, antecedents, outlist, label) "
                                "VALUES (%s, %s, %s, %s, %s, %s)",
                                (node_id, pid, j["type"],
                                 json.dumps(j["antecedents"]),
                                 json.dumps(j["outlist"]),
                                 j.get("label", "")),
                            )
                        changed = True

                    if is_out and not claim["raw_justifications"]:
                        retract_after.append(node_id)
                    elif is_out:
                        if not old_meta.get("_retracted"):
                            new_meta["_retracted"] = True
                            changed = True
                        cur.execute(
                            "UPDATE rms_nodes SET truth_value = 'OUT', "
                            "metadata = %s WHERE id = %s AND project_id = %s",
                            (json.dumps(new_meta), node_id, pid),
                        )
                    else:
                        if old_meta.get("_retracted"):
                            new_meta.pop("_retracted", None)
                            changed = True
                        if old_tv == "OUT":
                            assert_after.append(node_id)
                            changed = True
                        cur.execute(
                            "UPDATE rms_nodes SET metadata = %s "
                            "WHERE id = %s AND project_id = %s",
                            (json.dumps(new_meta), node_id, pid),
                        )

                    if changed:
                        beliefs_updated += 1
                    else:
                        beliefs_unchanged += 1
                else:
                    metadata = claim.get("metadata", {}).copy()
                    metadata.update({
                        "agent": agent_name,
                        "original_id": claim["id"],
                        "imported_from": source_path,
                    })

                    self._add_node_raw(
                        cur, node_id, claim["text"],
                        justifications=justs,
                        source=claim.get("source", ""),
                        source_url=claim.get("source_url", ""),
                        source_hash=claim.get("source_hash", ""),
                        date=claim.get("date", ""),
                        metadata=metadata,
                    )
                    existing.add(node_id)
                    beliefs_added += 1

                    if is_out and not claim["raw_justifications"]:
                        retract_after.append(node_id)
                    elif is_out:
                        cur.execute(
                            "UPDATE rms_nodes SET truth_value = 'OUT', "
                            "metadata = metadata || %s "
                            "WHERE id = %s AND project_id = %s",
                            (json.dumps({"_retracted": True}), node_id, pid),
                        )

        self.conn.commit()

        beliefs_removed = 0
        removed_ids = local_agent_ids - remote_ids
        if removed_ids:
            with self.conn.cursor() as cur:
                for node_id in sorted(removed_ids):
                    cur.execute(
                        "SELECT truth_value, metadata FROM rms_nodes "
                        "WHERE id = %s AND project_id = %s",
                        (node_id, pid),
                    )
                    row = cur.fetchone()
                    if not row:
                        continue
                    tv, meta = row
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    if tv == "IN":
                        cur.execute(
                            "UPDATE rms_nodes SET truth_value = 'OUT', "
                            "metadata = metadata || %s "
                            "WHERE id = %s AND project_id = %s",
                            (json.dumps({"_retracted": True}), node_id, pid),
                        )
                        self._log(cur, "retract", node_id, "OUT")
                        beliefs_removed += 1
                    elif not meta.get("_retracted"):
                        cur.execute(
                            "UPDATE rms_nodes SET metadata = metadata || %s "
                            "WHERE id = %s AND project_id = %s",
                            (json.dumps({"_retracted": True}), node_id, pid),
                        )
                        beliefs_removed += 1
            self.conn.commit()

        if assert_after:
            with self.conn.cursor() as cur:
                for node_id in assert_after:
                    cur.execute(
                        "UPDATE rms_nodes SET metadata = metadata - '_retracted' "
                        "WHERE id = %s AND project_id = %s",
                        (node_id, pid),
                    )
            self.conn.commit()

        self.propagate()

        beliefs_retracted = 0
        if retract_after:
            with self.conn.cursor() as cur:
                for node_id in retract_after:
                    cur.execute(
                        "SELECT truth_value FROM rms_nodes "
                        "WHERE id = %s AND project_id = %s",
                        (node_id, pid),
                    )
                    row = cur.fetchone()
                    if row and row[0] != "OUT":
                        cur.execute(
                            "UPDATE rms_nodes SET truth_value = 'OUT', "
                            "metadata = metadata || %s "
                            "WHERE id = %s AND project_id = %s",
                            (json.dumps({"_retracted": True}), node_id, pid),
                        )
                        self._log(cur, "retract", node_id, "OUT")
                    else:
                        cur.execute(
                            "UPDATE rms_nodes SET metadata = metadata || %s "
                            "WHERE id = %s AND project_id = %s",
                            (json.dumps({"_retracted": True}), node_id, pid),
                        )
                    beliefs_retracted += 1
            self.conn.commit()
            self.propagate()

        nogoods_imported = self._import_agent_nogoods(prefix=prefix,
                                                      nogoods=nogoods)

        return {
            "agent": agent_name,
            "prefix": prefix,
            "active_node": active_id,
            "created_premise": created_premise,
            "beliefs_added": beliefs_added,
            "beliefs_updated": beliefs_updated,
            "beliefs_removed": beliefs_removed,
            "beliefs_retracted": beliefs_retracted,
            "beliefs_unchanged": beliefs_unchanged,
            "beliefs_propagated": 0,
            "nogoods_imported": nogoods_imported,
        }

    def _import_agent_nogoods(self, prefix, nogoods):
        """Import prefixed nogoods for an agent."""
        pid = self.project_id
        count = 0
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM rms_nodes WHERE project_id = %s", (pid,),
            )
            all_nodes = {r[0] for r in cur.fetchall()}
            cur.execute(
                "SELECT id FROM rms_nogoods WHERE project_id = %s", (pid,),
            )
            existing_nogoods = {r[0] for r in cur.fetchall()}

            for ng in nogoods:
                prefixed_nodes = [f"{prefix}{n}" for n in ng["nodes"]]
                valid_nodes = [n for n in prefixed_nodes if n in all_nodes]
                nogood_id = f"{prefix}{ng['id']}"
                if len(valid_nodes) >= 2 and nogood_id not in existing_nogoods:
                    cur.execute(
                        "INSERT INTO rms_nogoods (id, project_id, nodes, "
                        "discovered, resolution) VALUES (%s, %s, %s, %s, %s)",
                        (nogood_id, pid, json.dumps(valid_nodes),
                         ng.get("discovered", ""), ng.get("resolution", "")),
                    )
                    existing_nogoods.add(nogood_id)
                    count += 1
        self.conn.commit()
        return count

    # ── Internal: propagation ───────────────────────────────────

    def _propagate(self, cur, changed_id):
        """BFS propagation of truth value changes through dependents.

        Returns (went_out, went_in) lists.
        """
        went_out = []
        went_in = []
        queue = deque([changed_id])
        visited = {changed_id}
        pid = self.project_id

        while queue:
            batch = []
            while queue:
                batch.append(queue.popleft())

            # Find all dependents of this batch
            dep_ids = self._find_dependents(cur, batch) - visited

            if not dep_ids:
                continue

            # Fetch current state of all dependents
            cur.execute(
                "SELECT id, truth_value, metadata FROM rms_nodes "
                "WHERE project_id = %s AND id = ANY(%s)",
                (pid, list(dep_ids)),
            )
            dep_states = {}
            for row in cur.fetchall():
                nid, tv, meta = row
                if isinstance(meta, str):
                    meta = json.loads(meta)
                dep_states[nid] = (tv, meta)

            # Fetch all justifications for these dependents
            cur.execute(
                "SELECT node_id, type, antecedents, outlist FROM rms_justifications "
                "WHERE project_id = %s AND node_id = ANY(%s)",
                (pid, list(dep_ids)),
            )
            justs_by_node = {}
            all_referenced = set()
            for row in cur.fetchall():
                nid, jtype, ants, outs = row
                if isinstance(ants, str):
                    ants = json.loads(ants)
                if isinstance(outs, str):
                    outs = json.loads(outs)
                justs_by_node.setdefault(nid, []).append((jtype, ants, outs))
                all_referenced.update(ants)
                all_referenced.update(outs)

            # Batch-fetch truth values for all referenced nodes
            truth_cache = {}
            if all_referenced:
                cur.execute(
                    "SELECT id, truth_value FROM rms_nodes "
                    "WHERE project_id = %s AND id = ANY(%s)",
                    (pid, list(all_referenced)),
                )
                for row in cur.fetchall():
                    truth_cache[row[0]] = row[1]

            # Evaluate each dependent
            for dep_id in dep_ids:
                if dep_id not in dep_states:
                    continue
                old_tv, meta = dep_states[dep_id]

                if meta.get("_retracted"):
                    continue

                justs = justs_by_node.get(dep_id, [])
                if not justs:
                    continue  # premise — keep current

                new_tv = "OUT"
                for jtype, ants, outs in justs:
                    inlist_ok = all(
                        truth_cache.get(a, "OUT") == "IN" for a in ants
                    )
                    outlist_ok = all(
                        truth_cache.get(o, "OUT") == "OUT" for o in outs
                    )
                    if inlist_ok and outlist_ok:
                        new_tv = "IN"
                        break

                if old_tv != new_tv:
                    cur.execute(
                        "UPDATE rms_nodes SET truth_value = %s "
                        "WHERE id = %s AND project_id = %s",
                        (new_tv, dep_id, pid),
                    )
                    self._log(cur, "propagate", dep_id, new_tv)
                    truth_cache[dep_id] = new_tv
                    visited.add(dep_id)
                    queue.append(dep_id)
                    if new_tv == "OUT":
                        went_out.append(dep_id)
                    else:
                        went_in.append(dep_id)

        return went_out, went_in

    def _compute_truth(self, cur, node_id):
        """Compute truth value from justifications."""
        pid = self.project_id
        cur.execute(
            "SELECT type, antecedents, outlist FROM rms_justifications "
            "WHERE node_id = %s AND project_id = %s",
            (node_id, pid),
        )
        justs = cur.fetchall()
        if not justs:
            return "IN"  # premise

        # Collect all referenced nodes
        all_refs = set()
        parsed = []
        for jtype, ants, outs in justs:
            if isinstance(ants, str):
                ants = json.loads(ants)
            if isinstance(outs, str):
                outs = json.loads(outs)
            parsed.append((jtype, ants, outs))
            all_refs.update(ants)
            all_refs.update(outs)

        # Batch-fetch truth values
        truth_cache = {}
        if all_refs:
            cur.execute(
                "SELECT id, truth_value FROM rms_nodes "
                "WHERE project_id = %s AND id = ANY(%s)",
                (pid, list(all_refs)),
            )
            for row in cur.fetchall():
                truth_cache[row[0]] = row[1]

        for jtype, ants, outs in parsed:
            inlist_ok = all(truth_cache.get(a, "OUT") == "IN" for a in ants)
            outlist_ok = all(truth_cache.get(o, "OUT") == "OUT" for o in outs)
            if inlist_ok and outlist_ok:
                return "IN"
        return "OUT"

    def _challenge_internal(self, cur, target_id, reason, challenge_id):
        pid = self.project_id

        # Verify target exists
        cur.execute(
            "SELECT truth_value, metadata FROM rms_nodes WHERE id = %s AND project_id = %s",
            (target_id, pid),
        )
        row = cur.fetchone()
        if not row:
            raise KeyError(f"Node '{target_id}' not found")
        old_value = row[0]
        target_meta = row[1]
        if isinstance(target_meta, str):
            target_meta = json.loads(target_meta)

        # Generate challenge ID
        if challenge_id is None:
            challenge_id = f"challenge-{target_id}"
            suffix = 1
            while True:
                cur.execute(
                    "SELECT 1 FROM rms_nodes WHERE id = %s AND project_id = %s",
                    (challenge_id, pid),
                )
                if not cur.fetchone():
                    break
                suffix += 1
                challenge_id = f"challenge-{target_id}-{suffix}"
        else:
            cur.execute(
                "SELECT 1 FROM rms_nodes WHERE id = %s AND project_id = %s",
                (challenge_id, pid),
            )
            if cur.fetchone():
                raise ValueError(f"Challenge node '{challenge_id}' already exists")

        # Create challenge node as premise
        now = datetime.now().isoformat(timespec="seconds")
        challenge_meta = {"challenge_target": target_id}
        cur.execute(
            "INSERT INTO rms_nodes (id, project_id, text, truth_value, source, date, metadata) "
            "VALUES (%s, %s, %s, 'IN', 'challenge', %s, %s)",
            (challenge_id, pid, reason, now, json.dumps(challenge_meta)),
        )
        self._log(cur, "add", challenge_id, "IN")

        # Add challenge to target's outlist
        cur.execute(
            "SELECT COUNT(*) FROM rms_justifications WHERE node_id = %s AND project_id = %s",
            (target_id, pid),
        )
        has_justifications = cur.fetchone()[0] > 0

        if has_justifications:
            cur.execute(
                "UPDATE rms_justifications SET outlist = outlist || %s::jsonb "
                "WHERE node_id = %s AND project_id = %s",
                (json.dumps([challenge_id]), target_id, pid),
            )
        else:
            cur.execute(
                "INSERT INTO rms_justifications (node_id, project_id, type, antecedents, outlist, label) "
                "VALUES (%s, %s, 'SL', '[]', %s, '')",
                (target_id, pid, json.dumps([challenge_id])),
            )

        # Update target metadata with challenges list
        challenges = target_meta.get("challenges", [])
        challenges.append(challenge_id)
        target_meta["challenges"] = challenges
        cur.execute(
            "UPDATE rms_nodes SET metadata = %s WHERE id = %s AND project_id = %s",
            (json.dumps(target_meta), target_id, pid),
        )

        # Recompute target truth and propagate
        new_value = self._compute_truth(cur, target_id)
        changed = []

        if old_value != new_value:
            cur.execute(
                "UPDATE rms_nodes SET truth_value = %s WHERE id = %s AND project_id = %s",
                (new_value, target_id, pid),
            )
            changed.append(target_id)
            self._log(cur, "challenge", target_id, new_value)
            went_out, went_in = self._propagate(cur, target_id)
            changed.extend(went_out)
            changed.extend(went_in)
        else:
            self._log(cur, "challenge", target_id, f"unchanged ({old_value})")

        return {"challenge_id": challenge_id, "target_id": target_id, "changed": changed}

    def _find_dependents(self, cur, node_ids):
        """Find nodes that have any of node_ids in their antecedents or outlist."""
        if not node_ids:
            return set()
        pid = self.project_id
        # Use JSONB containment: antecedents @> '["node_id"]'
        dep_ids = set()
        for nid in node_ids:
            needle = json.dumps([nid])
            cur.execute(
                "SELECT DISTINCT node_id FROM rms_justifications "
                "WHERE project_id = %s "
                "AND (antecedents @> %s::jsonb OR outlist @> %s::jsonb)",
                (pid, needle, needle),
            )
            for row in cur.fetchall():
                dep_ids.add(row[0])
        return dep_ids

    def _log(self, cur, action, target, value):
        cur.execute(
            "INSERT INTO rms_propagation_log (project_id, timestamp, action, target, value) "
            "VALUES (%s, %s, %s, %s, %s)",
            (self.project_id, datetime.now().isoformat(timespec="seconds"),
             action, target, value),
        )

    # ── Internal: nogoods + explain ─────────────────────────────

    def _find_culprits_internal(self, cur, nogood_node_ids):
        pid = self.project_id

        # Get truth values
        cur.execute(
            "SELECT id, truth_value FROM rms_nodes WHERE project_id = %s AND id = ANY(%s)",
            (pid, nogood_node_ids),
        )
        node_tvs = {r[0]: r[1] for r in cur.fetchall()}

        assumptions_by_node = {}
        all_premises = set()

        for nid in nogood_node_ids:
            if node_tvs.get(nid) != "IN":
                continue
            premises = []
            visited = set()
            self._trace_assumptions_recursive(cur, nid, premises, visited)
            assumptions_by_node[nid] = premises
            all_premises.update(premises)

        candidates = []
        for premise_id in all_premises:
            would_resolve = [
                nid for nid, assumptions in assumptions_by_node.items()
                if premise_id in assumptions
            ]
            if would_resolve:
                entrenchment = self._entrenchment(cur, premise_id)
                deps = self._find_dependents(cur, [premise_id])
                candidates.append({
                    "premise": premise_id,
                    "would_resolve": would_resolve,
                    "dependent_count": len(deps),
                    "entrenchment": entrenchment,
                })

        candidates.sort(key=lambda c: c["entrenchment"])
        return candidates

    def _entrenchment(self, cur, node_id):
        pid = self.project_id
        cur.execute(
            "SELECT source, source_hash, metadata FROM rms_nodes "
            "WHERE id = %s AND project_id = %s",
            (node_id, pid),
        )
        row = cur.fetchone()
        if not row:
            return 0
        source, source_hash, meta = row
        if isinstance(meta, str):
            meta = json.loads(meta)

        score = 0

        # Premises are more entrenched
        cur.execute(
            "SELECT COUNT(*) FROM rms_justifications "
            "WHERE node_id = %s AND project_id = %s",
            (node_id, pid),
        )
        if cur.fetchone()[0] == 0:
            score += 100

        if source:
            score += 50
        if source_hash:
            score += 25

        deps = self._find_dependents(cur, [node_id])
        score += len(deps) * 10

        btype = meta.get("beliefs_type", "").upper()
        type_scores = {
            "AXIOM": 90, "WARNING": 90,
            "OBSERVATION": 80,
            "DERIVED": 40,
            "PREDICTED": 30,
            "NOTE": 10,
        }
        score += type_scores.get(btype, 20)

        return score

    def _trace_assumptions_recursive(self, cur, node_id, premises, visited):
        if node_id in visited:
            return
        visited.add(node_id)
        pid = self.project_id

        cur.execute(
            "SELECT type, antecedents FROM rms_justifications "
            "WHERE node_id = %s AND project_id = %s",
            (node_id, pid),
        )
        justs = cur.fetchall()

        if not justs:
            if node_id not in premises:
                premises.append(node_id)
            return

        for jtype, ants in justs:
            if isinstance(ants, str):
                ants = json.loads(ants)
            for ant_id in ants:
                self._trace_assumptions_recursive(cur, ant_id, premises, visited)

    def _trace_access_tags_recursive(self, cur, node_id, all_tags, visited):
        if node_id in visited:
            return
        visited.add(node_id)
        pid = self.project_id

        cur.execute(
            "SELECT metadata FROM rms_nodes "
            "WHERE id = %s AND project_id = %s",
            (node_id, pid),
        )
        row = cur.fetchone()
        if not row:
            return
        meta = row[0]
        if isinstance(meta, str):
            meta = json.loads(meta)
        all_tags.update(meta.get("access_tags", []))

        cur.execute(
            "SELECT antecedents FROM rms_justifications "
            "WHERE node_id = %s AND project_id = %s",
            (node_id, pid),
        )
        for (ants,) in cur.fetchall():
            if isinstance(ants, str):
                ants = json.loads(ants)
            for ant_id in ants:
                self._trace_access_tags_recursive(
                    cur, ant_id, all_tags, visited)

    def _explain_recursive(self, cur, node_id, visible_to, visited):
        if node_id in visited:
            return []
        visited.add(node_id)
        pid = self.project_id

        cur.execute(
            "SELECT truth_value, metadata FROM rms_nodes "
            "WHERE id = %s AND project_id = %s",
            (node_id, pid),
        )
        row = cur.fetchone()
        if not row:
            return []
        tv, meta = row
        if isinstance(meta, str):
            meta = json.loads(meta)

        if visible_to is not None and not self._is_visible(meta, visible_to):
            return []

        cur.execute(
            "SELECT type, antecedents, outlist, label FROM rms_justifications "
            "WHERE node_id = %s AND project_id = %s ORDER BY id",
            (node_id, pid),
        )
        justs = cur.fetchall()

        steps = []

        if not justs:
            steps.append({
                "node": node_id,
                "truth_value": tv,
                "reason": "premise" if tv == "IN" else "retracted premise",
            })
            return steps

        if tv == "IN":
            # Find the valid justification
            for jtype, ants, outs, jlabel in justs:
                if isinstance(ants, str):
                    ants = json.loads(ants)
                if isinstance(outs, str):
                    outs = json.loads(outs)

                # Check validity
                if self._justification_valid_cached(cur, ants, outs):
                    step = {
                        "node": node_id,
                        "truth_value": "IN",
                        "reason": f"{jtype} justification valid",
                        "antecedents": list(ants),
                        "label": jlabel,
                    }
                    if outs:
                        step["outlist"] = list(outs)
                    steps.append(step)
                    for ant_id in ants:
                        steps.extend(self._explain_recursive(cur, ant_id, visible_to, visited))
                    break
        else:
            # All justifications invalid
            for jtype, ants, outs, jlabel in justs:
                if isinstance(ants, str):
                    ants = json.loads(ants)
                if isinstance(outs, str):
                    outs = json.loads(outs)

                # Find failed antecedents and violated outlist
                all_refs = set(ants) | set(outs)
                truth_cache = {}
                if all_refs:
                    cur.execute(
                        "SELECT id, truth_value FROM rms_nodes "
                        "WHERE project_id = %s AND id = ANY(%s)",
                        (pid, list(all_refs)),
                    )
                    truth_cache = {r[0]: r[1] for r in cur.fetchall()}

                failed = [a for a in ants if truth_cache.get(a, "OUT") == "OUT"]
                violated = [o for o in outs if truth_cache.get(o, "OUT") == "IN"]

                step = {
                    "node": node_id,
                    "truth_value": "OUT",
                    "reason": f"{jtype} justification invalid",
                    "failed_antecedents": failed,
                    "label": jlabel,
                }
                if violated:
                    step["violated_outlist"] = violated
                steps.append(step)

        return steps

    def _justification_valid_cached(self, cur, antecedents, outlist):
        pid = self.project_id
        all_refs = set(antecedents) | set(outlist)
        if not all_refs:
            return True
        cur.execute(
            "SELECT id, truth_value FROM rms_nodes "
            "WHERE project_id = %s AND id = ANY(%s)",
            (pid, list(all_refs)),
        )
        truth_cache = {r[0]: r[1] for r in cur.fetchall()}

        inlist_ok = all(truth_cache.get(a, "OUT") == "IN" for a in antecedents)
        outlist_ok = all(truth_cache.get(o, "OUT") == "OUT" for o in outlist)
        return inlist_ok and outlist_ok

    # ── Internal: formatting ────────────────────────────────────

    def _expand_neighbors(self, cur, matched_ids, visible_to):
        """Expand matched nodes to include 1-hop neighbors."""
        pid = self.project_id
        neighbor_ids = set()

        # Get justifications for matched nodes (dependencies)
        if matched_ids:
            cur.execute(
                "SELECT antecedents FROM rms_justifications "
                "WHERE project_id = %s AND node_id = ANY(%s)",
                (pid, matched_ids),
            )
            for row in cur.fetchall():
                ants = row[0]
                if isinstance(ants, str):
                    ants = json.loads(ants)
                neighbor_ids.update(ants)

        # Get dependents (nodes that reference matched nodes)
        dep_ids = self._find_dependents(cur, matched_ids)
        neighbor_ids.update(dep_ids)

        # Remove already-matched
        neighbor_ids -= set(matched_ids)

        if not neighbor_ids:
            return []

        # Fetch neighbor data
        cur.execute(
            "SELECT id, text, truth_value, source, metadata FROM rms_nodes "
            "WHERE project_id = %s AND id = ANY(%s) ORDER BY id",
            (pid, list(neighbor_ids)),
        )
        neighbors = []
        for row in cur.fetchall():
            nid, text, tv, source, meta = row
            if isinstance(meta, str):
                meta = json.loads(meta)
            if visible_to is not None and not self._is_visible(meta, visible_to):
                continue
            neighbors.append({
                "id": nid, "text": text, "truth_value": tv,
                "source": source, "metadata": meta,
            })
        return neighbors

    def _format_results(self, matched, neighbors, format):
        if format == "json":
            return self._format_json(matched, neighbors)
        elif format == "minimal":
            return self._format_minimal(matched, neighbors)
        elif format == "compact":
            return self._format_compact(matched, neighbors)
        else:
            return self._format_markdown(matched, neighbors)

    def _format_markdown(self, matched, neighbors):
        parts = []
        for m in matched:
            parts.append(f"### {m['id']}")
            parts.append(f"**Status:** {m['truth_value']}")
            parts.append(m["text"])
            if m.get("source"):
                parts.append(f"**Source:** {m['source']}")
            parts.append("")

        if neighbors:
            parts.append("---")
            parts.append("**Related nodes:**\n")
            for n in neighbors:
                parts.append(f"- **{n['id']}** ({n['truth_value']}): {n['text']}")
            parts.append("")

        return "\n".join(parts)

    def _format_json(self, matched, neighbors):
        results = []
        for m in matched:
            results.append({
                "id": m["id"], "text": m["text"],
                "truth_value": m["truth_value"],
                "source": m.get("source", ""), "match": True,
            })
        for n in neighbors:
            results.append({
                "id": n["id"], "text": n["text"],
                "truth_value": n["truth_value"],
                "source": n.get("source", ""), "match": False,
                "relation": "neighbor",
            })
        return json.dumps(results, indent=2)

    def _format_minimal(self, matched, neighbors):
        parts = [m["text"] for m in matched]
        if neighbors:
            parts.append("")
            parts.extend(n["text"] for n in neighbors)
        return "\n".join(parts)

    def _format_compact(self, matched, neighbors):
        lines = []
        for m in matched:
            lines.append(f"[{m['truth_value']}] {m['id']} — {m['text']}")
        for n in neighbors:
            lines.append(f"[{n['truth_value']}] {n['id']} — {n['text']}")
        return "\n".join(lines) if lines else "No results found."

    # ── Internal: helpers ───────────────────────────────────────

    def _validate_refs(self, cur, justifications):
        all_ids = set()
        for j in justifications:
            all_ids.update(j["antecedents"])
            all_ids.update(j["outlist"])
        if not all_ids:
            return
        cur.execute(
            "SELECT id FROM rms_nodes WHERE project_id = %s AND id = ANY(%s)",
            (self.project_id, list(all_ids)),
        )
        found = {row[0] for row in cur.fetchall()}
        missing = all_ids - found
        if missing:
            raise KeyError(f"Referenced nodes do not exist: {', '.join(sorted(missing))}")

    def _parse_justifications(self, sl, cp, unless, label):
        justs = []
        outlist = [o.strip() for o in unless.split(",") if o.strip()] if unless else []

        if sl:
            antecedents = [a.strip() for a in sl.split(",") if a.strip()]
            justs.append({
                "type": "SL",
                "antecedents": antecedents,
                "outlist": outlist,
                "label": label,
            })
        if cp:
            antecedents = [a.strip() for a in cp.split(",") if a.strip()]
            justs.append({
                "type": "CP",
                "antecedents": antecedents,
                "outlist": outlist,
                "label": label,
            })
        return justs

    def _is_visible(self, metadata, visible_to):
        tags = metadata.get("access_tags", [])
        if not tags:
            return True
        return all(t in visible_to for t in tags)

    def _inherit_access_tags(self, cur, node_id, justifications):
        pid = self.project_id
        all_ant_ids = set()
        for j in justifications:
            all_ant_ids.update(j["antecedents"])

        if not all_ant_ids:
            return

        cur.execute(
            "SELECT metadata FROM rms_nodes WHERE project_id = %s AND id = ANY(%s)",
            (pid, list(all_ant_ids)),
        )
        inherited = set()
        for row in cur.fetchall():
            meta = row[0]
            if isinstance(meta, str):
                meta = json.loads(meta)
            inherited.update(meta.get("access_tags", []))

        if not inherited:
            return

        cur.execute(
            "SELECT metadata FROM rms_nodes WHERE id = %s AND project_id = %s",
            (node_id, pid),
        )
        row = cur.fetchone()
        meta = row[0] if row else {}
        if isinstance(meta, str):
            meta = json.loads(meta)

        existing = set(meta.get("access_tags", []))
        merged = existing | inherited
        meta["access_tags"] = sorted(merged)

        cur.execute(
            "UPDATE rms_nodes SET metadata = %s WHERE id = %s AND project_id = %s",
            (json.dumps(meta), node_id, pid),
        )

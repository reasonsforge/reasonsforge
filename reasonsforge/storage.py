"""SQLite persistence for the dependency network.

Stores nodes, justifications, nogoods, and propagation log in a single
SQLite database. ACID transactions ensure propagation cascades are atomic.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import Node, Justification, Nogood
from .metadata import SCHEMA_VERSION, build_meta
from .network import Network


SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    truth_value TEXT NOT NULL DEFAULT 'IN',
    supporting_justification INTEGER DEFAULT NULL,
    source TEXT DEFAULT '',
    source_url TEXT DEFAULT '',
    source_hash TEXT DEFAULT '',
    text_hash TEXT DEFAULT '',
    date TEXT DEFAULT '',
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT '',
    updated_at TEXT DEFAULT '',
    reviewed_at TEXT DEFAULT '',
    verified_at TEXT DEFAULT '',
    retracted_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS justifications (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL REFERENCES nodes(id),
    type TEXT NOT NULL,
    antecedents_json TEXT NOT NULL DEFAULT '[]',
    outlist_json TEXT NOT NULL DEFAULT '[]',
    label TEXT DEFAULT '',
    content_hash TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS nogoods (
    id TEXT PRIMARY KEY,
    nodes_json TEXT NOT NULL DEFAULT '[]',
    discovered TEXT DEFAULT '',
    resolution TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS repos (
    name TEXT PRIMARY KEY,
    path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS propagation_log (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS network_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(id, text, tokenize="porter unicode61");
"""


class Storage:
    """SQLite persistence for a Network."""

    def __init__(self, db_path: str | Path, project_name: str = ""):
        self.db_path = Path(db_path)
        self._is_new = not self.db_path.exists()
        self._project_name = project_name
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        # Migrate existing databases: add source_url if missing
        cols = [c[1] for c in self.conn.execute("PRAGMA table_info(nodes)").fetchall()]
        if "source_url" not in cols:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN source_url TEXT DEFAULT ''")
        for col in ("created_at", "updated_at", "reviewed_at", "verified_at", "retracted_at"):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} TEXT DEFAULT ''")
        if "supporting_justification" not in cols:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN supporting_justification INTEGER DEFAULT NULL")
        if "text_hash" not in cols:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN text_hash TEXT DEFAULT ''")
        j_cols = [c[1] for c in self.conn.execute("PRAGMA table_info(justifications)").fetchall()]
        if "content_hash" not in j_cols:
            self.conn.execute("ALTER TABLE justifications ADD COLUMN content_hash TEXT DEFAULT ''")
        if self._is_new:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            project_name = self._project_name or self.db_path.stem
            for key, val in [
                ("schema_version", SCHEMA_VERSION),
                ("project_name", project_name),
                ("created_at", now),
                ("updated_at", now),
            ]:
                self.conn.execute(
                    "INSERT OR IGNORE INTO network_meta (key, value) VALUES (?, ?)",
                    (key, val),
                )
        self.conn.commit()

    def save(self, network: Network) -> None:
        """Persist the entire network state to SQLite."""
        with self.conn:
            # Clear and rewrite (simple strategy for small networks)
            self.conn.execute("DELETE FROM justifications")
            self.conn.execute("DROP TABLE IF EXISTS nodes_fts")
            self.conn.execute("CREATE VIRTUAL TABLE nodes_fts USING fts5(id, text, tokenize='porter unicode61')")
            self.conn.execute("DELETE FROM nodes")
            self.conn.execute("DELETE FROM nogoods")
            self.conn.execute("DELETE FROM repos")
            self.conn.execute("DELETE FROM propagation_log")
            self.conn.execute("DELETE FROM network_meta")

            for node in network.nodes.values():
                self.conn.execute(
                    "INSERT INTO nodes (id, text, truth_value, supporting_justification, "
                    "source, source_url, source_hash, text_hash, date, metadata_json, "
                    "created_at, updated_at, reviewed_at, verified_at, retracted_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        node.id,
                        node.text,
                        node.truth_value,
                        node.supporting_justification,
                        node.source,
                        node.source_url,
                        node.source_hash,
                        node.text_hash,
                        node.date,
                        json.dumps(node.metadata),
                        node.created_at,
                        node.updated_at,
                        node.reviewed_at,
                        node.verified_at,
                        node.retracted_at,
                    ),
                )
                self.conn.execute(
                    "INSERT INTO nodes_fts (id, text) VALUES (?, ?)",
                    (node.id, node.text),
                )
                for j in node.justifications:
                    self.conn.execute(
                        "INSERT INTO justifications (node_id, type, antecedents_json, outlist_json, label, content_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (node.id, j.type, json.dumps(j.antecedents), json.dumps(j.outlist), j.label, j.content_hash),
                    )

            for nogood in network.nogoods:
                self.conn.execute(
                    "INSERT INTO nogoods (id, nodes_json, discovered, resolution) "
                    "VALUES (?, ?, ?, ?)",
                    (nogood.id, json.dumps(nogood.nodes), nogood.discovered, nogood.resolution),
                )

            self.conn.execute(
                "INSERT INTO network_meta (key, value) VALUES (?, ?)",
                ("next_nogood_id", str(network._next_nogood_id)),
            )

            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            meta = dict(network.meta)
            meta.setdefault("schema_version", SCHEMA_VERSION)
            meta.setdefault("project_name", self.db_path.stem)
            meta.setdefault("created_at", now)
            meta["updated_at"] = now
            for key, val in meta.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO network_meta (key, value) VALUES (?, ?)",
                    (key, str(val)),
                )

            for name, path in network.repos.items():
                self.conn.execute(
                    "INSERT INTO repos (name, path) VALUES (?, ?)",
                    (name, path),
                )

            for entry in network.log:
                self.conn.execute(
                    "INSERT INTO propagation_log (timestamp, action, target, value) "
                    "VALUES (?, ?, ?, ?)",
                    (entry["timestamp"], entry["action"], entry["target"], entry["value"]),
                )

    def load(self) -> Network:
        """Load a Network from SQLite."""
        network = Network()

        # Load nodes (without justifications first, to avoid ordering issues)
        cols = [c[1] for c in self.conn.execute("PRAGMA table_info(nodes)").fetchall()]
        has_source_url = "source_url" in cols
        has_timestamps = "created_at" in cols
        has_supporting = "supporting_justification" in cols
        has_text_hash = "text_hash" in cols
        if has_timestamps:
            if has_supporting:
                cursor = self.conn.execute(
                    "SELECT id, text, truth_value, supporting_justification, "
                    "source, source_url, source_hash, date, metadata_json, "
                    "created_at, updated_at, reviewed_at, verified_at, retracted_at"
                    + (", text_hash" if has_text_hash else ", ''")
                    + " FROM nodes"
                )
            else:
                cursor = self.conn.execute(
                    "SELECT id, text, truth_value, NULL, "
                    "source, source_url, source_hash, date, metadata_json, "
                    "created_at, updated_at, reviewed_at, verified_at, retracted_at"
                    + (", text_hash" if has_text_hash else ", ''")
                    + " FROM nodes"
                )
        elif has_source_url:
            cursor = self.conn.execute(
                "SELECT id, text, truth_value, NULL, "
                "source, source_url, source_hash, date, metadata_json, '', '', '', '', '', '' FROM nodes"
            )
        else:
            cursor = self.conn.execute(
                "SELECT id, text, truth_value, NULL, "
                "source, '', source_hash, date, metadata_json, '', '', '', '', '', '' FROM nodes"
            )
        node_rows = cursor.fetchall()

        # Load justifications keyed by node_id
        j_cols = [c[1] for c in self.conn.execute("PRAGMA table_info(justifications)").fetchall()]
        has_content_hash = "content_hash" in j_cols
        just_cursor = self.conn.execute(
            "SELECT node_id, type, antecedents_json, outlist_json, label"
            + (", content_hash" if has_content_hash else ", ''")
            + " FROM justifications ORDER BY rowid"
        )
        justifications_by_node: dict[str, list[Justification]] = {}
        for node_id, jtype, ant_json, out_json, label, content_hash in just_cursor:
            j = Justification(
                type=jtype,
                antecedents=json.loads(ant_json),
                outlist=json.loads(out_json),
                label=label,
                content_hash=content_hash or "",
            )
            justifications_by_node.setdefault(node_id, []).append(j)

        # Build nodes directly (bypass add_node to preserve exact state)
        for row in node_rows:
            nid, text, truth_value, supporting_j, source, source_url, source_hash, \
                date, meta_json, created_at, updated_at, reviewed_at, verified_at, \
                retracted_at, text_hash = row
            node = Node(
                id=nid,
                text=text,
                truth_value=truth_value,
                justifications=justifications_by_node.get(nid, []),
                supporting_justification=supporting_j,
                source=source,
                source_url=source_url or "",
                source_hash=source_hash,
                text_hash=text_hash or "",
                date=date,
                metadata=json.loads(meta_json),
                created_at=created_at or "",
                updated_at=updated_at or "",
                reviewed_at=reviewed_at or "",
                verified_at=verified_at or "",
                retracted_at=retracted_at or "",
            )
            network.nodes[nid] = node

        # Rebuild dependent index from justifications (canonical method)
        network._rebuild_dependents()

        # Load nogoods
        ng_cursor = self.conn.execute(
            "SELECT id, nodes_json, discovered, resolution FROM nogoods"
        )
        for ng_id, nodes_json, discovered, resolution in ng_cursor:
            network.nogoods.append(Nogood(
                id=ng_id,
                nodes=json.loads(nodes_json),
                discovered=discovered,
                resolution=resolution,
            ))

        # Load network metadata — persisted counter takes priority,
        # otherwise derive from existing nogoods to avoid ID collisions
        loaded_counter = False
        try:
            meta_cursor = self.conn.execute("SELECT key, value FROM network_meta")
            for key, value in meta_cursor:
                if key == "next_nogood_id":
                    network._next_nogood_id = int(value)
                    loaded_counter = True
                else:
                    network.meta[key] = value
        except Exception:
            pass  # network_meta table may not exist in old databases
        if not loaded_counter and network.nogoods:
            import re
            max_id = 0
            for ng in network.nogoods:
                m = re.fullmatch(r"nogood-(\d+)", ng.id)
                if m:
                    max_id = max(max_id, int(m.group(1)))
            network._next_nogood_id = max_id + 1

        # Load repos
        try:
            repos_cursor = self.conn.execute("SELECT name, path FROM repos")
            for name, path in repos_cursor:
                network.repos[name] = path
        except Exception:
            pass  # repos table may not exist in old databases

        # Load log
        log_cursor = self.conn.execute(
            "SELECT timestamp, action, target, value FROM propagation_log ORDER BY rowid"
        )
        for ts, action, target, value in log_cursor:
            network.log.append({
                "timestamp": ts,
                "action": action,
                "target": target,
                "value": value,
            })

        return network

    def close(self) -> None:
        self.conn.close()

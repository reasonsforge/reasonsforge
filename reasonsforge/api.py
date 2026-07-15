"""Functional Python API for the Reason Maintenance System.

This module provides standalone functions that any Python caller can use
(CLI, LangGraph tools, scripts) without dealing with Storage lifecycle
or argparse. Each function opens the database, operates, saves, and closes.

All functions return dicts suitable for JSON serialization.
"""

import json
import logging
import re
import sqlite3
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from . import Justification
from . import pubsub
from .metadata import build_meta
from .network import Network
from .storage import Storage


logger = logging.getLogger(__name__)

DEFAULT_DB = "reasons.db"


def _is_visible(node, visible_to: list[str]) -> bool:
    """Check if a node is visible given the caller's access tags.

    A node is visible if its access_tags are all contained in visible_to.
    Nodes with no access_tags are always visible.
    """
    tags = node.metadata.get("access_tags", [])
    if not tags:
        return True
    visible_set = set(visible_to)
    return all(t in visible_set for t in tags)


def _resolve_namespace(node_id: str, namespace: str | None) -> str:
    """Prefix node_id with namespace if provided and not already namespaced.

    Skips prefixing if the node_id already contains a ':' (already namespaced,
    possibly from a different namespace for cross-namespace references).
    """
    if namespace and ":" not in node_id:
        return f"{namespace}:{node_id}"
    return node_id


def _with_network(db_path: str, write: bool = False):
    """Context manager pattern for load/operate/save."""
    class _Ctx:
        def __init__(self):
            self.store = Storage(db_path)
            self.network = self.store.load()
            self._before: dict[str, str] | None = None

        def __enter__(self):
            if write and pubsub.has_subscribers(db_path):
                self._before = {
                    nid: n.truth_value
                    for nid, n in self.network.nodes.items()
                }
            return self.network

        def __exit__(self, exc_type, exc_val, exc_tb):
            if write and exc_type is None:
                self.store.save(self.network)
                if self._before is not None:
                    after = {
                        nid: n.truth_value
                        for nid, n in self.network.nodes.items()
                    }
                    events = pubsub.compute_changes(
                        self._before, after, db_path
                    )
                    pubsub.publish(events, db_path)
            self.store.close()
            return False

    return _Ctx()


def _pg_dispatch(pg_conninfo, project_id, method_name, **kwargs):
    """Dispatch a call to the PostgreSQL backend."""
    from .pg import PgApi
    with PgApi(pg_conninfo, project_id) as pg:
        return getattr(pg, method_name)(**kwargs)


def init_db(db_path: str = DEFAULT_DB, force: bool = False,
            project_name: str = "",
            pg_conninfo=None, project_id=None) -> dict:
    """Initialize a new RMS database.

    Args:
        db_path: Path to the database file.
        force: Overwrite existing database if True.
        project_name: Name for this belief network (defaults to DB filename stem).

    Returns: {"db_path": str, "created": bool}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "init_db",
                            project_name=project_name)
    p = Path(db_path)
    if p.exists() and not force:
        raise FileExistsError(f"Database already exists: {db_path}")
    if p.exists() and force:
        p.unlink()
    store = Storage(db_path, project_name=project_name)
    store.close()
    return {"db_path": str(p), "created": True}


def ensure_namespace(namespace: str, db_path: str = DEFAULT_DB,
                     pg_conninfo=None, project_id=None) -> dict:
    """Ensure a namespace premise node exists (namespace:active).

    Creates the premise if it doesn't exist. This is the node that all
    beliefs in this namespace depend on — retracting it cascades OUT
    every belief from this namespace.

    Returns: {"namespace": str, "active_node": str, "created": bool}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "ensure_namespace",
                            namespace=namespace)
    active_id = f"{namespace}:active"
    with _with_network(db_path, write=True) as net:
        created = False
        if active_id not in net.nodes:
            net.add_node(
                id=active_id,
                text=f"Agent '{namespace}' beliefs are trusted",
                metadata={"agent": namespace, "role": "agent_premise"},
            )
            created = True
        return {"namespace": namespace, "active_node": active_id, "created": created}


def list_namespaces(db_path: str = DEFAULT_DB,
                    pg_conninfo=None, project_id=None) -> dict:
    """List all namespaces (agents) in the database.

    Detects namespaces by looking for nodes with ':active' suffix
    that have agent_premise role in metadata.

    Returns: {"namespaces": list[dict]}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "list_namespaces")
    with _with_network(db_path) as net:
        namespaces = []
        for nid, node in sorted(net.nodes.items()):
            if nid.endswith(":active") and node.metadata.get("role") == "agent_premise":
                ns = nid[:-len(":active")]
                # Count beliefs in this namespace
                count = sum(1 for n in net.nodes if n.startswith(f"{ns}:") and n != nid)
                in_count = sum(
                    1 for n, nd in net.nodes.items()
                    if n.startswith(f"{ns}:") and n != nid and nd.truth_value == "IN"
                )
                namespaces.append({
                    "namespace": ns,
                    "active_node": nid,
                    "active": node.truth_value == "IN",
                    "total_beliefs": count,
                    "in_beliefs": in_count,
                })
        return {"namespaces": namespaces}


SOURCE_TYPES = {"code", "document", "self-description", "derived"}


def add_node(
    node_id: str,
    text: str,
    sl: str = "",
    cp: str = "",
    unless: str = "",
    label: str = "",
    source: str = "",
    source_url: str = "",
    namespace: str | None = None,
    any_mode: bool = False,
    access_tags: list[str] | None = None,
    example: str | None = None,
    source_type: str = "",
    accepted_pr: str = "",
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Add a node to the network.

    Args:
        node_id: Node identifier
        text: Node text
        sl: Comma-separated antecedent IDs for SL justification
        cp: Comma-separated antecedent IDs for CP justification
        unless: Comma-separated outlist IDs (must be OUT for justification to hold)
        label: Justification label
        source: Provenance (repo:path)
        namespace: Optional namespace prefix (auto-creates ns:active premise)
        any_mode: If True, expand SL into one justification per antecedent (OR)
        access_tags: Data source provenance tags for access control
        db_path: Path to RMS database

    Returns: {"node_id": str, "truth_value": str, "type": str, "premise_count": int}
    """
    if pg_conninfo:
        if any_mode:
            raise NotImplementedError("any_mode is not supported with PostgreSQL")
        return _pg_dispatch(pg_conninfo, project_id, "add_node",
                            node_id=node_id, text=text, sl=sl, cp=cp, unless=unless,
                            label=label, source=source, source_url=source_url,
                            access_tags=access_tags, namespace=namespace,
                            example=example)
    outlist = [o.strip() for o in unless.split(",") if o.strip()] if unless else []
    justifications = []
    if sl:
        antecedents = [a.strip() for a in sl.split(",")]
        if any_mode and len(antecedents) > 1:
            for a in antecedents:
                justifications.append(Justification(type="SL", antecedents=[a], outlist=outlist, label=label))
        else:
            justifications.append(Justification(type="SL", antecedents=antecedents, outlist=outlist, label=label))
    elif cp:
        antecedents = [a.strip() for a in cp.split(",")]
        justifications.append(Justification(type="CP", antecedents=antecedents, outlist=outlist, label=label))
    elif outlist:
        # Outlist-only justification (no inlist) — premise that holds unless something is believed
        justifications.append(Justification(type="SL", antecedents=[], outlist=outlist, label=label))

    with _with_network(db_path, write=True) as net:
        # Namespace support: prefix node_id and add dependency on ns:active
        if namespace:
            node_id = _resolve_namespace(node_id, namespace)
            active_id = f"{namespace}:active"

            # Ensure the namespace premise exists
            if active_id not in net.nodes:
                net.add_node(
                    id=active_id,
                    text=f"Agent '{namespace}' beliefs are trusted",
                    metadata={"agent": namespace, "role": "agent_premise"},
                )

            # Add ns:active as antecedent to the justification
            if justifications:
                # Prepend active_id to existing antecedents
                j = justifications[0]
                if active_id not in j.antecedents:
                    j.antecedents.insert(0, active_id)
            else:
                # No explicit justification — create SL depending on ns:active
                justifications.append(Justification(
                    type="SL",
                    antecedents=[active_id],
                    outlist=outlist,
                    label=label or f"added by agent: {namespace}",
                ))

            # Also resolve namespace in antecedent references
            for j in justifications:
                j.antecedents = [_resolve_namespace(a, namespace) for a in j.antecedents]
                j.outlist = [_resolve_namespace(o, namespace) for o in j.outlist]

        if source_type and source_type not in SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type '{source_type}'. "
                f"Must be one of: {', '.join(sorted(SOURCE_TYPES))}"
            )

        metadata = {}
        if access_tags:
            metadata["access_tags"] = sorted(set(access_tags))
        if example is not None:
            metadata["example"] = example
        if source_type:
            metadata["source_type"] = source_type
        if accepted_pr:
            metadata["accepted_pr"] = accepted_pr

        node = net.add_node(
            id=node_id,
            text=text,
            justifications=justifications or None,
            source=source,
            source_url=source_url,
            metadata=metadata or None,
        )
        jtype = justifications[0].type if justifications else "premise"
        max_premises = max((len(j.antecedents) for j in justifications), default=0)
        return {
            "node_id": node_id,
            "truth_value": node.truth_value,
            "type": jtype,
            "premise_count": max_premises,
        }


def add_justification(
    node_id: str,
    sl: str = "",
    cp: str = "",
    unless: str = "",
    label: str = "",
    namespace: str | None = None,
    any_mode: bool = False,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Add a new justification to an existing node.

    Args:
        node_id: Node to add justification to
        sl: Comma-separated antecedent IDs for SL justification
        cp: Comma-separated antecedent IDs for CP justification
        unless: Comma-separated outlist IDs (must be OUT for justification to hold)
        label: Justification label
        namespace: Optional namespace prefix
        any_mode: If True, expand SL into one justification per antecedent (OR; not supported with PostgreSQL)
        db_path: Path to RMS database

    Returns: {"node_id", "old_truth_value", "new_truth_value", "changed", "premise_count"}
    """
    if pg_conninfo:
        if any_mode:
            raise NotImplementedError("any_mode is not supported with PostgreSQL")
        return _pg_dispatch(pg_conninfo, project_id, "add_justification",
                            node_id=node_id, sl=sl, cp=cp, unless=unless,
                            label=label, namespace=namespace)
    outlist = [o.strip() for o in unless.split(",") if o.strip()] if unless else []

    if sl:
        antecedents = [a.strip() for a in sl.split(",")]
        jtype = "SL"
    elif cp:
        antecedents = [a.strip() for a in cp.split(",")]
        jtype = "CP"
    elif outlist:
        antecedents = []
        jtype = "SL"
    else:
        raise ValueError("Must provide --sl, --cp, or --unless")

    with _with_network(db_path, write=True) as net:
        if namespace:
            node_id = _resolve_namespace(node_id, namespace)
            antecedents = [_resolve_namespace(a, namespace) for a in antecedents]
            outlist = [_resolve_namespace(o, namespace) for o in outlist]

        if any_mode and jtype == "SL" and len(antecedents) > 1:
            result = None
            for a in antecedents:
                j = Justification(type="SL", antecedents=[a], outlist=outlist, label=label)
                result = net.add_justification(node_id, j)
            result["premise_count"] = 1
            return result

        justification = Justification(
            type=jtype, antecedents=antecedents, outlist=outlist, label=label,
        )
        result = net.add_justification(node_id, justification)
        result["premise_count"] = len(antecedents)
        return result


def retract_node(node_id: str, reason: str = "", db_path: str = DEFAULT_DB,
                 pg_conninfo=None, project_id=None) -> dict:
    """Retract a node and cascade.

    Args:
        node_id: Node to retract
        reason: Why this node is being retracted
        db_path: Path to database

    Returns: {"changed", "went_out", "went_in", "restoration_hints"}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "retract_node",
                            node_id=node_id, reason=reason)
    with _with_network(db_path, write=True) as net:
        before = {nid: n.truth_value for nid, n in net.nodes.items()}
        changed = net.retract(node_id, reason=reason)
        went_out = [nid for nid in changed if before.get(nid) == "IN" and net.nodes[nid].truth_value == "OUT"]
        went_in = [nid for nid in changed if before.get(nid) == "OUT" and net.nodes[nid].truth_value == "IN"]

        hints = []
        for nid in went_out:
            if nid == node_id:
                continue
            node = net.nodes[nid]
            for j in node.justifications:
                if j.type == "SL" and len(j.antecedents) >= 2:
                    still_in = [a for a in j.antecedents if a in net.nodes and net.nodes[a].truth_value == "IN"]
                    if still_in:
                        hints.append({
                            "node_id": nid,
                            "all_premises": j.antecedents,
                            "surviving_premises": still_in,
                        })
                    break

        return {"changed": changed, "went_out": went_out, "went_in": went_in, "restoration_hints": hints}


def what_if_retract(node_id: str, db_path: str = DEFAULT_DB,
                   pg_conninfo=None, project_id=None) -> dict:
    """Simulate retracting a node without mutating the database.

    Loads the network read-only, performs the retraction in memory,
    and returns the cascade effects. The database is not modified.
    Tracks both nodes that go OUT (cascade) and nodes that go IN
    (restoration from outlist — gated beliefs whose blocker is removed).

    Returns: {"node_id": str, "retracted": list[dict], "restored": list[dict], ...}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "what_if_retract",
                            node_id=node_id)

    with _with_network(db_path, write=False) as net:
        if node_id not in net.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        node = net.nodes[node_id]
        if node.truth_value == "OUT":
            return {
                "node_id": node_id,
                "already_out": True,
                "retracted": [],
                "restored": [],
                "total_affected": 0,
            }

        # Snapshot truth values before
        before = {nid: n.truth_value for nid, n in net.nodes.items()}

        # Perform retraction in memory (not saved)
        changed = net.retract(node_id)

        # Separate into retracted (went OUT) and restored (went IN)
        retracted = []
        restored = []
        for nid in changed:
            if nid == node_id:
                continue
            n = net.nodes[nid]
            info = {
                "id": nid,
                "text": n.text,
                "depth": _cascade_depth(net, nid, node_id),
                "dependents": len(n.dependents),
            }
            if before[nid] == "IN" and n.truth_value == "OUT":
                retracted.append(info)
            elif before[nid] == "OUT" and n.truth_value == "IN":
                restored.append(info)

        retracted.sort(key=lambda c: (c["depth"], c["id"]))
        restored.sort(key=lambda c: (c["depth"], c["id"]))

        return {
            "node_id": node_id,
            "already_out": False,
            "retracted": retracted,
            "restored": restored,
            "total_affected": len(retracted) + len(restored),
        }


def what_if_assert(node_id: str, db_path: str = DEFAULT_DB,
                  pg_conninfo=None, project_id=None) -> dict:
    """Simulate asserting (restoring) a node without mutating the database.

    Shows what would change if a currently-OUT node were asserted back to IN.
    Tracks both nodes that go IN (restoration cascade) and nodes that go OUT
    (outlist-gated beliefs that lose their justification when this node goes IN).

    Returns: {"node_id": str, "retracted": list[dict], "restored": list[dict], ...}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "what_if_assert",
                            node_id=node_id)

    with _with_network(db_path, write=False) as net:
        if node_id not in net.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        node = net.nodes[node_id]
        if node.truth_value == "IN":
            return {
                "node_id": node_id,
                "already_in": True,
                "retracted": [],
                "restored": [],
                "total_affected": 0,
            }

        # Snapshot truth values before
        before = {nid: n.truth_value for nid, n in net.nodes.items()}

        # Perform assertion in memory (not saved)
        changed = net.assert_node(node_id)

        # Separate into restored (went IN) and retracted (went OUT)
        retracted = []
        restored = []
        for nid in changed:
            if nid == node_id:
                continue
            n = net.nodes[nid]
            info = {
                "id": nid,
                "text": n.text,
                "depth": _cascade_depth(net, nid, node_id),
                "dependents": len(n.dependents),
            }
            if before[nid] == "IN" and n.truth_value == "OUT":
                retracted.append(info)
            elif before[nid] == "OUT" and n.truth_value == "IN":
                restored.append(info)

        retracted.sort(key=lambda c: (c["depth"], c["id"]))
        restored.sort(key=lambda c: (c["depth"], c["id"]))

        return {
            "node_id": node_id,
            "already_in": False,
            "retracted": retracted,
            "restored": restored,
            "total_affected": len(retracted) + len(restored),
        }


def _cascade_depth(net, target_id: str, retracted_id: str) -> int:
    """Find the shortest justification path from retracted node to target."""
    from collections import deque
    visited = {retracted_id}
    queue = deque([(retracted_id, 0)])
    while queue:
        current_id, depth = queue.popleft()
        current = net.nodes[current_id]
        for dep_id in current.dependents:
            if dep_id in visited:
                continue
            if dep_id == target_id:
                return depth + 1
            visited.add(dep_id)
            queue.append((dep_id, depth + 1))
    return 0


def assert_node(node_id: str, db_path: str = DEFAULT_DB,
                pg_conninfo=None, project_id=None) -> dict:
    """Assert a node and cascade restoration.

    Returns: {"changed": list[str], "went_out": list[str], "went_in": list[str]}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "assert_node",
                            node_id=node_id)

    with _with_network(db_path, write=True) as net:
        before = {nid: n.truth_value for nid, n in net.nodes.items()}
        changed = net.assert_node(node_id)
        went_out = [nid for nid in changed if before.get(nid) == "IN" and net.nodes[nid].truth_value == "OUT"]
        went_in = [nid for nid in changed if before.get(nid) == "OUT" and net.nodes[nid].truth_value == "IN"]
        return {"changed": changed, "went_out": went_out, "went_in": went_in}


def propagate(db_path: str = DEFAULT_DB,
              pg_conninfo=None, project_id=None) -> dict:
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "propagate")
    with _with_network(db_path, write=True) as net:
        changed = net.recompute_all()
    return {"changed": changed}


def get_status(visible_to: list[str] | None = None, db_path: str = DEFAULT_DB,
               pg_conninfo=None, project_id=None) -> dict:
    """Get all nodes with truth values.

    Returns: {"nodes": list[dict], "in_count": int, "total": int}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "get_status",
                            visible_to=visible_to)

    with _with_network(db_path) as net:
        nodes = []
        for nid, node in sorted(net.nodes.items()):
            if visible_to is not None and not _is_visible(node, visible_to):
                continue
            nodes.append({
                "id": nid,
                "text": node.text,
                "truth_value": node.truth_value,
                "justification_count": len(node.justifications),
            })
        in_count = sum(1 for n in nodes if n["truth_value"] == "IN")
        return {"nodes": nodes, "in_count": in_count, "total": len(nodes)}


def show_node(node_id: str, visible_to: list[str] | None = None, db_path: str = DEFAULT_DB,
              pg_conninfo=None, project_id=None) -> dict:
    """Get full details for a node.

    Returns: dict with id, text, truth_value, source, justifications, dependents
    Raises PermissionError if node's access_tags are not a subset of visible_to.
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "show_node",
                            node_id=node_id, visible_to=visible_to)

    with _with_network(db_path) as net:
        if node_id not in net.nodes:
            raise KeyError(f"Node '{node_id}' not found")
        node = net.nodes[node_id]
        if visible_to is not None and not _is_visible(node, visible_to):
            raise PermissionError(
                f"Node '{node_id}' requires access tags not in {visible_to}"
            )
        return {
            "id": node.id,
            "text": node.text,
            "truth_value": node.truth_value,
            "supporting_justification": node.supporting_justification,
            "source": node.source,
            "source_url": node.source_url,
            "source_hash": node.source_hash,
            "text_hash": node.text_hash,
            "justifications": [
                {"type": j.type, "antecedents": j.antecedents, "outlist": j.outlist,
                 "label": j.label, "content_hash": j.content_hash}
                for j in node.justifications
            ],
            "dependents": sorted(node.dependents),
            "metadata": node.metadata,
            "created_at": node.created_at,
            "updated_at": node.updated_at,
            "reviewed_at": node.reviewed_at,
            "verified_at": node.verified_at,
            "retracted_at": node.retracted_at,
        }


def explain_node(node_id: str, visible_to: list[str] | None = None, db_path: str = DEFAULT_DB,
                 pg_conninfo=None, project_id=None) -> dict:
    """Explain why a node is IN or OUT.

    Returns: {"steps": list[dict]}
    Raises PermissionError if node's access_tags are not a subset of visible_to.
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "explain_node",
                            node_id=node_id, visible_to=visible_to)

    with _with_network(db_path) as net:
        if node_id not in net.nodes:
            raise KeyError(f"Node '{node_id}' not found")
        if visible_to is not None and not _is_visible(net.nodes[node_id], visible_to):
            raise PermissionError(
                f"Node '{node_id}' requires access tags not in {visible_to}"
            )
        steps = net.explain(node_id)
        return {"steps": steps}


def trace_assumptions(node_id: str, visible_to: list[str] | None = None, db_path: str = DEFAULT_DB,
                      pg_conninfo=None, project_id=None) -> dict:
    """Trace backward to find all premises a node rests on.

    Returns: {"node_id": str, "premises": list[str]}
    Raises PermissionError if node's access_tags are not a subset of visible_to.
    Filters returned premises by visible_to.
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "trace_assumptions",
                            node_id=node_id, visible_to=visible_to)

    with _with_network(db_path) as net:
        if node_id not in net.nodes:
            raise KeyError(f"Node '{node_id}' not found")
        if visible_to is not None and not _is_visible(net.nodes[node_id], visible_to):
            raise PermissionError(
                f"Node '{node_id}' requires access tags not in {visible_to}"
            )
        premises = net.trace_assumptions(node_id)
        if visible_to is not None:
            premises = [p for p in premises if p in net.nodes and _is_visible(net.nodes[p], visible_to)]
        return {"node_id": node_id, "premises": premises}


def trace_access_tags(node_id: str, visible_to: list[str] | None = None,
                      db_path: str = DEFAULT_DB,
                      pg_conninfo=None, project_id=None) -> dict:
    """Trace backward through dependency chains and return union of all access_tags.

    Returns: {"node_id": str, "access_tags": list[str]}
    Raises PermissionError if node's access_tags are not a subset of visible_to.
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "trace_access_tags",
                            node_id=node_id, visible_to=visible_to)
    with _with_network(db_path) as net:
        if node_id not in net.nodes:
            raise KeyError(f"Node '{node_id}' not found")
        if visible_to is not None and not _is_visible(net.nodes[node_id], visible_to):
            raise PermissionError(
                f"Node '{node_id}' requires access tags not in {visible_to}"
            )
        tags = net.trace_access_tags(node_id)
        return {"node_id": node_id, "access_tags": tags}


def find_culprits(node_ids: list[str], db_path: str = DEFAULT_DB,
                  pg_conninfo=None, project_id=None) -> dict:
    """Find premises that could be retracted to resolve a contradiction.

    Returns: {"culprits": list[dict]}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "find_culprits",
                            node_ids=node_ids)
    with _with_network(db_path) as net:
        culprits = net.find_culprits(node_ids)
        return {"culprits": culprits}


def convert_to_premise(node_id: str, db_path: str = DEFAULT_DB,
                       pg_conninfo=None, project_id=None) -> dict:
    """Strip justifications from a node, making it a premise (IN by default).

    Returns: {"node_id": str, "old_justifications": int, "truth_value": str, "changed": list[str]}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "convert_to_premise",
                            node_id=node_id)

    with _with_network(db_path, write=True) as net:
        return net.convert_to_premise(node_id)


def remove_justification(
    node_id: str, index: int, db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Remove a single justification by index and propagate."""
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "remove_justification",
                            node_id=node_id, index=index)
    with _with_network(db_path, write=True) as net:
        return net.remove_justification(node_id, index)


def summarize(
    summary_id: str,
    text: str,
    over: list[str],
    source: str = "",
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Create a summary node that abstracts over a group of nodes.

    Returns: {"summary_id": str, "over": list[str], "truth_value": str}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "summarize",
                            summary_id=summary_id, text=text,
                            over=over, source=source)
    with _with_network(db_path, write=True) as net:
        return net.summarize(summary_id, text, over, source=source)


def supersede(old_id: str, new_id: str, db_path: str = DEFAULT_DB,
              pg_conninfo=None, project_id=None) -> dict:
    """Mark old_id as superseded by new_id. Old goes OUT when new is IN.

    Returns: {"old_id": str, "new_id": str, "changed": list[str]}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "supersede",
                            old_id=old_id, new_id=new_id)
    with _with_network(db_path, write=True) as net:
        return net.supersede(old_id, new_id)


def supersede_with_text(
    old_id: str,
    new_text: str,
    new_id: str | None = None,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Create a successor node with new text and supersede old_id.

    The old node goes OUT via the outlist mechanism (reversible).
    Auto-generates new_id as {old_id}-v{N} if not specified.

    Returns: {"old_id": str, "new_id": str, "changed": list[str]}
    """
    with _with_network(db_path, write=True) as net:
        if old_id not in net.nodes:
            raise KeyError(f"Node '{old_id}' not found")

        if not new_id:
            base = f"{old_id}-v2"
            new_id = base
            suffix = 3
            while new_id in net.nodes:
                new_id = f"{old_id}-v{suffix}"
                suffix += 1

        old_node = net.nodes[old_id]
        net.add_node(
            id=new_id,
            text=new_text,
            source=old_node.source,
            source_url=old_node.source_url,
        )
        result = net.supersede(old_id, new_id)
        return result


def update_node(
    node_id: str,
    text: str | None = None,
    source: str | None = None,
    source_url: str | None = None,
    example: str | None = None,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Update a node's source, source_url, or example metadata.

    Text is immutable — use 'reasons supersede' to create a successor.

    Returns: {"node_id": str, "updated_fields": list[str]}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "update_node",
                            node_id=node_id, text=text, source=source,
                            source_url=source_url, example=example)
    with _with_network(db_path, write=True) as net:
        if node_id not in net.nodes:
            raise KeyError(f"Node '{node_id}' not found")
        if text is not None:
            raise ValueError(
                f"Text mutation is not allowed — beliefs are immutable propositions. "
                f"Use 'reasons supersede {node_id} --text \"...\"' to create a successor."
            )
        node = net.nodes[node_id]
        updated = []
        if source is not None:
            node.source = source
            updated.append("source")
        if source_url is not None:
            node.source_url = source_url
            updated.append("source_url")
        if example is not None:
            meta = node.metadata or {}
            meta["example"] = example
            node.metadata = meta
            updated.append("example")
        if updated:
            node.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {"node_id": node_id, "updated_fields": updated}


def set_metadata(
    node_id: str,
    key: str,
    value,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Set a single metadata key on a node."""
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "set_metadata",
                            node_id=node_id, key=key, value=value)
    with _with_network(db_path, write=True) as net:
        if node_id not in net.nodes:
            raise KeyError(f"Node '{node_id}' not found")
        node = net.nodes[node_id]
        meta = node.metadata or {}
        meta[key] = value
        node.metadata = meta
        node.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {"node_id": node_id, "key": key}


def mark_duplicate(
    source_id: str,
    canonical_id: str,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Mark a node as a duplicate of a canonical version.

    Retracts the source node and stores the duplicate-of relationship.

    Args:
        source_id: The duplicate node to retract
        canonical_id: The canonical node to reference
        db_path: Path to database

    Returns: {"source_id": str, "canonical_id": str, "changed": list[str]}
    """
    if source_id == canonical_id:
        raise ValueError("A node cannot be marked as a duplicate of itself")

    with _with_network(db_path, write=True) as net:
        if source_id not in net.nodes:
            raise KeyError(f"Node '{source_id}' not found")
        if canonical_id not in net.nodes:
            raise KeyError(f"Canonical node '{canonical_id}' not found")

        node = net.nodes[source_id]
        meta = node.metadata or {}
        meta["duplicate_of"] = canonical_id
        reason = f"Duplicate of {canonical_id}"
        meta["retract_reason"] = reason
        node.metadata = meta

        changed = net.retract(source_id, reason=reason)

        return {
            "source_id": source_id,
            "canonical_id": canonical_id,
            "changed": changed,
        }


def mark_superseded(
    old_id: str,
    new_id: str,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Mark a node as superseded by a newer/better version.

    Retracts the old node and stores the superseded-by relationship.

    Args:
        old_id: The obsolete node to retract
        new_id: The replacement node
        db_path: Path to database

    Returns: {"old_id": str, "new_id": str, "changed": list[str]}
    """
    if old_id == new_id:
        raise ValueError("A node cannot be marked as superseded by itself")

    with _with_network(db_path, write=True) as net:
        if old_id not in net.nodes:
            raise KeyError(f"Node '{old_id}' not found")
        if new_id not in net.nodes:
            raise KeyError(f"Replacement node '{new_id}' not found")

        node = net.nodes[old_id]
        meta = node.metadata or {}
        meta["superseded_by"] = new_id
        reason = f"Superseded by {new_id}"
        meta["retract_reason"] = reason
        node.metadata = meta

        changed = net.retract(old_id, reason=reason)

        return {
            "old_id": old_id,
            "new_id": new_id,
            "changed": changed,
        }


def defeat_justification(
    node_id: str,
    justification_index: int,
    reason: str,
    defeater_type: str = "invalid-inference",
    defeater_id: str | None = None,
    defeat_reason_type: str = "",
    db_path: str = DEFAULT_DB,
) -> dict:
    """Defeat a justification by adding a defeater belief to its outlist.

    Creates a defeater belief (premise node) and adds it to the specified
    justification's outlist, causing the belief to go OUT via graph-native
    TMS semantics rather than metadata strings.

    Args:
        node_id: The belief whose justification should be defeated
        justification_index: Index of the justification to defeat (0-based)
        reason: Explanation of why the justification is invalid
        defeater_type: Type of defeater (invalid-inference, over-generalizes, etc.)
        defeater_id: Custom defeater ID (default: {type}-{node_id}-j{index})
        defeat_reason_type: Logical failure mode classification
        db_path: Path to database

    Returns: {
        "node_id": str,
        "justification_index": int,
        "defeater_id": str,
        "defeater_type": str,
        "defeat_reason_type": str,
        "changed": list[str]
    }
    """
    with _with_network(db_path, write=True) as net:
        if node_id not in net.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        node = net.nodes[node_id]
        if not node.justifications:
            raise ValueError(f"Node '{node_id}' has no justifications to defeat")

        if justification_index < 0 or justification_index >= len(node.justifications):
            raise IndexError(
                f"Justification index {justification_index} out of range "
                f"(node has {len(node.justifications)} justifications)"
            )

        justification = node.justifications[justification_index]

        if not defeater_id:
            defeater_id = f"{defeater_type}-{node_id}-j{justification_index}"

        defeater_text = f"{reason} (defeats {node_id} justification {justification_index})"

        before = {nid: n.truth_value for nid, n in net.nodes.items()}

        meta = {
            "defeater_type": defeater_type,
            "defeats_node": node_id,
            "defeats_justification": justification_index,
        }
        if defeat_reason_type:
            meta["defeat_reason_type"] = defeat_reason_type

        defeater = net.add_node(
            id=defeater_id,
            text=defeater_text,
            metadata=meta,
        )

        if defeater_id not in justification.outlist:
            justification.outlist.append(defeater_id)
        defeater.dependents.add(node_id)
        node.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        changed = net.recompute_all()

        actually_changed = [
            nid for nid in changed
            if before.get(nid) != net.nodes[nid].truth_value
        ]

        return {
            "node_id": node_id,
            "justification_index": justification_index,
            "defeater_id": defeater_id,
            "defeater_type": defeater_type,
            "defeat_reason_type": defeat_reason_type,
            "changed": actually_changed,
        }


def defeat_with_scope(
    node_id: str,
    justification_index: int,
    scope_findings: list[dict],
    missing_property: str,
    defeater_type: str = "invalid-inference",
    defeater_id: str | None = None,
    defeat_reason_type: str = "",
    db_path: str = DEFAULT_DB,
) -> dict:
    """Defeat a justification with a derived defeater backed by scope beliefs.

    Instead of a bare premise defeater, creates scope beliefs (premises
    describing what each antecedent establishes) and a derived defeater
    whose SL justification cites them. Challenging any scope belief
    causes the defeater to go OUT, restoring the original belief.

    Args:
        node_id: The belief whose justification should be defeated
        justification_index: Index of the justification to defeat (0-based)
        scope_findings: List of dicts with keys: antecedent, establishes,
            does_not_establish
        missing_property: The property the derived belief claims but no
            antecedent establishes
        defeater_type: Type of defeater
        defeater_id: Custom defeater ID
        defeat_reason_type: Logical failure mode classification
        db_path: Path to database

    Returns: {
        "node_id": str,
        "justification_index": int,
        "defeater_id": str,
        "defeater_type": str,
        "defeat_reason_type": str,
        "scope_belief_ids": list[str],
        "changed": list[str]
    }
    """
    with _with_network(db_path, write=True) as net:
        if node_id not in net.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        node = net.nodes[node_id]
        if not node.justifications:
            raise ValueError(f"Node '{node_id}' has no justifications to defeat")

        if justification_index < 0 or justification_index >= len(node.justifications):
            raise IndexError(
                f"Justification index {justification_index} out of range "
                f"(node has {len(node.justifications)} justifications)"
            )

        justification = node.justifications[justification_index]

        if not defeater_id:
            defeater_id = f"{defeater_type}-{node_id}-j{justification_index}"

        if not scope_findings:
            raise ValueError("scope_findings must not be empty")

        before = {nid: n.truth_value for nid, n in net.nodes.items()}

        scope_belief_ids = []
        for sf in scope_findings:
            ant_id = sf["antecedent"]
            establishes = sf.get("establishes", "")
            does_not = sf.get("does_not_establish", "")
            scope_id = f"scope-{ant_id}-for-{node_id}-j{justification_index}"
            suffix = 1
            while scope_id in net.nodes:
                scope_id = f"scope-{ant_id}-for-{node_id}-j{justification_index}-{suffix}"
                suffix += 1
            scope_text = (
                f"{ant_id} establishes {establishes}"
                + (f", and does not establish {does_not}" if does_not else "")
            )
            net.add_node(
                id=scope_id,
                text=scope_text,
                metadata={
                    "scope_of": ant_id,
                    "for_defeater": defeater_id,
                },
            )
            scope_belief_ids.append(scope_id)

        defeater_text = (
            f"{node_id} claims {missing_property}, but no antecedent establishes it "
            f"(defeats {node_id} justification {justification_index})"
        )
        defeater_meta = {
            "defeater_type": defeater_type,
            "defeats_node": node_id,
            "defeats_justification": justification_index,
        }
        if defeat_reason_type:
            defeater_meta["defeat_reason_type"] = defeat_reason_type

        defeater = net.add_node(
            id=defeater_id,
            text=defeater_text,
            justifications=[
                Justification(
                    type="SL",
                    antecedents=scope_belief_ids,
                    label=f"scope-defeat of {node_id} j{justification_index}",
                ),
            ],
            metadata=defeater_meta,
        )

        if defeater_id not in justification.outlist:
            justification.outlist.append(defeater_id)
        defeater.dependents.add(node_id)
        node.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        changed = net.recompute_all()

        actually_changed = [
            nid for nid in changed
            if before.get(nid) != net.nodes[nid].truth_value
        ]

        return {
            "node_id": node_id,
            "justification_index": justification_index,
            "defeater_id": defeater_id,
            "defeater_type": defeater_type,
            "defeat_reason_type": defeat_reason_type,
            "scope_belief_ids": scope_belief_ids,
            "changed": actually_changed,
        }


def migrate_retract_to_defeaters(
    node_ids: list[str] | None = None,
    dry_run: bool = True,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Convert string-based retract_reason entries to graph-native defeaters.

    Finds OUT nodes with retract_reason metadata whose justifications would
    be satisfied if not for the _retracted flag. For each, creates a defeater
    node on the first satisfied justification and clears the retract metadata.

    Returns: {"migrated": [...], "skipped": [...], "errors": [...]}
    """
    with _with_network(db_path, write=not dry_run) as net:
        candidates = node_ids or list(net.nodes.keys())
        migrated = []
        skipped = []
        errors = []

        for nid in candidates:
            if nid not in net.nodes:
                errors.append({"id": nid, "reason": "not found"})
                continue

            node = net.nodes[nid]
            if node.truth_value != "OUT":
                continue
            retract_reason = (node.metadata or {}).get("retract_reason")
            if not retract_reason:
                continue
            if not node.justifications:
                skipped.append({"id": nid, "reason": "premise (no justifications)"})
                continue

            # Find first justification that would be satisfied
            satisfied_idx = None
            for i, j in enumerate(node.justifications):
                inlist_ok = all(
                    a in net.nodes and net.nodes[a].truth_value == "IN"
                    for a in j.antecedents
                )
                outlist_ok = all(
                    o not in net.nodes or net.nodes[o].truth_value == "OUT"
                    for o in j.outlist
                )
                if inlist_ok and outlist_ok:
                    satisfied_idx = i
                    break

            if satisfied_idx is None:
                skipped.append({"id": nid, "reason": "no satisfied justification"})
                continue

            if dry_run:
                migrated.append({
                    "id": nid,
                    "retract_reason": retract_reason,
                    "justification_index": satisfied_idx,
                })
                continue

            # Apply: inline defeat logic (can't call defeat_justification
            # because it opens its own _with_network context)
            defeater_id = f"migrated-retraction-{nid}-j{satisfied_idx}"
            defeater_text = (
                f"{retract_reason} (defeats {nid} justification {satisfied_idx})"
            )

            defeater = net.add_node(
                id=defeater_id,
                text=defeater_text,
                metadata={
                    "defeater_type": "migrated-retraction",
                    "defeats_node": nid,
                    "defeats_justification": satisfied_idx,
                },
            )

            justification = node.justifications[satisfied_idx]
            if defeater_id not in justification.outlist:
                justification.outlist.append(defeater_id)
            defeater.dependents.add(nid)
            node.updated_at = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )

            # Clear the string-based retract metadata
            node.metadata.pop("retract_reason", None)
            node.metadata.pop("_retracted", None)

            migrated.append({
                "id": nid,
                "defeater_id": defeater_id,
                "retract_reason": retract_reason,
                "justification_index": satisfied_idx,
            })

        if not dry_run and migrated:
            net.recompute_all()

        return {"migrated": migrated, "skipped": skipped, "errors": errors}


def classify_defeat_reason_types(
    defeater_type_filter: str | None = None,
    model: str = "claude",
    timeout: int = 300,
    dry_run: bool = True,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Classify unclassified defeaters by logical failure mode via LLM.

    Finds defeater nodes (those with defeats_node metadata) that have no
    defeat_reason_type, sends each to an LLM for classification, and
    writes the result to metadata.

    Args:
        defeater_type_filter: Only classify defeaters with this defeater_type
        model: LLM model for classification
        timeout: LLM timeout in seconds
        dry_run: If True, classify but don't write to db
        db_path: Path to database

    Returns: {"classified": [...], "skipped": [...], "errors": [...]}
    """
    from .review import classify_defeat_reason

    network = export_network(db_path=db_path)
    nodes = network.get("nodes", {})

    candidates = []
    skipped = []
    for nid, node in nodes.items():
        meta = node.get("metadata") or {}
        if not meta.get("defeats_node"):
            continue
        if meta.get("defeat_reason_type"):
            skipped.append({"id": nid, "reason": "already classified"})
            continue
        if defeater_type_filter and meta.get("defeater_type") != defeater_type_filter:
            skipped.append({"id": nid, "reason": f"defeater_type '{meta.get('defeater_type')}' != '{defeater_type_filter}'"})
            continue
        candidates.append(nid)

    classified = []
    errors = []

    for nid in candidates:
        node = nodes[nid]
        meta = node.get("metadata") or {}
        defeated_id = meta.get("defeats_node", "")
        defeated = nodes.get(defeated_id, {})

        reason_type = classify_defeat_reason(
            node.get("text", ""),
            defeated.get("text", ""),
            model, timeout,
        )

        if not reason_type:
            errors.append({"id": nid, "reason": "classification returned empty"})
            continue

        classified.append({"id": nid, "defeat_reason_type": reason_type})

        if not dry_run:
            set_metadata(nid, "defeat_reason_type", reason_type, db_path=db_path)

    return {"classified": classified, "skipped": skipped, "errors": errors}


def challenge(
    target_id: str,
    reason: str,
    challenge_id: str | None = None,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Challenge a node — creates a challenge node and the target goes OUT.

    Returns: {"challenge_id": str, "target_id": str, "changed": list[str]}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "challenge",
                            target_id=target_id, reason=reason,
                            challenge_id=challenge_id)
    with _with_network(db_path, write=True) as net:
        return net.challenge(target_id, reason, challenge_id=challenge_id)


def defend(
    target_id: str,
    challenge_id: str,
    reason: str,
    defense_id: str | None = None,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Defend a node against a challenge — neutralises the challenge, target restored.

    Returns: {"defense_id": str, "challenge_id": str, "target_id": str, "changed": list[str]}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "defend",
                            target_id=target_id, challenge_id=challenge_id,
                            reason=reason, defense_id=defense_id)
    with _with_network(db_path, write=True) as net:
        return net.defend(target_id, challenge_id, reason, defense_id=defense_id)


def add_nogood(node_ids: list[str], db_path: str = DEFAULT_DB,
               pg_conninfo=None, project_id=None) -> dict:
    """Record a contradiction and use backtracking to resolve.

    Returns: {"nogood_id": str, "nodes": list[str], "changed": list[str], "backtracked_to": str | None}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "add_nogood",
                            node_ids=node_ids)

    with _with_network(db_path, write=True) as net:
        # Find culprits before retraction for reporting
        all_in = all(
            nid in net.nodes and net.nodes[nid].truth_value == "IN"
            for nid in node_ids
        )
        culprits = net.find_culprits(node_ids) if all_in else []
        backtracked_to = culprits[0]["premise"] if culprits else None

        changed = net.add_nogood(node_ids)
        ng = net.nogoods[-1]
        return {
            "nogood_id": ng.id,
            "nodes": ng.nodes,
            "changed": changed,
            "backtracked_to": backtracked_to,
        }


def get_belief_set(db_path: str = DEFAULT_DB,
                   pg_conninfo=None, project_id=None) -> list[str]:
    """Return all node IDs currently IN."""
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "get_belief_set")
    with _with_network(db_path) as net:
        return net.get_belief_set()


def get_log(last: int | None = None, db_path: str = DEFAULT_DB,
            pg_conninfo=None, project_id=None) -> dict:
    """Get propagation history.

    Returns: {"entries": list[dict]}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "get_log",
                            last=last)

    with _with_network(db_path) as net:
        entries = net.log
        if last:
            entries = entries[-last:]
        return {"entries": entries}


def export_network(visible_to: list[str] | None = None, db_path: str = DEFAULT_DB,
                   pg_conninfo=None, project_id=None) -> dict:
    """Export the entire network as a dict.

    Returns: {"meta": dict, "nodes": dict, "nogoods": list, "repos": dict}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "export_network",
                            visible_to=visible_to)

    with _with_network(db_path) as net:
        nodes = {
            nid: {
                "text": n.text,
                "truth_value": n.truth_value,
                "supporting_justification": n.supporting_justification,
                "justifications": [
                    {"type": j.type, "antecedents": j.antecedents, "outlist": j.outlist,
                     "label": j.label, "content_hash": j.content_hash}
                    for j in n.justifications
                ],
                "source": n.source,
                "source_url": n.source_url,
                "source_hash": n.source_hash,
                "text_hash": n.text_hash,
                "date": n.date,
                "metadata": {k: v for k, v in n.metadata.items() if not k.startswith("_")},
                "created_at": n.created_at,
                "updated_at": n.updated_at,
                "reviewed_at": n.reviewed_at,
                "verified_at": n.verified_at,
                "retracted_at": n.retracted_at,
            }
            for nid, n in sorted(net.nodes.items())
            if visible_to is None or _is_visible(n, visible_to)
        }
        meta = build_meta(
            project_name=net.meta.get("project_name", ""),
            node_count=len(nodes),
            created_at=net.meta.get("created_at", ""),
        )
        return {
            "meta": meta,
            "nodes": nodes,
            "nogoods": [
                {"id": ng.id, "nodes": ng.nodes, "discovered": ng.discovered, "resolution": ng.resolution}
                for ng in net.nogoods
                if visible_to is None or all(n in net.nodes and _is_visible(net.nodes[n], visible_to) for n in ng.nodes)
            ],
            "repos": dict(net.repos),
        }


def import_beliefs(
    beliefs_file: str,
    nogoods_file: str | None = None,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Import a beliefs.md registry into the RMS network.

    Returns: {"claims_imported": int, "claims_skipped": int, "claims_retracted": int, "nogoods_imported": int}
    """
    beliefs_path = Path(beliefs_file)
    if not beliefs_path.exists():
        raise FileNotFoundError(f"File not found: {beliefs_file}")

    beliefs_text = beliefs_path.read_text()

    nogoods_text = None
    if nogoods_file:
        nogoods_path = Path(nogoods_file)
        if not nogoods_path.exists():
            raise FileNotFoundError(f"Nogoods file not found: {nogoods_file}")
        nogoods_text = nogoods_path.read_text()
    else:
        auto_nogoods = beliefs_path.parent / "nogoods.md"
        if auto_nogoods.exists():
            nogoods_text = auto_nogoods.read_text()

    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "import_beliefs",
                            beliefs_text=beliefs_text, nogoods_text=nogoods_text)

    from .import_beliefs import import_into_network
    with _with_network(db_path, write=True) as net:
        return import_into_network(net, beliefs_text, nogoods_text)


def import_agent(
    agent_name: str,
    beliefs_file: str,
    nogoods_file: str | None = None,
    only_in: bool = False,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Import another agent's beliefs into the local RMS with namespacing.

    Accepts beliefs.md (markdown) or network.json (JSON export) files.
    JSON files preserve full justification structure including outlists.

    Each belief is prefixed with 'agent_name:' and depends on a premise
    node 'agent_name:active'. Retracting that premise cascades OUT all
    beliefs from that agent.

    Returns: {"agent": str, "claims_imported": int, "claims_skipped": int, ...}
    """
    beliefs_path = Path(beliefs_file)
    if not beliefs_path.exists():
        raise FileNotFoundError(f"File not found: {beliefs_file}")

    if pg_conninfo:
        from .import_agent import (
            _normalize_json, _normalize_markdown,
            _normalize_nogoods_json, _normalize_nogoods_markdown,
        )
        if beliefs_path.suffix == ".json":
            import json as json_mod
            data = json_mod.loads(beliefs_path.read_text())
            claims = _normalize_json(data, only_in)
            nogoods = _normalize_nogoods_json(data)
        else:
            beliefs_text = beliefs_path.read_text()
            nogoods_text = None
            if nogoods_file:
                nogoods_path = Path(nogoods_file)
                if not nogoods_path.exists():
                    raise FileNotFoundError(f"Nogoods file not found: {nogoods_file}")
                nogoods_text = nogoods_path.read_text()
            else:
                auto_nogoods = beliefs_path.parent / "nogoods.md"
                if auto_nogoods.exists():
                    nogoods_text = auto_nogoods.read_text()
            claims = _normalize_markdown(beliefs_text, only_in)
            nogoods = _normalize_nogoods_markdown(nogoods_text)
        return _pg_dispatch(pg_conninfo, project_id, "import_agent",
                            agent_name=agent_name, claims=claims,
                            nogoods=nogoods, source_path=str(beliefs_path))

    if beliefs_path.suffix == ".json":
        from .import_agent import import_agent_json as _import_agent_json
        import json as json_mod

        data = json_mod.loads(beliefs_path.read_text())

        with _with_network(db_path, write=True) as net:
            return _import_agent_json(
                net,
                agent_name=agent_name,
                data=data,
                only_in=only_in,
                source_path=str(beliefs_path),
            )

    from .import_agent import import_agent as _import_agent

    beliefs_text = beliefs_path.read_text()

    nogoods_text = None
    if nogoods_file:
        nogoods_path = Path(nogoods_file)
        if not nogoods_path.exists():
            raise FileNotFoundError(f"Nogoods file not found: {nogoods_file}")
        nogoods_text = nogoods_path.read_text()
    else:
        auto_nogoods = beliefs_path.parent / "nogoods.md"
        if auto_nogoods.exists():
            nogoods_text = auto_nogoods.read_text()

    with _with_network(db_path, write=True) as net:
        return _import_agent(
            net,
            agent_name=agent_name,
            beliefs_text=beliefs_text,
            nogoods_text=nogoods_text,
            only_in=only_in,
            source_path=str(beliefs_path),
        )


def sync_agent(
    agent_name: str,
    beliefs_file: str,
    nogoods_file: str | None = None,
    only_in: bool = False,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Sync another agent's beliefs into the local RMS (remote wins).

    Accepts beliefs.md (markdown) or network.json (JSON export) files.
    Updates existing beliefs, adds new ones, retracts removed ones.

    Returns: {"agent": str, "beliefs_added": int, "beliefs_updated": int, ...}
    """
    beliefs_path = Path(beliefs_file)
    if not beliefs_path.exists():
        raise FileNotFoundError(f"File not found: {beliefs_file}")

    if pg_conninfo:
        from .import_agent import (
            _normalize_json, _normalize_markdown,
            _normalize_nogoods_json, _normalize_nogoods_markdown,
        )
        if beliefs_path.suffix == ".json":
            import json as json_mod
            data = json_mod.loads(beliefs_path.read_text())
            claims = _normalize_json(data, only_in)
            nogoods = _normalize_nogoods_json(data)
        else:
            beliefs_text = beliefs_path.read_text()
            nogoods_text = None
            if nogoods_file:
                nogoods_path = Path(nogoods_file)
                if not nogoods_path.exists():
                    raise FileNotFoundError(f"Nogoods file not found: {nogoods_file}")
                nogoods_text = nogoods_path.read_text()
            else:
                auto_nogoods = beliefs_path.parent / "nogoods.md"
                if auto_nogoods.exists():
                    nogoods_text = auto_nogoods.read_text()
            claims = _normalize_markdown(beliefs_text, only_in)
            nogoods = _normalize_nogoods_markdown(nogoods_text)
        return _pg_dispatch(pg_conninfo, project_id, "sync_agent",
                            agent_name=agent_name, claims=claims,
                            nogoods=nogoods, source_path=str(beliefs_path))

    if beliefs_path.suffix == ".json":
        from .import_agent import sync_agent_json as _sync_agent_json
        import json as json_mod

        data = json_mod.loads(beliefs_path.read_text())

        with _with_network(db_path, write=True) as net:
            return _sync_agent_json(
                net,
                agent_name=agent_name,
                data=data,
                only_in=only_in,
                source_path=str(beliefs_path),
            )

    from .import_agent import sync_agent as _sync_agent

    beliefs_text = beliefs_path.read_text()

    nogoods_text = None
    if nogoods_file:
        nogoods_path = Path(nogoods_file)
        if not nogoods_path.exists():
            raise FileNotFoundError(f"Nogoods file not found: {nogoods_file}")
        nogoods_text = nogoods_path.read_text()
    else:
        auto_nogoods = beliefs_path.parent / "nogoods.md"
        if auto_nogoods.exists():
            nogoods_text = auto_nogoods.read_text()

    with _with_network(db_path, write=True) as net:
        return _sync_agent(
            net,
            agent_name=agent_name,
            beliefs_text=beliefs_text,
            nogoods_text=nogoods_text,
            only_in=only_in,
            source_path=str(beliefs_path),
        )


def import_json(json_file: str, db_path: str = DEFAULT_DB,
                pg_conninfo=None, project_id=None) -> dict:
    """Import a network from a JSON file (produced by export).

    Reconstructs the full network: nodes with justifications, truth values,
    metadata, and nogoods. This is a lossless round-trip with export.

    Returns: {"nodes_imported": int, "nogoods_imported": int}
    """
    import json as json_mod

    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_file}")

    data = json_mod.loads(json_path.read_text())

    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "import_json", data=data)

    with _with_network(db_path, write=True) as net:
        imported_meta = data.get("meta")
        if imported_meta and isinstance(imported_meta, dict):
            for key in ("schema_version", "project_name", "created_at"):
                if key in imported_meta:
                    net.meta[key] = imported_meta[key]

        # Topological sort: add nodes whose antecedents are already in the network first
        remaining = dict(data.get("nodes", {}))
        added = set(net.nodes.keys())
        nodes_imported = 0
        skipped = 0

        max_passes = len(remaining) + 1
        for _ in range(max_passes):
            if not remaining:
                break
            next_remaining = {}
            for nid, ndata in remaining.items():
                if nid in added:
                    skipped += 1
                    continue
                # Check if all antecedents and outlist deps are available
                all_deps = set()
                for j in ndata.get("justifications", []):
                    all_deps.update(j.get("antecedents", []))
                    all_deps.update(j.get("outlist", []))
                deps_in_data = {d for d in all_deps if d in data.get("nodes", {})}
                if all(d in added for d in deps_in_data):
                    # Ready to add
                    justifications = None
                    jlist = ndata.get("justifications", [])
                    if jlist:
                        justifications = [
                            Justification(
                                type=j["type"],
                                antecedents=j.get("antecedents", []),
                                outlist=j.get("outlist", []),
                                label=j.get("label", ""),
                                content_hash=j.get("content_hash", ""),
                            )
                            for j in jlist
                        ]
                    node = net.add_node(
                        id=nid,
                        text=ndata.get("text", ""),
                        justifications=justifications,
                        source=ndata.get("source", ""),
                        source_url=ndata.get("source_url", ""),
                        source_hash=ndata.get("source_hash", ""),
                        date=ndata.get("date", ""),
                        metadata=ndata.get("metadata", {}),
                        created_at=ndata.get("created_at", ""),
                        updated_at=ndata.get("updated_at", ""),
                    )
                    # Restore exact truth value (may differ from computed if retracted)
                    target_tv = ndata.get("truth_value", "IN")
                    if node.truth_value != target_tv:
                        if target_tv == "OUT":
                            net.retract(nid)
                        else:
                            net.assert_node(nid)
                    # Restore exact timestamps AFTER retract/assert to avoid overwrite
                    node.reviewed_at = ndata.get("reviewed_at", "")
                    node.verified_at = ndata.get("verified_at", "")
                    node.retracted_at = ndata.get("retracted_at", "")
                    node.updated_at = ndata.get("updated_at", "")
                    added.add(nid)
                    nodes_imported += 1
                else:
                    next_remaining[nid] = ndata
            if len(next_remaining) == len(remaining):
                # No progress — add remaining anyway
                for nid, ndata in next_remaining.items():
                    if nid in added:
                        continue
                    justifications = None
                    jlist = ndata.get("justifications", [])
                    if jlist:
                        justifications = [
                            Justification(
                                type=j["type"],
                                antecedents=j.get("antecedents", []),
                                outlist=j.get("outlist", []),
                                label=j.get("label", ""),
                                content_hash=j.get("content_hash", ""),
                            )
                            for j in jlist
                        ]
                    node = net.add_node(
                        id=nid,
                        text=ndata.get("text", ""),
                        justifications=justifications,
                        source=ndata.get("source", ""),
                        source_url=ndata.get("source_url", ""),
                        source_hash=ndata.get("source_hash", ""),
                        date=ndata.get("date", ""),
                        metadata=ndata.get("metadata", {}),
                        created_at=ndata.get("created_at", ""),
                        updated_at=ndata.get("updated_at", ""),
                    )
                    target_tv = ndata.get("truth_value", "IN")
                    if node.truth_value != target_tv:
                        if target_tv == "OUT":
                            net.retract(nid)
                        else:
                            net.assert_node(nid)
                    node.reviewed_at = ndata.get("reviewed_at", "")
                    node.verified_at = ndata.get("verified_at", "")
                    node.retracted_at = ndata.get("retracted_at", "")
                    node.updated_at = ndata.get("updated_at", "")
                    added.add(nid)
                    nodes_imported += 1
                break
            remaining = next_remaining

        # Import nogoods

        from . import Nogood
        nogoods_imported = 0
        for ng_data in data.get("nogoods", []):
            nogood = Nogood(
                id=ng_data["id"],
                nodes=ng_data.get("nodes", []),
                discovered=ng_data.get("discovered", ""),
                resolution=ng_data.get("resolution", ""),
            )
            net.nogoods.append(nogood)
            m = re.fullmatch(r"nogood-(\d+)", nogood.id)
            if m:
                net._next_nogood_id = max(net._next_nogood_id, int(m.group(1)) + 1)
            nogoods_imported += 1

        # Import repos
        for name, path in data.get("repos", {}).items():
            net.repos[name] = path

        return {"nodes_imported": nodes_imported, "nogoods_imported": nogoods_imported}


def import_hf(repo_id: str, init: bool = False, token: str | None = None,
              db_path: str = DEFAULT_DB,
              pg_conninfo=None, project_id=None) -> dict:
    """Download network.json from a HuggingFace repo and import it.

    Args:
        repo_id: HuggingFace repo ID (user/repo) or full URL
        init: Initialize reasons.db before import if it doesn't exist
        token: Optional HuggingFace auth token

    Returns: {"nodes_imported": int, "nogoods_imported": int, "repo_id": str}
    """
    import json as json_mod
    import tempfile

    from .hf import download_network

    json_str = download_network(repo_id, token=token)

    db_exists = Path(db_path).exists()
    if not db_exists and (init or not pg_conninfo):
        data = json_mod.loads(json_str)
        project_name = data.get("meta", {}).get("project_name", "")
        init_db(db_path=db_path, force=False,
                project_name=project_name,
                pg_conninfo=pg_conninfo, project_id=project_id)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json_str)
        temp_path = f.name

    try:
        result = import_json(temp_path, db_path=db_path,
                             pg_conninfo=pg_conninfo, project_id=project_id)
    finally:
        Path(temp_path).unlink()

    from .hf import _parse_repo_id
    result["repo_id"] = _parse_repo_id(repo_id)
    return result


def publish_hf(
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    visible_to: list[str] | None = None,
    domain: list[str] | None = None,
    license: str = "mit",
    base_network: str | None = None,
    source_repos: list[str] | None = None,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Export and publish network to a HuggingFace repo.

    Uploads network.json, beliefs.md, and README.md (EEM card).

    Args:
        repo_id: HuggingFace repo ID (bare name, user/repo, or URL)
        token: HuggingFace auth token (falls back to HF_TOKEN env or cached)
        private: Create a private repo
        visible_to: Access tag filter for exported nodes

    Returns: {"repo_id": str, "url": str, "files_uploaded": list}
    """
    from .hf import _resolve_token, create_repo, resolve_repo_id, upload_file

    resolved_token = _resolve_token(token)
    if not resolved_token:
        raise RuntimeError(
            "HuggingFace token required for publishing. "
            "Run 'huggingface-cli login', set HF_TOKEN, or pass --token."
        )

    parsed_id = resolve_repo_id(repo_id)
    create_repo(parsed_id, token=resolved_token, private=private)

    backend = dict(db_path=db_path, pg_conninfo=pg_conninfo, project_id=project_id)
    network_json = json.dumps(
        export_network(visible_to=visible_to, **backend), indent=2
    )

    beliefs_md = export_markdown(visible_to=visible_to, **backend)

    card_md = export_card(
        visible_to=visible_to, domain=domain, license=license,
        base_network=base_network, source_repos=source_repos,
        **backend,
    )

    files = []
    for filename, content in [
        ("network.json", network_json),
        ("beliefs.md", beliefs_md),
        ("README.md", card_md),
    ]:
        upload_file(parsed_id, filename, content.encode("utf-8"), resolved_token)
        files.append(filename)

    return {
        "repo_id": parsed_id,
        "url": f"https://huggingface.co/{parsed_id}",
        "files_uploaded": files,
    }


def import_api(url: str | None = None, agent_id: str | None = None,
               api_key: str | None = None, init: bool = False,
               db_path: str = DEFAULT_DB,
               pg_conninfo=None, project_id=None) -> dict:
    """Import beliefs from agentic-mind-service.

    Downloads the full network via GET /export and imports it locally.

    Returns: {"nodes_imported": int, "nogoods_imported": int}
    """
    import tempfile

    from .mind_service import _resolve_config, fetch_export

    resolved_url, resolved_agent_id, resolved_api_key = _resolve_config(
        url, agent_id, api_key)
    json_str = fetch_export(resolved_url, resolved_agent_id, resolved_api_key)

    db_exists = Path(db_path).exists()
    if not db_exists and (init or not pg_conninfo):
        data = json.loads(json_str)
        project_name = data.get("meta", {}).get("project_name", "")
        init_db(db_path=db_path, force=False,
                project_name=project_name,
                pg_conninfo=pg_conninfo, project_id=project_id)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json_str)
        temp_path = f.name

    try:
        return import_json(temp_path, db_path=db_path,
                           pg_conninfo=pg_conninfo, project_id=project_id)
    finally:
        Path(temp_path).unlink()


def export_api(url: str | None = None, agent_id: str | None = None,
               api_key: str | None = None,
               db_path: str = DEFAULT_DB,
               pg_conninfo=None, project_id=None) -> dict:
    """Export local beliefs to agentic-mind-service.

    Pushes each node via POST /beliefs. Idempotent on node_id.

    Returns: {"nodes_exported": int, "errors": int}
    """
    from .mind_service import _resolve_config, push_belief

    resolved_url, resolved_agent_id, resolved_api_key = _resolve_config(
        url, agent_id, api_key)

    data = export_network(db_path=db_path, pg_conninfo=pg_conninfo,
                          project_id=project_id)

    exported = 0
    errors = 0
    for nid, ndata in data.get("nodes", {}).items():
        sl_parts = []
        for j in ndata.get("justifications", []):
            if j.get("type") == "SL" and j.get("antecedents"):
                sl_parts.extend(j["antecedents"])
        sl = ",".join(sl_parts) if sl_parts else ""

        try:
            push_belief(
                resolved_url, resolved_agent_id, resolved_api_key,
                node_id=nid,
                text=ndata.get("text", ""),
                sl=sl,
                source=ndata.get("source", ""),
            )
            exported += 1
        except Exception as exc:
            logger.warning("Failed to push node %s: %s", nid, exc)
            errors += 1

    return {"nodes_exported": exported, "errors": errors}


def derive_prompt(domain: str | None = None, db_path: str = DEFAULT_DB) -> dict:
    """Build a derive prompt from the current network.

    Returns: {"prompt": str, "stats": dict}
    """
    from .derive import build_prompt

    data = export_network(db_path=db_path)
    nodes = data.get("nodes", {})
    if not nodes:
        raise ValueError("No nodes in the network")

    prompt, stats = build_prompt(nodes, domain=domain)
    return {"prompt": prompt, "stats": stats}


def derive_apply(proposals: list[dict], db_path: str = DEFAULT_DB) -> dict:
    """Apply validated derive proposals to the network.

    Returns: {"added": list[dict], "failed": list[dict]}
    """
    from .derive import apply_proposals
    results = apply_proposals(proposals, db_path=db_path)

    added = []
    failed = []
    for p, result in results:
        if isinstance(result, dict):
            added.append({"id": p["id"], "truth_value": result["truth_value"]})
        else:
            failed.append({"id": p["id"], "error": result})

    return {"added": added, "failed": failed}


def add_repo(name: str, path: str, db_path: str = DEFAULT_DB,
             pg_conninfo=None, project_id=None) -> dict:
    """Add a repo to the network.

    Returns: {"name": str, "path": str}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "add_repo",
                            name=name, path=path)
    with _with_network(db_path, write=True) as net:
        net.repos[name] = path
        return {"name": name, "path": path}


def list_repos(db_path: str = DEFAULT_DB,
               pg_conninfo=None, project_id=None) -> dict:
    """List all repos.

    Returns: {"repos": dict[str, str]}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "list_repos")
    with _with_network(db_path) as net:
        return {"repos": dict(net.repos)}


def export_markdown(visible_to: list[str] | None = None, db_path: str = DEFAULT_DB,
                    pg_conninfo=None, project_id=None) -> str:
    """Export the network as beliefs.md-compatible markdown.

    Returns: the markdown string
    """
    from .export_markdown import export_markdown as _export

    if pg_conninfo:
        data = export_network(visible_to=visible_to, pg_conninfo=pg_conninfo,
                              project_id=project_id)
        from . import Node, Justification, Nogood
        from .network import Network
        net = Network()
        for nid, ndata in data.get("nodes", {}).items():
            node = Node(nid, ndata.get("text", ""))
            node.truth_value = ndata.get("truth_value", "OUT")
            node.source = ndata.get("source", "")
            node.source_url = ndata.get("source_url", "")
            node.source_hash = ndata.get("source_hash", "")
            node.date = ndata.get("date", "")
            node.metadata = ndata.get("metadata", {})
            for jdata in ndata.get("justifications", []):
                j = Justification(
                    type=jdata.get("type", "SL"),
                    antecedents=jdata.get("antecedents", []),
                    outlist=jdata.get("outlist", []),
                    label=jdata.get("label", ""),
                )
                node.justifications.append(j)
            net.nodes[nid] = node
        for nid, node in net.nodes.items():
            for j in node.justifications:
                for ant_id in j.antecedents:
                    if ant_id in net.nodes:
                        net.nodes[ant_id].dependents.add(nid)
                for out_id in j.outlist:
                    if out_id in net.nodes:
                        net.nodes[out_id].dependents.add(nid)
        for ngdata in data.get("nogoods", []):
            net.nogoods.append(Nogood(
                id=ngdata.get("id", ""),
                nodes=ngdata.get("nodes", []),
                discovered=ngdata.get("discovered", ""),
                resolution=ngdata.get("resolution", ""),
            ))
        repos = data.get("repos", {})
        meta = data.get("meta")
        if meta:
            net.meta = dict(meta)
        return _export(net, repos=repos)

    with _with_network(db_path) as net:
        if visible_to is not None:
            from .network import Network
            filtered = Network()
            for nid, node in net.nodes.items():
                if _is_visible(node, visible_to):
                    filtered.nodes[nid] = node
            filtered.nogoods = [ng for ng in net.nogoods if all(n in filtered.nodes for n in ng.nodes)]
            filtered.repos = net.repos
            filtered.meta = net.meta
            return _export(filtered, repos=filtered.repos)
        return _export(net, repos=net.repos)


def export_card(visible_to: list[str] | None = None, db_path: str = DEFAULT_DB,
                pg_conninfo=None, project_id=None,
                domain=None, license="mit", base_network=None,
                source_repos=None) -> str:
    """Export the network as a HuggingFace EEM card (README.md).

    Returns: the markdown string
    """
    from .export_card import export_card as _export_card

    card_kwargs = dict(domain=domain, license=license,
                       base_network=base_network, source_repos=source_repos)

    if pg_conninfo:
        data = export_network(visible_to=visible_to, pg_conninfo=pg_conninfo,
                              project_id=project_id)
        from . import Node, Justification, Nogood
        from .network import Network
        net = Network()
        for nid, ndata in data.get("nodes", {}).items():
            node = Node(nid, ndata.get("text", ""))
            node.truth_value = ndata.get("truth_value", "OUT")
            node.source = ndata.get("source", "")
            node.source_url = ndata.get("source_url", "")
            node.source_hash = ndata.get("source_hash", "")
            node.date = ndata.get("date", "")
            node.metadata = ndata.get("metadata", {})
            for jdata in ndata.get("justifications", []):
                j = Justification(
                    type=jdata.get("type", "SL"),
                    antecedents=jdata.get("antecedents", []),
                    outlist=jdata.get("outlist", []),
                    label=jdata.get("label", ""),
                )
                node.justifications.append(j)
            net.nodes[nid] = node
        for nid, node in net.nodes.items():
            for j in node.justifications:
                for ant_id in j.antecedents:
                    if ant_id in net.nodes:
                        net.nodes[ant_id].dependents.add(nid)
                for out_id in j.outlist:
                    if out_id in net.nodes:
                        net.nodes[out_id].dependents.add(nid)
        for ngdata in data.get("nogoods", []):
            net.nogoods.append(Nogood(
                id=ngdata.get("id", ""),
                nodes=ngdata.get("nodes", []),
                discovered=ngdata.get("discovered", ""),
                resolution=ngdata.get("resolution", ""),
            ))
        meta = data.get("meta")
        if meta:
            net.meta = dict(meta)
        return _export_card(net, **card_kwargs)

    with _with_network(db_path) as net:
        if visible_to is not None:
            from .network import Network
            filtered = Network()
            for nid, node in net.nodes.items():
                if _is_visible(node, visible_to):
                    filtered.nodes[nid] = node
            filtered.nogoods = [ng for ng in net.nogoods if all(n in filtered.nodes for n in ng.nodes)]
            filtered.repos = net.repos
            filtered.meta = net.meta
            return _export_card(filtered, **card_kwargs)
        return _export_card(net, **card_kwargs)


def check_stale(
    repos: dict[str, str] | None = None,
    upgrade_hashes: bool = False,
    git_aware: bool = False,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Check all IN nodes for source file staleness.

    If upgrade_hashes=True, truncated hashes that prefix-match the current
    full hash are upgraded in place and saved to the database.

    If git_aware=True, nodes with pinned_sha skip content hashing when
    no commits touched the file since the pinned SHA.

    Returns: {"stale": list[dict], "checked": int, "stale_count": int,
              "upgraded": int, "sha_bumped": int}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "check_stale",
                            repos=repos, upgrade_hashes=upgrade_hashes,
                            git_aware=git_aware)
    from pathlib import Path as P
    from .check_stale import check_stale as _check

    db_dir = P(db_path).resolve().parent

    needs_write = upgrade_hashes or git_aware
    with _with_network(db_path, write=needs_write) as net:
        repo_paths = repos
        if repo_paths is None and net.repos:
            repo_paths = net.repos
        if repo_paths:
            repo_paths = {k: P(v) for k, v in repo_paths.items()}

        in_with_source = sum(
            1 for n in net.nodes.values()
            if n.truth_value == "IN" and n.source and n.source_hash
        )
        results, upgraded, sha_bumped = _check(net, repo_paths, db_dir=db_dir,
                                               upgrade_hashes=upgrade_hashes,
                                               git_aware=git_aware)
        return {
            "stale": results,
            "checked": in_with_source,
            "stale_count": len(results),
            "upgraded": upgraded,
            "sha_bumped": sha_bumped,
        }


def check_integrity(db_path: str = DEFAULT_DB) -> dict:
    """Verify Merkle hashes for all nodes and justifications.

    Returns: {"text_mutations": [...], "chain_mutations": [...], "missing_hashes": int}
    """
    from .merkle import verify_all
    with _with_network(db_path, write=False) as net:
        result = verify_all(net)
    text_mutations = [f for f in result["findings"] if f["type"] == "text_mutation"]
    chain_mutations = [f for f in result["findings"] if f["type"] == "chain_mutation"]
    return {
        "text_mutations": text_mutations,
        "chain_mutations": chain_mutations,
        "missing_hashes": result["missing_hashes"],
    }


def backfill_hashes(db_path: str = DEFAULT_DB) -> dict:
    """Compute and store hashes for nodes/justifications missing them."""
    from .merkle import backfill_hashes as _backfill
    with _with_network(db_path, write=True) as net:
        return _backfill(net)


def hash_sources(
    force: bool = False,
    repos: dict[str, str] | None = None,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Backfill source hashes for nodes with source paths but no stored hash.

    Returns: {"hashed": list[dict], "count": int}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "hash_sources",
                            force=force, repos=repos)
    from pathlib import Path as P
    from .check_stale import hash_sources as _hash

    db_dir = P(db_path).resolve().parent

    with _with_network(db_path, write=True) as net:
        repo_paths = repos
        if repo_paths is None and net.repos:
            repo_paths = net.repos
        if repo_paths:
            repo_paths = {k: P(v) for k, v in repo_paths.items()}

        results = _hash(net, repo_paths, force=force, db_dir=db_dir)
        return {"hashed": results, "count": len(results)}


def pin_sources(
    force: bool = False,
    pin_urls: bool = False,
    repos: dict[str, str] | None = None,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Pin IN nodes to their current git commit SHA.

    Stores pinned_sha and verified_at in node metadata.

    Returns: {"pinned": list[dict], "count": int}
    """
    from pathlib import Path as P
    from .check_stale import pin_sources as _pin

    db_dir = P(db_path).resolve().parent

    with _with_network(db_path, write=True) as net:
        repo_paths = repos
        if repo_paths is None and net.repos:
            repo_paths = net.repos
        if repo_paths:
            repo_paths = {k: P(v) for k, v in repo_paths.items()}

        results = _pin(net, repo_paths, db_dir=db_dir,
                        force=force, pin_urls=pin_urls)
        return {"pinned": results, "count": len(results)}


def pin_update(
    node_ids: list[str],
    repos: dict[str, str] | None = None,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Bump pinned_sha to current HEAD for specified nodes.

    Returns: {"updated": list[dict], "count": int, "errors": int}
    """
    from pathlib import Path as P
    from .check_stale import pin_update as _update

    db_dir = P(db_path).resolve().parent

    with _with_network(db_path, write=True) as net:
        repo_paths = repos
        if repo_paths is None and net.repos:
            repo_paths = net.repos
        if repo_paths:
            repo_paths = {k: P(v) for k, v in repo_paths.items()}

        results = _update(net, node_ids, repo_paths, db_dir=db_dir)
        errors = sum(1 for r in results if "error" in r)
        return {"updated": results, "count": len(results) - errors,
                "errors": errors}


def pin_lines(
    node_id: str,
    line_start: int,
    line_end: int,
    repos: dict[str, str] | None = None,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Set pinned_lines metadata on a node.

    If the node doesn't have a pinned_sha yet, auto-pins it.

    Returns: {"node_id", "pinned_lines", "pinned_sha", "auto_pinned"}
    """
    from pathlib import Path as P
    from .check_stale import pin_lines as _pin_lines

    db_dir = P(db_path).resolve().parent

    with _with_network(db_path, write=True) as net:
        repo_paths = repos
        if repo_paths is None and net.repos:
            repo_paths = net.repos
        if repo_paths:
            repo_paths = {k: P(v) for k, v in repo_paths.items()}

        return _pin_lines(net, node_id, line_start, line_end,
                          repos=repo_paths, db_dir=db_dir)


def compact(budget: int = 500, truncate: bool = True, visible_to: list[str] | None = None, db_path: str = DEFAULT_DB,
            pg_conninfo=None, project_id=None) -> str:
    """Generate a token-budgeted belief state summary.

    Returns: the compact summary string
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "compact",
                            budget=budget, truncate=truncate, visible_to=visible_to)
    from .compact import compact as _compact

    with _with_network(db_path) as net:
        if visible_to is not None:
            from .network import Network
            filtered = Network()
            for nid, node in net.nodes.items():
                if _is_visible(node, visible_to):
                    filtered.nodes[nid] = node
            filtered.nogoods = [ng for ng in net.nogoods if all(n in filtered.nodes for n in ng.nodes)]
            return _compact(filtered, budget=budget, truncate=truncate)
        return _compact(net, budget=budget, truncate=truncate)


def lookup(query: str, visible_to: list[str] | None = None, db_path: str = DEFAULT_DB,
           pg_conninfo=None, project_id=None) -> str:
    """Simple all-terms search over the full belief block — ID, text, source,
    dependencies, and metadata. Matches the same search corpus and output
    format as lookup_beliefs on a flat beliefs.md file.

    Args:
        query: search terms (all must appear, case-insensitive)
        visible_to: only return nodes whose access_tags are a subset
        db_path: path to RMS database

    Returns: formatted string with matching beliefs (full blocks)
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "lookup",
                            query=query, visible_to=visible_to)
    with _with_network(db_path) as net:
        query_terms = query.lower().split()
        matches = []
        for nid, node in sorted(net.nodes.items()):
            if visible_to is not None and not _is_visible(node, visible_to):
                continue
            # Build the full searchable block — same fields as beliefs.md
            block_parts = [nid, node.text]
            if node.source:
                block_parts.append(node.source)
            if node.source_hash:
                block_parts.append(node.source_hash)
            if node.date:
                block_parts.append(node.date)
            for j in node.justifications:
                block_parts.extend(j.antecedents)
            for dep_id in node.dependents:
                block_parts.append(dep_id)
            block_lower = " ".join(block_parts).lower()
            if all(term in block_lower for term in query_terms):
                matches.append(node)

        if not matches:
            return f"No beliefs found matching '{query}'"

        parts = [f"Found {len(matches)} matching belief(s):", ""]
        for node in matches[:20]:
            parts.append(f"### {node.id} [{node.truth_value}]")
            parts.append(node.text)
            if node.source:
                parts.append(f"- Source: {node.source}")
            if node.source_hash:
                parts.append(f"- Source hash: {node.source_hash}")
            if node.date:
                parts.append(f"- Date: {node.date}")
            deps = []
            for j in node.justifications:
                deps.extend(j.antecedents)
            if deps:
                parts.append(f"- Depends on: {', '.join(deps)}")
            parts.append("")

        return "\n".join(parts)


def search(query: str, visible_to: list[str] | None = None, db_path: str = DEFAULT_DB,
           format: str = "markdown", depth: int = 1,
           pg_conninfo=None, project_id=None) -> str:
    """Search nodes using full-text search with neighbor expansion.

    Uses SQLite FTS5 for ranked all-terms matching. Returns matched nodes
    plus their neighbors (dependencies and dependents) formatted
    as readable markdown.

    Falls back to substring matching if FTS5 table is not available.

    Args:
        query: search terms (FTS5 matches all terms in any order)
        visible_to: only return nodes whose access_tags are a subset
        db_path: path to RMS database
        format: output format — "markdown" (default), "json", or "minimal"
        depth: number of hops to expand along justification chains (default: 1)

    Returns: formatted string with matched nodes and neighbors
    """
    if pg_conninfo:
        if depth != 1:
            raise NotImplementedError("depth is not supported with PostgreSQL")
        return _pg_dispatch(pg_conninfo, project_id, "search",
                            query=query, visible_to=visible_to, format=format)
    with _with_network(db_path) as net:
        matched_ids = _fts_search(query, db_path)

        # Fallback to substring if FTS returned nothing or isn't available
        if not matched_ids:
            matched_ids = _substring_search(query, net)

        if not matched_ids:
            return "No results found."

        # Apply access filtering
        if visible_to is not None:
            matched_ids = [
                nid for nid in matched_ids
                if nid in net.nodes and _is_visible(net.nodes[nid], visible_to)
            ]
            if not matched_ids:
                return "No results found."

        # Expand to include neighbors (BFS along dependency graph)
        neighbor_ids = set()
        frontier = set(matched_ids)
        visited = set(matched_ids)
        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                if nid not in net.nodes:
                    continue
                node = net.nodes[nid]
                for j in node.justifications:
                    for ant_id in j.antecedents:
                        if ant_id in net.nodes and ant_id not in visited:
                            neighbor_ids.add(ant_id)
                            next_frontier.add(ant_id)
                for dep_id in node.dependents:
                    if dep_id in net.nodes and dep_id not in visited:
                        neighbor_ids.add(dep_id)
                        next_frontier.add(dep_id)
            visited |= next_frontier
            frontier = next_frontier

        # Remove already-matched nodes from neighbors
        neighbor_ids -= set(matched_ids)

        # Apply access filtering to neighbors too
        if visible_to is not None:
            neighbor_ids = {
                nid for nid in neighbor_ids
                if nid in net.nodes and _is_visible(net.nodes[nid], visible_to)
            }

        if format == "json":
            return _format_json(net, matched_ids, neighbor_ids)
        elif format == "minimal":
            return _format_minimal(net, matched_ids, neighbor_ids)
        elif format == "compact":
            return _format_compact(net, matched_ids, neighbor_ids)
        else:
            return _format_markdown(net, matched_ids, neighbor_ids)


def _fts_query(conn, terms: list[str]) -> list[str]:
    fts_query = " ".join(f'"{t}"' for t in terms)
    cursor = conn.execute(
        "SELECT id FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY rank LIMIT 20",
        (fts_query,),
    )
    return [row[0] for row in cursor.fetchall()]


_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "about", "against", "along", "among",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "also", "now", "how", "what", "when", "where",
    "which", "who", "whom", "why", "this", "that", "these", "those",
    "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "we", "you", "your", "my", "our", "me", "us", "him",
    "if", "then", "else", "while", "until", "unless",
    "there", "here", "up", "out", "off",
    "specific", "specifically", "particular", "particularly",
    "included", "including", "within",
})


def _fts_search(query: str, db_path: str) -> list[str]:
    """Search using FTS5 full-text index with porter stemming and progressive relaxation."""
    try:
        conn = sqlite3.connect(db_path)
        try:
            raw_terms = re.findall(r'\w+', query)
            terms = [t for t in raw_terms if t.lower() not in _STOP_WORDS and len(t) > 1]
            if not terms:
                terms = [t for t in raw_terms if len(t) > 1]
            if not terms:
                return []

            results = _fts_query(conn, terms)

            _MAX_RELAXATION_QUERIES = 50
            if not results and len(terms) > 2:
                min_terms = max(1, len(terms) // 2)
                budget = _MAX_RELAXATION_QUERIES
                for n in range(len(terms) - 1, min_terms - 1, -1):
                    for combo in combinations(terms, n):
                        budget -= 1
                        if budget < 0:
                            return results
                        results = _fts_query(conn, list(combo))
                        if results:
                            return results

            return results
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []


def _substring_search(query: str, net) -> list[str]:
    """Fallback: substring matching on node id and text."""
    q = query.lower()
    results = []
    for nid, node in sorted(net.nodes.items()):
        if q in nid.lower() or q in node.text.lower():
            results.append(nid)
    return results


def _format_markdown(net, matched_ids: list[str], neighbor_ids: set[str]) -> str:
    """Format results as readable markdown with neighbors."""
    parts = []
    for nid in matched_ids:
        node = net.nodes[nid]
        parts.append(f"### {nid}")
        parts.append(f"**Status:** {node.truth_value}")
        parts.append(f"{node.text}")
        if node.source:
            parts.append(f"**Source:** {node.source}")
        if node.justifications:
            deps = []
            for j in node.justifications:
                deps.extend(j.antecedents)
            if deps:
                parts.append(f"**Depends on:** {', '.join(deps)}")
        if node.dependents:
            parts.append(f"**Depended on by:** {', '.join(sorted(node.dependents))}")
        parts.append("")

    if neighbor_ids:
        parts.append("---")
        parts.append("**Related nodes:**\n")
        for nid in sorted(neighbor_ids):
            node = net.nodes[nid]
            parts.append(f"- **{nid}** ({node.truth_value}): {node.text}")
        parts.append("")

    return "\n".join(parts)


def _format_json(net, matched_ids: list[str], neighbor_ids: set[str]) -> str:
    """Format results as JSON."""
    results = []
    for nid in matched_ids:
        node = net.nodes[nid]
        results.append({
            "id": nid,
            "text": node.text,
            "truth_value": node.truth_value,
            "source": node.source,
            "match": True,
        })
    for nid in sorted(neighbor_ids):
        node = net.nodes[nid]
        results.append({
            "id": nid,
            "text": node.text,
            "truth_value": node.truth_value,
            "source": node.source,
            "match": False,
            "relation": "neighbor",
        })
    return json.dumps(results, indent=2)


def _format_minimal(net, matched_ids: list[str], neighbor_ids: set[str]) -> str:
    """Format results as plain text, claims only."""
    parts = []
    for nid in matched_ids:
        parts.append(net.nodes[nid].text)
    if neighbor_ids:
        parts.append("")
        for nid in sorted(neighbor_ids):
            parts.append(net.nodes[nid].text)
    return "\n".join(parts)


def _format_compact(net, matched_ids: list[str], neighbor_ids: set[str]) -> str:
    """Format results as one line per belief: [STATUS] id — text."""
    lines = []
    for nid in matched_ids:
        node = net.nodes[nid]
        lines.append(f"[{node.truth_value}] {nid} — {node.text}")
    for nid in sorted(neighbor_ids):
        node = net.nodes[nid]
        lines.append(f"[{node.truth_value}] {nid} — {node.text}")
    return "\n".join(lines) if lines else "No results found."


def _node_depth(nid, net, memo=None):
    """Compute depth of a node: 0 for premises, max(antecedent depths)+1 for derived."""
    if memo is None:
        memo = {}
    if nid in memo:
        return memo[nid]
    node = net.nodes.get(nid)
    if not node or not node.justifications:
        memo[nid] = 0
        return 0
    memo[nid] = 0  # cycle guard
    si = node.supporting_justification
    if si is not None and 0 <= si < len(node.justifications):
        justifications = [node.justifications[si]]
    else:
        justifications = node.justifications
    max_d = 0
    for j in justifications:
        for a in j.antecedents:
            max_d = max(max_d, _node_depth(a, net, memo))
    memo[nid] = max_d + 1
    return max_d + 1


def list_nodes(
    status: str | None = None,
    premises_only: bool = False,
    has_dependents: bool = False,
    challenged: bool = False,
    namespace: str | None = None,
    min_depth: int | None = None,
    max_depth: int | None = None,
    visible_to: list[str] | None = None,
    not_reviewed_since: int | None = None,
    never_reviewed: bool = False,
    by_impact: bool = False,
    label: str | None = None,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """List nodes with optional filters.

    Returns: {"nodes": list[dict], "count": int}
    """
    if pg_conninfo:
        unsupported = []
        if challenged:
            unsupported.append("challenged")
        if min_depth is not None:
            unsupported.append("min_depth")
        if max_depth is not None:
            unsupported.append("max_depth")
        if not_reviewed_since is not None:
            unsupported.append("not_reviewed_since")
        if never_reviewed:
            unsupported.append("never_reviewed")
        if by_impact:
            unsupported.append("by_impact")
        if unsupported:
            raise NotImplementedError(
                f"{', '.join(unsupported)} not supported with PostgreSQL")
        return _pg_dispatch(pg_conninfo, project_id, "list_nodes",
                            status=status, premises_only=premises_only,
                            has_dependents=has_dependents, namespace=namespace,
                            visible_to=visible_to, label=label)
    from datetime import timedelta

    with _with_network(db_path) as net:
        memo = {} if (min_depth is not None or max_depth is not None) else None
        if not_reviewed_since is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=not_reviewed_since)
        else:
            cutoff = None
        nodes = []
        for nid, node in sorted(net.nodes.items()):
            if namespace and not nid.startswith(f"{namespace}:"):
                continue
            if status and node.truth_value != status:
                continue
            if premises_only and node.justifications:
                continue
            if has_dependents and not node.dependents:
                continue
            if challenged and not node.metadata.get("challenges"):
                continue
            if label and not any(j.label == label for j in node.justifications):
                continue
            if visible_to is not None and not _is_visible(node, visible_to):
                continue
            if memo is not None:
                d = _node_depth(nid, net, memo)
                if min_depth is not None and d < min_depth:
                    continue
                if max_depth is not None and d > max_depth:
                    continue
            if never_reviewed:
                if not node.justifications:
                    continue
                if node.reviewed_at or node.metadata.get("last_reviewed"):
                    continue
            if cutoff is not None:
                if not node.justifications:
                    continue
                last = node.reviewed_at or node.metadata.get("last_reviewed", "")
                if last:
                    try:
                        dt = datetime.fromisoformat(last)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if dt >= cutoff:
                            continue
                    except ValueError:
                        pass
            nodes.append({
                "id": nid,
                "text": node.text,
                "truth_value": node.truth_value,
                "justification_count": len(node.justifications),
                "dependent_count": len(node.dependents),
                "challenges": node.metadata.get("challenges", []),
                "last_reviewed": node.reviewed_at or node.metadata.get("last_reviewed"),
                "review_result": node.metadata.get("review_result"),
                "source_type": node.metadata.get("source_type", ""),
            })
        if by_impact:
            nodes.sort(key=lambda n: -n["dependent_count"])
        return {"nodes": nodes, "count": len(nodes)}


_TOPIC_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "and", "or", "not", "as", "by", "via", "can", "with",
    "from", "than", "that", "this", "be", "has", "have", "it", "its",
    "no", "do", "if", "so", "up", "out", "all", "but", "get", "set",
    "only", "per", "use", "may", "one", "two", "new", "any", "each",
    "must", "when", "how", "also", "into", "over", "more", "both",
    "same", "own", "used", "using", "based", "does", "then",
}


def topics(
    limit: int = 20,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Extract topics from node IDs by word frequency.

    Returns: {"topics": [{"topic": str, "count": int}, ...], "total_nodes": int}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "topics", limit=limit)
    with _with_network(db_path) as net:
        word_counts: dict[str, int] = {}
        for nid in net.nodes:
            for word in re.split(r'[-._:]', nid):
                if word and len(word) > 2 and word not in _TOPIC_STOP_WORDS:
                    word_counts[word] = word_counts.get(word, 0) + 1
        ranked = sorted(word_counts, key=lambda w: (-word_counts[w], w))[:limit]
        return {
            "topics": [{"topic": t, "count": word_counts[t]} for t in ranked],
            "total_nodes": len(net.nodes),
        }


def build_wiki(
    output_dir: str = "wiki",
    status: str | None = None,
    max_topics: int = 20,
    cluster: bool = False,
    n_clusters: int | None = None,
    seed: int | None = None,
    embedding_model: str | None = None,
    visible_to: list[str] | None = None,
    model: str = "",
    timeout: int = 300,
    parallel: int = 0,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Export beliefs as interlinked markdown wiki pages grouped by topic or cluster.

    Returns: {"output_dir": str, "pages": int, "total_nodes": int}
    """
    from .build_wiki import _assign_topics, build_wiki as _build_wiki

    nodes_result = list_nodes(status=status, visible_to=visible_to, db_path=db_path)
    node_ids = [n["id"] for n in nodes_result["nodes"]]

    if not node_ids:
        import os
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "index.md"), "w") as f:
            f.write("# Belief Wiki\n\n*No beliefs found.*\n")
        return {"output_dir": output_dir, "pages": 0, "total_nodes": 0}

    node_details = {}
    for nid in node_ids:
        try:
            node_details[nid] = show_node(nid, visible_to=visible_to, db_path=db_path)
        except (KeyError, PermissionError):
            pass

    if cluster:
        cluster_result = list_clusters(
            status=status or "",
            n_clusters=n_clusters,
            seed=seed,
            embedding_model=embedding_model,
            visible_to=visible_to,
            db_path=db_path,
        )
        groups = {}
        for c in cluster_result["clusters"]:
            ids_in_cluster = [b["id"] for b in c["beliefs"]]
            word_counts: dict[str, int] = {}
            for nid in ids_in_cluster:
                for word in re.split(r'[-._:]', nid):
                    if word and len(word) > 2 and word not in _TOPIC_STOP_WORDS:
                        word_counts[word] = word_counts.get(word, 0) + 1
            label = max(word_counts, key=word_counts.get) if word_counts else f"cluster-{c['id']}"
            if label in groups:
                label = f"{label}-{c['id']}"
            groups[label] = ids_in_cluster
    else:
        topics_result = topics(limit=max_topics, db_path=db_path)
        groups = _assign_topics(node_ids, topics_result["topics"])

    return _build_wiki(node_details, groups, output_dir,
                       model=model, timeout=timeout, parallel=parallel)


def list_gated(
    visible_to: list[str] | None = None,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Find OUT nodes blocked by IN outlist nodes (active gates).

    Returns: {"blockers": {blocker_id: {"text": str, "gated": [{"id": str, "text": str}]}},
              "gated_count": int, "blocker_count": int}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "list_gated",
                            visible_to=visible_to)
    with _with_network(db_path) as net:
        blockers: dict[str, dict] = {}
        for nid, node in sorted(net.nodes.items()):
            if node.truth_value != "OUT":
                continue
            if node.metadata.get("superseded_by"):
                continue
            if visible_to is not None and not _is_visible(node, visible_to):
                continue
            for j in node.justifications:
                for outlist_id in j.outlist:
                    if outlist_id not in net.nodes:
                        continue
                    out_node = net.nodes[outlist_id]
                    if out_node.truth_value != "IN":
                        continue
                    if outlist_id not in blockers:
                        blockers[outlist_id] = {
                            "text": out_node.text,
                            "gated": [],
                        }
                    if not any(g["id"] == nid for g in blockers[outlist_id]["gated"]):
                        blockers[outlist_id]["gated"].append({
                            "id": nid,
                            "text": node.text,
                        })
        gated_count = sum(len(b["gated"]) for b in blockers.values())
        return {"blockers": blockers, "gated_count": gated_count, "blocker_count": len(blockers)}


def report_gated(
    visible_to: list[str] | None = None,
    model: str = "",
    timeout: int = 300,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Generate a problems/open-issues report from gated beliefs.

    Returns: {"report": str, "blocker_count": int, "gated_count": int,
              "retracted_count": int}
    """
    from .report_gated import report_gated as _report_gated

    backend = dict(db_path=db_path, pg_conninfo=pg_conninfo,
                   project_id=project_id)
    vis = dict(visible_to=visible_to)

    in_nodes = list_nodes(status="IN", **vis, **backend)
    out_nodes = list_nodes(status="OUT", **vis, **backend)
    in_count = in_nodes["count"]
    out_count = out_nodes["count"]

    out_premises = list_nodes(status="OUT", premises_only=True, **vis,
                              **backend)
    retracted = []
    for node in out_premises["nodes"]:
        detail = show_node(node["id"], **vis, **backend)
        reason = detail.get("metadata", {}).get("retract_reason")
        if reason:
            retracted.append({
                "id": node["id"],
                "text": node["text"],
                "retract_reason": reason,
                "dependent_count": node["dependent_count"],
            })

    gated_data = list_gated(**vis, **backend)

    blocker_details = {}
    for bid in gated_data.get("blockers", {}):
        try:
            detail = show_node(bid, **vis, **backend)
        except (KeyError, PermissionError):
            detail = {}
        blocker_details[bid] = {
            "dependent_count": len(detail.get("dependents", [])),
        }

    report = _report_gated(
        gated_data, retracted, blocker_details,
        in_count, out_count, model=model, timeout=timeout,
    )
    return {
        "report": report,
        "blocker_count": gated_data["blocker_count"],
        "gated_count": gated_data["gated_count"],
        "retracted_count": len(retracted),
    }


def report_belief(
    node_id: str,
    sources_db: str | None = None,
    model: str = "",
    timeout: int = 300,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Generate an evidence report tracing a belief to its root premises.

    Returns: {"report": str, "premise_count": int}
    """
    from .report_belief import report_belief as _report_belief

    backend = dict(db_path=db_path, pg_conninfo=pg_conninfo,
                   project_id=project_id)

    node_detail = show_node(node_id, **backend)
    explain_data = explain_node(node_id, **backend)
    trace_data = trace_assumptions(node_id, **backend)

    premises_data = []
    for pid in trace_data["premises"]:
        try:
            detail = show_node(pid, **backend)
            premises_data.append({
                "id": pid,
                "text": detail.get("text", ""),
                "truth_value": detail.get("truth_value", ""),
                "source": detail.get("source", ""),
            })
        except (KeyError, PermissionError):
            premises_data.append({"id": pid, "text": "", "truth_value": "UNKNOWN", "source": ""})

    source_chunks = {}
    if sources_db:
        from .ask import search_source_chunks
        for pd in premises_data:
            if pd["text"]:
                try:
                    chunks = search_source_chunks(pd["text"], sources_db, top_k=3)
                    source_chunks[pd["id"]] = chunks
                except Exception:
                    source_chunks[pd["id"]] = []

    report = _report_belief(
        node_id, node_detail, explain_data["steps"], premises_data,
        source_chunks, model=model, timeout=timeout,
    )
    return {
        "report": report,
        "premise_count": len(premises_data),
    }


def verify_belief(
    node_id: str,
    trace: bool = False,
    retract: bool = False,
    dry_run: bool = False,
    model: str = "claude",
    timeout: int = 120,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Verify a belief against its source documents.

    If trace=True and the belief is derived, traces to leaf premises and
    verifies each one.  Stamps verified_at on CONFIRMED beliefs.  If
    retract=True, retracts STALE beliefs.

    Returns: {"results": dict[id, verdict_dict], "verified": list[str],
              "stale": list[str], "beliefs_checked": list[dict],
              "is_derived": bool, "dry_run": bool}
    """
    from .verify import read_source, verify_beliefs
    from datetime import datetime, timezone

    backend = dict(db_path=db_path, pg_conninfo=pg_conninfo,
                   project_id=project_id)

    detail = show_node(node_id, **backend)
    has_justifications = bool(detail.get("justifications"))

    if has_justifications and trace:
        trace_data = trace_assumptions(node_id, **backend)
        premise_ids = trace_data["premises"]
        beliefs_to_check = []
        seen = set()
        for pid in premise_ids:
            if pid in seen:
                continue
            seen.add(pid)
            try:
                pd = show_node(pid, **backend)
                beliefs_to_check.append({
                    "id": pid,
                    "text": pd.get("text", ""),
                    "truth_value": pd.get("truth_value", ""),
                    "source": pd.get("source", ""),
                    "source_url": pd.get("source_url", ""),
                })
            except (KeyError, PermissionError):
                beliefs_to_check.append({
                    "id": pid, "text": "", "truth_value": "UNKNOWN",
                    "source": "", "source_url": "",
                })
    else:
        beliefs_to_check = [{
            "id": node_id,
            "text": detail.get("text", ""),
            "truth_value": detail.get("truth_value", ""),
            "source": detail.get("source", ""),
            "source_url": detail.get("source_url", ""),
        }]

    beliefs_with_sources = []
    for b in beliefs_to_check:
        source_content = read_source(b["source"], db_path=db_path) if b["source"] else None
        beliefs_with_sources.append((b, source_content))

    if dry_run:
        return {
            "results": {},
            "verified": [],
            "stale": [],
            "beliefs_checked": beliefs_to_check,
            "is_derived": has_justifications,
            "dry_run": True,
        }

    verdicts = verify_beliefs(beliefs_with_sources, model=model, timeout=timeout)

    verified = []
    stale = []
    retract_failed = []
    stamp_failed = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for b, _ in beliefs_with_sources:
        bid = b["id"]
        verdict = verdicts.get(bid, {})
        v = verdict.get("verdict", "INCONCLUSIVE")

        if v == "CONFIRMED":
            verified.append(bid)
        elif v == "STALE":
            stale.append(bid)
            if retract:
                try:
                    reason = verdict.get("reason", "stale per verify")
                    retract_node(bid, reason=f"verify: {reason}", **backend)
                except Exception:
                    retract_failed.append(bid)

    if not pg_conninfo and verified:
        try:
            with _with_network(db_path, write=True) as net:
                for bid in verified:
                    if bid in net.nodes:
                        net.nodes[bid].verified_at = now
                        net.nodes[bid].updated_at = now
                    else:
                        stamp_failed.append(bid)
        except Exception:
            stamp_failed = list(verified)

    result = {
        "results": verdicts,
        "verified": verified,
        "stale": stale,
        "beliefs_checked": beliefs_to_check,
        "is_derived": has_justifications,
        "dry_run": False,
    }
    if retract_failed:
        result["retract_failed"] = retract_failed
    if stamp_failed:
        result["stamp_failed"] = stamp_failed
    return result


NEGATIVE_BATCH_SIZE = 50

NEGATIVE_TERMS = [
    'bug', 'defect', 'missing', 'fail', 'error', 'broken', 'incorrect',
    'wrong', 'risk', 'gap', 'lack', 'vulnerable', 'insecure', 'stale',
    'outdated', 'deprecated', 'fragile', 'brittle', 'hack', 'workaround',
    'technical debt', 'tech debt', 'not implemented', 'unimplemented',
    'incomplete', 'inconsistent', 'unclear', 'confusing', 'problem',
    'known issue', 'security issue', 'concern', 'warning', 'danger',
    'threat', 'weakness', 'limitation', 'constraint', 'bottleneck',
    'blocker', 'obstacle', 'undermines', 'concentrated',
    'single point of failure', 'no tests', 'untested', 'not tested',
    'hard-coded', 'hardcoded', 'tight coupling', 'tightly coupled',
    'monolithic', 'legacy', 'unmaintained',
    'blocked', 'blocking', 'deferred', 'stalled', 'unresolved',
    'disabled', 'skipped',
    'sole', 'empty', 'placeholder', 'regression', 'corrupted',
    'orphaned', 'zombie',
    'absent', 'absence', 'invisible', 'silent', 'hollow', 'vacuum',
    'debt', 'overhead', 'degradation', 'deficit', 'friction',
    'undocumented', 'unverified',
    'dysfunction', 'opacity', 'opaque', 'isolated', 'permanently',
]

NEGATIVE_CLASSIFY_PROMPT = """\
You are classifying beliefs from a Truth Maintenance System.
Each belief below passed a keyword filter for negative terms.
Identify which are GENUINELY NEGATIVE — they assert something is
a problem, defect, risk, gap, limitation, or concern.

EXCLUDE beliefs that merely DESCRIBE error handling, failure modes,
or warning mechanisms as part of normal system behavior.

Return ONLY a JSON array of the IDs of genuinely negative beliefs.
Example: ["belief-1", "belief-3"]
If none are genuinely negative, return: []

## Candidates

{candidates}"""


def list_negative(
    visible_to: list[str] | None = None,
    model: str = "claude",
    skip_llm: bool = False,
    db_path: str = DEFAULT_DB,
    pg_conninfo=None, project_id=None,
) -> dict:
    """Find IN beliefs that describe problems, defects, or risks.

    Uses keyword pre-filtering then LLM classification.
    With skip_llm=True, returns keyword-filtered candidates directly.

    Returns: {"negative": [{"id": str, "text": str}, ...],
              "count": int, "candidates": int, "total": int}
    """
    if pg_conninfo:
        return _pg_dispatch(pg_conninfo, project_id, "list_negative",
                            visible_to=visible_to, model=model,
                            skip_llm=skip_llm)
    from . import ask

    with _with_network(db_path) as net:
        in_nodes = []
        for nid, node in sorted(net.nodes.items()):
            if node.truth_value != "IN":
                continue
            if visible_to is not None and not _is_visible(node, visible_to):
                continue
            in_nodes.append((nid, node.text))

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


def _classify_review_result(result: dict) -> str:
    if not result.get("valid", True):
        return "invalid"
    if not result.get("sufficient", True):
        return "insufficient"
    if not result.get("necessary", True):
        return "unnecessary"
    return "pass"


def review_beliefs(
    belief_ids: list[str] | None = None,
    model: str = "claude",
    timeout: int = 300,
    min_depth: int | None = None,
    depends_on: str | None = None,
    namespace: str | None = None,
    sample: int | None = None,
    visible_to: list[str] | None = None,
    dry_run: bool = False,
    on_batch: Callable | None = None,
    include_out: bool = False,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Review derived beliefs for validity, sufficiency, and necessity.

    Uses an LLM to evaluate whether each derived belief's reasoning
    from antecedents to conclusion is sound. Sets reviewed_at and updated_at
    timestamps and writes review_result metadata to each reviewed node
    (unless dry_run=True).

    Returns: {"results": [...], "reviewed": int, "invalid": int,
              "insufficient": int, "unnecessary": int, "total_derived": int}
    """
    from .derive import _get_depth
    from .review import review_beliefs as _review

    result = export_network(db_path=db_path)
    nodes = result.get("nodes", {})

    all_derived = {
        k: v for k, v in nodes.items()
        if v.get("truth_value") == "IN"
        and v.get("justifications")
        and len(v["justifications"]) > 0
    }
    total_derived = len(all_derived)

    if include_out:
        candidates = {
            k: v for k, v in nodes.items()
            if v.get("justifications")
            and len(v["justifications"]) > 0
        }
    else:
        candidates = dict(all_derived)

    if belief_ids:
        candidates = {k: v for k, v in candidates.items() if k in belief_ids}

    if visible_to is not None:
        tags = set(visible_to)
        candidates = {
            k: v for k, v in candidates.items()
            if not v.get("metadata", {}).get("access_tags")
            or all(t in tags for t in v["metadata"]["access_tags"])
        }

    if min_depth is not None:
        memo = {}
        candidates = {
            k: v for k, v in candidates.items()
            if _get_depth(k, nodes, all_derived, memo) >= min_depth
        }

    if depends_on:
        candidates = {
            k: v for k, v in candidates.items()
            if any(
                depends_on in j.get("antecedents", [])
                for j in v.get("justifications", [])
            )
        }

    if namespace is not None:
        if namespace == "":
            candidates = {k: v for k, v in candidates.items() if ":" not in k}
        else:
            candidates = {
                k: v for k, v in candidates.items()
                if k.startswith(f"{namespace}:")
            }

    if sample is not None and len(candidates) > sample:
        import random
        sampled_keys = random.sample(sorted(candidates.keys()), sample)
        candidates = {k: candidates[k] for k in sampled_keys}

    review_ids = sorted(candidates.keys())
    results = _review(nodes, belief_ids=review_ids, model=model, timeout=timeout,
                      on_batch=on_batch)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not dry_run and results:
        result_map = {r["id"]: r for r in results}
        with _with_network(db_path, write=True) as net:
            for nid in review_ids:
                if nid in net.nodes and nid in result_map:
                    net.nodes[nid].reviewed_at = now
                    net.nodes[nid].updated_at = now
                    net.nodes[nid].metadata["review_result"] = _classify_review_result(result_map[nid])

    invalid = sum(1 for r in results if not r.get("valid", True))
    insufficient = sum(1 for r in results if not r.get("sufficient", True))
    unnecessary = sum(1 for r in results if not r.get("necessary", True))

    return {
        "results": results,
        "reviewed": len(review_ids),
        "invalid": invalid,
        "insufficient": insufficient,
        "unnecessary": unnecessary,
        "total_derived": total_derived,
        "timestamp": now,
    }


def review_justifications(
    belief_ids: list[str] | None = None,
    model: str = "claude",
    timeout: int = 300,
    min_antecedents: int = 2,
    on_batch: Callable | None = None,
    parallel: int = 0,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Review SL justifications for ALL vs ANY misclassification.

    Uses an LLM to evaluate whether each multi-antecedent SL justification
    should be conjunctive (ALL) or disjunctive (ANY). Read-only — does not
    modify the database.

    Returns: {"results": [...], "reviewed": int, "convert_any": int,
              "convert_mixed": int, "keep_all": int, "total_candidates": int}
    """
    from .review_justifications import (
        review_justifications as _review,
        _has_multi_antecedent_sl,
    )

    result = export_network(db_path=db_path)
    nodes = result.get("nodes", {})

    candidates = {
        nid: node for nid, node in nodes.items()
        if node.get("truth_value") == "IN"
        and _has_multi_antecedent_sl(node, min_antecedents)
    }
    if belief_ids:
        candidates = {k: v for k, v in candidates.items() if k in belief_ids}
    total_candidates = len(candidates)

    results = _review(nodes, belief_ids=belief_ids, model=model, timeout=timeout,
                      min_antecedents=min_antecedents, on_batch=on_batch,
                      parallel=parallel)

    convert_any = sum(1 for r in results if r.get("classification") == "ANY")
    convert_mixed = sum(1 for r in results if r.get("classification") == "MIXED")
    keep_all = sum(1 for r in results if r.get("classification") == "ALL")

    return {
        "results": results,
        "reviewed": len(results),
        "convert_any": convert_any,
        "convert_mixed": convert_mixed,
        "keep_all": keep_all,
        "total_candidates": total_candidates,
    }


def review_premises(
    belief_ids: list[str] | None = None,
    model: str = "claude",
    timeout: int = 300,
    sample: int | None = None,
    visible_to: list[str] | None = None,
    dry_run: bool = False,
    on_batch: Callable | None = None,
    parallel: int = 0,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Review premises against their source material for factual accuracy.

    Uses an LLM to evaluate whether each premise accurately reflects what
    its source document says. Writes last_premise_reviewed and
    premise_review_result metadata to each reviewed node (unless dry_run).

    Returns: {"results": [...], "reviewed": int, "inaccurate": int,
              "overgeneralized": int, "total_premises": int,
              "skipped_no_source": int}
    """
    import sys

    from .check_stale import resolve_source_path
    from .review_premises import review_premises as _review

    result = export_network(db_path=db_path)
    nodes = result.get("nodes", {})
    repos = result.get("repos", {})
    repos = {k: Path(v) if isinstance(v, str) else v for k, v in repos.items()}

    db_dir = Path(db_path).parent if db_path else None

    all_premises = {
        k: v for k, v in nodes.items()
        if v.get("truth_value") == "IN"
        and (not v.get("justifications") or len(v["justifications"]) == 0)
    }
    total_premises = len(all_premises)

    candidates = dict(all_premises)

    if belief_ids:
        candidates = {k: v for k, v in candidates.items() if k in belief_ids}

    if visible_to is not None:
        tags = set(visible_to)
        candidates = {
            k: v for k, v in candidates.items()
            if not v.get("metadata", {}).get("access_tags")
            or all(t in tags for t in v["metadata"]["access_tags"])
        }

    if sample is not None and len(candidates) > sample:
        import random
        sampled_keys = random.sample(sorted(candidates.keys()), sample)
        candidates = {k: candidates[k] for k in sampled_keys}

    source_contents = {}
    review_ids = []
    skipped = 0

    for nid in sorted(candidates.keys()):
        source = candidates[nid].get("source", "")
        if not source:
            skipped += 1
            continue
        if source not in source_contents:
            agent = candidates[nid].get("metadata", {}).get("agent")
            path = resolve_source_path(source, repos=repos, db_dir=db_dir, agent=agent)
            if path and path.exists():
                try:
                    source_contents[source] = path.read_text()
                except (OSError, UnicodeDecodeError):
                    skipped += 1
                    continue
            else:
                print(f"  SKIP {nid}: source not found: {source}", file=sys.stderr)
                skipped += 1
                continue
        review_ids.append(nid)

    results = _review(nodes, premise_ids=review_ids, source_contents=source_contents,
                      model=model, timeout=timeout, on_batch=on_batch,
                      parallel=parallel)

    now = datetime.now().isoformat(timespec="seconds")

    if not dry_run and results:
        result_map = {r["id"]: r for r in results}
        with _with_network(db_path, write=True) as net:
            for nid in review_ids:
                if nid in net.nodes and nid in result_map:
                    r = result_map[nid]
                    net.nodes[nid].metadata["last_premise_reviewed"] = now
                    if r.get("accurate", True) and r.get("well_scoped", True):
                        net.nodes[nid].metadata["premise_review_result"] = "pass"
                    elif not r.get("accurate", True):
                        net.nodes[nid].metadata["premise_review_result"] = r.get("error_type", "inaccurate")
                    else:
                        net.nodes[nid].metadata["premise_review_result"] = "overgeneralized"

    inaccurate = sum(1 for r in results if not r.get("accurate", True))
    overgeneralized = sum(1 for r in results
                         if r.get("accurate", True) and not r.get("well_scoped", True))

    return {
        "results": results,
        "reviewed": len(review_ids),
        "inaccurate": inaccurate,
        "overgeneralized": overgeneralized,
        "total_premises": total_premises,
        "skipped_no_source": skipped,
        "timestamp": now,
    }


def repair_premises(
    review_file: str | None = None,
    belief_ids: list[str] | None = None,
    model: str = "claude",
    timeout: int = 300,
    dry_run: bool = False,
    parallel: int = 0,
    on_result: Callable | None = None,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Repair inaccurate premises by rewriting from source or retracting.

    Takes review results (from review_file or by re-reviewing belief_ids)
    and either rewrites premises to match source material or retracts them.

    Returns: {"results": [...], "total_inaccurate": int, "rewritten": int,
              "retracted": int, "failed": int}
    """
    import sys
    from pathlib import Path

    from .check_stale import resolve_source_path
    from .repair_premises import repair_premises as _repair

    if review_file:
        import json as json_mod
        data = json_mod.loads(Path(review_file).read_text())
        review_results_list = data.get("results", [])
    elif belief_ids:
        review_result = review_premises(
            belief_ids=belief_ids, model=model, timeout=timeout,
            dry_run=True, db_path=db_path,
        )
        review_results_list = review_result.get("results", [])
    else:
        raise ValueError("Either review_file or belief_ids must be provided")

    inaccurate = [r for r in review_results_list if not r.get("accurate", True)]
    if not inaccurate:
        return {"results": [], "total_inaccurate": 0, "rewritten": 0,
                "retracted": 0, "failed": 0}

    review_map = {r["id"]: r for r in inaccurate}

    result = export_network(db_path=db_path)
    nodes = result.get("nodes", {})
    repos = result.get("repos", {})
    repos = {k: Path(v) if isinstance(v, str) else v for k, v in repos.items()}
    db_dir = Path(db_path).parent if db_path else None

    source_contents = {}
    repair_ids = []

    for nid in sorted(review_map.keys()):
        if nid not in nodes:
            continue
        source = nodes[nid].get("source", "")
        if not source:
            continue
        if source not in source_contents:
            agent = nodes[nid].get("metadata", {}).get("agent")
            path = resolve_source_path(source, repos=repos, db_dir=db_dir, agent=agent)
            if path and path.exists():
                try:
                    source_contents[source] = path.read_text()
                except (OSError, UnicodeDecodeError):
                    continue
            else:
                print(f"  SKIP {nid}: source not found: {source}", file=sys.stderr)
                continue
        repair_ids.append(nid)

    results = _repair(nodes, premise_ids=repair_ids, source_contents=source_contents,
                      review_results=review_map, model=model, timeout=timeout,
                      parallel=parallel, on_result=on_result)

    if not dry_run:
        for r in results:
            nid = r["id"]
            action = r.get("action")
            if action == "rewrite" and r.get("corrected_text"):
                try:
                    sup = supersede_with_text(nid, r["corrected_text"], db_path=db_path)
                    r["new_id"] = sup["new_id"]
                    set_metadata(sup["new_id"], "repair_action", "rewritten", db_path=db_path)
                except Exception as e:
                    print(f"  ERROR updating {nid}: {e}", file=sys.stderr)
                    r["action"] = "error"
            elif action == "retract":
                try:
                    retract_node(nid,
                                 reason=f"repair-premises: {r.get('rationale', 'inaccurate')}",
                                 db_path=db_path)
                    set_metadata(nid, "repair_action", "retracted", db_path=db_path)
                except Exception as e:
                    print(f"  ERROR retracting {nid}: {e}", file=sys.stderr)
                    r["action"] = "error"

    rewritten = sum(1 for r in results if r.get("action") == "rewrite")
    retracted = sum(1 for r in results if r.get("action") == "retract")
    failed = sum(1 for r in results if r.get("action") == "error")

    return {
        "results": results,
        "total_inaccurate": len(inaccurate),
        "rewritten": rewritten,
        "retracted": retracted,
        "failed": failed,
    }


def propose_update(
    belief_ids: list[str] | None = None,
    model: str = "claude",
    timeout: int = 600,
    stale_only: bool = False,
    namespace: str | None = None,
    sample: int | None = None,
    on_batch: Callable | None = None,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Propose structured updates or retractions for beliefs.

    Sends beliefs to an LLM for evaluation, classifying each by failure
    mode and update basis. Computes cascade impact for retract proposals.

    Returns: {"proposals": [...], "reviewed": int, "timestamp": str}
    """
    from .propose_update import propose_updates as _propose

    result = export_network(db_path=db_path)
    nodes = result.get("nodes", {})

    candidates = {
        k: v for k, v in nodes.items()
        if v.get("truth_value") == "IN"
    }

    if belief_ids:
        candidates = {k: v for k, v in candidates.items() if k in belief_ids}

    if stale_only:
        candidates = {
            k: v for k, v in candidates.items()
            if (v.get("metadata") or {}).get("stale_reason")
        }

    if namespace is not None:
        if namespace == "":
            candidates = {k: v for k, v in candidates.items() if ":" not in k}
        else:
            candidates = {
                k: v for k, v in candidates.items()
                if k.startswith(f"{namespace}:")
            }

    if sample is not None and len(candidates) > sample:
        import random
        sampled_keys = random.sample(sorted(candidates.keys()), sample)
        candidates = {k: candidates[k] for k in sampled_keys}

    review_ids = sorted(candidates.keys())

    # Enrich nodes with dependents (export_network doesn't include them)
    with _with_network(db_path) as net:
        for nid in nodes:
            if nid in net.nodes:
                nodes[nid]["dependents"] = sorted(net.nodes[nid].dependents)

    proposals = _propose(nodes, belief_ids=review_ids, model=model,
                         timeout=timeout, on_batch=on_batch)

    cascades = {}
    for p in proposals:
        if p["id"] not in nodes:
            continue
        try:
            cascades[p["id"]] = what_if_retract(p["id"], db_path=db_path)
        except KeyError:
            pass

    now = datetime.now().isoformat(timespec="seconds")
    return {
        "proposals": proposals,
        "cascades": cascades,
        "reviewed": len(review_ids),
        "timestamp": now,
    }


def repair_smuggled(
    review_file: str | None = None,
    belief_ids: list[str] | None = None,
    model: str = "claude",
    timeout: int = 300,
    dry_run: bool = False,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Repair smuggled premises by searching for and linking existing premises.

    Two input modes:
        review_file: path to a review-beliefs JSON report — extract invalid results
        belief_ids: re-review these beliefs inline, then repair invalids

    Returns dict with repairs list and summary counts.
    """
    from .repair import repair_smuggled_beliefs

    if review_file:
        import json as _json
        with open(review_file) as f:
            report = _json.load(f)
        review_results = report.get("results", [])
    elif belief_ids:
        result = review_beliefs(
            belief_ids=belief_ids, model=model, timeout=timeout,
            dry_run=True, db_path=db_path,
        )
        review_results = result["results"]
    else:
        raise ValueError("Provide either review_file or belief_ids")

    invalid = [r for r in review_results if not r.get("valid", True)]

    if not invalid:
        return {
            "repairs": [],
            "total_invalid": 0,
            "repaired": 0,
            "no_candidates": 0,
            "no_match": 0,
            "extraction_failed": 0,
            "errors": 0,
        }

    net = export_network(db_path=db_path)
    nodes = net.get("nodes", {})

    repairs = repair_smuggled_beliefs(
        invalid, nodes, model=model, timeout=timeout,
        db_path=db_path, dry_run=dry_run,
    )

    return {
        "repairs": repairs,
        "total_invalid": len(invalid),
        "repaired": sum(1 for r in repairs if r["status"] == "repaired"),
        "no_candidates": sum(1 for r in repairs if r["status"] == "no_candidates"),
        "no_match": sum(1 for r in repairs if r["status"] == "no_match"),
        "extraction_failed": sum(1 for r in repairs if r["status"] == "extraction_failed"),
        "errors": sum(1 for r in repairs if r["status"] == "error"),
    }


def repair(
    review_file: str | None = None,
    belief_ids: list[str] | None = None,
    model: str = "claude",
    timeout: int = 300,
    dry_run: bool = False,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Repair flagged beliefs: triage into search-and-link, soften, abandon, or research.

    Two input modes:
        review_file: path to a review-beliefs JSON report
        belief_ids: re-review these beliefs inline, then repair invalids

    Returns dict with results list and summary counts.
    """
    from .repair import repair_beliefs

    if review_file:
        import json as _json
        with open(review_file) as f:
            report = _json.load(f)
        review_results = report.get("results", [])
    elif belief_ids:
        result = review_beliefs(
            belief_ids=belief_ids, model=model, timeout=timeout,
            dry_run=True, db_path=db_path,
        )
        review_results = result["results"]
    else:
        raise ValueError("Provide either review_file or belief_ids")

    invalid = [r for r in review_results if not r.get("valid", True)]

    if not invalid:
        return {
            "results": [],
            "total_invalid": 0,
            "linked": 0,
            "softened": 0,
            "abandoned": 0,
            "needs_research": 0,
            "failed": 0,
            "errors": 0,
        }

    net = export_network(db_path=db_path)
    nodes = net.get("nodes", {})

    results = repair_beliefs(
        invalid, nodes, model=model, timeout=timeout,
        db_path=db_path, dry_run=dry_run,
    )

    return {
        "results": results,
        "total_invalid": len(invalid),
        "linked": sum(1 for r in results if r["status"] == "linked"),
        "softened": sum(1 for r in results if r["status"] == "softened"),
        "abandoned": sum(1 for r in results if r["status"] == "abandoned"),
        "needs_research": sum(1 for r in results if r["status"] == "needs_research"),
        "failed": sum(1 for r in results if r["status"] in
                       ("triage_failed", "no_candidates", "no_match",
                        "soften_failed", "extraction_failed")),
        "errors": sum(1 for r in results if r["status"] == "error"),
    }


research = repair


def detect_contradictions(
    belief_ids: list[str] | None = None,
    model: str = "claude",
    timeout: int = 300,
    sample: int | None = None,
    auto_apply: bool = False,
    semantic: bool = False,
    embedding_model: str | None = None,
    output_path: str | None = None,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Detect contradictions between IN beliefs via LLM analysis.

    When semantic=True, beliefs are clustered by embedding similarity
    before sending to the LLM, so topically related beliefs are
    analyzed together.

    When output_path is set, results are written incrementally after
    each batch/cluster completes.

    Returns: {"contradictions": [...], "checked": int, "found": int,
              "applied": int, "total_in": int}
    """
    from .contradictions import detect_contradictions as _detect
    if semantic:
        from .contradictions import detect_contradictions_semantic as _detect_semantic

    result = export_network(db_path=db_path)
    nodes = result.get("nodes", {})

    candidates = {
        k: v for k, v in nodes.items()
        if v.get("truth_value") == "IN"
    }
    total_in = len(candidates)

    if belief_ids:
        candidates = {k: v for k, v in candidates.items() if k in belief_ids}

    if sample is not None and len(candidates) > sample:
        import random
        sampled_keys = random.sample(sorted(candidates.keys()), sample)
        candidates = {k: candidates[k] for k in sampled_keys}

    check_ids = sorted(candidates.keys())
    if semantic:
        contradictions = _detect_semantic(nodes, belief_ids=check_ids,
                                          model=model, timeout=timeout,
                                          embedding_model=embedding_model,
                                          output_path=output_path)
    else:
        contradictions = _detect(nodes, belief_ids=check_ids, model=model,
                                timeout=timeout,
                                output_path=output_path)

    applied = 0
    applied_details = []
    if auto_apply:
        for c in contradictions:
            try:
                nogood_result = add_nogood(c["claims"], db_path=db_path)
                applied += 1
                applied_details.append({
                    "id": c["id"],
                    "nogood_id": nogood_result.get("nogood_id"),
                    "changed": nogood_result.get("changed", []),
                })
            except Exception as e:
                print(f"  WARN: failed to apply {c['id']}: {e}",
                      file=sys.stderr)

    return {
        "contradictions": contradictions,
        "checked": len(check_ids),
        "found": len(contradictions),
        "applied": applied,
        "applied_details": applied_details,
        "total_in": total_in,
    }


def write_contradiction_plan(contradictions: list[dict], output_path: str,
                             append: bool = False) -> str:
    """Write (or append to) a contradiction plan file for human review.

    Format is parseable by parse_contradiction_plan(). Each nogood lists
    claims and analysis. Change [APPLY] to [SKIP] or delete entries to
    exclude them.
    """
    path = Path(output_path)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        if not append:
            f.write("# Contradiction Plan\n\n")
            f.write("Review each nogood below. Delete any you want to skip,\n")
            f.write("or change APPLY to SKIP. Then run:\n")
            f.write(f"  reasons contradictions --accept {output_path}\n\n")
            f.write("---\n\n")

        for c in contradictions:
            severity = c.get("severity", "")
            f.write(f"### NOGOOD {c['id']} [APPLY]\n")
            if severity:
                f.write(f"- Severity: {severity}\n")
            f.write(f"- Claims: {', '.join(c['claims'])}\n")
            if c.get("analysis"):
                f.write(f"- Analysis: {c['analysis']}\n")
            f.write("\n")

    return str(path)


def parse_contradiction_plan(plan_text: str) -> list[dict]:
    """Parse a contradiction plan file into actionable entries.

    Returns list of {"id": str, "claims": list[str]} for APPLY-marked entries.
    """
    import re
    entries = []
    current_id = None
    current_claims = []

    for line in plan_text.splitlines():
        m = re.match(r"###\s+NOGOOD\s+(\S+)\s+\[(APPLY|SKIP)\]", line)
        if m:
            if current_id and current_claims:
                entries.append({"id": current_id, "claims": current_claims})
            current_id = m.group(1) if m.group(2) == "APPLY" else None
            current_claims = []
            continue

        if current_id and line.strip().startswith("- Claims:"):
            claims_str = line.split(":", 1)[1].strip()
            current_claims = [c.strip().strip("`") for c in claims_str.split(",")
                              if c.strip()]

    if current_id and current_claims:
        entries.append({"id": current_id, "claims": current_claims})

    return entries


def apply_contradiction_plan(
    plan: list[dict],
    db_path: str = DEFAULT_DB,
) -> dict:
    """Apply a reviewed contradiction plan: record nogoods and backtrack.

    Args:
        plan: list of {"id": str, "claims": list[str]} from parse_contradiction_plan
        db_path: Path to database

    Returns: {"applied": int, "nogoods": list[dict], "errors": list[str]}
    """
    nogoods = []
    errors = []
    for entry in plan:
        try:
            result = add_nogood(entry["claims"], db_path=db_path)
            nogoods.append({
                "id": entry["id"],
                "nogood_id": result.get("nogood_id"),
                "changed": result.get("changed", []),
                "backtracked_to": result.get("backtracked_to"),
            })
        except Exception as e:
            errors.append(f"Failed to apply {entry['id']}: {e}")
    return {"applied": len(nogoods), "nogoods": nogoods, "errors": errors}


def _rewrite_dependents(net, old_id: str, new_id: str):
    """Rewrite justifications that reference old_id to point at new_id.

    Updates both the justification antecedents/outlist and the dependents
    reverse index so that derived beliefs survive deduplication.
    """
    old_node = net.nodes[old_id]
    new_node = net.nodes[new_id]
    for dep_id in list(old_node.dependents):
        dep = net.nodes[dep_id]
        for j in dep.justifications:
            if old_id in j.antecedents:
                j.antecedents = [new_id if a == old_id else a for a in j.antecedents]
                new_node.dependents.add(dep_id)
            if old_id in j.outlist:
                j.outlist = [new_id if o == old_id else o for o in j.outlist]
                new_node.dependents.add(dep_id)
        old_node.dependents.discard(dep_id)


def list_clusters(
    status: str = "IN",
    n_clusters: int | None = None,
    seed: int | None = None,
    embedding_model: str | None = None,
    visible_to: list[str] | None = None,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Cluster beliefs by semantic similarity and return full assignments."""
    from .cluster import list_clusters as _list_clusters, DEFAULT_MODEL

    with _with_network(db_path) as net:
        beliefs = {}
        for nid, n in sorted(net.nodes.items()):
            if status and n.truth_value != status:
                continue
            if visible_to is not None and not _is_visible(n, visible_to):
                continue
            beliefs[nid] = n.text

    if not beliefs:
        return {"clusters": [], "n_clusters": 0, "embedding_model": embedding_model or DEFAULT_MODEL}

    return _list_clusters(
        beliefs,
        n_clusters=n_clusters,
        seed=seed,
        model_name=embedding_model or DEFAULT_MODEL,
    )


def deduplicate(
    threshold: float = 0.5,
    auto: bool = False,
    semantic: bool = False,
    embedding_model: str | None = None,
    db_path: str = DEFAULT_DB,
) -> dict:
    """Find clusters of IN beliefs with similar IDs or text (likely duplicates).

    Uses Jaccard similarity on ID tokens by default, or embedding cosine
    similarity when semantic=True.

    Args:
        threshold: Minimum similarity to consider a pair (default: 0.5)
        auto: If True, retract all but the most-connected belief in each cluster
        semantic: If True, use embedding cosine similarity instead of ID tokens
        embedding_model: Sentence-transformers model (semantic mode only)
        db_path: Path to database

    Returns: {"clusters": list[dict], "retracted": list[str]}
    """
    if semantic:
        from .cluster import ClusterCache, DEFAULT_MODEL
        import numpy as np
    else:
        from .derive import _tokenize_id, _jaccard

    with _with_network(db_path, write=auto) as net:
        in_nodes = [(nid, n) for nid, n in sorted(net.nodes.items())
                    if n.truth_value == "IN"]

        if not in_nodes:
            return {"clusters": [], "retracted": []}

        if semantic:
            beliefs = {nid: n.text for nid, n in in_nodes}
            cache = ClusterCache(embedding_model or DEFAULT_MODEL)
            ids, embeddings = cache.embed(beliefs)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normed = embeddings / norms
            sim_matrix = normed @ normed.T
        else:
            tokens = {nid: _tokenize_id(nid) for nid, _ in in_nodes}

        parent = {nid: nid for nid, _ in in_nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        if semantic:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if sim_matrix[i, j] >= threshold:
                        union(ids[i], ids[j])
        else:
            for i, (nid_a, _) in enumerate(in_nodes):
                for nid_b, _ in in_nodes[i + 1:]:
                    if _jaccard(tokens[nid_a], tokens[nid_b]) >= threshold:
                        union(nid_a, nid_b)

        # Collect clusters (only groups of 2+)
        from collections import defaultdict
        groups = defaultdict(list)
        for nid, _ in in_nodes:
            groups[find(nid)].append(nid)

        clusters = []
        retracted = []
        for members in groups.values():
            if len(members) < 2:
                continue
            cluster = {
                "beliefs": [
                    {"id": nid, "text": net.nodes[nid].text,
                     "dependents": len(net.nodes[nid].dependents)}
                    for nid in sorted(members)
                ],
                "size": len(members),
            }
            keep = max(members, key=lambda nid: (len(net.nodes[nid].dependents), nid))
            cluster["kept"] = keep
            clusters.append(cluster)

            if auto:
                for nid in members:
                    if nid != keep:
                        _rewrite_dependents(net, old_id=nid, new_id=keep)
                        net.retract(nid)
                        retracted.append(nid)

        clusters.sort(key=lambda c: -c["size"])
        return {"clusters": clusters, "retracted": retracted}


def write_dedup_plan(clusters: list[dict], output_path: str) -> str:
    """Write a dedup plan file for human review.

    Format is parseable by parse_dedup_plan(). Each cluster lists the
    kept belief and the beliefs to retract. Remove clusters or lines
    you disagree with before accepting.
    """
    path = Path(output_path)
    with open(path, "w") as f:
        f.write("# Deduplication Plan\n\n")
        f.write("Review each cluster below. Delete any cluster you want to skip,\n")
        f.write("or change which belief is KEEP vs RETRACT. Then run:\n")
        f.write("  reasons deduplicate --accept proposed-dedup.md\n\n")
        f.write("---\n\n")

        for i, cluster in enumerate(clusters, 1):
            f.write(f"## Cluster {i} ({cluster['size']} beliefs)\n\n")
            kept = cluster.get("kept")
            for b in cluster["beliefs"]:
                action = "KEEP" if b["id"] == kept else "RETRACT"
                deps = f"  ({b['dependents']} dependents)" if b["dependents"] else ""
                f.write(f"- [{action}] `{b['id']}`{deps}\n")
                f.write(f"  {b['text']}\n")
            f.write("\n")

    return str(path)


def parse_dedup_plan(plan_text: str) -> list[dict]:
    """Parse a dedup plan file into actionable clusters.

    Returns list of {"keep": str, "retract": list[str]} dicts.
    """
    import re
    clusters = []
    current_keep = None
    current_retract = []

    for line in plan_text.splitlines():
        if line.startswith("## Cluster"):
            if current_keep and current_retract:
                clusters.append({"keep": current_keep, "retract": current_retract})
            current_keep = None
            current_retract = []
            continue

        m = re.match(r"- \[(KEEP|RETRACT)\] `(\S+?)`", line)
        if m:
            action, node_id = m.group(1), m.group(2)
            if action == "KEEP":
                current_keep = node_id
            else:
                current_retract.append(node_id)

    if current_keep and current_retract:
        clusters.append({"keep": current_keep, "retract": current_retract})

    return clusters


def apply_dedup_plan(
    plan: list[dict],
    db_path: str = DEFAULT_DB,
) -> dict:
    """Apply a reviewed dedup plan: rewrite justifications and retract duplicates.

    Args:
        plan: list of {"keep": str, "retract": list[str]} from parse_dedup_plan
        db_path: Path to database

    Returns: {"applied": int, "retracted": list[str], "errors": list[str]}
    """
    with _with_network(db_path, write=True) as net:
        retracted = []
        errors = []
        for cluster in plan:
            keep = cluster["keep"]
            if keep not in net.nodes:
                errors.append(f"keep node not found: {keep}")
                continue
            for old_id in cluster["retract"]:
                if old_id not in net.nodes:
                    errors.append(f"retract node not found: {old_id}")
                    continue
                if net.nodes[old_id].truth_value == "OUT":
                    continue
                _rewrite_dependents(net, old_id=old_id, new_id=keep)
                net.retract(old_id)
                retracted.append(old_id)
        return {"applied": len(plan), "retracted": retracted, "errors": errors}

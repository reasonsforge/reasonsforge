"""Import and sync another agent's beliefs into the local RMS network.

Creates namespaced nodes (agent:belief-id) so beliefs from multiple agents
can coexist without collision. Each agent gets a premise node (agent:active)
and a relay node (agent:inactive) that provides a kill switch — retracting
agent:active makes agent:inactive go IN, which cascades OUT every belief
from that agent via outlist.

The agent:active premise is NOT placed in antecedents (which would provide
an always-valid fallback defeating per-belief retraction). Instead,
agent:inactive is placed in the outlist of each imported belief.

Usage:
    reasons import-agent aap-expert ~/git/aap-expert/beliefs.md
    reasons import-agent rhel-expert ~/git/rhel-expert/network.json
    reasons import-agent rhel-expert ~/git/rhel-expert/beliefs.md --only-in
    reasons sync-agent aap-expert ~/git/aap-expert/beliefs.md
"""

from pathlib import Path

from . import Justification, Nogood
from .import_beliefs import parse_beliefs, parse_nogoods
from .network import Network


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fixup_dependents(network):
    """Re-register dependents for all nodes.

    Outlist nodes may have been added after the nodes that reference them,
    so add_node couldn't register the dependency at creation time.
    """
    network._rebuild_dependents()


def _ensure_agent_nodes(network, agent_name, source_path=""):
    """Create agent:active and agent:inactive nodes if they don't exist.

    Returns (active_id, inactive_id, created_premise).
    """
    active_id = f"{agent_name}:active"
    inactive_id = f"{agent_name}:inactive"

    created_premise = False
    if active_id not in network.nodes:
        network.add_node(
            id=active_id,
            text=f"Agent '{agent_name}' beliefs are trusted",
            source=source_path,
            metadata={"agent": agent_name, "role": "agent_premise"},
        )
        created_premise = True

    if inactive_id not in network.nodes:
        network.add_node(
            id=inactive_id,
            text=f"Agent '{agent_name}' kill switch — IN when active is OUT",
            justifications=[Justification(type="SL", antecedents=[], outlist=[active_id])],
            source=source_path,
            metadata={"agent": agent_name, "role": "agent_inactive"},
        )

    return active_id, inactive_id, created_premise


def _justifications_match(old, new):
    """Check if two justification lists are equivalent."""
    if len(old) != len(new):
        return False
    for a, b in zip(old, new):
        if (a.type != b.type or a.antecedents != b.antecedents
                or a.outlist != b.outlist or a.label != b.label):
            return False
    return True


def _update_node_justifications(network, node_id, new_justifications):
    """Replace justifications on an existing node, fixing dependent registrations."""
    network.nodes[node_id].justifications = new_justifications
    network._rebuild_dependents()


# ---------------------------------------------------------------------------
# Format normalization — both formats produce the same claim structure:
#   {id, text, is_out, source, source_hash, date, metadata, raw_justifications}
# where raw_justifications = [{type, antecedents, outlist, label}] with
# unprefixed IDs.  Filtering (which antecedents/outlist to keep) is done
# here so the shared logic just prefixes everything.
# ---------------------------------------------------------------------------

def _normalize_markdown(beliefs_text, only_in=False):
    """Convert markdown beliefs to normalized claim dicts."""
    claims = parse_beliefs(beliefs_text)
    if only_in:
        claims = [c for c in claims if c["status"] == "IN"]

    claim_ids = {c["id"] for c in claims}

    normalized = []
    for c in claims:
        is_out = c["status"] in ("STALE", "OUT")
        meta = {}
        if c["type"]:
            meta["beliefs_type"] = c["type"]

        if is_out:
            raw_justs = []
        else:
            antecedents = [d for d in c["depends_on"] if d in claim_ids]
            outlist = [o for o in c.get("unless", []) if o in claim_ids]
            raw_justs = [{"type": "SL", "antecedents": antecedents,
                          "outlist": outlist, "label": None}]

        normalized.append({
            "id": c["id"],
            "text": c["text"],
            "is_out": is_out,
            "source": c["source"],
            "source_hash": c["source_hash"],
            "date": c["date"],
            "metadata": meta,
            "raw_justifications": raw_justs,
        })
    return normalized


def _normalize_json(data, only_in=False):
    """Convert JSON nodes to normalized claim dicts."""
    nodes = data.get("nodes", {})
    if only_in:
        nodes = {k: v for k, v in nodes.items() if v.get("truth_value") == "IN"}

    node_ids = set(nodes.keys())

    normalized = []
    for nid, ndata in nodes.items():
        is_out = ndata.get("truth_value") == "OUT"
        meta = dict(ndata.get("metadata", {}))

        raw_justs = []
        for j in ndata.get("justifications", []):
            raw_justs.append({
                "type": j.get("type", "SL"),
                "antecedents": list(j.get("antecedents", [])),
                "outlist": [o for o in j.get("outlist", []) if o in node_ids],
                "label": j.get("label"),
            })

        normalized.append({
            "id": nid,
            "text": ndata.get("text", ""),
            "is_out": is_out,
            "source": ndata.get("source", ""),
            "source_url": ndata.get("source_url", ""),
            "source_hash": ndata.get("source_hash", ""),
            "date": ndata.get("date", ""),
            "metadata": meta,
            "raw_justifications": raw_justs,
        })
    return normalized


def _normalize_nogoods_markdown(nogoods_text):
    """Convert markdown nogoods to normalized dicts."""
    if not nogoods_text:
        return []
    return [
        {"id": ng["id"], "nodes": ng["affects"],
         "discovered": ng["discovered"], "resolution": ng["resolution"]}
        for ng in parse_nogoods(nogoods_text)
    ]


def _normalize_nogoods_json(data):
    """Convert JSON nogoods to normalized dicts."""
    return [
        {"id": ng["id"], "nodes": ng.get("nodes", []),
         "discovered": ng.get("discovered", ""), "resolution": ng.get("resolution", "")}
        for ng in data.get("nogoods", [])
    ]


# ---------------------------------------------------------------------------
# Shared operations
# ---------------------------------------------------------------------------

def _topo_sort_claims(claims):
    """Topological sort claims by antecedent dependencies."""
    claim_ids = {c["id"] for c in claims}
    ordered = []
    added = set()
    remaining = list(claims)
    max_passes = len(remaining) + 1
    for _ in range(max_passes):
        if not remaining:
            break
        next_remaining = []
        for c in remaining:
            deps = set()
            for j in c["raw_justifications"]:
                deps.update(a for a in j["antecedents"] if a in claim_ids)
            if all(d in added for d in deps):
                ordered.append(c)
                added.add(c["id"])
            else:
                next_remaining.append(c)
        if len(next_remaining) == len(remaining):
            ordered.extend(next_remaining)
            break
        remaining = next_remaining
    return ordered


def _build_justifications(claim, prefix, inactive_id, agent_name):
    """Build Justification objects from a normalized claim."""
    justs = []
    for rj in claim["raw_justifications"]:
        antecedents = [f"{prefix}{a}" for a in rj["antecedents"]]
        outlist = [inactive_id] + [f"{prefix}{o}" for o in rj["outlist"]]
        label = rj["label"] or f"imported from agent: {agent_name}"
        justs.append(Justification(
            type=rj["type"], antecedents=antecedents,
            outlist=outlist, label=label,
        ))

    if not justs:
        justs = [Justification(
            type="SL", antecedents=[], outlist=[inactive_id],
            label=f"imported from agent: {agent_name}",
        )]

    return justs


def _import_nogoods(network, prefix, nogoods):
    """Import normalized nogoods into the network."""
    count = 0
    existing_ids = {n.id for n in network.nogoods}
    for ng in nogoods:
        prefixed_nodes = [f"{prefix}{n}" for n in ng["nodes"]]
        valid_nodes = [n for n in prefixed_nodes if n in network.nodes]
        nogood_id = f"{prefix}{ng['id']}"
        if len(valid_nodes) >= 2 and nogood_id not in existing_ids:
            network.nogoods.append(Nogood(
                id=nogood_id,
                nodes=valid_nodes,
                discovered=ng.get("discovered", ""),
                resolution=ng.get("resolution", ""),
            ))
            existing_ids.add(nogood_id)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Shared import / sync logic
# ---------------------------------------------------------------------------

def _import_claims(network, agent_name, claims, source_path, nogoods):
    """Import normalized claims into the network."""
    prefix = f"{agent_name}:"
    active_id, inactive_id, created_premise = _ensure_agent_nodes(
        network, agent_name, source_path
    )

    if source_path:
        network.repos[agent_name] = str(Path(source_path).resolve().parent)

    ordered = _topo_sort_claims(claims)

    imported = 0
    skipped = 0
    retracted = 0
    retract_after = []

    for claim in ordered:
        node_id = f"{prefix}{claim['id']}"

        if node_id in network.nodes:
            skipped += 1
            continue

        justifications = _build_justifications(claim, prefix, inactive_id, agent_name)

        metadata = claim["metadata"].copy()
        metadata.update({
            "agent": agent_name,
            "original_id": claim["id"],
            "imported_from": source_path,
        })

        network.add_node(
            id=node_id,
            text=claim["text"],
            justifications=justifications if justifications else None,
            source=claim["source"],
            source_url=claim.get("source_url", ""),
            source_hash=claim["source_hash"],
            date=claim["date"],
            metadata=metadata,
        )
        imported += 1

        if claim["is_out"] and not claim["raw_justifications"]:
            retract_after.append(node_id)
        elif claim["is_out"]:
            network.nodes[node_id].metadata["_retracted"] = True
            network.nodes[node_id].truth_value = "OUT"

    nogoods_imported = _import_nogoods(network, prefix, nogoods)

    _fixup_dependents(network)
    propagated = len(network.recompute_all())

    for node_id in retract_after:
        network.retract(node_id)
        retracted += 1

    return {
        "agent": agent_name,
        "prefix": prefix,
        "active_node": active_id,
        "created_premise": created_premise,
        "claims_imported": imported,
        "claims_skipped": skipped,
        "claims_retracted": retracted,
        "claims_propagated": propagated,
        "nogoods_imported": nogoods_imported,
    }


def _sync_claims(network, agent_name, claims, source_path, nogoods):
    """Sync normalized claims into the network (remote wins)."""
    prefix = f"{agent_name}:"
    active_id, inactive_id, created_premise = _ensure_agent_nodes(
        network, agent_name, source_path
    )

    if source_path:
        network.repos[agent_name] = str(Path(source_path).resolve().parent)

    remote_ids = {f"{prefix}{c['id']}" for c in claims}

    infra_ids = {active_id, inactive_id}
    local_agent_ids = {
        nid for nid in network.nodes
        if nid.startswith(prefix) and nid not in infra_ids
    }

    ordered = _topo_sort_claims(claims)

    beliefs_added = 0
    beliefs_updated = 0
    beliefs_unchanged = 0
    retract_after = []
    assert_after = []

    for claim in ordered:
        node_id = f"{prefix}{claim['id']}"
        is_out = claim["is_out"]

        if node_id in network.nodes:
            node = network.nodes[node_id]
            changed = False

            if node.text != claim["text"]:
                node.text = claim["text"]
                changed = True

            if claim["source"] and node.source != claim["source"]:
                node.source = claim["source"]
                changed = True
            if claim.get("source_url") and node.source_url != claim["source_url"]:
                node.source_url = claim["source_url"]
                changed = True
            if claim["source_hash"] and node.source_hash != claim["source_hash"]:
                node.source_hash = claim["source_hash"]
                changed = True
            if claim["date"] and node.date != claim["date"]:
                node.date = claim["date"]
                changed = True

            for k, v in claim["metadata"].items():
                if not k.startswith("_"):
                    node.metadata[k] = v
            node.metadata["imported_from"] = source_path

            if is_out and not claim["raw_justifications"]:
                new_justs = _build_justifications(
                    claim, prefix, inactive_id, agent_name
                )
                if not _justifications_match(node.justifications, new_justs):
                    _update_node_justifications(network, node_id, new_justs)
                    changed = True
                retract_after.append(node_id)
            elif is_out:
                # Preserve justifications so the node can resurrect when the
                # remote flips it to IN. _retracted is set so both _propagate()
                # and recompute_all() skip the node — it stays OUT until the
                # remote explicitly sends IN (the else branch clears _retracted).
                new_justs = _build_justifications(
                    claim, prefix, inactive_id, agent_name
                )
                if not _justifications_match(node.justifications, new_justs):
                    _update_node_justifications(network, node_id, new_justs)
                    changed = True
                if not node.metadata.get("_retracted"):
                    node.metadata["_retracted"] = True
                    changed = True
                if node.truth_value != "OUT":
                    node.truth_value = "OUT"
                    changed = True
            else:
                new_justs = _build_justifications(
                    claim, prefix, inactive_id, agent_name
                )
                if not _justifications_match(node.justifications, new_justs):
                    _update_node_justifications(network, node_id, new_justs)
                    changed = True

                if node.metadata.get("_retracted"):
                    node.metadata.pop("_retracted", None)
                    changed = True

                if node.truth_value == "OUT":
                    assert_after.append(node_id)
                    changed = True

            if changed:
                beliefs_updated += 1
            else:
                beliefs_unchanged += 1
        else:
            justifications = _build_justifications(
                claim, prefix, inactive_id, agent_name
            )

            metadata = claim["metadata"].copy()
            metadata.update({
                "agent": agent_name,
                "original_id": claim["id"],
                "imported_from": source_path,
            })

            network.add_node(
                id=node_id,
                text=claim["text"],
                justifications=justifications if justifications else None,
                source=claim["source"],
                source_url=claim.get("source_url", ""),
                source_hash=claim["source_hash"],
                date=claim["date"],
                metadata=metadata,
            )
            beliefs_added += 1

            if is_out and not claim["raw_justifications"]:
                retract_after.append(node_id)
            elif is_out:
                # Set _retracted directly instead of using retract_after,
                # matching the existing-node sync path which relies on
                # _retracted to keep OUT-with-justifications nodes locked.
                network.nodes[node_id].metadata["_retracted"] = True
                network.nodes[node_id].truth_value = "OUT"

    beliefs_removed = 0
    removed_ids = local_agent_ids - remote_ids
    for node_id in sorted(removed_ids):
        node = network.nodes[node_id]
        if node.truth_value == "IN":
            network.retract(node_id)
            beliefs_removed += 1
        elif not node.metadata.get("_retracted"):
            node.metadata["_retracted"] = True
            beliefs_removed += 1

    _fixup_dependents(network)

    for node_id in assert_after:
        node = network.nodes[node_id]
        node.metadata.pop("_retracted", None)
        if node.truth_value == "OUT":
            network.assert_node(node_id)

    propagated = len(network.recompute_all())

    beliefs_retracted = 0
    for node_id in retract_after:
        if node_id in network.nodes:
            node = network.nodes[node_id]
            if node.truth_value != "OUT":
                network.retract(node_id)
            else:
                node.metadata["_retracted"] = True
            beliefs_retracted += 1

    nogoods_imported = _import_nogoods(network, prefix, nogoods)

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
        "beliefs_propagated": propagated,
        "nogoods_imported": nogoods_imported,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def import_agent(
    network: Network,
    agent_name: str,
    beliefs_text: str,
    nogoods_text: str | None = None,
    only_in: bool = False,
    source_path: str = "",
) -> dict:
    """Import another agent's beliefs into the network with namespacing.

    Each belief is prefixed with 'agent_name:' to avoid ID collisions.
    A premise node 'agent_name:active' is created along with a relay node
    'agent_name:inactive' (IN when active is OUT). Imported beliefs have
    inactive in their outlist — retracting active cascades everything OUT.

    Beliefs that are OUT/STALE in the source are marked with _retracted
    metadata so recompute_all cannot resurrect them. JSON imports preserve
    justifications on OUT nodes (enabling future resurrection when the remote
    flips to IN); markdown imports strip them (bare premise path).
    """
    claims = _normalize_markdown(beliefs_text, only_in)
    nogoods = _normalize_nogoods_markdown(nogoods_text)
    return _import_claims(network, agent_name, claims, source_path, nogoods)


def import_agent_json(
    network: Network,
    agent_name: str,
    data: dict,
    only_in: bool = False,
    source_path: str = "",
) -> dict:
    """Import an agent's beliefs from JSON export with namespacing.

    JSON format preserves full justification structure including outlists,
    providing lossless import of non-monotonic relationships.
    """
    claims = _normalize_json(data, only_in)
    nogoods = _normalize_nogoods_json(data)
    return _import_claims(network, agent_name, claims, source_path, nogoods)


def sync_agent(
    network: Network,
    agent_name: str,
    beliefs_text: str,
    nogoods_text: str | None = None,
    only_in: bool = False,
    source_path: str = "",
) -> dict:
    """Sync another agent's beliefs into the network (remote wins).

    Compares the remote beliefs against existing local agent nodes:
    - New beliefs in remote -> added
    - Beliefs removed from remote -> retracted
    - Changed text/justifications/truth values -> updated
    - _retracted flag cleared when remote says IN
    """
    claims = _normalize_markdown(beliefs_text, only_in)
    nogoods = _normalize_nogoods_markdown(nogoods_text)
    return _sync_claims(network, agent_name, claims, source_path, nogoods)


def sync_agent_json(
    network: Network,
    agent_name: str,
    data: dict,
    only_in: bool = False,
    source_path: str = "",
) -> dict:
    """Sync an agent's beliefs from JSON export (remote wins).

    Same semantics as sync_agent but for JSON format which preserves
    full justification structure including outlists.
    """
    claims = _normalize_json(data, only_in)
    nogoods = _normalize_nogoods_json(data)
    return _sync_claims(network, agent_name, claims, source_path, nogoods)

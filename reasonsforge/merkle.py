"""Merkle tree hashing for justification chain integrity.

Each node stores a text_hash (SHA-256 of its text). Each justification
stores a content_hash — a Merkle root incorporating the node's text and
the recursive Merkle hashes of all antecedents. Any text mutation
anywhere in the chain causes a hash mismatch at verification time.
"""

import hashlib


def compute_text_hash(text: str) -> str:
    """SHA-256 hex digest of node text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fresh_merkle_hash(node_id, net, cache, visiting):
    """Compute a node's Merkle hash from current text (never uses stored hashes)."""
    if node_id in cache:
        return cache[node_id]
    if node_id in visiting:
        return compute_text_hash("")
    visiting.add(node_id)

    node = net.nodes[node_id]
    text_hash = compute_text_hash(node.text)

    if not node.justifications or node.supporting_justification is None:
        result = text_hash
    else:
        j = node.justifications[node.supporting_justification]
        ch = _fresh_content_hash(text_hash, j, net, cache, visiting)
        result = hashlib.sha256(
            (text_hash + "|" + ch).encode("utf-8")
        ).hexdigest()

    cache[node_id] = result
    return result


def _fresh_content_hash(text_hash, justification, net, cache, visiting):
    """Compute a justification's content hash from current text."""
    ant_hashes = sorted(
        _fresh_merkle_hash(a, net, cache, visiting)
        for a in justification.antecedents
        if a in net.nodes
    )
    parts = text_hash + "|" + "|".join(ant_hashes) if ant_hashes else text_hash
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def compute_merkle_hash(node_id, net, cache=None):
    """Recursive Merkle hash for a node, computed from current text."""
    if cache is None:
        cache = {}
    return _fresh_merkle_hash(node_id, net, cache, set())


def compute_content_hash(node, justification, net, cache=None):
    """Merkle root for a justification, computed from current text.

    sha256(node.text_hash | sorted(antecedent_merkle_hashes))
    """
    if cache is None:
        cache = {}
    text_hash = compute_text_hash(node.text)
    return _fresh_content_hash(text_hash, justification, net, cache, set())


def verify_node(node):
    """Check node.text_hash against sha256(node.text).

    Returns a finding dict if mismatch, None if clean or no stored hash.
    """
    if not node.text_hash:
        return None
    current = compute_text_hash(node.text)
    if current != node.text_hash:
        return {
            "type": "text_mutation",
            "node_id": node.id,
            "stored_hash": node.text_hash,
            "current_hash": current,
        }
    return None


def verify_justification(node, j_index, net, cache=None):
    """Check justification.content_hash against recomputed value.

    Returns a finding dict if mismatch, None if clean or no stored hash.
    """
    if j_index < 0 or j_index >= len(node.justifications):
        return None
    j = node.justifications[j_index]
    if not j.content_hash:
        return None
    current = compute_content_hash(node, j, net, cache)
    if current != j.content_hash:
        return {
            "type": "chain_mutation",
            "node_id": node.id,
            "justification_index": j_index,
            "stored_hash": j.content_hash,
            "current_hash": current,
        }
    return None


def verify_all(net):
    """Check all nodes and justifications, return list of mismatches."""
    findings = []
    missing = 0
    cache = {}
    for nid, node in sorted(net.nodes.items()):
        f = verify_node(node)
        if f:
            findings.append(f)
        elif not node.text_hash:
            missing += 1
        for ji in range(len(node.justifications)):
            f = verify_justification(node, ji, net, cache)
            if f:
                findings.append(f)
            elif not node.justifications[ji].content_hash:
                missing += 1
    return {"findings": findings, "missing_hashes": missing}


def backfill_hashes(net):
    """Compute and store hashes for all nodes/justifications missing them."""
    nodes_updated = 0
    justifications_updated = 0
    cache = {}
    for nid, node in net.nodes.items():
        if not node.text_hash:
            node.text_hash = compute_text_hash(node.text)
            nodes_updated += 1
        for j in node.justifications:
            if not j.content_hash:
                j.content_hash = compute_content_hash(node, j, net, cache)
                justifications_updated += 1
    return {
        "nodes_updated": nodes_updated,
        "justifications_updated": justifications_updated,
    }

"""Tests for Merkle tree hashing and integrity verification."""

import hashlib

import pytest
from reasonsforge import api
from reasonsforge.merkle import (
    compute_text_hash,
    compute_merkle_hash,
    compute_content_hash,
    verify_all,
    backfill_hashes,
)


def test_text_hash_set_on_creation(tmp_path):
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("p1", "Safety is established", db_path=str(db))
    node = api.show_node("p1", db_path=str(db))
    expected = hashlib.sha256(b"Safety is established").hexdigest()
    assert node["text_hash"] == expected


def test_content_hash_set_on_creation(tmp_path):
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("p1", "P1 text", db_path=str(db))
    api.add_node("d1", "D1 text", sl="p1", db_path=str(db))
    node = api.show_node("d1", db_path=str(db))
    assert len(node["justifications"]) == 1
    assert node["justifications"][0]["content_hash"] != ""


def test_content_hash_incorporates_antecedents(tmp_path):
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("p1", "P1 text", db_path=str(db))
    api.add_node("p2", "P2 text", db_path=str(db))
    api.add_node("d1", "D1 text", sl="p1,p2", db_path=str(db))

    d1 = api.show_node("d1", db_path=str(db))
    original_hash = d1["justifications"][0]["content_hash"]

    # Simulate unauthorized text mutation via Storage layer
    from reasonsforge.storage import Storage
    from reasonsforge.merkle import compute_text_hash
    store = Storage(str(db))
    net = store.load()
    net.nodes["p1"].text = "P1 CHANGED"
    net.nodes["p1"].text_hash = compute_text_hash("P1 CHANGED")
    store.save(net)
    store.close()

    # Verify detects the chain mutation
    result = api.check_integrity(db_path=str(db))
    chain_ids = [f["node_id"] for f in result["chain_mutations"]]
    assert "d1" in chain_ids


def test_merkle_transitive(tmp_path):
    """A -> B -> C chain: mutating C's text flags B's and A's justifications."""
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("c", "C text", db_path=str(db))
    api.add_node("b", "B text", sl="c", db_path=str(db))
    api.add_node("a", "A text", sl="b", db_path=str(db))

    result = api.check_integrity(db_path=str(db))
    assert len(result["text_mutations"]) == 0
    assert len(result["chain_mutations"]) == 0

    # Simulate unauthorized text mutation via Storage layer
    from reasonsforge.storage import Storage
    from reasonsforge.merkle import compute_text_hash
    store = Storage(str(db))
    net = store.load()
    net.nodes["c"].text = "C CHANGED"
    net.nodes["c"].text_hash = compute_text_hash("C CHANGED")
    store.save(net)
    store.close()

    result = api.check_integrity(db_path=str(db))
    chain_ids = [f["node_id"] for f in result["chain_mutations"]]
    # B's justification cites C directly — should be flagged
    assert "b" in chain_ids
    # A's justification cites B — B's merkle hash changed because C changed
    assert "a" in chain_ids


def test_verify_clean(tmp_path):
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))
    result = api.check_integrity(db_path=str(db))
    assert len(result["text_mutations"]) == 0
    assert len(result["chain_mutations"]) == 0
    assert result["missing_hashes"] == 0


def test_verify_text_mutation(tmp_path):
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("p1", "Original text", db_path=str(db))

    # Directly mutate text without going through update_node
    # (simulating a raw DB edit or bug)
    from reasonsforge.api import _with_network
    with _with_network(str(db), write=True) as net:
        net.nodes["p1"].text = "Tampered text"

    result = api.check_integrity(db_path=str(db))
    assert len(result["text_mutations"]) == 1
    assert result["text_mutations"][0]["node_id"] == "p1"


def test_verify_chain_mutation(tmp_path):
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))

    # Simulate text mutation with updated text_hash (e.g. direct SQL)
    from reasonsforge.storage import Storage
    from reasonsforge.merkle import compute_text_hash
    store = Storage(str(db))
    net = store.load()
    net.nodes["p1"].text = "P1 CHANGED"
    net.nodes["p1"].text_hash = compute_text_hash("P1 CHANGED")
    store.save(net)
    store.close()

    result = api.check_integrity(db_path=str(db))
    # text_hash matches text, so no text_mutation
    assert len(result["text_mutations"]) == 0
    # But d1's justification hash is stale
    assert len(result["chain_mutations"]) == 1
    assert result["chain_mutations"][0]["node_id"] == "d1"


def test_backfill_hashes(tmp_path):
    """Nodes/justifications without hashes get them computed."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    # Create nodes, then strip their hashes to simulate pre-existing data
    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))

    from reasonsforge.api import _with_network
    with _with_network(str(db), write=True) as net:
        net.nodes["p1"].text_hash = ""
        net.nodes["d1"].text_hash = ""
        for j in net.nodes["d1"].justifications:
            j.content_hash = ""

    result = api.check_integrity(db_path=str(db))
    assert result["missing_hashes"] >= 2

    result = api.backfill_hashes(db_path=str(db))
    assert result["nodes_updated"] == 2
    assert result["justifications_updated"] == 1

    result = api.check_integrity(db_path=str(db))
    assert result["missing_hashes"] == 0
    assert len(result["text_mutations"]) == 0
    assert len(result["chain_mutations"]) == 0


def test_update_node_rejects_text(tmp_path):
    """update_node rejects text mutation — beliefs are immutable."""
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("p1", "P1", db_path=str(db))

    with pytest.raises(ValueError, match="immutable"):
        api.update_node("p1", text="P1 CHANGED", db_path=str(db))


def test_export_import_preserves_hashes(tmp_path):
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))

    export = api.export_network(db_path=str(db))
    assert export["nodes"]["p1"]["text_hash"] != ""
    assert export["nodes"]["d1"]["justifications"][0]["content_hash"] != ""

    db2 = tmp_path / "test2.db"
    api.init_db(str(db2))
    import json
    json_path = tmp_path / "export.json"
    json_path.write_text(json.dumps(export))
    api.import_json(str(json_path), db_path=str(db2))

    result = api.check_integrity(db_path=str(db2))
    assert len(result["text_mutations"]) == 0
    assert len(result["chain_mutations"]) == 0


def test_cycle_safety(tmp_path):
    """Verify doesn't infinite-loop on hypothetical cycle in antecedents."""
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("a", "A", db_path=str(db))
    api.add_node("b", "B", sl="a", db_path=str(db))

    # Artificially create a cycle: make a's antecedents point to b
    from reasonsforge import Justification
    from reasonsforge.api import _with_network
    with _with_network(str(db), write=True) as net:
        net.nodes["a"].justifications.append(
            Justification(type="SL", antecedents=["b"])
        )
        net.nodes["a"].justifications[-1].content_hash = "fake"

    # Should not hang
    result = api.check_integrity(db_path=str(db))
    assert isinstance(result, dict)


def test_supersede_preserves_hashes(tmp_path):
    """Superseding a node gives the successor its own hashes."""
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("p1", "Original", db_path=str(db))

    result = api.supersede_with_text("p1", "Updated version", db_path=str(db))
    new_id = result["new_id"]

    new_node = api.show_node(new_id, db_path=str(db))
    assert new_node["text_hash"] == hashlib.sha256(b"Updated version").hexdigest()

    integrity = api.check_integrity(db_path=str(db))
    assert len(integrity["text_mutations"]) == 0
    assert len(integrity["chain_mutations"]) == 0


def test_multiple_justifications_each_hashed(tmp_path):
    """Each justification on a node gets its own content_hash."""
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("p2", "P2", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))
    api.add_justification("d1", sl="p2", db_path=str(db))

    d1 = api.show_node("d1", db_path=str(db))
    assert len(d1["justifications"]) == 2
    assert d1["justifications"][0]["content_hash"] != ""
    assert d1["justifications"][1]["content_hash"] != ""
    # Different antecedents should produce different hashes
    assert d1["justifications"][0]["content_hash"] != d1["justifications"][1]["content_hash"]

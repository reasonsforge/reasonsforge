"""Tests for duplicate-of and superseded-by metadata."""

import pytest
from reasonsforge import api


def test_mark_duplicate(tmp_path):
    """Test marking a node as duplicate of a canonical version."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    # Add canonical and duplicate nodes
    api.add_node("canonical-node", "This is the canonical version", db_path=str(db))
    api.add_node("duplicate-node", "This is a duplicate", db_path=str(db))

    # Mark as duplicate
    result = api.mark_duplicate("duplicate-node", "canonical-node", db_path=str(db))

    assert result["source_id"] == "duplicate-node"
    assert result["canonical_id"] == "canonical-node"
    assert "duplicate-node" in result["changed"]

    # Verify the duplicate node is OUT with metadata
    node = api.show_node("duplicate-node", db_path=str(db))
    assert node["truth_value"] == "OUT"
    assert node["metadata"]["duplicate_of"] == "canonical-node"
    assert "Duplicate of canonical-node" in node["metadata"]["retract_reason"]


def test_mark_superseded(tmp_path):
    """Test marking a node as superseded by a newer version."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    # Add old and new nodes
    api.add_node("old-belief", "Old understanding", db_path=str(db))
    api.add_node("new-belief", "Updated understanding", db_path=str(db))

    # Mark as superseded
    result = api.mark_superseded("old-belief", "new-belief", db_path=str(db))

    assert result["old_id"] == "old-belief"
    assert result["new_id"] == "new-belief"
    assert "old-belief" in result["changed"]

    # Verify the old node is OUT with metadata
    node = api.show_node("old-belief", db_path=str(db))
    assert node["truth_value"] == "OUT"
    assert node["metadata"]["superseded_by"] == "new-belief"
    assert "Superseded by new-belief" in node["metadata"]["retract_reason"]


def test_mark_duplicate_with_cascade(tmp_path):
    """Test that marking a duplicate cascades to dependents."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    # Add canonical and duplicate nodes
    api.add_node("canonical", "Canonical belief", db_path=str(db))
    api.add_node("duplicate", "Duplicate belief", db_path=str(db))

    # Add a dependent
    api.add_node("dependent", "Depends on duplicate", sl="duplicate", db_path=str(db))

    # Verify dependent is IN
    dep = api.show_node("dependent", db_path=str(db))
    assert dep["truth_value"] == "IN"

    # Mark as duplicate
    result = api.mark_duplicate("duplicate", "canonical", db_path=str(db))

    # Verify cascade
    assert "dependent" in result["changed"]

    # Verify dependent went OUT
    dep = api.show_node("dependent", db_path=str(db))
    assert dep["truth_value"] == "OUT"


def test_mark_superseded_with_cascade(tmp_path):
    """Test that marking superseded cascades to dependents."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("old", "Old belief", db_path=str(db))
    api.add_node("new", "New belief", db_path=str(db))
    api.add_node("dependent", "Depends on old", sl="old", db_path=str(db))

    dep = api.show_node("dependent", db_path=str(db))
    assert dep["truth_value"] == "IN"

    result = api.mark_superseded("old", "new", db_path=str(db))
    assert "dependent" in result["changed"]

    dep = api.show_node("dependent", db_path=str(db))
    assert dep["truth_value"] == "OUT"


def test_mark_duplicate_errors(tmp_path):
    """Test error handling for mark_duplicate."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("exists", "This exists", db_path=str(db))

    # Source doesn't exist
    with pytest.raises(KeyError, match="not found"):
        api.mark_duplicate("missing", "exists", db_path=str(db))

    # Canonical doesn't exist
    with pytest.raises(KeyError, match="not found"):
        api.mark_duplicate("exists", "missing", db_path=str(db))


def test_mark_superseded_errors(tmp_path):
    """Test error handling for mark_superseded."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("exists", "This exists", db_path=str(db))

    with pytest.raises(KeyError, match="not found"):
        api.mark_superseded("missing", "exists", db_path=str(db))

    with pytest.raises(KeyError, match="not found"):
        api.mark_superseded("exists", "missing", db_path=str(db))


def test_metadata_in_export(tmp_path):
    """Test that duplicate/superseded metadata is exported correctly."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    # Add nodes and mark relationships
    api.add_node("canonical", "Canonical", db_path=str(db))
    api.add_node("duplicate", "Duplicate", db_path=str(db))
    api.add_node("old", "Old", db_path=str(db))
    api.add_node("new", "New", db_path=str(db))

    api.mark_duplicate("duplicate", "canonical", db_path=str(db))
    api.mark_superseded("old", "new", db_path=str(db))

    # Export and verify metadata
    data = api.export_network(db_path=str(db))

    dup = data["nodes"]["duplicate"]
    assert dup["metadata"]["duplicate_of"] == "canonical"
    assert dup["truth_value"] == "OUT"

    old = data["nodes"]["old"]
    assert old["metadata"]["superseded_by"] == "new"
    assert old["truth_value"] == "OUT"


def test_metadata_in_show_output(tmp_path):
    """Test that duplicate/superseded relationships appear in show command."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("canonical", "Canonical", db_path=str(db))
    api.add_node("duplicate", "Duplicate", db_path=str(db))
    api.mark_duplicate("duplicate", "canonical", db_path=str(db))

    node = api.show_node("duplicate", db_path=str(db))
    assert "duplicate_of" in node["metadata"]
    assert node["metadata"]["duplicate_of"] == "canonical"
    assert "retract_reason" in node["metadata"]


def test_mark_duplicate_self_reference(tmp_path):
    """Test that marking a node as duplicate of itself is rejected."""
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("node-a", "Some belief", db_path=str(db))

    with pytest.raises(ValueError, match="cannot be marked as a duplicate of itself"):
        api.mark_duplicate("node-a", "node-a", db_path=str(db))

    node = api.show_node("node-a", db_path=str(db))
    assert node["truth_value"] == "IN"


def test_mark_superseded_self_reference(tmp_path):
    """Test that marking a node as superseded by itself is rejected."""
    db = tmp_path / "test.db"
    api.init_db(str(db))
    api.add_node("node-a", "Some belief", db_path=str(db))

    with pytest.raises(ValueError, match="cannot be marked as superseded by itself"):
        api.mark_superseded("node-a", "node-a", db_path=str(db))

    node = api.show_node("node-a", db_path=str(db))
    assert node["truth_value"] == "IN"


def test_mark_duplicate_already_out(tmp_path):
    """Test marking an already-OUT node as duplicate."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("canonical", "Canonical", db_path=str(db))
    api.add_node("duplicate", "Duplicate", db_path=str(db))
    api.retract_node("duplicate", reason="Initially retracted", db_path=str(db))

    result = api.mark_duplicate("duplicate", "canonical", db_path=str(db))
    assert result["source_id"] == "duplicate"

    node = api.show_node("duplicate", db_path=str(db))
    assert node["truth_value"] == "OUT"
    assert node["metadata"]["duplicate_of"] == "canonical"
    assert node["metadata"]["retract_reason"] == "Duplicate of canonical"


def test_mark_superseded_already_out(tmp_path):
    """Test marking an already-OUT node as superseded."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("old", "Old belief", db_path=str(db))
    api.add_node("new", "New belief", db_path=str(db))
    api.retract_node("old", reason="Initially retracted", db_path=str(db))

    result = api.mark_superseded("old", "new", db_path=str(db))
    assert result["old_id"] == "old"

    node = api.show_node("old", db_path=str(db))
    assert node["truth_value"] == "OUT"
    assert node["metadata"]["superseded_by"] == "new"
    assert node["metadata"]["retract_reason"] == "Superseded by new"

"""Tests for graph-native justification defeaters."""

import pytest
from reasonsforge import api


def test_defeat_justification_basic(tmp_path):
    """Test basic justification defeat mechanism."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    # Add premises and derived belief
    api.add_node("premise-a", "Premise A", db_path=str(db))
    api.add_node("premise-b", "Premise B", db_path=str(db))
    api.add_node("belief", "Derived from A and B", sl="premise-a,premise-b", db_path=str(db))

    # Verify belief is IN
    node = api.show_node("belief", db_path=str(db))
    assert node["truth_value"] == "IN"

    # Defeat the justification
    result = api.defeat_justification(
        "belief", 0, "Over-generalizes",
        defeater_type="over-generalizes",
        db_path=str(db)
    )

    assert result["node_id"] == "belief"
    assert result["justification_index"] == 0
    assert result["defeater_type"] == "over-generalizes"
    assert "over-generalizes-belief-j0" == result["defeater_id"]
    assert "belief" in result["changed"]

    # Verify belief went OUT
    node = api.show_node("belief", db_path=str(db))
    assert node["truth_value"] == "OUT"

    # Verify defeater exists and is IN
    defeater = api.show_node(result["defeater_id"], db_path=str(db))
    assert defeater["truth_value"] == "IN"
    assert defeater["metadata"]["defeater_type"] == "over-generalizes"
    assert defeater["metadata"]["defeats_node"] == "belief"
    assert defeater["metadata"]["defeats_justification"] == 0


def test_defeat_justification_added_to_outlist(tmp_path):
    """Test that defeater is added to justification's outlist."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p", "Premise", db_path=str(db))
    api.add_node("b", "Belief", sl="p", db_path=str(db))

    # Defeat the justification
    result = api.defeat_justification(
        "b", 0, "Invalid inference",
        db_path=str(db)
    )

    # Check justification's outlist
    node = api.show_node("b", db_path=str(db))
    assert len(node["justifications"]) == 1
    assert result["defeater_id"] in node["justifications"][0]["outlist"]


def test_defeat_restoration_on_retract(tmp_path):
    """Test that retracting defeater restores the belief."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p", "Premise", db_path=str(db))
    api.add_node("b", "Belief", sl="p", db_path=str(db))

    # Defeat
    result = api.defeat_justification("b", 0, "Test defeat", db_path=str(db))
    defeater_id = result["defeater_id"]

    # Verify OUT
    node = api.show_node("b", db_path=str(db))
    assert node["truth_value"] == "OUT"

    # Retract defeater
    retract_result = api.retract_node(defeater_id, db_path=str(db))

    # Verify belief restored to IN
    node = api.show_node("b", db_path=str(db))
    assert node["truth_value"] == "IN"
    assert "b" in retract_result["changed"]


def test_defeat_cascade_to_dependents(tmp_path):
    """Test that defeating cascades to dependent beliefs."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    # Chain: p -> b1 -> b2
    api.add_node("p", "Premise", db_path=str(db))
    api.add_node("b1", "Belief 1", sl="p", db_path=str(db))
    api.add_node("b2", "Belief 2", sl="b1", db_path=str(db))

    # All should be IN
    assert api.show_node("b1", db_path=str(db))["truth_value"] == "IN"
    assert api.show_node("b2", db_path=str(db))["truth_value"] == "IN"

    # Defeat b1's justification
    result = api.defeat_justification("b1", 0, "Test", db_path=str(db))

    # Both b1 and b2 should go OUT
    assert "b1" in result["changed"]
    assert "b2" in result["changed"]
    assert api.show_node("b1", db_path=str(db))["truth_value"] == "OUT"
    assert api.show_node("b2", db_path=str(db))["truth_value"] == "OUT"


def test_defeat_multiple_justifications(tmp_path):
    """Test defeating specific justification when node has multiple."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "Premise 1", db_path=str(db))
    api.add_node("p2", "Premise 2", db_path=str(db))
    api.add_node("b", "Belief", sl="p1", db_path=str(db))

    # Add second justification
    api.add_justification("b", sl="p2", db_path=str(db))

    # Belief has 2 justifications
    node = api.show_node("b", db_path=str(db))
    assert len(node["justifications"]) == 2
    assert node["truth_value"] == "IN"

    # Defeat first justification
    result = api.defeat_justification("b", 0, "First is invalid", db_path=str(db))

    # Belief should still be IN (second justification is still valid)
    node = api.show_node("b", db_path=str(db))
    assert node["truth_value"] == "IN"

    # Defeat second justification
    result2 = api.defeat_justification("b", 1, "Second is invalid", db_path=str(db))

    # Now belief should be OUT (no valid justifications)
    node = api.show_node("b", db_path=str(db))
    assert node["truth_value"] == "OUT"


def test_defeat_custom_id(tmp_path):
    """Test custom defeater ID."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p", "Premise", db_path=str(db))
    api.add_node("b", "Belief", sl="p", db_path=str(db))

    # Use custom defeater ID
    result = api.defeat_justification(
        "b", 0, "Custom defeat",
        defeater_id="my-custom-defeater",
        db_path=str(db)
    )

    assert result["defeater_id"] == "my-custom-defeater"

    # Verify custom ID works
    defeater = api.show_node("my-custom-defeater", db_path=str(db))
    assert defeater["truth_value"] == "IN"


def test_defeat_errors(tmp_path):
    """Test error handling."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p", "Premise", db_path=str(db))
    api.add_node("b", "Belief", sl="p", db_path=str(db))

    # Node doesn't exist
    with pytest.raises(KeyError, match="not found"):
        api.defeat_justification("missing", 0, "Test", db_path=str(db))

    # Node has no justifications
    with pytest.raises(ValueError, match="no justifications"):
        api.defeat_justification("p", 0, "Test", db_path=str(db))

    # Index out of range
    with pytest.raises(IndexError, match="out of range"):
        api.defeat_justification("b", 5, "Test", db_path=str(db))


def test_defeater_types(tmp_path):
    """Test different defeater types."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p", "Premise", db_path=str(db))

    for dtype in ["invalid-inference", "over-generalizes", "duplicate-of", "superseded-by"]:
        belief_id = f"belief-{dtype}"
        api.add_node(belief_id, f"Belief {dtype}", sl="p", db_path=str(db))

        result = api.defeat_justification(
            belief_id, 0, f"Test {dtype}",
            defeater_type=dtype,
            db_path=str(db)
        )

        assert result["defeater_type"] == dtype
        defeater = api.show_node(result["defeater_id"], db_path=str(db))
        assert defeater["metadata"]["defeater_type"] == dtype


def test_defeater_in_export(tmp_path):
    """Test that defeaters are preserved in export."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p", "Premise", db_path=str(db))
    api.add_node("b", "Belief", sl="p", db_path=str(db))

    result = api.defeat_justification(
        "b", 0, "Test defeat",
        defeater_type="over-generalizes",
        db_path=str(db)
    )

    # Export and verify
    data = api.export_network(db_path=str(db))

    # Defeater should be in export
    defeater = data["nodes"][result["defeater_id"]]
    assert defeater["metadata"]["defeater_type"] == "over-generalizes"
    assert defeater["metadata"]["defeats_node"] == "b"

    # Belief's justification should have defeater in outlist
    belief = data["nodes"]["b"]
    assert result["defeater_id"] in belief["justifications"][0]["outlist"]


def test_premise_defeat(tmp_path):
    """Test that premises can't be defeated (they have no justifications)."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("premise", "A premise", db_path=str(db))

    # Premises have no justifications
    with pytest.raises(ValueError, match="no justifications"):
        api.defeat_justification("premise", 0, "Test", db_path=str(db))

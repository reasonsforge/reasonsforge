"""Tests for justified defeaters with scope beliefs."""

import pytest
from reasonsforge import api


def test_defeat_with_scope_basic(tmp_path):
    """Test basic scope-belief defeat mechanism."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("premise-a", "A establishes safety", db_path=str(db))
    api.add_node("premise-b", "B establishes correctness", db_path=str(db))
    api.add_node("belief", "System is safe, correct, and traceable",
                 sl="premise-a,premise-b", db_path=str(db))

    node = api.show_node("belief", db_path=str(db))
    assert node["truth_value"] == "IN"

    result = api.defeat_with_scope(
        "belief", 0,
        scope_findings=[
            {"antecedent": "premise-a", "establishes": "safety",
             "does_not_establish": "traceability"},
            {"antecedent": "premise-b", "establishes": "correctness",
             "does_not_establish": "traceability"},
        ],
        missing_property="traceability",
        db_path=str(db),
    )

    assert result["node_id"] == "belief"
    assert result["justification_index"] == 0
    assert len(result["scope_belief_ids"]) == 2
    assert "belief" in result["changed"]

    node = api.show_node("belief", db_path=str(db))
    assert node["truth_value"] == "OUT"


def test_scope_beliefs_are_premises(tmp_path):
    """Verify scope beliefs are created as IN premises with correct metadata."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "P1 text", db_path=str(db))
    api.add_node("derived", "Derived text", sl="p1", db_path=str(db))

    result = api.defeat_with_scope(
        "derived", 0,
        scope_findings=[
            {"antecedent": "p1", "establishes": "X", "does_not_establish": "Y"},
        ],
        missing_property="Y",
        db_path=str(db),
    )

    scope_id = result["scope_belief_ids"][0]
    scope = api.show_node(scope_id, db_path=str(db))
    assert scope["truth_value"] == "IN"
    assert scope["justifications"] == []
    assert scope["metadata"]["scope_of"] == "p1"
    assert scope["metadata"]["for_defeater"] == result["defeater_id"]
    assert "establishes X" in scope["text"]
    assert "does not establish Y" in scope["text"]


def test_defeater_is_derived(tmp_path):
    """Verify the defeater is a derived node with SL justification."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))

    result = api.defeat_with_scope(
        "d1", 0,
        scope_findings=[
            {"antecedent": "p1", "establishes": "X", "does_not_establish": "Y"},
        ],
        missing_property="Y",
        db_path=str(db),
    )

    defeater = api.show_node(result["defeater_id"], db_path=str(db))
    assert defeater["truth_value"] == "IN"
    assert len(defeater["justifications"]) == 1
    j = defeater["justifications"][0]
    assert j["type"] == "SL"
    assert j["antecedents"] == result["scope_belief_ids"]
    assert defeater["metadata"]["defeats_node"] == "d1"
    assert defeater["metadata"]["defeats_justification"] == 0


def test_challenge_scope_restores_target(tmp_path):
    """Challenging a scope belief should restore the original belief."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("p2", "P2", db_path=str(db))
    api.add_node("target", "Target", sl="p1,p2", db_path=str(db))

    result = api.defeat_with_scope(
        "target", 0,
        scope_findings=[
            {"antecedent": "p1", "establishes": "X", "does_not_establish": "Z"},
            {"antecedent": "p2", "establishes": "Y", "does_not_establish": "Z"},
        ],
        missing_property="Z",
        db_path=str(db),
    )

    assert api.show_node("target", db_path=str(db))["truth_value"] == "OUT"

    scope_id = result["scope_belief_ids"][0]
    api.retract_node(scope_id, db_path=str(db))

    assert api.show_node(scope_id, db_path=str(db))["truth_value"] == "OUT"
    defeater = api.show_node(result["defeater_id"], db_path=str(db))
    assert defeater["truth_value"] == "OUT"
    assert api.show_node("target", db_path=str(db))["truth_value"] == "IN"


def test_defeat_with_scope_in_export(tmp_path):
    """Verify scope beliefs and justified defeater survive export/import."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))

    result = api.defeat_with_scope(
        "d1", 0,
        scope_findings=[
            {"antecedent": "p1", "establishes": "X", "does_not_establish": "Y"},
        ],
        missing_property="Y",
        db_path=str(db),
    )

    export = api.export_network(db_path=str(db))
    scope_id = result["scope_belief_ids"][0]
    assert scope_id in export["nodes"]
    assert result["defeater_id"] in export["nodes"]

    defeater_data = export["nodes"][result["defeater_id"]]
    assert len(defeater_data["justifications"]) == 1
    assert defeater_data["justifications"][0]["antecedents"] == [scope_id]

    db2 = tmp_path / "test2.db"
    api.init_db(str(db2))
    import json
    json_path = tmp_path / "export.json"
    json_path.write_text(json.dumps(export))
    api.import_json(str(json_path), db_path=str(db2))

    d1 = api.show_node("d1", db_path=str(db2))
    assert d1["truth_value"] == "OUT"


def test_defeat_with_scope_errors(tmp_path):
    """Test error cases for defeat_with_scope."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))

    with pytest.raises(KeyError, match="not found"):
        api.defeat_with_scope("nonexistent", 0, [{"antecedent": "p1",
            "establishes": "X"}], "Y", db_path=str(db))

    with pytest.raises(ValueError, match="no justifications"):
        api.defeat_with_scope("p1", 0, [{"antecedent": "p1",
            "establishes": "X"}], "Y", db_path=str(db))

    with pytest.raises(IndexError, match="out of range"):
        api.defeat_with_scope("d1", 5, [{"antecedent": "p1",
            "establishes": "X"}], "Y", db_path=str(db))

    with pytest.raises(ValueError, match="scope_findings must not be empty"):
        api.defeat_with_scope("d1", 0, [], "Y", db_path=str(db))


def test_scope_ids_are_unique(tmp_path):
    """Multiple defeats on same node don't collide on scope belief IDs."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))

    r1 = api.defeat_with_scope(
        "d1", 0,
        scope_findings=[
            {"antecedent": "p1", "establishes": "X", "does_not_establish": "Y"},
        ],
        missing_property="Y",
        defeater_id="defeater-1",
        db_path=str(db),
    )

    api.add_justification("d1", sl="p1", db_path=str(db))

    r2 = api.defeat_with_scope(
        "d1", 1,
        scope_findings=[
            {"antecedent": "p1", "establishes": "X", "does_not_establish": "Z"},
        ],
        missing_property="Z",
        defeater_id="defeater-2",
        db_path=str(db),
    )

    all_ids = set(r1["scope_belief_ids"] + r2["scope_belief_ids"])
    assert len(all_ids) == 2


def test_cascade_through_scope_defeat(tmp_path):
    """Defeating a belief cascades to its dependents."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))
    api.add_node("d2", "D2 depends on D1", sl="d1", db_path=str(db))

    assert api.show_node("d2", db_path=str(db))["truth_value"] == "IN"

    result = api.defeat_with_scope(
        "d1", 0,
        scope_findings=[
            {"antecedent": "p1", "establishes": "X", "does_not_establish": "Y"},
        ],
        missing_property="Y",
        db_path=str(db),
    )

    assert api.show_node("d1", db_path=str(db))["truth_value"] == "OUT"
    assert api.show_node("d2", db_path=str(db))["truth_value"] == "OUT"


def test_defeat_reason_type_stored(tmp_path):
    """defeat_reason_type is stored in metadata and returned."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))

    result = api.defeat_with_scope(
        "d1", 0,
        scope_findings=[
            {"antecedent": "p1", "establishes": "X", "does_not_establish": "Y"},
        ],
        missing_property="Y",
        defeat_reason_type="unsupported-conjunct",
        db_path=str(db),
    )

    assert result["defeat_reason_type"] == "unsupported-conjunct"
    defeater = api.show_node(result["defeater_id"], db_path=str(db))
    assert defeater["metadata"]["defeat_reason_type"] == "unsupported-conjunct"


def test_defeat_reason_type_omitted(tmp_path):
    """defeat_reason_type defaults to empty and is not stored when empty."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))

    result = api.defeat_with_scope(
        "d1", 0,
        scope_findings=[
            {"antecedent": "p1", "establishes": "X", "does_not_establish": "Y"},
        ],
        missing_property="Y",
        db_path=str(db),
    )

    assert result["defeat_reason_type"] == ""
    defeater = api.show_node(result["defeater_id"], db_path=str(db))
    assert "defeat_reason_type" not in defeater["metadata"]


def test_defeat_justification_reason_type(tmp_path):
    """defeat_justification stores defeat_reason_type in metadata."""
    db = tmp_path / "test.db"
    api.init_db(str(db))

    api.add_node("p1", "P1", db_path=str(db))
    api.add_node("d1", "D1", sl="p1", db_path=str(db))

    result = api.defeat_justification(
        "d1", 0, "overclaims scope",
        defeat_reason_type="scope-mismatch",
        db_path=str(db),
    )

    assert result["defeat_reason_type"] == "scope-mismatch"
    defeater = api.show_node(result["defeater_id"], db_path=str(db))
    assert defeater["metadata"]["defeat_reason_type"] == "scope-mismatch"

"""Tests for migrate-defeaters command."""

import pytest
from reasonsforge import api


def test_basic_migration(tmp_path):
    """OUT node with retract_reason + satisfied justification gets defeated via outlist."""
    db = str(tmp_path / "test.db")
    api.init_db(db)
    api.add_node("base", "Base belief", db_path=db)
    api.add_node("derived", "Derived belief", sl="base", db_path=db)

    assert api.show_node("derived", db_path=db)["truth_value"] == "IN"
    api.retract_node("derived", reason="Invalid reasoning", db_path=db)
    assert api.show_node("derived", db_path=db)["truth_value"] == "OUT"

    result = api.migrate_retract_to_defeaters(dry_run=False, db_path=db)

    assert len(result["migrated"]) == 1
    assert result["migrated"][0]["id"] == "derived"
    assert result["migrated"][0]["defeater_id"] == "migrated-retraction-derived-j0"

    node = api.show_node("derived", db_path=db)
    assert node["truth_value"] == "OUT"
    assert "retract_reason" not in node["metadata"]

    defeater = api.show_node("migrated-retraction-derived-j0", db_path=db)
    assert defeater["metadata"]["defeater_type"] == "migrated-retraction"
    assert defeater["metadata"]["defeats_node"] == "derived"


def test_dry_run(tmp_path):
    """Dry run returns candidates without modifying anything."""
    db = str(tmp_path / "test.db")
    api.init_db(db)
    api.add_node("base", "Base belief", db_path=db)
    api.add_node("derived", "Derived belief", sl="base", db_path=db)
    api.retract_node("derived", reason="Invalid", db_path=db)

    result = api.migrate_retract_to_defeaters(dry_run=True, db_path=db)

    assert len(result["migrated"]) == 1
    assert result["migrated"][0]["id"] == "derived"
    assert "defeater_id" not in result["migrated"][0]

    node = api.show_node("derived", db_path=db)
    assert "retract_reason" in node["metadata"]
    assert node["metadata"]["retract_reason"] == "Invalid"


def test_skip_unsatisfied_justification(tmp_path):
    """OUT node where antecedent is also OUT should be skipped."""
    db = str(tmp_path / "test.db")
    api.init_db(db)
    api.add_node("base", "Base belief", db_path=db)
    api.add_node("derived", "Derived belief", sl="base", db_path=db)

    api.retract_node("base", db_path=db)
    assert api.show_node("derived", db_path=db)["truth_value"] == "OUT"

    api.set_metadata("derived", "retract_reason", "Some reason", db_path=db)

    result = api.migrate_retract_to_defeaters(dry_run=False, db_path=db)

    assert len(result["migrated"]) == 0
    assert any(s["id"] == "derived" for s in result["skipped"])


def test_skip_no_retract_reason(tmp_path):
    """OUT node without retract_reason is not a migration candidate."""
    db = str(tmp_path / "test.db")
    api.init_db(db)
    api.add_node("base", "Base belief", db_path=db)
    api.add_node("derived", "Derived belief", sl="base", db_path=db)
    api.retract_node("base", db_path=db)

    result = api.migrate_retract_to_defeaters(dry_run=False, db_path=db)

    assert len(result["migrated"]) == 0


def test_metadata_cleared(tmp_path):
    """Retract_reason and _retracted metadata are cleared after migration."""
    db = str(tmp_path / "test.db")
    api.init_db(db)
    api.add_node("base", "Base belief", db_path=db)
    api.add_node("derived", "Derived belief", sl="base", db_path=db)
    api.retract_node("derived", reason="Old reason", db_path=db)

    api.migrate_retract_to_defeaters(dry_run=False, db_path=db)

    node = api.show_node("derived", db_path=db)
    assert "retract_reason" not in node["metadata"]
    assert "_retracted" not in node["metadata"]


def test_reversibility(tmp_path):
    """Retracting the migrated defeater restores the original belief to IN."""
    db = str(tmp_path / "test.db")
    api.init_db(db)
    api.add_node("base", "Base belief", db_path=db)
    api.add_node("derived", "Derived belief", sl="base", db_path=db)
    api.retract_node("derived", reason="Wrong", db_path=db)

    api.migrate_retract_to_defeaters(dry_run=False, db_path=db)
    assert api.show_node("derived", db_path=db)["truth_value"] == "OUT"

    api.retract_node("migrated-retraction-derived-j0", db_path=db)
    assert api.show_node("derived", db_path=db)["truth_value"] == "IN"


def test_node_ids_filter(tmp_path):
    """Only specified node_ids are migrated when filter is provided."""
    db = str(tmp_path / "test.db")
    api.init_db(db)
    api.add_node("base", "Base belief", db_path=db)
    api.add_node("d1", "Derived 1", sl="base", db_path=db)
    api.add_node("d2", "Derived 2", sl="base", db_path=db)
    api.retract_node("d1", reason="Reason 1", db_path=db)
    api.retract_node("d2", reason="Reason 2", db_path=db)

    result = api.migrate_retract_to_defeaters(
        node_ids=["d1"], dry_run=False, db_path=db
    )

    assert len(result["migrated"]) == 1
    assert result["migrated"][0]["id"] == "d1"

    node_d2 = api.show_node("d2", db_path=db)
    assert "retract_reason" in node_d2["metadata"]


def test_error_on_missing_node(tmp_path):
    """Missing node_ids are reported in errors."""
    db = str(tmp_path / "test.db")
    api.init_db(db)

    result = api.migrate_retract_to_defeaters(
        node_ids=["nonexistent"], dry_run=False, db_path=db
    )

    assert len(result["errors"]) == 1
    assert result["errors"][0]["id"] == "nonexistent"

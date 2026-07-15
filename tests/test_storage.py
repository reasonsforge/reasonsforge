"""Tests for SQLite persistence."""

import tempfile
from pathlib import Path

import pytest

from reasonsforge import Justification
from reasonsforge.network import Network
from reasonsforge.storage import Storage


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_reasons.db"


class TestRoundTrip:
    """Save and load preserves network state."""

    def test_empty_network(self, db_path):
        net = Network()
        store = Storage(db_path)
        store.save(net)
        loaded = store.load()
        assert loaded.nodes == {}
        assert loaded.nogoods == []
        store.close()

    def test_premises(self, db_path):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")

        store = Storage(db_path)
        store.save(net)
        loaded = store.load()
        assert set(loaded.nodes.keys()) == {"a", "b"}
        assert loaded.nodes["a"].truth_value == "IN"
        assert loaded.nodes["a"].text == "Premise A"
        store.close()

    def test_derived_nodes(self, db_path):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"], label="test")],
        )

        store = Storage(db_path)
        store.save(net)
        loaded = store.load()

        b = loaded.nodes["b"]
        assert b.truth_value == "IN"
        assert len(b.justifications) == 1
        assert b.justifications[0].type == "SL"
        assert b.justifications[0].antecedents == ["a"]
        assert b.justifications[0].label == "test"
        # Dependent index rebuilt
        assert "b" in loaded.nodes["a"].dependents
        store.close()

    def test_retracted_state(self, db_path):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.retract("a")

        store = Storage(db_path)
        store.save(net)
        loaded = store.load()
        assert loaded.nodes["a"].truth_value == "OUT"
        assert loaded.nodes["b"].truth_value == "OUT"
        store.close()

    def test_nogoods_persisted(self, db_path):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("d", "Premise D")
        net.add_nogood(["a", "d"])

        store = Storage(db_path)
        store.save(net)
        loaded = store.load()
        assert len(loaded.nogoods) == 1
        assert loaded.nogoods[0].nodes == ["a", "d"]
        store.close()

    def test_log_persisted(self, db_path):
        net = Network()
        net.add_node("a", "Premise A")

        store = Storage(db_path)
        store.save(net)
        loaded = store.load()
        assert len(loaded.log) > 0
        assert any(e["action"] == "add" for e in loaded.log)
        store.close()

    def test_metadata_persisted(self, db_path):
        net = Network()
        net.add_node("a", "Premise A", source="repo:file.py", source_hash="abc123",
                      metadata={"confidence": 0.9})

        store = Storage(db_path)
        store.save(net)
        loaded = store.load()
        assert loaded.nodes["a"].source == "repo:file.py"
        assert loaded.nodes["a"].source_hash == "abc123"
        assert loaded.nodes["a"].metadata == {"confidence": 0.9}
        store.close()

    def test_propagation_works_after_load(self, db_path):
        """Load a network, then operate on it — propagation still works."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )

        store = Storage(db_path)
        store.save(net)
        loaded = store.load()

        # Retract in the loaded network
        loaded.retract("a")
        assert loaded.nodes["b"].truth_value == "OUT"

        # Restore
        loaded.assert_node("a")
        assert loaded.nodes["b"].truth_value == "IN"
        store.close()

    def test_save_overwrites(self, db_path):
        """Saving twice replaces the previous state."""
        net = Network()
        net.add_node("a", "Premise A")

        store = Storage(db_path)
        store.save(net)

        net.add_node("b", "Premise B")
        store.save(net)

        loaded = store.load()
        assert set(loaded.nodes.keys()) == {"a", "b"}
        store.close()

"""Tests for non-monotonic justifications (outlist).

Doyle's key innovation: SL justifications have both an inlist (must be IN)
and an outlist (must be OUT). "Believe X unless Y is believed."
"""

import pytest

from reasonsforge import Justification
from reasonsforge.network import Network
from reasonsforge.storage import Storage
from reasonsforge import api


class TestOutlistBasic:
    """Outlist makes justifications invalid when outlist nodes go IN."""

    def test_outlist_valid_when_out(self):
        """X holds unless Y — Y is OUT, so X is IN."""
        net = Network()
        net.add_node("y", "Counter-evidence Y")
        net.retract("y")
        net.add_node("x", "Default X", justifications=[
            Justification(type="SL", antecedents=[], outlist=["y"]),
        ])
        assert net.nodes["x"].truth_value == "IN"

    def test_outlist_invalid_when_in(self):
        """X holds unless Y — Y is IN, so X is OUT."""
        net = Network()
        net.add_node("y", "Counter-evidence Y")
        net.add_node("x", "Default X", justifications=[
            Justification(type="SL", antecedents=[], outlist=["y"]),
        ])
        assert net.nodes["x"].truth_value == "OUT"

    def test_outlist_with_inlist(self):
        """X holds if A is IN and Y is OUT."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("y", "Counter-evidence Y")
        net.retract("y")
        net.add_node("x", "X from A unless Y", justifications=[
            Justification(type="SL", antecedents=["a"], outlist=["y"]),
        ])
        assert net.nodes["x"].truth_value == "IN"

    def test_outlist_with_inlist_both_fail(self):
        """X requires A IN and Y OUT — A is OUT, so X is OUT regardless of Y."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("y", "Counter-evidence Y")
        net.retract("a")
        net.retract("y")
        net.add_node("x", "X from A unless Y", justifications=[
            Justification(type="SL", antecedents=["a"], outlist=["y"]),
        ])
        assert net.nodes["x"].truth_value == "OUT"

    def test_outlist_node_not_in_network(self):
        """Outlist node doesn't exist — treated as OUT (absent = not believed)."""
        net = Network()
        net.add_node("x", "Default X", justifications=[
            Justification(type="SL", antecedents=[], outlist=["nonexistent"]),
        ])
        assert net.nodes["x"].truth_value == "IN"


class TestOutlistCascading:
    """Outlist triggers cascades when outlist nodes change status."""

    def test_assert_outlist_node_retracts_dependent(self):
        """X unless Y. Assert Y → X goes OUT."""
        net = Network()
        net.add_node("y", "Counter-evidence Y")
        net.retract("y")
        net.add_node("x", "Default X", justifications=[
            Justification(type="SL", antecedents=[], outlist=["y"]),
        ])
        assert net.nodes["x"].truth_value == "IN"

        # Assert Y — X should go OUT
        changed = net.assert_node("y")
        assert net.nodes["x"].truth_value == "OUT"
        assert "x" in changed

    def test_retract_outlist_node_restores_dependent(self):
        """X unless Y. Y is IN (X is OUT). Retract Y → X restored."""
        net = Network()
        net.add_node("y", "Counter-evidence Y")
        net.add_node("x", "Default X", justifications=[
            Justification(type="SL", antecedents=[], outlist=["y"]),
        ])
        assert net.nodes["x"].truth_value == "OUT"

        # Retract Y — X should come back IN
        changed = net.retract("y")
        assert net.nodes["x"].truth_value == "IN"
        assert "x" in changed

    def test_outlist_cascade_through_chain(self):
        """X unless Y, Z depends on X. Assert Y → both X and Z go OUT."""
        net = Network()
        net.add_node("y", "Counter-evidence Y")
        net.retract("y")
        net.add_node("x", "Default X", justifications=[
            Justification(type="SL", antecedents=[], outlist=["y"]),
        ])
        net.add_node("z", "Derived Z", justifications=[
            Justification(type="SL", antecedents=["x"]),
        ])
        assert net.nodes["z"].truth_value == "IN"

        changed = net.assert_node("y")
        assert net.nodes["x"].truth_value == "OUT"
        assert net.nodes["z"].truth_value == "OUT"

    def test_multiple_outlist_nodes(self):
        """X unless Y or Z — both must be OUT."""
        net = Network()
        net.add_node("y", "Y")
        net.add_node("z", "Z")
        net.retract("y")
        net.retract("z")
        net.add_node("x", "Default X", justifications=[
            Justification(type="SL", antecedents=[], outlist=["y", "z"]),
        ])
        assert net.nodes["x"].truth_value == "IN"

        # Assert just Y — X goes OUT
        net.assert_node("y")
        assert net.nodes["x"].truth_value == "OUT"

        # Retract Y again — X comes back
        net.retract("y")
        assert net.nodes["x"].truth_value == "IN"


class TestOutlistWithMultipleJustifications:
    """Outlist interacts with multiple justifications correctly."""

    def test_alternate_justification_survives_outlist(self):
        """X has two justifications: SL(unless Y) and SL(A). Y goes IN but A keeps X alive."""
        net = Network()
        net.add_node("y", "Counter-evidence Y")
        net.add_node("a", "Premise A")
        net.retract("y")
        net.add_node("x", "X with backup", justifications=[
            Justification(type="SL", antecedents=[], outlist=["y"]),
            Justification(type="SL", antecedents=["a"]),
        ])
        assert net.nodes["x"].truth_value == "IN"

        # Assert Y — first justification fails, but second keeps X IN
        net.assert_node("y")
        assert net.nodes["x"].truth_value == "IN"

        # Now retract A — both justifications fail, X goes OUT
        net.retract("a")
        assert net.nodes["x"].truth_value == "OUT"


class TestOutlistExplain:
    """Explain traces show outlist information."""

    def test_explain_shows_outlist_on_valid(self):
        net = Network()
        net.add_node("y", "Y")
        net.retract("y")
        net.add_node("x", "Default X", justifications=[
            Justification(type="SL", antecedents=[], outlist=["y"]),
        ])
        steps = net.explain("x")
        assert steps[0]["truth_value"] == "IN"
        assert steps[0]["outlist"] == ["y"]

    def test_explain_shows_violated_outlist(self):
        net = Network()
        net.add_node("y", "Y")
        net.add_node("x", "Default X", justifications=[
            Justification(type="SL", antecedents=[], outlist=["y"]),
        ])
        steps = net.explain("x")
        assert steps[0]["truth_value"] == "OUT"
        assert steps[0]["violated_outlist"] == ["y"]


class TestOutlistPersistence:
    """Outlist survives SQLite round-trip."""

    def test_round_trip(self, tmp_path):
        db_path = tmp_path / "test.db"

        net = Network()
        net.add_node("y", "Y")
        net.retract("y")
        net.add_node("x", "Default X", justifications=[
            Justification(type="SL", antecedents=[], outlist=["y"], label="default unless Y"),
        ])

        store = Storage(db_path)
        store.save(net)
        loaded = store.load()
        store.close()

        j = loaded.nodes["x"].justifications[0]
        assert j.outlist == ["y"]
        assert j.label == "default unless Y"
        assert loaded.nodes["x"].truth_value == "IN"

        # Outlist propagation works after load
        loaded.assert_node("y")
        assert loaded.nodes["x"].truth_value == "OUT"


class TestOutlistAPI:
    """API layer supports --unless."""

    def test_add_with_unless(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("y", "Y", db_path=db)
        api.retract_node("y", db_path=db)
        result = api.add_node("x", "Default X", unless="y", db_path=db)
        assert result["truth_value"] == "IN"

        # Assert Y — X goes OUT
        api.assert_node("y", db_path=db)
        status = api.get_status(db_path=db)
        x_node = [n for n in status["nodes"] if n["id"] == "x"][0]
        assert x_node["truth_value"] == "OUT"

    def test_add_with_sl_and_unless(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("y", "Y", db_path=db)
        api.retract_node("y", db_path=db)
        result = api.add_node("x", "X from A unless Y", sl="a", unless="y", db_path=db)
        assert result["truth_value"] == "IN"

    def test_unless_only_no_sl(self, tmp_path):
        """Unless without --sl creates a premise-like node that holds unless something is believed."""
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("y", "Y", db_path=db)
        api.retract_node("y", db_path=db)
        result = api.add_node("x", "Default X", unless="y", db_path=db)
        assert result["truth_value"] == "IN"
        assert result["type"] == "SL"

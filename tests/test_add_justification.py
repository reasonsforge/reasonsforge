"""Tests for add-justification: appending justifications to existing nodes."""

import pytest

from reasonsforge import Justification, api
from reasonsforge.network import Network


class TestNetworkAddJustification:
    def test_add_justification_to_derived_node(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("c", "Premise C")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        assert net.nodes["b"].truth_value == "IN"

        result = net.add_justification(
            "b", Justification(type="SL", antecedents=["c"])
        )
        assert result["node_id"] == "b"
        assert result["old_truth_value"] == "IN"
        assert result["new_truth_value"] == "IN"
        assert len(net.nodes["b"].justifications) == 2

    def test_added_justification_keeps_node_in_after_retract(self):
        """Node with two justifications stays IN when one antecedent is retracted."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("c", "Premise C")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )

        net.add_justification("b", Justification(type="SL", antecedents=["c"]))
        net.retract("a")

        assert net.nodes["b"].truth_value == "IN"

    def test_add_justification_restores_out_node(self):
        """Adding a valid justification to an OUT node makes it IN."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("c", "Premise C")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.retract("a")
        assert net.nodes["b"].truth_value == "OUT"

        result = net.add_justification(
            "b", Justification(type="SL", antecedents=["c"])
        )
        assert result["old_truth_value"] == "OUT"
        assert result["new_truth_value"] == "IN"
        assert "b" in result["changed"]

    def test_add_justification_cascades(self):
        """Adding a justification that restores a node cascades to dependents."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("c", "Premise C")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.add_node(
            "d", "Depends on B",
            justifications=[Justification(type="SL", antecedents=["b"])],
        )
        net.retract("a")
        assert net.nodes["d"].truth_value == "OUT"

        result = net.add_justification(
            "b", Justification(type="SL", antecedents=["c"])
        )
        assert net.nodes["b"].truth_value == "IN"
        assert net.nodes["d"].truth_value == "IN"
        assert "d" in result["changed"]

    def test_add_justification_with_outlist(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("enemy", "Enemy")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.retract("a")
        assert net.nodes["b"].truth_value == "OUT"

        net.add_justification(
            "b", Justification(type="SL", antecedents=[], outlist=["enemy"])
        )
        # enemy is IN, so this justification is invalid — b stays OUT
        assert net.nodes["b"].truth_value == "OUT"

        net.retract("enemy")
        # Now enemy is OUT, outlist satisfied — b should be IN
        assert net.nodes["b"].truth_value == "IN"

    def test_add_justification_to_premise(self):
        """Adding a justification to a premise gives it a backup route."""
        net = Network()
        net.add_node("p", "Premise")
        net.add_node("c", "Support C")

        net.add_justification(
            "p", Justification(type="SL", antecedents=["c"])
        )
        assert len(net.nodes["p"].justifications) == 1
        assert net.nodes["p"].truth_value == "IN"

    def test_add_justification_nonexistent_node(self):
        net = Network()
        with pytest.raises(KeyError, match="not found"):
            net.add_justification(
                "ghost", Justification(type="SL", antecedents=[])
            )

    def test_add_justification_registers_dependents(self):
        """Antecedent and outlist nodes get the target registered as dependent."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("e", "Enemy")
        net.add_node("b", "Premise B")

        net.add_justification(
            "b", Justification(type="SL", antecedents=["a"], outlist=["e"])
        )
        assert "b" in net.nodes["a"].dependents
        assert "b" in net.nodes["e"].dependents


class TestAPIAddJustification:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "reasons.db")
        api.init_db(db_path=db_path)
        return db_path

    def test_add_sl_justification(self, db):
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("c", "Premise C", db_path=db)
        api.add_node("b", "Derived B", sl="a", db_path=db)

        result = api.add_justification("b", sl="c", db_path=db)
        assert result["node_id"] == "b"
        assert result["new_truth_value"] == "IN"

        node = api.show_node("b", db_path=db)
        assert len(node["justifications"]) == 2

    def test_add_cp_justification(self, db):
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Derived B", sl="a", db_path=db)

        result = api.add_justification("b", cp="a", db_path=db)
        assert result["new_truth_value"] == "IN"

        node = api.show_node("b", db_path=db)
        assert len(node["justifications"]) == 2
        assert node["justifications"][1]["type"] == "CP"

    def test_add_unless_only_justification(self, db):
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("enemy", "Enemy", db_path=db)
        api.add_node("b", "Derived B", sl="a", db_path=db)

        result = api.add_justification("b", unless="enemy", db_path=db)
        assert result["new_truth_value"] == "IN"

    def test_add_justification_no_args_raises(self, db):
        api.add_node("a", "Premise A", db_path=db)
        with pytest.raises(ValueError, match="Must provide"):
            api.add_justification("a", db_path=db)

    def test_add_justification_with_namespace(self, db):
        api.add_node("a", "Premise A", namespace="ns", db_path=db)
        api.add_node("c", "Support C", namespace="ns", db_path=db)
        api.add_node("b", "Derived B", sl="a", namespace="ns", db_path=db)

        result = api.add_justification("b", sl="c", namespace="ns", db_path=db)
        assert result["node_id"] == "ns:b"

    def test_add_justification_restores_out_node(self, db):
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("c", "Premise C", db_path=db)
        api.add_node("b", "Derived B", sl="a", db_path=db)
        api.retract_node("a", db_path=db)

        node = api.show_node("b", db_path=db)
        assert node["truth_value"] == "OUT"

        result = api.add_justification("b", sl="c", db_path=db)
        assert result["old_truth_value"] == "OUT"
        assert result["new_truth_value"] == "IN"

    def test_add_justification_persists(self, db):
        """Justification survives save/load cycle."""
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("c", "Premise C", db_path=db)
        api.add_node("b", "Derived B", sl="a", db_path=db)
        api.add_justification("b", sl="c", db_path=db)

        node = api.show_node("b", db_path=db)
        assert len(node["justifications"]) == 2

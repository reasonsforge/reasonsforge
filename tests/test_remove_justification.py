"""Tests for remove-justification: removing individual justifications by index."""

import pytest

from reasonsforge import Justification, api
from reasonsforge.network import Network


class TestNetworkRemoveJustification:
    def test_remove_one_of_two(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node(
            "c", "Derived C",
            justifications=[
                Justification(type="SL", antecedents=["a"], label="first"),
                Justification(type="SL", antecedents=["b"], label="second"),
            ],
        )
        result = net.remove_justification("c", 0)
        assert result["removed"]["label"] == "first"
        assert result["remaining"] == 1
        assert len(net.nodes["c"].justifications) == 1
        assert net.nodes["c"].justifications[0].label == "second"

    def test_remove_causes_out(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.retract("b")
        net.add_node(
            "c", "Derived C",
            justifications=[
                Justification(type="SL", antecedents=["a"]),
                Justification(type="SL", antecedents=["b"]),
            ],
        )
        assert net.nodes["c"].truth_value == "IN"
        result = net.remove_justification("c", 0)
        assert result["old_truth_value"] == "IN"
        assert result["new_truth_value"] == "OUT"

    def test_remove_propagates_cascade(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.retract("b")
        net.add_node(
            "c", "Derived C",
            justifications=[
                Justification(type="SL", antecedents=["a"]),
                Justification(type="SL", antecedents=["b"]),
            ],
        )
        net.add_node(
            "d", "Derived D",
            justifications=[Justification(type="SL", antecedents=["c"])],
        )
        assert net.nodes["d"].truth_value == "IN"
        result = net.remove_justification("c", 0)
        assert "d" in result["changed"]
        assert net.nodes["d"].truth_value == "OUT"

    def test_remove_cleans_dependents(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node(
            "c", "Derived C",
            justifications=[
                Justification(type="SL", antecedents=["a"]),
                Justification(type="SL", antecedents=["b"]),
            ],
        )
        assert "c" in net.nodes["a"].dependents
        assert "c" in net.nodes["b"].dependents
        net.remove_justification("c", 0)
        assert "c" not in net.nodes["a"].dependents
        assert "c" in net.nodes["b"].dependents

    def test_remove_keeps_shared_dependent(self):
        """If a node appears in multiple justifications, don't remove from dependents."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node(
            "c", "Derived C",
            justifications=[
                Justification(type="SL", antecedents=["a", "b"]),
                Justification(type="SL", antecedents=["a"]),
            ],
        )
        net.remove_justification("c", 0)
        assert "c" in net.nodes["a"].dependents

    def test_remove_outlist_cleans_dependents(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("blocker", "Blocker")
        net.retract("blocker")
        net.add_node(
            "c", "Gated C",
            justifications=[
                Justification(type="SL", antecedents=["a"], outlist=["blocker"]),
                Justification(type="SL", antecedents=["a"]),
            ],
        )
        assert "c" in net.nodes["blocker"].dependents
        net.remove_justification("c", 0)
        assert "c" not in net.nodes["blocker"].dependents

    def test_error_on_premise(self):
        net = Network()
        net.add_node("a", "Premise A")
        with pytest.raises(ValueError, match="premise"):
            net.remove_justification("a", 0)

    def test_error_on_single_justification(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        with pytest.raises(ValueError, match="only one justification"):
            net.remove_justification("b", 0)

    def test_error_on_bad_index(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node(
            "c", "Derived C",
            justifications=[
                Justification(type="SL", antecedents=["a"]),
                Justification(type="SL", antecedents=["b"]),
            ],
        )
        with pytest.raises(IndexError, match="out of range"):
            net.remove_justification("c", 5)

    def test_error_on_missing_node(self):
        net = Network()
        with pytest.raises(KeyError, match="not found"):
            net.remove_justification("nonexistent", 0)


class TestApiRemoveJustification:
    def test_api_remove_justification(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Premise B", db_path=db)
        api.add_node("c", "Derived C", sl="a", db_path=db)
        api.add_justification("c", sl="b", db_path=db)

        node = api.show_node("c", db_path=db)
        assert len(node["justifications"]) == 2

        result = api.remove_justification("c", 0, db_path=db)
        assert result["remaining"] == 1

        node = api.show_node("c", db_path=db)
        assert len(node["justifications"]) == 1


class TestCliRemoveJustification:
    def test_cli_remove_justification(self, tmp_path):
        import subprocess
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Premise B", db_path=db)
        api.add_node("c", "Derived C", sl="a", db_path=db)
        api.add_justification("c", sl="b", label="keep-this", db_path=db)

        result = subprocess.run(
            ["uv", "run", "reasons", "--db", db, "remove-justification", "c", "0"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Removed justification 0" in result.stdout

        node = api.show_node("c", db_path=db)
        assert len(node["justifications"]) == 1
        assert node["justifications"][0]["label"] == "keep-this"

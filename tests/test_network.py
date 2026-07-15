"""Tests for the dependency network and propagation algorithm."""

import pytest

from reasonsforge import Node, Justification, Nogood
from reasonsforge.network import Network


class TestAddNode:
    """Adding nodes to the network."""

    def test_add_premise(self):
        net = Network()
        node = net.add_node("a", "Premise A")
        assert node.truth_value == "IN"
        assert node.justifications == []

    def test_add_derived_node(self):
        net = Network()
        net.add_node("a", "Premise A")
        node = net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        assert node.truth_value == "IN"

    def test_add_derived_node_antecedent_out(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.retract("a")
        node = net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        assert node.truth_value == "OUT"

    def test_add_duplicate_raises(self):
        net = Network()
        net.add_node("a", "Premise A")
        with pytest.raises(ValueError, match="already exists"):
            net.add_node("a", "Duplicate")

    def test_dependents_registered(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        assert "b" in net.nodes["a"].dependents


class TestRetraction:
    """Retraction cascades — Doyle's core mechanism."""

    def test_retract_premise(self):
        net = Network()
        net.add_node("a", "Premise A")
        changed = net.retract("a")
        assert net.nodes["a"].truth_value == "OUT"
        assert "a" in changed

    def test_retract_cascades_to_dependent(self):
        """A → B: retract A → B goes OUT."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        changed = net.retract("a")
        assert net.nodes["b"].truth_value == "OUT"
        assert "b" in changed

    def test_retract_cascades_through_chain(self):
        """A → B → C: retract A → both B and C go OUT."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.add_node(
            "c", "Derived C",
            justifications=[Justification(type="SL", antecedents=["b"])],
        )
        changed = net.retract("a")
        assert net.nodes["b"].truth_value == "OUT"
        assert net.nodes["c"].truth_value == "OUT"
        assert set(changed) == {"a", "b", "c"}

    def test_retract_already_out_is_noop(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.retract("a")
        changed = net.retract("a")
        assert changed == []

    def test_retract_nonexistent_raises(self):
        net = Network()
        with pytest.raises(KeyError, match="not found"):
            net.retract("missing")

    def test_retract_does_not_cascade_with_alternate_justification(self):
        """B has two justifications: SL(A) and SL(C). Retracting A leaves B IN via C."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("c", "Premise C")
        net.add_node(
            "b", "Derived B",
            justifications=[
                Justification(type="SL", antecedents=["a"]),
                Justification(type="SL", antecedents=["c"]),
            ],
        )
        changed = net.retract("a")
        assert net.nodes["b"].truth_value == "IN"
        assert "b" not in changed


    def test_retract_already_out_pins_retracted(self):
        """Retracting an already-OUT node should still set _retracted.

        If B is OUT because its antecedent A is OUT, explicitly retracting B
        should pin it so it doesn't resurrect when A comes back.
        """
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )

        net.retract("a")
        assert net.nodes["b"].truth_value == "OUT"

        # B is already OUT — retract it explicitly to pin it
        result = net.retract("b")
        assert result == []  # no state change
        assert net.nodes["b"].metadata.get("_retracted") is True

        # Restore A — C would come back but B should stay pinned
        net.assert_node("a")
        assert net.nodes["a"].truth_value == "IN"
        assert net.nodes["b"].truth_value == "OUT", "pinned node resurrected"

        # recompute_all should also respect the pin
        net.recompute_all()
        assert net.nodes["b"].truth_value == "OUT", "recompute resurrected pinned node"


class TestRestoration:
    """Restoration — re-asserting a node restores dependents."""

    def test_assert_restores_dependent(self):
        """A → B: retract A → assert A → B is IN again."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.retract("a")
        assert net.nodes["b"].truth_value == "OUT"

        changed = net.assert_node("a")
        assert net.nodes["a"].truth_value == "IN"
        assert net.nodes["b"].truth_value == "IN"
        assert set(changed) == {"a", "b"}

    def test_assert_restores_chain(self):
        """A → B → C: retract A → assert A → all restored."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.add_node(
            "c", "Derived C",
            justifications=[Justification(type="SL", antecedents=["b"])],
        )
        net.retract("a")
        assert net.nodes["c"].truth_value == "OUT"

        changed = net.assert_node("a")
        assert net.nodes["b"].truth_value == "IN"
        assert net.nodes["c"].truth_value == "IN"

    def test_assert_already_in_is_noop(self):
        net = Network()
        net.add_node("a", "Premise A")
        changed = net.assert_node("a")
        assert changed == []

    def test_assert_nonexistent_raises(self):
        net = Network()
        with pytest.raises(KeyError, match="not found"):
            net.assert_node("missing")


class TestMultipleAntecedents:
    """Nodes depending on multiple antecedents (conjunction)."""

    def test_sl_requires_all_antecedents(self):
        """D depends on SL(A, B, C) — all must be IN."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node("c", "Premise C")
        net.add_node(
            "d", "Derived D",
            justifications=[Justification(type="SL", antecedents=["a", "b", "c"])],
        )
        assert net.nodes["d"].truth_value == "IN"

        net.retract("b")
        assert net.nodes["d"].truth_value == "OUT"

        net.assert_node("b")
        assert net.nodes["d"].truth_value == "IN"


class TestNogood:
    """Contradiction detection and resolution."""

    def test_nogood_retracts_one(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("d", "Premise D")
        changed = net.add_nogood(["a", "d"])
        # One of them should be OUT
        values = {net.nodes["a"].truth_value, net.nodes["d"].truth_value}
        assert "OUT" in values
        assert len(net.nogoods) == 1

    def test_nogood_recorded(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("d", "Premise D")
        net.add_nogood(["a", "d"])
        assert net.nogoods[0].nodes == ["a", "d"]
        assert net.nogoods[0].id == "nogood-001"

    def test_nogood_inactive_when_one_already_out(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("d", "Premise D")
        net.retract("d")
        changed = net.add_nogood(["a", "d"])
        assert changed == []  # no retraction needed
        assert net.nodes["a"].truth_value == "IN"

    def test_nogood_nonexistent_raises(self):
        net = Network()
        net.add_node("a", "Premise A")
        with pytest.raises(KeyError, match="not found"):
            net.add_nogood(["a", "missing"])


class TestExplain:
    """Explain traces — why is a node IN or OUT."""

    def test_explain_premise_in(self):
        net = Network()
        net.add_node("a", "Premise A")
        steps = net.explain("a")
        assert len(steps) == 1
        assert steps[0]["reason"] == "premise"
        assert steps[0]["truth_value"] == "IN"

    def test_explain_premise_retracted(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.retract("a")
        steps = net.explain("a")
        assert steps[0]["reason"] == "retracted premise"

    def test_explain_derived_in(self):
        """B depends on A — explain B should trace back to A."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"], label="A supports B")],
        )
        steps = net.explain("b")
        assert steps[0]["node"] == "b"
        assert steps[0]["truth_value"] == "IN"
        assert steps[0]["label"] == "A supports B"
        # Should include A in the trace
        assert any(s["node"] == "a" for s in steps)

    def test_explain_chain(self):
        """A → B → C: explain C traces through B to A."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.add_node(
            "c", "Derived C",
            justifications=[Justification(type="SL", antecedents=["b"])],
        )
        steps = net.explain("c")
        nodes_in_trace = [s["node"] for s in steps]
        assert "c" in nodes_in_trace
        assert "b" in nodes_in_trace
        assert "a" in nodes_in_trace

    def test_explain_derived_out(self):
        """B depends on A, A retracted — explain B shows failed antecedent."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.retract("a")
        steps = net.explain("b")
        assert steps[0]["truth_value"] == "OUT"
        assert "a" in steps[0]["failed_antecedents"]

    def test_explain_nonexistent_raises(self):
        net = Network()
        with pytest.raises(KeyError, match="not found"):
            net.explain("missing")


class TestBeliefSet:
    """get_belief_set returns all IN nodes."""

    def test_belief_set(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node(
            "c", "Derived C",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        assert set(net.get_belief_set()) == {"a", "b", "c"}

        net.retract("a")
        assert set(net.get_belief_set()) == {"b"}


class TestLog:
    """Propagation audit trail."""

    def test_log_records_add(self):
        net = Network()
        net.add_node("a", "Premise A")
        assert any(e["action"] == "add" and e["target"] == "a" for e in net.log)

    def test_log_records_retract_and_propagate(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.retract("a")
        actions = [(e["action"], e["target"]) for e in net.log]
        assert ("retract", "a") in actions
        assert ("propagate", "b") in actions


class TestDiamondDependency:
    """Diamond pattern: A → B, A → C, B+C → D."""

    def test_diamond_retract_and_restore(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.add_node(
            "c", "Derived C",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.add_node(
            "d", "Derived D",
            justifications=[Justification(type="SL", antecedents=["b", "c"])],
        )
        assert net.nodes["d"].truth_value == "IN"

        net.retract("a")
        assert net.nodes["b"].truth_value == "OUT"
        assert net.nodes["c"].truth_value == "OUT"
        assert net.nodes["d"].truth_value == "OUT"

        net.assert_node("a")
        assert net.nodes["b"].truth_value == "IN"
        assert net.nodes["c"].truth_value == "IN"
        assert net.nodes["d"].truth_value == "IN"

    def test_diamond_with_retracted_intermediate(self):
        """A → B, A → C, B+C → D. Retract B explicitly, then change A.

        B is retracted (sticky) so propagation skips it. D should stay OUT
        because B remains OUT. C should still respond to A's changes.
        """
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node(
            "b", "Derived B",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.add_node(
            "c", "Derived C",
            justifications=[Justification(type="SL", antecedents=["a"])],
        )
        net.add_node(
            "d", "Derived D",
            justifications=[Justification(type="SL", antecedents=["b", "c"])],
        )

        # Explicitly retract B — sticky
        net.retract("b")
        assert net.nodes["b"].truth_value == "OUT"
        assert net.nodes["d"].truth_value == "OUT"  # B is OUT → D invalid

        # Retract and restore A — C follows, B stays retracted, D stays OUT
        net.retract("a")
        assert net.nodes["c"].truth_value == "OUT"

        net.assert_node("a")
        assert net.nodes["c"].truth_value == "IN"
        assert net.nodes["b"].truth_value == "OUT", "retracted B should not resurrect"
        assert net.nodes["d"].truth_value == "OUT", "D needs B which is retracted"

        # recompute_all should not resurrect B either
        net.recompute_all()
        assert net.nodes["b"].truth_value == "OUT", "recompute resurrected retracted B"
        assert net.nodes["d"].truth_value == "OUT", "recompute resurrected D via B"

        # Explicitly asserting B clears _retracted — D should come back
        net.assert_node("b")
        assert net.nodes["b"].truth_value == "IN"
        assert net.nodes["d"].truth_value == "IN"


class TestDanglingDependents:
    """Propagation handles dangling dependent references gracefully."""

    def test_propagate_skips_dangling_dependent(self):
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.add("nonexistent")
        net.retract("a")  # triggers _propagate — should not crash

    def test_propagate_logs_dangling_warning(self):
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.add("ghost")
        net.retract("a")
        warnings = [e for e in net.log if e["action"] == "warn"]
        assert len(warnings) == 1
        assert warnings[0]["target"] == "ghost"
        assert "dangling" in warnings[0]["value"]

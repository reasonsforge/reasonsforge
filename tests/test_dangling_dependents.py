"""Tests for dangling dependent guard in _propagate (issue #22).

Validates that propagation skips dangling dependent references with a warning
instead of crashing with KeyError.
"""

from reasonsforge import Justification, Node
from reasonsforge.network import Network


class TestDanglingDependentNoCrash:
    """Propagation must not crash when dependents reference nonexistent nodes."""

    def test_retract_with_single_dangling_dependent(self):
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.add("nonexistent")
        net.retract("a")

    def test_assert_with_single_dangling_dependent(self):
        net = Network()
        net.add_node("a", "premise A")
        net.retract("a")
        net.nodes["a"].dependents.add("nonexistent")
        net.assert_node("a")

    def test_multiple_dangling_refs_on_one_node(self):
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.update({"ghost1", "ghost2", "ghost3"})
        net.retract("a")

    def test_dangling_ref_alongside_valid_dependent(self):
        """Valid dependents still propagate even when a dangling ref is present."""
        net = Network()
        net.add_node("a", "premise A")
        net.add_node("b", "derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        assert net.nodes["b"].truth_value == "IN"
        net.nodes["a"].dependents.add("ghost")
        net.retract("a")
        assert net.nodes["b"].truth_value == "OUT"


class TestDanglingDependentWarningLogs:
    """Warning logs must be emitted for dangling dependents."""

    def test_single_warning_logged(self):
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.add("ghost")
        net.retract("a")
        warnings = [e for e in net.log if e["action"] == "warn"]
        assert len(warnings) == 1
        assert warnings[0]["target"] == "ghost"
        assert "dangling" in warnings[0]["value"]

    def test_warning_includes_parent_node_id(self):
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.add("ghost")
        net.retract("a")
        warnings = [e for e in net.log if e["action"] == "warn"]
        assert "a" in warnings[0]["value"], "warning should identify the parent node"

    def test_multiple_dangling_produce_multiple_warnings(self):
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.update({"ghost1", "ghost2"})
        net.retract("a")
        warnings = [e for e in net.log if e["action"] == "warn"]
        assert len(warnings) == 2
        targets = {w["target"] for w in warnings}
        assert targets == {"ghost1", "ghost2"}

    def test_warning_on_assert_path(self):
        """assert_node also triggers _propagate — verify warning fires."""
        net = Network()
        net.add_node("a", "premise A")
        net.retract("a")
        net.nodes["a"].dependents.add("ghost")
        net.log.clear()
        net.assert_node("a")
        warnings = [e for e in net.log if e["action"] == "warn"]
        assert len(warnings) == 1
        assert warnings[0]["target"] == "ghost"

    def test_warning_has_timestamp(self):
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.add("ghost")
        net.retract("a")
        warnings = [e for e in net.log if e["action"] == "warn"]
        assert "timestamp" in warnings[0]


class TestDanglingDependentPropagationContinues:
    """Propagation must complete correctly past dangling refs."""

    def test_valid_dependent_still_retracted(self):
        """Dangling ref before valid dependent doesn't block cascade."""
        net = Network()
        net.add_node("a", "premise A")
        net.add_node("b", "derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.nodes["a"].dependents.add("ghost")
        changed = net.retract("a")
        assert "b" in changed
        assert net.nodes["b"].truth_value == "OUT"

    def test_valid_dependent_still_restored(self):
        """Dangling ref doesn't block restoration via assert_node."""
        net = Network()
        net.add_node("a", "premise A")
        net.add_node("b", "derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.retract("a")
        assert net.nodes["b"].truth_value == "OUT"
        net.nodes["a"].dependents.add("ghost")
        changed = net.assert_node("a")
        assert "b" in changed
        assert net.nodes["b"].truth_value == "IN"

    def test_chain_with_dangling_in_middle(self):
        """A -> B -> C chain where B has a dangling ref. C still updates."""
        net = Network()
        net.add_node("a", "premise A")
        net.add_node("b", "derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.add_node("c", "derived C", justifications=[
            Justification(type="SL", antecedents=["b"]),
        ])
        net.nodes["b"].dependents.add("ghost")
        changed = net.retract("a")
        assert "b" in changed
        assert "c" in changed
        assert net.nodes["c"].truth_value == "OUT"

    def test_diamond_with_dangling(self):
        """Diamond: A -> B, A -> C, B&C -> D. Dangling ref on A."""
        net = Network()
        net.add_node("a", "premise A")
        net.add_node("b", "derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.add_node("c", "derived C", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.add_node("d", "derived D", justifications=[
            Justification(type="SL", antecedents=["b", "c"]),
        ])
        net.nodes["a"].dependents.add("ghost")
        changed = net.retract("a")
        assert net.nodes["d"].truth_value == "OUT"
        assert "ghost" not in changed


class TestDanglingDependentEdgeCases:
    """Edge cases around dangling dependent handling."""

    def test_same_dangling_in_multiple_nodes(self):
        """Same ghost ID in two different nodes' dependents — warns for each."""
        net = Network()
        net.add_node("a", "premise A")
        net.add_node("b", "derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.nodes["a"].dependents.add("ghost")
        net.nodes["b"].dependents.add("ghost")
        changed = net.retract("a")
        warnings = [e for e in net.log if e["action"] == "warn"]
        ghost_warnings = [w for w in warnings if w["target"] == "ghost"]
        assert len(ghost_warnings) >= 1

    def test_dangling_not_in_changed_list(self):
        """Dangling dependents must not appear in the returned changed list."""
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.add("ghost")
        changed = net.retract("a")
        assert "ghost" not in changed

    def test_challenge_with_dangling(self):
        """challenge() path triggers _propagate — should handle dangling."""
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.add("ghost")
        net.challenge("a", "test challenge")
        warnings = [e for e in net.log if e["action"] == "warn"]
        ghost_warnings = [w for w in warnings if w["target"] == "ghost"]
        assert len(ghost_warnings) >= 1

    def test_add_justification_with_dangling(self):
        """add_justification() can trigger _propagate — handle dangling."""
        net = Network()
        net.add_node("a", "premise A")
        net.add_node("b", "premise B")
        net.nodes["b"].dependents.add("ghost")
        net.add_justification("b", Justification(type="SL", antecedents=["a"]))
        net.retract("a")
        warnings = [e for e in net.log if e["action"] == "warn"]
        ghost_warnings = [w for w in warnings if w["target"] == "ghost"]
        assert len(ghost_warnings) >= 1

    def test_dangling_does_not_enter_visited(self):
        """A dangling ID should not be added to the visited set, so if
        it later becomes a real node, it wouldn't be incorrectly skipped."""
        net = Network()
        net.add_node("a", "premise A")
        net.nodes["a"].dependents.add("ghost")
        net.retract("a")
        # Now add the previously-dangling node and verify it works normally
        net.add_node("ghost", "now real", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        assert net.nodes["ghost"].truth_value == "OUT"  # a is OUT
        net.assert_node("a")
        assert net.nodes["ghost"].truth_value == "IN"

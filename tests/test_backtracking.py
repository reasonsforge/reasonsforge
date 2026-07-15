"""Tests for dependency-directed backtracking.

When a contradiction (nogood) is discovered, the system traces backward
through the justification graph to find the premises responsible, then
retracts the premise with minimal disruption instead of an arbitrary node.
"""

import pytest

from reasonsforge import Justification
from reasonsforge.network import Network
from reasonsforge import api


class TestTraceAssumptions:
    """trace_assumptions walks backward to find premises."""

    def test_premise_traces_to_itself(self):
        net = Network()
        net.add_node("a", "Premise A")
        result = net.trace_assumptions("a")
        assert result == ["a"]

    def test_derived_traces_to_premise(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        result = net.trace_assumptions("b")
        assert result == ["a"]

    def test_chain_traces_to_root(self):
        """A → B → C traces back to A."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.add_node("c", "Derived C", justifications=[
            Justification(type="SL", antecedents=["b"]),
        ])
        result = net.trace_assumptions("c")
        assert result == ["a"]

    def test_diamond_traces_to_root(self):
        """A → B, A → C, B+C → D — all trace back to A."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.add_node("c", "Derived C", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.add_node("d", "Derived D", justifications=[
            Justification(type="SL", antecedents=["b", "c"]),
        ])
        result = net.trace_assumptions("d")
        assert result == ["a"]

    def test_multiple_premises(self):
        """A, B → C — traces to both A and B."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node("c", "Derived C", justifications=[
            Justification(type="SL", antecedents=["a", "b"]),
        ])
        result = net.trace_assumptions("c")
        assert set(result) == {"a", "b"}

    def test_deep_chain_multiple_roots(self):
        """A → B → D, C → D — traces to A and C."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("c", "Premise C")
        net.add_node("b", "Derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.add_node("d", "Derived D", justifications=[
            Justification(type="SL", antecedents=["b", "c"]),
        ])
        result = net.trace_assumptions("d")
        assert set(result) == {"a", "c"}

    def test_nonexistent_raises(self):
        net = Network()
        with pytest.raises(KeyError):
            net.trace_assumptions("missing")


class TestFindCulprits:
    """find_culprits identifies premises responsible for a contradiction."""

    def test_two_premises_contradicting(self):
        """A and B contradict — both are culprits."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        culprits = net.find_culprits(["a", "b"])
        premise_ids = [c["premise"] for c in culprits]
        assert set(premise_ids) == {"a", "b"}

    def test_derived_nodes_trace_to_premise(self):
        """A → C, B contradicts C — culprit is A (premise behind C), not C itself."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node("c", "Derived C", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        culprits = net.find_culprits(["b", "c"])
        premise_ids = [c["premise"] for c in culprits]
        # Both a and b are culprits — a supports c, b is itself a premise
        assert set(premise_ids) == {"a", "b"}

    def test_sorted_by_dependents(self):
        """Culprit with fewer dependents comes first."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        # Give A more dependents than B
        net.add_node("x", "X", justifications=[Justification(type="SL", antecedents=["a"])])
        net.add_node("y", "Y", justifications=[Justification(type="SL", antecedents=["a"])])
        culprits = net.find_culprits(["a", "b"])
        # B has fewer dependents, should be first
        assert culprits[0]["premise"] == "b"

    def test_would_resolve_field(self):
        """Culprit reports which nogood nodes it would resolve."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node("c", "Derived C", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        culprits = net.find_culprits(["b", "c"])
        a_culprit = [c for c in culprits if c["premise"] == "a"][0]
        assert "c" in a_culprit["would_resolve"]


class TestBacktrackingInNogood:
    """add_nogood uses backtracking to retract a premise, not an arbitrary node."""

    def test_retracts_premise_not_derived(self):
        """A → C, B. Nogood(B, C). Should retract a premise, not C."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node("c", "Derived C", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        changed = net.add_nogood(["b", "c"])
        # Should have retracted a premise (a or b), not c
        # B has fewer dependents than A, so B is the victim
        assert net.nodes["b"].truth_value == "OUT"
        # C stays IN because A still supports it
        assert net.nodes["c"].truth_value == "IN"

    def test_cascade_from_retracted_premise(self):
        """A → B → C. Nogood(C, D). Retracts A → B and C cascade OUT."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("d", "Premise D")
        net.add_node("b", "Derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.add_node("c", "Derived C", justifications=[
            Justification(type="SL", antecedents=["b"]),
        ])
        # D has 0 dependents, A has 1 (b). D should be retracted.
        changed = net.add_nogood(["c", "d"])
        assert net.nodes["d"].truth_value == "OUT"
        # C and B stay IN
        assert net.nodes["c"].truth_value == "IN"

    def test_backtracking_logs_culprit(self):
        """The log should record which premise was backtracked to."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_nogood(["a", "b"])
        backtrack_entries = [e for e in net.log if e["action"] == "backtrack"]
        assert len(backtrack_entries) == 1
        assert "culprit" in backtrack_entries[0]["value"]

    def test_shared_premise_resolves_both(self):
        """A → B, A → C. Nogood(B, C). Retracting A resolves both."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.add_node("c", "Derived C", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        changed = net.add_nogood(["b", "c"])
        # A is the only premise — retracting it resolves both
        assert net.nodes["a"].truth_value == "OUT"
        assert net.nodes["b"].truth_value == "OUT"
        assert net.nodes["c"].truth_value == "OUT"

    def test_inactive_nogood_no_backtracking(self):
        """If one node is already OUT, no backtracking needed."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.retract("b")
        changed = net.add_nogood(["a", "b"])
        assert changed == []
        # No backtrack log entries
        backtrack_entries = [e for e in net.log if e["action"] == "backtrack"]
        assert len(backtrack_entries) == 0


class TestEntrenchment:
    """Entrenchment scoring prefers retracting speculation over evidence."""

    def test_evidence_premise_more_entrenched_than_derived(self):
        """Evidence (premise) should be protected over speculation (derived)."""
        net = Network()
        net.add_node("evidence", "Finite trees have gap closing ~1/N",
                      source="physics-quantum-lattice:results.md")
        net.add_node("basis", "W is a Bethe lattice")
        net.add_node("speculation", "Tree is the genealogy",
                      justifications=[Justification(type="SL", antecedents=["basis"])])

        # Evidence is more entrenched (premise + has source)
        assert net._entrenchment("evidence") > net._entrenchment("basis")

    def test_nogood_retracts_speculation_not_evidence(self):
        """The physics-speculation bug: evidence should survive, speculation should die."""
        net = Network()
        # Evidence: a premise with source (simulation result)
        net.add_node("finite-tree-gap-closes",
                      "Finite trees have spectral gap closing ~1/N",
                      source="physics-quantum-lattice:entries/2026/02/22/combined-laplacian-results.md")
        # Speculation: derived from another premise
        net.add_node("w-is-bethe-lattice", "W dimension is a Bethe lattice")
        net.add_node("tree-as-genealogy",
                      "Picture A: the tree is the duplication genealogy",
                      justifications=[Justification(type="SL", antecedents=["w-is-bethe-lattice"])])

        # These two contradict each other
        changed = net.add_nogood(["tree-as-genealogy", "finite-tree-gap-closes"])

        # Evidence should survive — speculation's premise should be retracted
        assert net.nodes["finite-tree-gap-closes"].truth_value == "IN"
        # The speculation or its premise should be OUT
        assert (net.nodes["tree-as-genealogy"].truth_value == "OUT" or
                net.nodes["w-is-bethe-lattice"].truth_value == "OUT")

    def test_sourced_premise_over_unsourced(self):
        """Premise with source is more entrenched than premise without."""
        net = Network()
        net.add_node("sourced", "Has a source", source="repo/file.md")
        net.add_node("unsourced", "No source")
        assert net._entrenchment("sourced") > net._entrenchment("unsourced")

    def test_beliefs_type_affects_entrenchment(self):
        """OBSERVATION > DERIVED in entrenchment."""
        net = Network()
        net.add_node("obs", "An observation", metadata={"beliefs_type": "OBSERVATION"})
        net.add_node("der", "A derivation", metadata={"beliefs_type": "DERIVED"})
        assert net._entrenchment("obs") > net._entrenchment("der")

    def test_more_dependents_more_entrenched(self):
        """Nodes with more dependents are more entrenched."""
        net = Network()
        net.add_node("popular", "Many depend on this")
        net.add_node("lonely", "Nothing depends on this")
        net.add_node("d1", "Dep 1", justifications=[Justification(type="SL", antecedents=["popular"])])
        net.add_node("d2", "Dep 2", justifications=[Justification(type="SL", antecedents=["popular"])])
        assert net._entrenchment("popular") > net._entrenchment("lonely")


class TestBacktrackingAPI:
    """API layer for trace and backtracking."""

    def test_trace_api(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Derived B", sl="a", db_path=db)
        api.add_node("c", "Derived C", sl="b", db_path=db)

        result = api.trace_assumptions("c", db_path=db)
        assert result["node_id"] == "c"
        assert result["premises"] == ["a"]

    def test_find_culprits_api(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Premise B", db_path=db)

        result = api.find_culprits(["a", "b"], db_path=db)
        premise_ids = [c["premise"] for c in result["culprits"]]
        assert set(premise_ids) == {"a", "b"}

    def test_nogood_reports_backtrack(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Premise B", db_path=db)

        result = api.add_nogood(["a", "b"], db_path=db)
        assert result["backtracked_to"] is not None

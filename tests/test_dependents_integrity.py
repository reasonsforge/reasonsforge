"""Tests for dependents index integrity — _rebuild_dependents() and verify_dependents().

Validates that the dependents reverse index stays consistent across every
mutation path in the Network, that corruption is detected, and that
_rebuild_dependents() repairs it.
"""

import tempfile
from pathlib import Path

import pytest

from reasonsforge import Justification
from reasonsforge.network import Network
from reasonsforge.storage import Storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_clean(net: Network):
    """Assert the dependents index has no inconsistencies."""
    errors = net.verify_dependents()
    assert errors == [], f"Dependents index inconsistent: {errors}"


# ---------------------------------------------------------------------------
# Core: _rebuild_dependents and verify_dependents
# ---------------------------------------------------------------------------

class TestRebuildDependents:
    """_rebuild_dependents() is the canonical rebuild."""

    def test_empty_network(self):
        net = Network()
        net._rebuild_dependents()
        assert_clean(net)

    def test_premises_only(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net._rebuild_dependents()
        assert net.nodes["a"].dependents == set()
        assert net.nodes["b"].dependents == set()
        assert_clean(net)

    def test_single_dependency(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        net._rebuild_dependents()
        assert net.nodes["a"].dependents == {"b"}
        assert net.nodes["b"].dependents == set()
        assert_clean(net)

    def test_outlist_dependency(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Unless A",
                     justifications=[Justification(type="SL", antecedents=[], outlist=["a"])])
        net._rebuild_dependents()
        assert "b" in net.nodes["a"].dependents
        assert_clean(net)

    def test_both_antecedent_and_outlist(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node("c", "Derived C",
                     justifications=[Justification(type="SL", antecedents=["a"], outlist=["b"])])
        net._rebuild_dependents()
        assert "c" in net.nodes["a"].dependents
        assert "c" in net.nodes["b"].dependents
        assert_clean(net)

    def test_rebuild_clears_stale_entries(self):
        """_rebuild_dependents clears everything before rebuilding."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        # Inject fake dependent
        net.nodes["a"].dependents.add("phantom")
        errors_before = net.verify_dependents()
        assert len(errors_before) > 0

        net._rebuild_dependents()
        assert "phantom" not in net.nodes["a"].dependents
        assert_clean(net)


class TestVerifyDependents:
    """verify_dependents() detects inconsistencies without fixing them."""

    def test_clean_network_returns_empty(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        assert net.verify_dependents() == []

    def test_detects_extra_dependent(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.nodes["a"].dependents.add("ghost")
        errors = net.verify_dependents()
        assert any("extra" in e and "ghost" in e for e in errors)

    def test_detects_missing_dependent(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        net.nodes["a"].dependents.discard("b")
        errors = net.verify_dependents()
        assert any("missing" in e and "b" in e for e in errors)

    def test_does_not_modify_state(self):
        """verify is read-only — it should not change the dependents."""
        net = Network()
        net.add_node("a", "Premise A")
        net.nodes["a"].dependents.add("ghost")
        net.verify_dependents()
        assert "ghost" in net.nodes["a"].dependents


# ---------------------------------------------------------------------------
# Mutation path coverage
# ---------------------------------------------------------------------------

class TestMutationPathIntegrity:
    """Every mutation in Network must leave dependents consistent."""

    def test_after_add_node_with_justification(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node("c", "Derived C",
                     justifications=[Justification(type="SL", antecedents=["a", "b"])])
        assert_clean(net)

    def test_after_retract_and_restore(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        net.retract("a")
        assert_clean(net)
        net.assert_node("a")
        assert_clean(net)

    def test_after_add_justification(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node("c", "Derived C",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        net.add_justification("c", Justification(type="SL", antecedents=["b"]))
        assert "c" in net.nodes["b"].dependents
        assert_clean(net)

    def test_after_supersede(self):
        net = Network()
        net.add_node("old", "Old belief")
        net.add_node("new", "New belief")
        net.supersede("old", "new")
        assert "old" in net.nodes["new"].dependents
        assert_clean(net)

    def test_after_challenge(self):
        net = Network()
        net.add_node("target", "Target belief")
        result = net.challenge("target", "I disagree")
        cid = result["challenge_id"]
        assert "target" in net.nodes[cid].dependents
        assert_clean(net)

    def test_after_challenge_and_defend(self):
        net = Network()
        net.add_node("target", "Target belief")
        ch = net.challenge("target", "I disagree")
        net.defend("target", ch["challenge_id"], "Counterargument")
        assert_clean(net)

    def test_after_convert_to_premise(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        assert "b" in net.nodes["a"].dependents
        net.convert_to_premise("b")
        assert "b" not in net.nodes["a"].dependents
        assert_clean(net)

    def test_after_add_nogood(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_nogood(["a", "b"])
        assert_clean(net)

    def test_after_summarize(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.summarize("summary", "Summary of A and B", over=["a", "b"])
        assert "summary" in net.nodes["a"].dependents
        assert "summary" in net.nodes["b"].dependents
        assert_clean(net)


# ---------------------------------------------------------------------------
# Corruption detection and repair
# ---------------------------------------------------------------------------

class TestCorruptionAndRepair:
    """Detect corruption, then repair with _rebuild_dependents."""

    def test_detect_and_repair_extra(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        net.nodes["b"].dependents.add("a")  # bogus reverse entry
        assert len(net.verify_dependents()) > 0
        net._rebuild_dependents()
        assert_clean(net)

    def test_detect_and_repair_missing(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        net.nodes["a"].dependents.clear()
        assert len(net.verify_dependents()) > 0
        net._rebuild_dependents()
        assert_clean(net)

    def test_rebuild_idempotent(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        net._rebuild_dependents()
        snapshot1 = {nid: set(n.dependents) for nid, n in net.nodes.items()}
        net._rebuild_dependents()
        snapshot2 = {nid: set(n.dependents) for nid, n in net.nodes.items()}
        assert snapshot1 == snapshot2


# ---------------------------------------------------------------------------
# Storage round-trip
# ---------------------------------------------------------------------------

class TestStorageRoundTripDependents:
    """Save → load should produce a clean dependents index."""

    def test_round_trip_preserves_dependents(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node("c", "Derived C",
                     justifications=[Justification(type="SL", antecedents=["a", "b"])])
        net.add_node("d", "Unless B",
                     justifications=[Justification(type="SL", antecedents=[], outlist=["b"])])

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            store = Storage(db_path)
            store.save(net)
            loaded = store.load()
            store.close()

        assert_clean(loaded)
        assert "c" in loaded.nodes["a"].dependents
        assert "c" in loaded.nodes["b"].dependents
        assert "d" in loaded.nodes["b"].dependents

    def test_round_trip_complex_graph(self):
        """Diamond pattern: A → B, A → C, B+C → D."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        net.add_node("c", "Derived C",
                     justifications=[Justification(type="SL", antecedents=["a"])])
        net.add_node("d", "Derived D",
                     justifications=[Justification(type="SL", antecedents=["b", "c"])])

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            store = Storage(db_path)
            store.save(net)
            loaded = store.load()
            store.close()

        assert_clean(loaded)
        assert loaded.nodes["a"].dependents == {"b", "c"}
        assert loaded.nodes["b"].dependents == {"d"}
        assert loaded.nodes["c"].dependents == {"d"}
        assert loaded.nodes["d"].dependents == set()


# ---------------------------------------------------------------------------
# Edge cases from reviewer notes
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Reviewer-flagged edge cases and boundary conditions."""

    def test_dangling_antecedent_reference(self):
        """Justification references a node that doesn't exist in the network."""
        net = Network()
        net.add_node("a", "Depends on missing",
                     justifications=[Justification(type="SL", antecedents=["missing"])])
        assert_clean(net)
        net._rebuild_dependents()
        assert_clean(net)

    def test_multiple_justifications_same_antecedent(self):
        """Two justifications on the same node both reference the same antecedent."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Double-justified",
                     justifications=[
                         Justification(type="SL", antecedents=["a"]),
                         Justification(type="SL", antecedents=["a"]),
                     ])
        assert_clean(net)
        assert net.nodes["a"].dependents == {"b"}

    def test_self_referencing_justification(self):
        """A node that references itself in its antecedents (degenerate case)."""
        net = Network()
        net.add_node("a", "Self-referencing")
        # Manually add a self-referencing justification after creation
        net.nodes["a"].justifications.append(
            Justification(type="SL", antecedents=["a"])
        )
        net.nodes["a"].dependents.add("a")
        assert_clean(net)

    def test_api_rewrite_dependents_stays_clean(self):
        """_rewrite_dependents (used by deduplication) should leave index clean."""
        from reasonsforge.api import _rewrite_dependents
        net = Network()
        net.add_node("old", "Old belief")
        net.add_node("new", "New belief")
        net.add_node("derived", "Depends on old",
                     justifications=[Justification(type="SL", antecedents=["old"])])
        assert "derived" in net.nodes["old"].dependents

        _rewrite_dependents(net, "old", "new")

        assert "derived" in net.nodes["new"].dependents
        assert "derived" not in net.nodes["old"].dependents
        assert net.nodes["derived"].justifications[0].antecedents == ["new"]
        assert_clean(net)

    def test_api_rewrite_dependents_outlist(self):
        """_rewrite_dependents handles outlist references too."""
        from reasonsforge.api import _rewrite_dependents
        net = Network()
        net.add_node("old", "Old belief")
        net.add_node("new", "New belief")
        net.add_node("unless-old", "Unless old",
                     justifications=[Justification(type="SL", antecedents=[], outlist=["old"])])
        assert "unless-old" in net.nodes["old"].dependents

        _rewrite_dependents(net, "old", "new")

        assert "unless-old" in net.nodes["new"].dependents
        assert "unless-old" not in net.nodes["old"].dependents
        assert_clean(net)

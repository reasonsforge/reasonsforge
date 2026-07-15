"""Tests for summarization justifications.

Doyle's SUM/GF justifications let you abstract away internal details.
A summary node replaces a group of nodes in compact output.
"""

import pytest

from reasonsforge import Justification
from reasonsforge.network import Network
from reasonsforge.storage import Storage
from reasonsforge.compact import compact
from reasonsforge import api


class TestSummarizeBasic:

    def test_create_summary(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_node("c", "Derived C", justifications=[
            Justification(type="SL", antecedents=["a", "b"]),
        ])
        result = net.summarize("summary-abc", "A, B, and C together", over=["a", "b", "c"])
        assert result["summary_id"] == "summary-abc"
        assert result["truth_value"] == "IN"
        assert net.nodes["summary-abc"].metadata["summarizes"] == ["a", "b", "c"]

    def test_summary_depends_on_covered_nodes(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.summarize("s", "Summary", over=["a", "b"])
        # Summary has SL justification on a and b
        j = net.nodes["s"].justifications[0]
        assert j.type == "SL"
        assert set(j.antecedents) == {"a", "b"}

    def test_summary_out_when_covered_node_out(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.summarize("s", "Summary", over=["a", "b"])
        assert net.nodes["s"].truth_value == "IN"

        net.retract("a")
        assert net.nodes["s"].truth_value == "OUT"

    def test_summary_restored_when_covered_restored(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.summarize("s", "Summary", over=["a", "b"])
        net.retract("a")
        assert net.nodes["s"].truth_value == "OUT"

        net.assert_node("a")
        assert net.nodes["s"].truth_value == "IN"

    def test_covered_nodes_get_metadata(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.summarize("s", "Summary", over=["a", "b"])
        assert "s" in net.nodes["a"].metadata["summarized_by"]
        assert "s" in net.nodes["b"].metadata["summarized_by"]

    def test_nonexistent_node_raises(self):
        net = Network()
        net.add_node("a", "Premise A")
        with pytest.raises(KeyError):
            net.summarize("s", "Summary", over=["a", "missing"])

    def test_duplicate_summary_raises(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.summarize("s", "Summary", over=["a"])
        with pytest.raises(ValueError):
            net.summarize("s", "Duplicate", over=["a"])


class TestSummarizeInCompact:
    """Summaries replace covered nodes in compact output."""

    def test_summary_replaces_covered_in_compact(self):
        net = Network()
        net.add_node("a", "Premise A detail")
        net.add_node("b", "Premise B detail")
        net.add_node("c", "Premise C uncovered")
        net.summarize("s", "Summary of A and B", over=["a", "b"])

        result = compact(net, budget=5000)
        # Summary should appear
        assert "summary" in result.lower()
        assert "s:" in result
        # Covered nodes should be hidden
        assert "a: Premise A detail" not in result
        assert "b: Premise B detail" not in result
        # Uncovered node should appear
        assert "c:" in result

    def test_hidden_count_shown(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.summarize("s", "Summary", over=["a", "b"])
        result = compact(net, budget=5000)
        assert "hidden by summaries" in result

    def test_no_summary_no_hiding(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        result = compact(net, budget=5000)
        assert "hidden by summaries" not in result
        assert "a:" in result
        assert "b:" in result

    def test_out_summary_does_not_hide(self):
        """If the summary is OUT, covered nodes should not be hidden."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.summarize("s", "Summary", over=["a", "b"])
        net.retract("a")
        # Summary is OUT (a is OUT), both a and b are not hidden
        result = compact(net, budget=5000)
        # b should appear in IN section (it's still IN)
        assert "b:" in result

    def test_covers_count_shown(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.summarize("s", "Summary", over=["a", "b"])
        result = compact(net, budget=5000)
        assert "covers 2 nodes" in result


class TestSummarizePersistence:

    def test_round_trip(self, tmp_path):
        db = tmp_path / "test.db"
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.summarize("s", "Summary", over=["a", "b"])

        store = Storage(db)
        store.save(net)
        loaded = store.load()
        store.close()

        assert loaded.nodes["s"].metadata["summarizes"] == ["a", "b"]
        assert loaded.nodes["s"].truth_value == "IN"
        assert "s" in loaded.nodes["a"].metadata["summarized_by"]

        # Cascading works after load
        loaded.retract("a")
        assert loaded.nodes["s"].truth_value == "OUT"


class TestSummarizeAPI:

    def test_summarize_api(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Premise B", db_path=db)

        result = api.summarize("s", "Summary of A and B", over=["a", "b"], db_path=db)
        assert result["summary_id"] == "s"
        assert result["truth_value"] == "IN"
        assert result["over"] == ["a", "b"]

    def test_summarize_affects_compact(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Premise B", db_path=db)
        api.summarize("s", "Summary", over=["a", "b"], db_path=db)

        result = api.compact(budget=5000, db_path=db)
        assert "hidden by summaries" in result

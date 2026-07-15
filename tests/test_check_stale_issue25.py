"""Tests for issue #25: check_stale reports missing source files.

Validates that check_stale() returns source_deleted results instead of
silently skipping nodes whose source files no longer exist on disk.
"""

import hashlib
from pathlib import Path

import pytest

from reasonsforge.network import Network
from reasonsforge.check_stale import check_stale, hash_file


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class TestSourceDeletedResult:
    """Core behavior: missing source files produce source_deleted results."""

    def test_missing_source_returns_source_deleted(self, tmp_path):
        net = Network()
        net.add_node("a", "Belief A", source="r/gone.md", source_hash="abc123")

        results, *_ = check_stale(net, repos={"r": tmp_path})

        assert len(results) == 1
        r = results[0]
        assert r["node_id"] == "a"
        assert r["reason"] == "source_deleted"
        assert r["new_hash"] is None
        assert r["source_path"] is None
        assert r["old_hash"] == "abc123"
        assert r["source"] == "r/gone.md"

    def test_source_deleted_has_all_required_keys(self, tmp_path):
        net = Network()
        net.add_node("a", "Belief A", source="r/gone.md", source_hash="xyz")

        results, *_ = check_stale(net, repos={"r": tmp_path})

        required_keys = {"node_id", "old_hash", "new_hash", "source", "source_path", "reason"}
        assert set(results[0].keys()) == required_keys

    def test_content_changed_has_same_keys_as_source_deleted(self, tmp_path):
        f = tmp_path / "src.md"
        f.write_text("original")
        h = _hash("original")

        net = Network()
        net.add_node("a", "Belief A", source="r/src.md", source_hash=h)
        net.add_node("b", "Belief B", source="r/gone.md", source_hash="old")

        f.write_text("changed")

        results, *_ = check_stale(net, repos={"r": tmp_path})

        assert len(results) == 2
        changed = next(r for r in results if r["reason"] == "content_changed")
        deleted = next(r for r in results if r["reason"] == "source_deleted")
        assert set(changed.keys()) == set(deleted.keys())


class TestMixedResults:
    """Both deleted and changed sources in the same run."""

    def test_mixed_deleted_and_changed(self, tmp_path):
        changing = tmp_path / "changing.md"
        changing.write_text("old content")
        old_hash = _hash("old content")

        stable = tmp_path / "stable.md"
        stable.write_text("stable content")
        stable_hash = _hash("stable content")

        net = Network()
        net.add_node("changed", "Changed belief", source="r/changing.md", source_hash=old_hash)
        net.add_node("deleted", "Deleted belief", source="r/gone.md", source_hash="somehash")
        net.add_node("fresh", "Fresh belief", source="r/stable.md", source_hash=stable_hash)

        changing.write_text("new content")

        results, *_ = check_stale(net, repos={"r": tmp_path})

        reasons = {r["node_id"]: r["reason"] for r in results}
        assert reasons["changed"] == "content_changed"
        assert reasons["deleted"] == "source_deleted"
        assert "fresh" not in reasons

    def test_multiple_nodes_same_deleted_source(self, tmp_path):
        net = Network()
        net.add_node("a", "Belief A", source="r/shared.md", source_hash="h1")
        net.add_node("b", "Belief B", source="r/shared.md", source_hash="h2")

        results, *_ = check_stale(net, repos={"r": tmp_path})

        assert len(results) == 2
        assert all(r["reason"] == "source_deleted" for r in results)
        ids = {r["node_id"] for r in results}
        assert ids == {"a", "b"}


class TestSkipBehavior:
    """Nodes that should still be skipped by check_stale."""

    def test_skips_out_nodes_with_missing_source(self, tmp_path):
        net = Network()
        net.add_node("a", "Belief A", source="r/gone.md", source_hash="abc")
        net.retract("a")

        results, *_ = check_stale(net, repos={"r": tmp_path})
        assert results == []

    def test_skips_nodes_without_source(self, tmp_path):
        net = Network()
        net.add_node("a", "Premise with no source")

        results, *_ = check_stale(net, repos={"r": tmp_path})
        assert results == []

    def test_skips_nodes_without_hash(self, tmp_path):
        net = Network()
        net.add_node("a", "Has source but no hash", source="r/file.md")

        results, *_ = check_stale(net, repos={"r": tmp_path})
        assert results == []

    def test_skips_nodes_with_empty_source_string(self, tmp_path):
        net = Network()
        net.add_node("a", "Empty source", source="", source_hash="abc")

        results, *_ = check_stale(net, repos={"r": tmp_path})
        assert results == []


class TestEdgeCases:
    """Edge cases from reviewer notes."""

    def test_empty_file_is_not_deleted(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("")
        empty_hash = _hash("")

        net = Network()
        net.add_node("a", "From empty file", source="r/empty.md", source_hash=empty_hash)

        results, *_ = check_stale(net, repos={"r": tmp_path})
        assert results == []

    def test_empty_file_with_wrong_hash_is_content_changed(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("")

        net = Network()
        net.add_node("a", "From empty file", source="r/empty.md", source_hash="wronghash")

        results, *_ = check_stale(net, repos={"r": tmp_path})
        assert len(results) == 1
        assert results[0]["reason"] == "content_changed"
        assert results[0]["new_hash"] == _hash("")

    def test_no_repos_mapping_with_missing_file(self):
        net = Network()
        net.add_node("a", "Belief", source="nonexistent-repo/file.md", source_hash="abc")

        results, *_ = check_stale(net, repos=None)
        assert len(results) == 1
        assert results[0]["reason"] == "source_deleted"

    def test_fresh_node_not_in_results(self, tmp_path):
        f = tmp_path / "src.md"
        f.write_text("content")
        h = _hash("content")

        net = Network()
        net.add_node("a", "Belief A", source="r/src.md", source_hash=h)

        results, *_ = check_stale(net, repos={"r": tmp_path})
        assert results == []

    def test_results_sorted_by_node_id(self, tmp_path):
        net = Network()
        net.add_node("z-node", "Z", source="r/z.md", source_hash="h")
        net.add_node("a-node", "A", source="r/a.md", source_hash="h")
        net.add_node("m-node", "M", source="r/m.md", source_hash="h")

        results, *_ = check_stale(net, repos={"r": tmp_path})
        ids = [r["node_id"] for r in results]
        assert ids == sorted(ids)

    def test_content_changed_populates_source_path(self, tmp_path):
        f = tmp_path / "src.md"
        f.write_text("old")
        h = _hash("old")

        net = Network()
        net.add_node("a", "Belief", source="r/src.md", source_hash=h)
        f.write_text("new")

        results, *_ = check_stale(net, repos={"r": tmp_path})
        assert results[0]["source_path"] == str(f)
        assert results[0]["reason"] == "content_changed"

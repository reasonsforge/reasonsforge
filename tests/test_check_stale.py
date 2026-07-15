"""Tests for check-stale."""

import hashlib
from pathlib import Path

import pytest

from reasonsforge.network import Network
from reasonsforge.check_stale import check_stale, hash_file, hash_sources, resolve_source_path


class TestHashFile:

    def test_hashes_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("hello world")
        h = hash_file(f)
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert h == expected

    def test_hash_changes_with_content(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("version 1")
        h1 = hash_file(f)
        f.write_text("version 2")
        h2 = hash_file(f)
        assert h1 != h2


class TestResolveSourcePath:

    def test_resolve_with_repos(self, tmp_path):
        f = tmp_path / "entry.md"
        f.write_text("content")
        repos = {"myrepo": tmp_path}
        result = resolve_source_path("myrepo/entry.md", repos)
        assert result == f

    def test_resolve_missing_file(self, tmp_path):
        repos = {"myrepo": tmp_path}
        result = resolve_source_path("myrepo/nonexistent.md", repos)
        assert result is None

    def test_resolve_empty_source(self):
        result = resolve_source_path("")
        assert result is None

    def test_resolve_relative_to_db_dir(self, tmp_path):
        entry_dir = tmp_path / "entries" / "2026"
        entry_dir.mkdir(parents=True)
        f = entry_dir / "topic.md"
        f.write_text("content")
        result = resolve_source_path("entries/2026/topic.md", db_dir=tmp_path)
        assert result == f

    def test_db_dir_takes_precedence(self, tmp_path):
        db_dir = tmp_path / "expert"
        repo_dir = tmp_path / "repo"
        for d in (db_dir / "entries", repo_dir / "entries"):
            d.mkdir(parents=True)
        (db_dir / "entries" / "topic.md").write_text("db version")
        (repo_dir / "entries" / "topic.md").write_text("repo version")
        result = resolve_source_path("entries/topic.md", repos={"entries": repo_dir}, db_dir=db_dir)
        assert result == db_dir / "entries" / "topic.md"

    def test_resolve_via_agent_repo(self, tmp_path):
        agent_dir = tmp_path / "agent-repo"
        entry_dir = agent_dir / "entries" / "2026"
        entry_dir.mkdir(parents=True)
        f = entry_dir / "topic.md"
        f.write_text("content")
        repos = {"code": agent_dir}
        result = resolve_source_path("entries/2026/topic.md", repos=repos, agent="code")
        assert result == f

    def test_agent_repo_takes_precedence_over_split(self, tmp_path):
        agent_dir = tmp_path / "agent-repo"
        entries_dir = tmp_path / "entries-repo"
        for d in (agent_dir / "entries", entries_dir):
            d.mkdir(parents=True)
        (agent_dir / "entries" / "topic.md").write_text("agent version")
        (entries_dir / "topic.md").write_text("entries version")
        repos = {"code": agent_dir, "entries": entries_dir}
        result = resolve_source_path("entries/topic.md", repos=repos, agent="code")
        assert result == agent_dir / "entries" / "topic.md"

    def test_falls_back_to_repo_when_not_in_db_dir(self, tmp_path):
        db_dir = tmp_path / "expert"
        db_dir.mkdir()
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        f = repo_dir / "file.md"
        f.write_text("content")
        result = resolve_source_path("myrepo/file.md", repos={"myrepo": repo_dir}, db_dir=db_dir)
        assert result == f


class TestCheckStale:

    def test_fresh_node_via_db_dir(self, tmp_path):
        entry_dir = tmp_path / "entries"
        entry_dir.mkdir()
        f = entry_dir / "topic.md"
        f.write_text("original content")
        h = hashlib.sha256(b"original content").hexdigest()

        net = Network()
        net.add_node("a", "Premise A", source="entries/topic.md", source_hash=h)

        results, *_ = check_stale(net, db_dir=tmp_path)
        assert results == []

    def test_fresh_node(self, tmp_path):
        f = tmp_path / "source.md"
        f.write_text("original content")
        h = hashlib.sha256(b"original content").hexdigest()

        net = Network()
        net.add_node("a", "Premise A", source="myrepo/source.md", source_hash=h)

        results, *_ = check_stale(net, repos={"myrepo": tmp_path})
        assert results == []

    def test_stale_node(self, tmp_path):
        f = tmp_path / "source.md"
        f.write_text("original content")
        old_hash = hashlib.sha256(b"original content").hexdigest()

        net = Network()
        net.add_node("a", "Premise A", source="myrepo/source.md", source_hash=old_hash)

        # Change the file
        f.write_text("updated content")

        results, *_ = check_stale(net, repos={"myrepo": tmp_path})
        assert len(results) == 1
        assert results[0]["node_id"] == "a"
        assert results[0]["old_hash"] == old_hash
        assert results[0]["new_hash"] != old_hash
        assert results[0]["reason"] == "content_changed"

    def test_skips_out_nodes(self, tmp_path):
        f = tmp_path / "source.md"
        f.write_text("original")
        old_hash = hashlib.sha256(b"original").hexdigest()

        net = Network()
        net.add_node("a", "Premise A", source="myrepo/source.md", source_hash=old_hash)
        net.retract("a")

        f.write_text("changed")

        results, *_ = check_stale(net, repos={"myrepo": tmp_path})
        assert results == []

    def test_skips_nodes_without_hash(self, tmp_path):
        net = Network()
        net.add_node("a", "Premise A", source="myrepo/source.md")  # no hash

        results, *_ = check_stale(net, repos={"myrepo": tmp_path})
        assert results == []

    def test_reports_missing_source_files(self, tmp_path):
        net = Network()
        net.add_node("a", "Premise A", source="myrepo/missing.md", source_hash="abc123")

        results, *_ = check_stale(net, repos={"myrepo": tmp_path})
        assert len(results) == 1
        assert results[0]["node_id"] == "a"
        assert results[0]["reason"] == "source_deleted"
        assert results[0]["new_hash"] is None
        assert results[0]["source_path"] is None
        assert results[0]["old_hash"] == "abc123"

    def test_multiple_stale(self, tmp_path):
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("old a")
        f2.write_text("old b")

        net = Network()
        net.add_node("a", "Node A", source="r/a.md", source_hash=hashlib.sha256(b"old a").hexdigest())
        net.add_node("b", "Node B", source="r/b.md", source_hash=hashlib.sha256(b"old b").hexdigest())

        f1.write_text("new a")
        f2.write_text("new b")

        results, *_ = check_stale(net, repos={"r": tmp_path})
        assert len(results) == 2

    def test_agent_imported_node_resolves_via_agent_repo(self, tmp_path):
        agent_dir = tmp_path / "agent-repo"
        entry_dir = agent_dir / "entries"
        entry_dir.mkdir(parents=True)
        f = entry_dir / "topic.md"
        f.write_text("original content")
        h = hashlib.sha256(b"original content").hexdigest()

        net = Network()
        net.repos["code"] = str(agent_dir)
        net.add_node(
            "code:topic-belief", "Topic belief",
            source="entries/topic.md", source_hash=h,
            metadata={"agent": "code"},
        )

        results, *_ = check_stale(net)
        assert results == []

    def test_agent_imported_node_detects_stale(self, tmp_path):
        agent_dir = tmp_path / "agent-repo"
        entry_dir = agent_dir / "entries"
        entry_dir.mkdir(parents=True)
        f = entry_dir / "topic.md"
        f.write_text("original content")
        h = hashlib.sha256(b"original content").hexdigest()

        net = Network()
        net.repos["code"] = str(agent_dir)
        net.add_node(
            "code:topic-belief", "Topic belief",
            source="entries/topic.md", source_hash=h,
            metadata={"agent": "code"},
        )

        f.write_text("updated content")
        results, *_ = check_stale(net)
        assert len(results) == 1
        assert results[0]["node_id"] == "code:topic-belief"
        assert results[0]["reason"] == "content_changed"


class TestPrefixHashUpgrade:

    def test_prefix_hash_treated_as_fresh(self, tmp_path):
        f = tmp_path / "source.md"
        f.write_text("hello world")
        full_hash = hashlib.sha256(b"hello world").hexdigest()
        truncated = full_hash[:16]

        net = Network()
        net.add_node("a", "Premise A", source="myrepo/source.md", source_hash=truncated)

        results, upgraded, *_ = check_stale(net, repos={"myrepo": tmp_path},
                                        upgrade_hashes=True)
        assert results == []
        assert upgraded == 1
        assert net.nodes["a"].source_hash == full_hash

    def test_prefix_hash_without_flag_warns(self, tmp_path):
        f = tmp_path / "source.md"
        f.write_text("hello world")
        full_hash = hashlib.sha256(b"hello world").hexdigest()
        truncated = full_hash[:16]

        net = Network()
        net.add_node("a", "Premise A", source="myrepo/source.md", source_hash=truncated)

        results, upgraded, *_ = check_stale(net, repos={"myrepo": tmp_path})
        assert len(results) == 1
        assert results[0]["reason"] == "truncated_hash"
        assert upgraded == 0
        assert net.nodes["a"].source_hash == truncated

    def test_genuine_change_still_detected(self, tmp_path):
        f = tmp_path / "source.md"
        f.write_text("original content")
        old_hash = hashlib.sha256(b"original content").hexdigest()[:16]

        net = Network()
        net.add_node("a", "Premise A", source="myrepo/source.md", source_hash=old_hash)

        f.write_text("different content")

        results, upgraded, *_ = check_stale(net, repos={"myrepo": tmp_path},
                                        upgrade_hashes=True)
        assert len(results) == 1
        assert results[0]["reason"] == "content_changed"
        assert upgraded == 0

    def test_full_hash_match_unchanged(self, tmp_path):
        f = tmp_path / "source.md"
        f.write_text("hello world")
        full_hash = hashlib.sha256(b"hello world").hexdigest()

        net = Network()
        net.add_node("a", "Premise A", source="myrepo/source.md", source_hash=full_hash)

        results, upgraded, *_ = check_stale(net, repos={"myrepo": tmp_path},
                                        upgrade_hashes=True)
        assert results == []
        assert upgraded == 0
        assert net.nodes["a"].source_hash == full_hash


class TestHashSources:

    def test_backfills_via_db_dir(self, tmp_path):
        entry_dir = tmp_path / "entries"
        entry_dir.mkdir()
        f = entry_dir / "topic.md"
        f.write_text("content")

        net = Network()
        net.add_node("a", "Node A", source="entries/topic.md")

        results = hash_sources(net, db_dir=tmp_path)
        assert len(results) == 1
        assert results[0]["node_id"] == "a"
        assert net.nodes["a"].source_hash != ""

    def test_backfills_empty_hash(self, tmp_path):
        f = tmp_path / "source.md"
        f.write_text("content")

        net = Network()
        net.add_node("a", "Node A", source="r/source.md")
        assert net.nodes["a"].source_hash == ""

        results = hash_sources(net, repos={"r": tmp_path})
        assert len(results) == 1
        assert results[0]["node_id"] == "a"
        assert results[0]["was_empty"] is True
        assert net.nodes["a"].source_hash != ""

    def test_skips_existing_hash(self, tmp_path):
        f = tmp_path / "source.md"
        f.write_text("content")

        net = Network()
        net.add_node("a", "Node A", source="r/source.md", source_hash="existing")

        results = hash_sources(net, repos={"r": tmp_path})
        assert len(results) == 0
        assert net.nodes["a"].source_hash == "existing"

    def test_force_rehashes(self, tmp_path):
        f = tmp_path / "source.md"
        f.write_text("content")

        net = Network()
        net.add_node("a", "Node A", source="r/source.md", source_hash="old")

        results = hash_sources(net, repos={"r": tmp_path}, force=True)
        assert len(results) == 1
        assert results[0]["was_empty"] is False
        assert net.nodes["a"].source_hash != "old"

    def test_skips_missing_source_files(self, tmp_path):
        net = Network()
        net.add_node("a", "Node A", source="r/missing.md")

        results = hash_sources(net, repos={"r": tmp_path})
        assert len(results) == 0

    def test_skips_nodes_without_source(self):
        net = Network()
        net.add_node("a", "Node A")

        results = hash_sources(net)
        assert len(results) == 0

    def test_multiple_nodes(self, tmp_path):
        (tmp_path / "a.md").write_text("aaa")
        (tmp_path / "b.md").write_text("bbb")

        net = Network()
        net.add_node("a", "Node A", source="r/a.md")
        net.add_node("b", "Node B", source="r/b.md")
        net.add_node("c", "Node C", source="r/c.md")  # missing file

        results = hash_sources(net, repos={"r": tmp_path})
        assert len(results) == 2
        hashed_ids = [r["node_id"] for r in results]
        assert "a" in hashed_ids
        assert "b" in hashed_ids

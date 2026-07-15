"""Tests for the topics command."""

import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from reasonsforge import api


def run_cli(*args, db_path=None):
    from reasonsforge.cli import main
    argv = ["reasons"]
    if db_path:
        argv += ["--db", db_path]
    argv += list(args)
    stdout, stderr = StringIO(), StringIO()
    with patch.object(sys, "argv", argv), \
         patch.object(sys, "stdout", stdout), \
         patch.object(sys, "stderr", stderr):
        try:
            main()
            code = 0
        except SystemExit as e:
            code = e.code if e.code is not None else 0
    return stdout.getvalue(), stderr.getvalue(), code


class TestTopicsApi:
    def test_empty_db(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = api.topics(db_path=db)
        assert result["topics"] == []
        assert result["total_nodes"] == 0

    def test_extracts_words_from_ids(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("belief-about-caching", "Caching is good", db_path=db)
        api.add_node("belief-about-testing", "Testing matters", db_path=db)
        api.add_node("caching-strategy-redis", "Use Redis", db_path=db)
        result = api.topics(db_path=db)
        topic_names = [t["topic"] for t in result["topics"]]
        assert "caching" in topic_names
        assert "belief" in topic_names
        assert "about" in topic_names
        caching = next(t for t in result["topics"] if t["topic"] == "caching")
        assert caching["count"] == 2

    def test_filters_stop_words(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("the-use-of-caching", "Cache it", db_path=db)
        result = api.topics(db_path=db)
        topic_names = [t["topic"] for t in result["topics"]]
        assert "the" not in topic_names
        assert "use" not in topic_names
        assert "caching" in topic_names

    def test_filters_short_words(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("ab-cd-caching", "Cache", db_path=db)
        result = api.topics(db_path=db)
        topic_names = [t["topic"] for t in result["topics"]]
        assert "ab" not in topic_names
        assert "cd" not in topic_names
        assert "caching" in topic_names

    def test_splits_on_delimiters(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("module.class:method_name", "A method", db_path=db)
        result = api.topics(db_path=db)
        topic_names = [t["topic"] for t in result["topics"]]
        assert "module" in topic_names
        assert "class" in topic_names
        assert "method" in topic_names
        assert "name" in topic_names

    def test_limit(self, tmp_path):
        db = str(tmp_path / "test.db")
        for i in range(30):
            api.add_node(f"topic-{i}-word{i}", f"Belief {i}", db_path=db)
        result = api.topics(limit=5, db_path=db)
        assert len(result["topics"]) == 5

    def test_sorted_by_frequency(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("alpha-beta", "One", db_path=db)
        api.add_node("alpha-gamma", "Two", db_path=db)
        api.add_node("alpha-delta", "Three", db_path=db)
        api.add_node("beta-gamma", "Four", db_path=db)
        result = api.topics(db_path=db)
        assert result["topics"][0]["topic"] == "alpha"
        assert result["topics"][0]["count"] == 3

    def test_total_nodes(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("node-one", "First", db_path=db)
        api.add_node("node-two", "Second", db_path=db)
        result = api.topics(db_path=db)
        assert result["total_nodes"] == 2


class TestCmdTopics:
    def test_help(self):
        stdout, stderr, code = run_cli("topics", "--help")
        assert code == 0
        assert "topics" in stdout.lower()

    def test_basic_output(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("belief-about-caching", "Cache it", db_path=db)
        api.add_node("belief-about-testing", "Test it", db_path=db)
        stdout, stderr, code = run_cli("topics", db_path=db)
        assert code == 0
        assert "belief" in stdout
        assert "topics from" in stdout

    def test_json_output(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("belief-about-caching", "Cache", db_path=db)
        stdout, stderr, code = run_cli("topics", "--json", db_path=db)
        assert code == 0
        data = json.loads(stdout)
        assert "topics" in data
        assert "total_nodes" in data

    def test_limit_flag(self, tmp_path):
        db = str(tmp_path / "test.db")
        for i in range(20):
            api.add_node(f"topic-{i}-word{i}", f"Belief {i}", db_path=db)
        stdout, stderr, code = run_cli("topics", "--limit", "3", db_path=db)
        assert code == 0
        assert "3 topics from" in stdout

    def test_empty_db(self, tmp_path):
        db = str(tmp_path / "test.db")
        stdout, stderr, code = run_cli("topics", db_path=db)
        assert code == 0
        assert "No topics found" in stdout

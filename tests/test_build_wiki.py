"""Tests for the build-wiki command."""

import os
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from reasonsforge import api
from reasonsforge.build_wiki import (
    _assign_topics, _format_beliefs_for_prompt, _format_node, _linkify,
    _page_name, build_wiki, generate_wiki_page,
)


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


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    api.add_node("caching-strategy-redis", "Use Redis for caching", db_path=db_path)
    api.add_node("caching-ttl-policy", "Set TTL on cache keys", db_path=db_path)
    api.add_node("testing-unit-coverage", "Aim for high unit test coverage", db_path=db_path)
    api.add_node("testing-integration-api", "Integration tests for API endpoints", db_path=db_path)
    api.add_node("deploy-ci-pipeline", "CI pipeline runs on every push",
                 sl="testing-unit-coverage", db_path=db_path)
    return db_path


class TestAssignTopics:

    def test_assigns_to_matching_topic(self):
        topics = [{"topic": "caching", "count": 3}, {"topic": "testing", "count": 2}]
        node_ids = ["caching-strategy-redis", "testing-unit-coverage"]
        groups = _assign_topics(node_ids, topics)
        assert "caching-strategy-redis" in groups["caching"]
        assert "testing-unit-coverage" in groups["testing"]

    def test_unmatched_goes_to_other(self):
        topics = [{"topic": "caching", "count": 3}]
        node_ids = ["caching-redis", "xyz-unknown-thing"]
        groups = _assign_topics(node_ids, topics)
        assert "xyz-unknown-thing" in groups["Other"]

    def test_empty_nodes(self):
        topics = [{"topic": "caching", "count": 1}]
        groups = _assign_topics([], topics)
        assert groups == {}

    def test_no_topics(self):
        groups = _assign_topics(["some-node"], [])
        assert "some-node" in groups["Other"]


class TestPageName:

    def test_simple(self):
        assert _page_name("caching") == "caching"

    def test_special_chars(self):
        assert _page_name("My Topic!") == "my-topic"

    def test_spaces(self):
        assert _page_name("hello world") == "hello-world"

    def test_empty_fallback(self):
        assert _page_name("!!!") == "other"


class TestFormatNode:

    def test_basic_output(self):
        detail = {
            "text": "Use Redis for caching",
            "truth_value": "IN",
            "justifications": [],
            "dependents": [],
        }
        result = _format_node("caching-redis", detail, {})
        assert "### caching-redis" in result
        assert "**Status:** IN" in result
        assert "Use Redis for caching" in result

    def test_cross_references(self):
        detail = {
            "text": "CI pipeline",
            "truth_value": "IN",
            "justifications": [{"antecedents": ["testing-unit"], "outlist": []}],
            "dependents": ["deploy-prod"],
        }
        node_to_page = {
            "testing-unit": "testing.md",
            "deploy-prod": "deploy.md",
        }
        result = _format_node("ci-pipeline", detail, node_to_page)
        assert "[testing-unit](testing.md#testing-unit)" in result
        assert "[deploy-prod](deploy.md#deploy-prod)" in result
        assert "**Depends on:**" in result
        assert "**Supports:**" in result

    def test_no_links_for_unknown_nodes(self):
        detail = {
            "text": "Some belief",
            "truth_value": "OUT",
            "justifications": [{"antecedents": ["unknown-node"], "outlist": []}],
            "dependents": [],
        }
        result = _format_node("test-node", detail, {})
        assert "unknown-node" in result
        assert "[unknown-node]" not in result

    def test_duplicate_of_metadata(self):
        detail = {
            "text": "Duplicate belief",
            "truth_value": "OUT",
            "justifications": [],
            "dependents": [],
            "metadata": {"duplicate_of": "canonical-node"},
        }
        node_to_page = {"canonical-node": "core.md"}
        result = _format_node("dup-node", detail, node_to_page)
        assert "**Duplicate of:** [canonical-node](core.md#canonical-node)" in result

    def test_duplicate_of_no_page(self):
        detail = {
            "text": "Duplicate belief",
            "truth_value": "OUT",
            "justifications": [],
            "dependents": [],
            "metadata": {"duplicate_of": "unknown-canonical"},
        }
        result = _format_node("dup-node", detail, {})
        assert "**Duplicate of:** unknown-canonical" in result
        assert "[unknown-canonical]" not in result

    def test_superseded_by_metadata(self):
        detail = {
            "text": "Old belief",
            "truth_value": "OUT",
            "justifications": [],
            "dependents": [],
            "metadata": {"superseded_by": "new-belief"},
        }
        node_to_page = {"new-belief": "updates.md"}
        result = _format_node("old-belief", detail, node_to_page)
        assert "**Superseded by:** [new-belief](updates.md#new-belief)" in result

    def test_none_metadata(self):
        detail = {
            "text": "Node with None metadata",
            "truth_value": "IN",
            "justifications": [],
            "dependents": [],
            "metadata": None,
        }
        result = _format_node("none-meta", detail, {})
        assert "### none-meta" in result
        assert "Duplicate of" not in result
        assert "Superseded by" not in result

    def test_defeated_by_metadata(self):
        defeater_detail = {
            "text": "This defeats target-node",
            "truth_value": "IN",
            "justifications": [],
            "dependents": [],
            "metadata": {
                "defeats_node": "target-node",
                "defeater_type": "invalid-inference",
            },
        }
        target_detail = {
            "text": "A defeated belief",
            "truth_value": "OUT",
            "justifications": [{"antecedents": ["base"], "outlist": ["defeater-1"]}],
            "dependents": [],
        }
        all_details = {
            "target-node": target_detail,
            "defeater-1": defeater_detail,
        }
        node_to_page = {"defeater-1": "misc.md"}
        result = _format_node("target-node", target_detail, node_to_page,
                              all_details=all_details)
        assert "**Defeated by:** [defeater-1](misc.md#defeater-1) (invalid-inference)" in result

    def test_defeated_by_no_page(self):
        defeater_detail = {
            "text": "This defeats target-node",
            "truth_value": "IN",
            "justifications": [],
            "dependents": [],
            "metadata": {"defeats_node": "target-node", "defeater_type": "defeater"},
        }
        target_detail = {
            "text": "A defeated belief",
            "truth_value": "OUT",
            "justifications": [{"antecedents": [], "outlist": ["def-1"]}],
            "dependents": [],
        }
        all_details = {"target-node": target_detail, "def-1": defeater_detail}
        result = _format_node("target-node", target_detail, {},
                              all_details=all_details)
        assert "**Defeated by:** def-1 (defeater)" in result

    def test_defeated_by_with_reason_type(self):
        defeater_detail = {
            "text": "This defeats target-node",
            "truth_value": "IN",
            "justifications": [],
            "dependents": [],
            "metadata": {
                "defeats_node": "target-node",
                "defeater_type": "invalid-inference",
                "defeat_reason_type": "scope-mismatch",
            },
        }
        target_detail = {
            "text": "A defeated belief",
            "truth_value": "OUT",
            "justifications": [{"antecedents": ["base"], "outlist": ["defeater-1"]}],
            "dependents": [],
        }
        all_details = {
            "target-node": target_detail,
            "defeater-1": defeater_detail,
        }
        node_to_page = {"defeater-1": "misc.md"}
        result = _format_node("target-node", target_detail, node_to_page,
                              all_details=all_details)
        assert "**Defeated by:** [defeater-1](misc.md#defeater-1) (invalid-inference, scope-mismatch)" in result

    def test_no_defeaters_without_all_details(self):
        detail = {
            "text": "A belief",
            "truth_value": "OUT",
            "justifications": [{"antecedents": [], "outlist": ["some-defeater"]}],
            "dependents": [],
        }
        result = _format_node("test-node", detail, {})
        assert "Defeated by" not in result


class TestBuildWiki:

    def test_creates_files(self, tmp_path):
        output_dir = str(tmp_path / "wiki")
        details = {
            "node-a": {"text": "A", "truth_value": "IN", "justifications": [], "dependents": []},
            "node-b": {"text": "B", "truth_value": "IN", "justifications": [], "dependents": []},
        }
        groups = {"alpha": ["node-a"], "beta": ["node-b"]}
        result = build_wiki(details, groups, output_dir)
        assert result["pages"] == 2
        assert result["total_nodes"] == 2
        assert os.path.isfile(os.path.join(output_dir, "index.md"))
        assert os.path.isfile(os.path.join(output_dir, "alpha.md"))
        assert os.path.isfile(os.path.join(output_dir, "beta.md"))

    def test_index_has_links(self, tmp_path):
        output_dir = str(tmp_path / "wiki")
        details = {
            "node-a": {"text": "A", "truth_value": "IN", "justifications": [], "dependents": []},
        }
        groups = {"mytopic": ["node-a"]}
        build_wiki(details, groups, output_dir)
        index = open(os.path.join(output_dir, "index.md")).read()
        assert "[mytopic](mytopic.md)" in index
        assert "1 beliefs across 1 pages" in index

    def test_page_has_back_link(self, tmp_path):
        output_dir = str(tmp_path / "wiki")
        details = {
            "node-a": {"text": "A", "truth_value": "IN", "justifications": [], "dependents": []},
        }
        groups = {"topic": ["node-a"]}
        build_wiki(details, groups, output_dir)
        page = open(os.path.join(output_dir, "topic.md")).read()
        assert "[Back to index](index.md)" in page

    def test_index_slug_collision(self, tmp_path):
        output_dir = str(tmp_path / "wiki")
        details = {
            "node-a": {"text": "A", "truth_value": "IN", "justifications": [], "dependents": []},
        }
        groups = {"index": ["node-a"]}
        result = build_wiki(details, groups, output_dir)
        assert result["pages"] == 1
        assert os.path.isfile(os.path.join(output_dir, "index.md"))
        assert os.path.isfile(os.path.join(output_dir, "index-topic.md"))
        index = open(os.path.join(output_dir, "index.md")).read()
        assert "Belief Wiki" in index

    def test_duplicate_slug_collision(self, tmp_path):
        output_dir = str(tmp_path / "wiki")
        details = {
            "node-a": {"text": "A", "truth_value": "IN", "justifications": [], "dependents": []},
            "node-b": {"text": "B", "truth_value": "IN", "justifications": [], "dependents": []},
        }
        groups = {"My Topic": ["node-a"], "My-Topic!": ["node-b"]}
        result = build_wiki(details, groups, output_dir)
        assert result["pages"] == 2
        files = [f for f in os.listdir(output_dir) if f != "index.md"]
        assert len(files) == 2


class TestBuildWikiApi:

    def test_creates_wiki_directory(self, db, tmp_path):
        output_dir = str(tmp_path / "wiki_out")
        result = api.build_wiki(output_dir=output_dir, db_path=db)
        assert result["total_nodes"] == 5
        assert result["pages"] > 0
        assert os.path.isfile(os.path.join(output_dir, "index.md"))

    def test_status_filter(self, db, tmp_path):
        api.retract_node("caching-strategy-redis", db_path=db)
        output_dir = str(tmp_path / "wiki_in")
        result = api.build_wiki(output_dir=output_dir, status="IN", db_path=db)
        assert result["total_nodes"] < 5

    def test_empty_db(self, tmp_path):
        db = str(tmp_path / "empty.db")
        output_dir = str(tmp_path / "wiki_empty")
        result = api.build_wiki(output_dir=output_dir, db_path=db)
        assert result["total_nodes"] == 0
        assert result["pages"] == 0
        index = open(os.path.join(output_dir, "index.md")).read()
        assert "No beliefs found" in index

    def test_cross_references_resolve(self, db, tmp_path):
        output_dir = str(tmp_path / "wiki_xref")
        api.build_wiki(output_dir=output_dir, db_path=db)
        found_depends = False
        for fname in os.listdir(output_dir):
            if fname == "index.md":
                continue
            content = open(os.path.join(output_dir, fname)).read()
            if "**Depends on:**" in content:
                found_depends = True
        assert found_depends


class TestBuildWikiCli:

    def test_help(self):
        stdout, stderr, code = run_cli("build-wiki", "--help")
        assert code == 0
        assert "build-wiki" in stdout or "wiki" in stdout

    def test_creates_output(self, db, tmp_path):
        output_dir = str(tmp_path / "cli_wiki")
        stdout, stderr, code = run_cli("build-wiki", "-o", output_dir, db_path=db)
        assert code == 0
        assert "Wiki written to" in stdout
        assert os.path.isfile(os.path.join(output_dir, "index.md"))

    def test_status_filter(self, db, tmp_path):
        output_dir = str(tmp_path / "cli_wiki_in")
        stdout, stderr, code = run_cli("build-wiki", "-o", output_dir, "--status", "IN", db_path=db)
        assert code == 0


class TestLinkify:

    def test_creates_cross_page_links(self):
        content = "depends on node-alpha and node-beta"
        node_to_page = {"node-alpha": "page-a.md", "node-beta": "page-b.md"}
        result = _linkify(content, "page-c.md", node_to_page,
                          node_to_page.keys())
        assert "[node-alpha](page-a.md#node-alpha)" in result
        assert "[node-beta](page-b.md#node-beta)" in result

    def test_skips_same_page(self):
        content = "mentions node-alpha"
        node_to_page = {"node-alpha": "page-a.md"}
        result = _linkify(content, "page-a.md", node_to_page,
                          node_to_page.keys())
        assert result == content

    def test_skips_already_linked(self):
        content = "see [node-alpha](page-a.md#node-alpha)"
        node_to_page = {"node-alpha": "page-a.md"}
        result = _linkify(content, "page-b.md", node_to_page,
                          node_to_page.keys())
        assert result.count("[node-alpha]") == 1

    def test_longest_first(self):
        content = "about node-alpha-extended"
        node_to_page = {
            "node-alpha": "a.md",
            "node-alpha-extended": "b.md",
        }
        result = _linkify(content, "c.md", node_to_page, node_to_page.keys())
        assert "[node-alpha-extended](b.md#node-alpha-extended)" in result


class TestFormatBeliefsForPrompt:

    def test_formats_beliefs(self):
        details = {
            "node-a": {
                "text": "A is true",
                "truth_value": "IN",
                "justifications": [{"antecedents": ["node-b"], "outlist": []}],
                "dependents": ["node-c"],
            },
        }
        result = _format_beliefs_for_prompt(["node-a"], details)
        assert "### node-a" in result
        assert "Status: IN" in result
        assert "Text: A is true" in result
        assert "Depends on: node-b" in result
        assert "Supports: node-c" in result

    def test_skips_missing_nodes(self):
        result = _format_beliefs_for_prompt(["missing"], {})
        assert result.strip() == ""


class TestGenerateWikiPage:

    def test_calls_invoke_model(self):
        details = {
            "node-a": {
                "text": "A is true",
                "truth_value": "IN",
                "justifications": [],
                "dependents": [],
            },
        }
        with patch("reasonsforge.llm.invoke_model",
                    return_value="Generated wiki content") as mock:
            result = generate_wiki_page("mytopic", ["node-a"], details,
                                        "claude", 300)
            assert result == "Generated wiki content"
            mock.assert_called_once()
            prompt = mock.call_args[0][0]
            assert "mytopic" in prompt
            assert "node-a" in prompt


class TestBuildWikiLlm:

    def test_llm_mode_uses_generated_content(self, tmp_path):
        output_dir = str(tmp_path / "wiki")
        details = {
            "node-a": {"text": "A", "truth_value": "IN",
                       "justifications": [], "dependents": []},
        }
        groups = {"alpha": ["node-a"]}
        with patch("reasonsforge.build_wiki.generate_wiki_page",
                    return_value="LLM wrote this page"):
            result = build_wiki(details, groups, output_dir, model="claude")
        page = open(os.path.join(output_dir, "alpha.md")).read()
        assert "LLM wrote this page" in page
        assert result["pages"] == 1

    def test_no_llm_mode_no_invoke(self, tmp_path):
        output_dir = str(tmp_path / "wiki")
        details = {
            "node-a": {"text": "A", "truth_value": "IN",
                       "justifications": [], "dependents": []},
        }
        groups = {"alpha": ["node-a"]}
        with patch("reasonsforge.build_wiki.generate_wiki_page") as mock:
            build_wiki(details, groups, output_dir)
            mock.assert_not_called()

    def test_llm_failure_falls_back(self, tmp_path):
        output_dir = str(tmp_path / "wiki")
        details = {
            "node-a": {"text": "A", "truth_value": "IN",
                       "justifications": [], "dependents": []},
        }
        groups = {"alpha": ["node-a"]}
        with patch("reasonsforge.build_wiki.generate_wiki_page",
                    side_effect=RuntimeError("LLM failed")):
            build_wiki(details, groups, output_dir, model="claude")
        page = open(os.path.join(output_dir, "alpha.md")).read()
        assert "### node-a" in page

    def test_api_passes_model_through(self, db, tmp_path):
        output_dir = str(tmp_path / "wiki_llm")
        with patch("reasonsforge.build_wiki.generate_wiki_page",
                    return_value="Generated page") as mock:
            result = api.build_wiki(output_dir=output_dir, model="claude",
                                    db_path=db)
        assert result["pages"] > 0
        assert mock.call_count == result["pages"]

    def test_parallel_generates_all_pages(self, tmp_path):
        output_dir = str(tmp_path / "wiki")
        details = {
            "node-a": {"text": "A", "truth_value": "IN",
                       "justifications": [], "dependents": []},
            "node-b": {"text": "B", "truth_value": "IN",
                       "justifications": [], "dependents": []},
        }
        groups = {"alpha": ["node-a"], "beta": ["node-b"]}
        with patch("reasonsforge.build_wiki.generate_wiki_page",
                    return_value="Parallel content"):
            result = build_wiki(details, groups, output_dir,
                                model="claude", parallel=2)
        assert result["pages"] == 2
        for name in ["alpha.md", "beta.md"]:
            page = open(os.path.join(output_dir, name)).read()
            assert "Parallel content" in page

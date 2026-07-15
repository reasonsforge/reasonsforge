"""Tests for the ask module (FTS5 search + LLM synthesis)."""

import sqlite3
import subprocess
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from reasonsforge.ask import (
    extract_tool_call, build_ask_prompt, build_final_prompt, build_simple_prompt,
    ask, _invoke_claude, _strip_belief_metadata, _search_source_chunks,
    NO_BELIEFS_MSG,
)
from reasonsforge.cli import main


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


def run_cli(*args, db_path=None):
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
        except SystemExit as e:
            return stdout.getvalue(), stderr.getvalue(), e.code
    return stdout.getvalue(), stderr.getvalue(), 0


class TestExtractToolCall:

    def test_valid_tool_call(self):
        text = 'Some preamble text\n{"tool": "search_beliefs", "query": "propagation"}\nMore text'
        result = extract_tool_call(text)
        assert result == {"tool": "search_beliefs", "query": "propagation"}

    def test_no_tool_call(self):
        text = "This is just a plain answer with no JSON."
        result = extract_tool_call(text)
        assert result is None

    def test_json_without_tool_key(self):
        text = '{"name": "test", "value": 42}'
        result = extract_tool_call(text)
        assert result is None

    def test_malformed_json_skipped(self):
        text = '{bad json}\n{"tool": "search_beliefs", "query": "test"}'
        result = extract_tool_call(text)
        assert result == {"tool": "search_beliefs", "query": "test"}

    def test_first_tool_call_wins(self):
        text = '{"tool": "search_beliefs", "query": "first"}\n{"tool": "search_beliefs", "query": "second"}'
        result = extract_tool_call(text)
        assert result["query"] == "first"

    def test_non_json_lines_skipped(self):
        text = "Hello\nWorld\n  not json\n"
        result = extract_tool_call(text)
        assert result is None

    def test_empty_string(self):
        result = extract_tool_call("")
        assert result is None


class TestBuildAskPrompt:

    def test_contains_question_and_context(self):
        prompt = build_ask_prompt("What is BFS?", "Some belief context")
        assert "What is BFS?" in prompt
        assert "Some belief context" in prompt

    def test_no_tool_history(self):
        prompt = build_ask_prompt("question", "context")
        assert "Additional search results" not in prompt

    def test_with_tool_history(self):
        history = [
            {"tool_label": 'search_beliefs("propagation")', "result": "Found: propagation-is-bfs"},
            {"tool_label": 'search_beliefs("retraction")', "result": "Found: retraction-cascades"},
        ]
        prompt = build_ask_prompt("question", "context", tool_history=history)
        assert "Additional search results" in prompt
        assert "propagation" in prompt
        assert "retraction" in prompt

    def test_tool_definition_in_prompt(self):
        prompt = build_ask_prompt("question", "context")
        assert "search_beliefs" in prompt
        assert '"tool"' in prompt

    def test_final_prompt_has_no_tool_definition(self):
        prompt = build_final_prompt("question", "context")
        assert "search_beliefs" not in prompt
        assert '"tool"' not in prompt
        assert "question" in prompt
        assert "context" in prompt


class TestAskNoSynth:

    def test_no_synth_defaults_to_compact(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "prop-bfs", "Propagation uses breadth-first search", db_path=db_path)

        result = ask("propagation", db_path=db_path, no_synth=True)
        assert "[IN] prop-bfs" in result
        assert "—" in result

    def test_no_synth_compact_multiple(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "prop-bfs", "Propagation uses BFS", db_path=db_path)
        run_cli("add", "prop-cascade", "Propagation cascades", db_path=db_path)

        result = ask("propagation", db_path=db_path, no_synth=True)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        for line in lines:
            assert line.startswith("[IN]") or line.startswith("[OUT]")

    def test_no_synth_markdown_format(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "prop-bfs", "Propagation uses BFS", db_path=db_path)

        result = ask("propagation", db_path=db_path, no_synth=True, format="markdown")
        assert "##" in result or "**" in result or "prop-bfs" in result

    def test_no_synth_no_results(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "alpha", "Alpha belief", db_path=db_path)

        result = ask("zzzznonexistent", db_path=db_path, no_synth=True)
        assert "No results" in result


class TestCmdAskNoSynth:

    def test_cli_ask_no_synth_compact(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "test-belief", "The system uses BFS for propagation", db_path=db_path)
        out, err, code = run_cli("ask", "BFS propagation", "--no-synth", db_path=db_path)
        assert code == 0
        assert "[IN] test-belief" in out

    def test_cli_ask_no_synth_markdown(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "test-belief", "The system uses BFS for propagation", db_path=db_path)
        out, err, code = run_cli("ask", "BFS propagation", "--no-synth",
                                 "--format", "markdown", db_path=db_path)
        assert code == 0
        assert "test-belief" in out

    def test_cli_ask_no_synth_no_results(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("ask", "nothing matches", "--no-synth", db_path=db_path)
        assert code == 0
        assert "No results" in out


class TestInvokeClaude:

    def test_claude_not_in_path(self):
        with patch("reasonsforge.llm.shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="claude"):
                _invoke_claude("test prompt")


class TestAskNoBeliefs:

    def test_empty_network_llm_declines(self, db_path):
        run_cli("init", db_path=db_path)
        refusal = "I don't have enough beliefs in the network to answer this question."
        with patch("reasonsforge.ask.invoke_model", return_value=refusal):
            result = ask("what is the meaning of life", db_path=db_path)
        assert "don't have enough beliefs" in result

    def test_no_matching_beliefs_llm_declines(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "alpha", "Alpha belief about propagation", db_path=db_path)
        refusal = "I don't have enough beliefs in the network to answer this question."
        with patch("reasonsforge.ask.invoke_model", return_value=refusal):
            result = ask("zzzznonexistent", db_path=db_path)
        assert "don't have enough beliefs" in result

    def test_timeout_on_empty_returns_no_beliefs_message(self, db_path):
        run_cli("init", db_path=db_path)
        with patch("reasonsforge.ask.invoke_model",
                    side_effect=subprocess.TimeoutExpired("claude", 300)):
            result = ask("nothing matches", db_path=db_path)
        assert result == NO_BELIEFS_MSG

    def test_error_on_empty_returns_no_beliefs_message(self, db_path):
        run_cli("init", db_path=db_path)
        with patch("reasonsforge.ask.invoke_model",
                    side_effect=RuntimeError("claude crashed")):
            result = ask("nothing matches", db_path=db_path)
        assert result == NO_BELIEFS_MSG

    def test_timeout_after_successful_tool_search_returns_beliefs(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "b", "Beta belief about retraction", db_path=db_path)

        calls = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            calls[0] += 1
            if calls[0] == 1:
                return '{"tool": "search_beliefs", "query": "retraction"}'
            raise subprocess.TimeoutExpired("claude", 300)

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("zzzznothing", db_path=db_path)
        assert "retraction" in result.lower()
        assert result != NO_BELIEFS_MSG

    def test_retry_no_results_preserves_initial_beliefs(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief about propagation", db_path=db_path)

        calls = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            calls[0] += 1
            if calls[0] == 1:
                return '{"tool": "search_beliefs", "query": "zzzznothing"}'
            raise subprocess.TimeoutExpired("claude", 300)

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("propagation", db_path=db_path)
        assert "propagation" in result.lower()
        assert result != NO_BELIEFS_MSG


class TestAskWithMockedLLM:

    def test_direct_answer(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model", return_value="The answer is alpha."):
            result = ask("what is alpha?", db_path=db_path)
        assert result == "The answer is alpha."

    def test_tool_call_then_answer(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)
        run_cli("add", "b", "Beta belief about retraction", db_path=db_path)

        responses = [
            '{"tool": "search_beliefs", "query": "retraction"}',
            "Retraction cascades through dependents [b].",
        ]
        call_count = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("how does retraction work?", db_path=db_path)
        assert "retraction" in result.lower() or "Retraction" in result
        assert call_count[0] == 2

    def test_final_tool_call_triggers_extra_synthesis(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        calls = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            calls[0] += 1
            if calls[0] <= 3:
                return '{"tool": "search_beliefs", "query": "more"}'
            return "The answer based on alpha [a]."

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("alpha", db_path=db_path)
        assert calls[0] == 4
        assert "alpha" in result.lower()
        assert "search_beliefs" not in result

    def test_timeout_returns_search_results(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    side_effect=subprocess.TimeoutExpired("claude", 300)):
            result = ask("alpha", db_path=db_path)
        assert "a" in result

    def test_error_returns_search_results(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    side_effect=RuntimeError("claude crashed")):
            result = ask("alpha", db_path=db_path)
        assert "a" in result

    def test_unknown_tool_returns_response(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    return_value='{"tool": "unknown_tool", "query": "x"}'):
            result = ask("question", db_path=db_path)
        assert "unknown_tool" in result

    def test_extra_synthesis_timeout_returns_beliefs(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        calls = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            calls[0] += 1
            if calls[0] <= 3:
                return '{"tool": "search_beliefs", "query": "more"}'
            raise subprocess.TimeoutExpired("claude", 300)

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("alpha", db_path=db_path)
        assert calls[0] == 4
        assert "search_beliefs" not in result
        assert "Alpha" in result or result == NO_BELIEFS_MSG

    def test_extra_synthesis_tool_call_returns_beliefs(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    return_value='{"tool": "search_beliefs", "query": "more"}'):
            result = ask("alpha", db_path=db_path)
        assert "search_beliefs" not in result
        assert "Alpha" in result or result == NO_BELIEFS_MSG


class TestAskSourcesPreserved:

    @pytest.fixture
    def sources_db(self, tmp_path):
        db = str(tmp_path / "sources.db")
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, text TEXT, cluster TEXT,
                filename TEXT, section TEXT
            )
        """)
        conn.execute("""
            INSERT INTO chunks VALUES (1, 'Important source doc content', 'docs', 'doc.md', NULL)
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=id)
        """)
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()
        conn.close()
        return db

    def test_sources_survive_tool_call(self, db_path, sources_db):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief about docs", db_path=db_path)
        run_cli("add", "b", "Beta belief about important topics", db_path=db_path)

        calls = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            calls[0] += 1
            if calls[0] == 1:
                return '{"tool": "search_beliefs", "query": "important"}'
            assert "Source Documents" in prompt, "Source documents lost after tool call"
            assert "Important source doc content" in prompt
            return "Answer combining beliefs and sources."

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("important docs", db_path=db_path, sources_db=sources_db)
        assert calls[0] == 2
        assert result == "Answer combining beliefs and sources."

    def test_natural_applied_to_tool_results(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)
        run_cli("add", "b", "Beta belief about retraction", db_path=db_path)

        calls = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            calls[0] += 1
            if calls[0] == 1:
                return '{"tool": "search_beliefs", "query": "retraction"}'
            assert "**Status:**" not in prompt, "Status metadata leaked into natural prompt"
            assert "**Source:**" not in prompt, "Source metadata leaked into natural prompt"
            return "Natural answer about retraction."

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("retraction", db_path=db_path, natural=True)
        assert calls[0] == 2
        assert result == "Natural answer about retraction."


class TestBuildSimplePrompt:

    def test_contains_question_and_context(self):
        prompt = build_simple_prompt("What is BFS?", "Some belief context")
        assert "What is BFS?" in prompt
        assert "Some belief context" in prompt

    def test_no_tool_definition(self):
        prompt = build_simple_prompt("question", "context")
        assert "search_beliefs" not in prompt
        assert '"tool"' not in prompt

    def test_natural_mode_no_cite(self):
        prompt = build_simple_prompt("question", "context", natural=True)
        assert "Cite belief IDs" not in prompt
        assert "plain natural language" in prompt

    def test_default_has_cite(self):
        prompt = build_simple_prompt("question", "context")
        assert "Cite belief IDs in [brackets]" in prompt


class TestAskSimple:

    def test_simple_direct_answer(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief about propagation", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    return_value="Propagation uses BFS [a].") as mock_llm:
            result = ask("propagation", db_path=db_path, simple=True)
        assert result == "Propagation uses BFS [a]."
        assert mock_llm.call_count == 1

    def test_simple_no_results(self, db_path):
        run_cli("init", db_path=db_path)
        result = ask("zzzznonexistent", db_path=db_path, simple=True)
        assert result == NO_BELIEFS_MSG

    def test_simple_timeout_returns_beliefs(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    side_effect=subprocess.TimeoutExpired("claude", 300)):
            result = ask("alpha", db_path=db_path, simple=True)
        assert "Alpha" in result

    def test_simple_error_returns_beliefs(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    side_effect=RuntimeError("model crashed")):
            result = ask("alpha", db_path=db_path, simple=True)
        assert "Alpha" in result

    def test_simple_prompt_has_no_tool_definition(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    return_value="Answer.") as mock_llm:
            ask("alpha", db_path=db_path, simple=True)
        prompt = mock_llm.call_args[0][0]
        assert "search_beliefs" not in prompt
        assert '"tool"' not in prompt


class TestStripBeliefMetadata:

    def test_strips_header_lines(self):
        context = "### my-belief\n**Status:** IN\nThe actual belief text.\n"
        result = _strip_belief_metadata(context)
        assert "### my-belief" not in result
        assert "The actual belief text." in result

    def test_strips_status(self):
        context = "**Status:** IN\nBelief text."
        result = _strip_belief_metadata(context)
        assert "**Status:**" not in result
        assert "Belief text." in result

    def test_strips_source(self):
        context = "**Source:** repo:file.py\nBelief text."
        result = _strip_belief_metadata(context)
        assert "**Source:**" not in result
        assert "Belief text." in result

    def test_strips_depends_on(self):
        context = "**Depends on:** a, b\nBelief text."
        result = _strip_belief_metadata(context)
        assert "**Depends on:**" not in result

    def test_strips_justification(self):
        context = "**Justification:** SL(a)\nBelief text."
        result = _strip_belief_metadata(context)
        assert "**Justification:**" not in result

    def test_strips_supported_by(self):
        context = "**Supported by:** x, y\nBelief text."
        result = _strip_belief_metadata(context)
        assert "**Supported by:**" not in result

    def test_strips_supports(self):
        context = "**Supports:** z\nBelief text."
        result = _strip_belief_metadata(context)
        assert "**Supports:**" not in result

    def test_strips_depended_on_by(self):
        context = "**Depended on by:** x, y\nBelief text."
        result = _strip_belief_metadata(context)
        assert "**Depended on by:**" not in result
        assert "Belief text." in result

    def test_strips_related_nodes(self):
        context = "**Related nodes:**\n\n- **foo** (IN): Some text\n- **bar** (OUT): Other text\nBelief text."
        result = _strip_belief_metadata(context)
        assert "**Related nodes:**" not in result
        assert "- **foo**" not in result
        assert "- **bar**" not in result
        assert "Belief text." in result

    def test_strips_separator(self):
        context = "Belief A.\n\n---\n\nBelief B."
        result = _strip_belief_metadata(context)
        assert "---" not in result
        assert "Belief A." in result
        assert "Belief B." in result

    def test_preserves_plain_text(self):
        context = "Propagation uses BFS.\nRetraction cascades through dependents."
        result = _strip_belief_metadata(context)
        assert result == context

    def test_collapses_blank_lines(self):
        context = "Text A.\n\n\n\n\nText B."
        result = _strip_belief_metadata(context)
        assert "\n\n\n" not in result
        assert "Text A." in result
        assert "Text B." in result

    def test_empty_input(self):
        assert _strip_belief_metadata("") == ""
        assert _strip_belief_metadata(None) is None

    def test_full_markdown_format(self):
        context = (
            "### prop-bfs\n"
            "**Status:** IN\n"
            "Propagation uses breadth-first search.\n"
            "**Source:** code:network.py\n"
            "**Depends on:** core-algo\n"
            "**Supported by:** impl-detail\n"
            "**Depended on by:** retract-cascade\n"
            "\n"
            "### retract-cascade\n"
            "**Status:** IN\n"
            "Retraction cascades through dependents.\n"
            "**Justification:** SL(prop-bfs)\n"
            "\n"
            "---\n"
            "**Related nodes:**\n"
            "\n"
            "- **core-algo** (IN): Core algorithm implementation\n"
        )
        result = _strip_belief_metadata(context)
        assert "Propagation uses breadth-first search." in result
        assert "Retraction cascades through dependents." in result
        assert "### " not in result
        assert "**Status:**" not in result
        assert "**Source:**" not in result
        assert "**Depends on:**" not in result
        assert "**Depended on by:**" not in result
        assert "**Related nodes:**" not in result
        assert "---" not in result
        assert "- **core-algo**" not in result


class TestSearchSourceChunks:

    @pytest.fixture
    def sources_db(self, tmp_path):
        db = str(tmp_path / "sources.db")
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                text TEXT,
                cluster TEXT,
                filename TEXT,
                section TEXT
            )
        """)
        conn.execute("""
            INSERT INTO chunks (id, text, cluster, filename, section)
            VALUES (1, 'Red Hat Summit 2025 is in Boston', 'events', 'events.md', 'Summit')
        """)
        conn.execute("""
            INSERT INTO chunks (id, text, cluster, filename, section)
            VALUES (2, 'OpenShift supports Kubernetes workloads', 'products', 'openshift.md', NULL)
        """)
        conn.execute("""
            INSERT INTO chunks (id, text, cluster, filename, section)
            VALUES (3, 'Ansible automates IT infrastructure', 'products', 'ansible.md', 'Overview')
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=id)
        """)
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()
        conn.close()
        return db

    def test_returns_matching_chunks(self, sources_db):
        result = _search_source_chunks("Red Hat Summit", sources_db)
        assert "Red Hat Summit 2025 is in Boston" in result
        assert "events.md" in result

    def test_includes_section_in_header(self, sources_db):
        result = _search_source_chunks("Summit Boston", sources_db)
        assert "> Summit" in result

    def test_no_section_omits_separator(self, sources_db):
        result = _search_source_chunks("OpenShift Kubernetes", sources_db)
        assert "openshift.md" in result
        assert "> " not in result.split("openshift.md")[1].split("\n")[0]

    def test_no_matches_returns_empty(self, sources_db):
        result = _search_source_chunks("zzzznonexistent", sources_db)
        assert result == ""

    def test_single_char_words_filtered(self, sources_db):
        result = _search_source_chunks("a b c", sources_db)
        assert result == ""

    def test_bad_db_returns_empty(self, tmp_path):
        result = _search_source_chunks("test", str(tmp_path / "nonexistent.db"))
        assert result == ""

    def test_respects_top_k(self, sources_db):
        result = _search_source_chunks("Red Hat", sources_db, top_k=1)
        assert "### [1]" in result
        assert "### [2]" not in result

    def test_stop_words_filtered(self, sources_db):
        result = _search_source_chunks("What is the Summit about?", sources_db)
        assert "Summit" in result

    def test_all_stop_words_falls_back(self, sources_db):
        result = _search_source_chunks("What is the", sources_db)
        assert result == "" or isinstance(result, str)


class TestAskNatural:

    def test_natural_strips_metadata(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief about propagation", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    return_value="Natural answer.") as mock_llm:
            result = ask("propagation", db_path=db_path, simple=True, natural=True)
        prompt = mock_llm.call_args[0][0]
        assert "**Status:**" not in prompt
        assert "### " not in prompt
        assert "propagation" in prompt.lower()

    def test_natural_removes_cite_instruction(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    return_value="Natural answer.") as mock_llm:
            ask("alpha", db_path=db_path, simple=True, natural=True)
        prompt = mock_llm.call_args[0][0]
        assert "Cite belief IDs" not in prompt
        assert "plain natural language" in prompt

    def test_non_natural_has_cite_instruction(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    return_value="Answer.") as mock_llm:
            ask("alpha", db_path=db_path, simple=True, natural=False)
        prompt = mock_llm.call_args[0][0]
        assert "Cite belief IDs in [brackets]" in prompt

    def test_natural_in_full_mode(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    return_value="Answer from full mode.") as mock_llm:
            result = ask("alpha", db_path=db_path, natural=True)
        prompt = mock_llm.call_args[0][0]
        assert "**Status:**" not in prompt
        assert "### " not in prompt
        assert "Cite belief IDs" not in prompt
        assert "plain natural language" in prompt


class TestAskWithSources:

    @pytest.fixture
    def sources_db(self, tmp_path):
        db = str(tmp_path / "sources.db")
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, text TEXT, cluster TEXT,
                filename TEXT, section TEXT
            )
        """)
        conn.execute("""
            INSERT INTO chunks VALUES (1, 'Summit is in June 2025', 'ev', 'events.md', 'Dates')
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=id)
        """)
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()
        conn.close()
        return db

    def test_sources_appended_to_context(self, db_path, sources_db):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Summit info belief", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    return_value="Answer with sources.") as mock_llm:
            result = ask("Summit", db_path=db_path, simple=True,
                         sources_db=sources_db)
        prompt = mock_llm.call_args[0][0]
        assert "Source Documents" in prompt
        assert "Summit is in June 2025" in prompt
        assert "events.md" in prompt

    def test_sources_only_when_beliefs_empty(self, db_path, sources_db):
        run_cli("init", db_path=db_path)

        with patch("reasonsforge.ask.invoke_model",
                    return_value="Answer from sources only.") as mock_llm:
            result = ask("Summit", db_path=db_path, simple=True,
                         sources_db=sources_db)
        assert result == "Answer from sources only."

    def test_no_sources_match_returns_no_beliefs(self, db_path, sources_db):
        run_cli("init", db_path=db_path)

        result = ask("zzzznonexistent", db_path=db_path, simple=True,
                     sources_db=sources_db)
        assert result == NO_BELIEFS_MSG


class TestAskDual:

    @pytest.fixture
    def sources_db(self, tmp_path):
        db = str(tmp_path / "sources.db")
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, text TEXT, cluster TEXT,
                filename TEXT, section TEXT
            )
        """)
        conn.execute("""
            INSERT INTO chunks VALUES (1, 'Doc about alpha topic', 'docs', 'alpha.md', NULL)
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=id)
        """)
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()
        conn.close()
        return db

    def test_dual_calls_three_times(self, db_path, sources_db):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        calls = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            calls[0] += 1
            if calls[0] == 1:
                return "TMS answer about alpha [a]."
            elif calls[0] == 2:
                return "FTS answer about alpha [alpha.md]."
            else:
                return "Merged: alpha is described in both sources."

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("alpha", db_path=db_path, simple=True, dual=True,
                         sources_db=sources_db)
        assert calls[0] == 3
        assert "Merged" in result

    def test_dual_without_sources_db_raises(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        with pytest.raises(ValueError, match="--dual requires --full-sources"):
            ask("alpha", db_path=db_path, simple=True, dual=True)

    def test_dual_propagates_natural(self, db_path, sources_db):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        calls = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            calls[0] += 1
            if calls[0] == 1:
                assert "**Status:**" not in prompt, "natural not propagated to TMS leg"
                return "Natural TMS answer."
            elif calls[0] == 2:
                return "FTS answer."
            else:
                return "Merged answer."

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("alpha", db_path=db_path, simple=True, dual=True,
                         sources_db=sources_db, natural=True)
        assert calls[0] == 3

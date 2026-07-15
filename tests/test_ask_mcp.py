"""Tests for MCP server integration in ask."""

import json
import sqlite3
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from reasonsforge.ask import (
    _build_tools_section,
    _build_mcp_instructions,
    build_ask_prompt,
    extract_tool_call,
    ask,
)
from reasonsforge.cli import main


class FakeBridge:
    """Minimal McpBridge substitute for testing."""

    def __init__(self, tools=None, instructions=""):
        self._tools = tools or []
        self._instructions = instructions

    def list_tools(self):
        return self._tools

    def get_instructions(self):
        return self._instructions

    def call_tool(self, name, arguments):
        return json.dumps({"tool": name, "args": arguments, "result": "ok"})


class TestBuildToolsSection:

    def test_no_mcp_servers(self):
        section = _build_tools_section([])
        assert "search_beliefs" in section

    def test_includes_mcp_tools(self):
        bridge = FakeBridge(tools=[
            {
                "name": "execute_query",
                "description": "Execute a read-only SQL query",
                "input_schema": {
                    "properties": {
                        "sql": {"description": "A single SELECT statement", "type": "string"}
                    }
                },
            }
        ])
        section = _build_tools_section([bridge])
        assert "search_beliefs" in section
        assert "execute_query" in section
        assert "SELECT" in section

    def test_multiple_servers(self):
        b1 = FakeBridge(tools=[
            {"name": "tool_a", "description": "Tool A", "input_schema": {"properties": {}}},
        ])
        b2 = FakeBridge(tools=[
            {"name": "tool_b", "description": "Tool B", "input_schema": {"properties": {}}},
        ])
        section = _build_tools_section([b1, b2])
        assert "tool_a" in section
        assert "tool_b" in section

    def test_no_params_no_trailing_comma(self):
        bridge = FakeBridge(tools=[
            {"name": "list_tables", "description": "List tables", "input_schema": {"properties": {}}},
        ])
        section = _build_tools_section([bridge])
        assert '{"tool": "list_tables"}' in section
        assert '{"tool": "list_tables", }' not in section


class TestBuildMcpInstructions:

    def test_no_instructions(self):
        bridge = FakeBridge(instructions="")
        result = _build_mcp_instructions([bridge])
        assert result == ""

    def test_collects_instructions(self):
        b1 = FakeBridge(instructions="Use mart X for sales data")
        b2 = FakeBridge(instructions="Use API Y for user data")
        result = _build_mcp_instructions([b1, b2])
        assert "mart X" in result
        assert "API Y" in result


class TestAskPromptWithMcp:

    def test_default_tools_section(self):
        prompt = build_ask_prompt("question", "context")
        assert "one tool available" in prompt
        assert '{"tool": "search_beliefs"' in prompt
        assert '{{' not in prompt

    def test_custom_tools_section(self):
        prompt = build_ask_prompt("question", "context",
                                  tools_section="Custom tools here")
        assert "Custom tools here" in prompt
        assert "one tool available" not in prompt

    def test_mcp_instructions_injected(self):
        prompt = build_ask_prompt("question", "context",
                                  mcp_instructions="Use snowflake for queries")
        assert "Data Source Instructions" in prompt
        assert "Use snowflake for queries" in prompt

    def test_no_mcp_instructions(self):
        prompt = build_ask_prompt("question", "context", mcp_instructions="")
        assert "Data Source Instructions" not in prompt


class TestExtractToolCallMcp:

    def test_mcp_tool_call(self):
        text = '{"tool": "execute_query", "sql": "SELECT 1"}'
        result = extract_tool_call(text)
        assert result["tool"] == "execute_query"
        assert result["sql"] == "SELECT 1"

    def test_search_beliefs_unchanged(self):
        text = '{"tool": "search_beliefs", "query": "retraction"}'
        result = extract_tool_call(text)
        assert result["tool"] == "search_beliefs"
        assert result["query"] == "retraction"


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
        except SystemExit:
            pass


class TestAskWithMcpDispatch:

    def test_mcp_tool_dispatched(self, db_path):
        """ask() dispatches MCP tool calls to the bridge."""
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        bridge = FakeBridge(
            tools=[{"name": "run_query", "description": "Run SQL",
                    "input_schema": {"properties": {"sql": {"description": "SQL", "type": "string"}}}}],
        )
        calls = []
        orig_call = bridge.call_tool

        def tracking_call(name, args):
            calls.append((name, args))
            return orig_call(name, args)

        bridge.call_tool = tracking_call

        responses = [
            '{"tool": "run_query", "sql": "SELECT 1"}',
            "The query returned 1.",
        ]
        idx = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            r = responses[idx[0]]
            idx[0] += 1
            return r

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("run a query", db_path=db_path, mcp_servers=[bridge])

        assert result == "The query returned 1."
        assert len(calls) == 1
        assert calls[0] == ("run_query", {"sql": "SELECT 1"})

    def test_mcp_tool_error_handled(self, db_path):
        """MCP tool errors are captured in tool history, not raised."""
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        bridge = FakeBridge(
            tools=[{"name": "bad_tool", "description": "Fails",
                    "input_schema": {"properties": {}}}],
        )
        bridge.call_tool = lambda name, args: (_ for _ in ()).throw(
            RuntimeError("connection lost"))

        responses = [
            '{"tool": "bad_tool"}',
            "The tool failed, but alpha is relevant.",
        ]
        idx = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            r = responses[idx[0]]
            idx[0] += 1
            return r

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("alpha", db_path=db_path, mcp_servers=[bridge])

        assert "alpha" in result.lower()

    def test_max_iterations_bumped_with_mcp(self, db_path):
        """With MCP servers, max iterations increases to 5."""
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Alpha belief", db_path=db_path)

        bridge = FakeBridge(
            tools=[{"name": "some_tool", "description": "A tool",
                    "input_schema": {"properties": {}}}],
        )

        calls = [0]

        def mock_invoke(prompt, model="claude", timeout=300):
            calls[0] += 1
            if calls[0] <= 5:
                return '{"tool": "search_beliefs", "query": "more"}'
            return "Final answer."

        with patch("reasonsforge.ask.invoke_model", side_effect=mock_invoke):
            result = ask("alpha", db_path=db_path, mcp_servers=[bridge])

        assert calls[0] == 6
        assert "Final answer" in result

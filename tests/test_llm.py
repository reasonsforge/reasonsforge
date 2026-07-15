"""Tests for the shared LLM invocation module."""

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

import reasonsforge.llm as llm_module
from reasonsforge.llm import (
    _get_langfuse_handler,
    _invoke_api,
    _parse_cli_json,
    _record_cost,
    format_cost_summary,
    get_cost_summary,
    invoke_model,
    reset_cost_tracker,
    resolve_model_cmd,
)


class TestResolveModelCmd:

    def test_resolve_claude(self):
        assert resolve_model_cmd("claude") == ["claude", "-p", "--output-format", "json"]

    def test_resolve_gemini(self):
        assert resolve_model_cmd("gemini") == ["gemini", "--skip-trust", "-o", "json", "-p", ""]

    def test_resolve_gemini_submodel(self):
        assert resolve_model_cmd("gemini:gemini-2.5-flash") == [
            "gemini", "--skip-trust", "-m", "gemini-2.5-flash", "-o", "json", "-p", ""
        ]

    def test_resolve_gemini_submodel_short(self):
        assert resolve_model_cmd("gemini:flash") == [
            "gemini", "--skip-trust", "-m", "flash", "-o", "json", "-p", ""
        ]

    def test_resolve_ollama_model(self):
        assert resolve_model_cmd("ollama:gemma3:4b") == ["ollama", "run", "gemma3:4b"]

    def test_resolve_ollama_with_tag(self):
        assert resolve_model_cmd("ollama:qwen3.5:27b") == ["ollama", "run", "qwen3.5:27b"]

    def test_resolve_claude_submodel(self):
        assert resolve_model_cmd("claude:sonnet") == ["claude", "-p", "--model", "sonnet", "--output-format", "json"]

    def test_resolve_claude_submodel_full_name(self):
        assert resolve_model_cmd("claude:claude-sonnet-4-6") == ["claude", "-p", "--model", "claude-sonnet-4-6", "--output-format", "json"]

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_model_cmd("gpt-4")


class TestInvokeModel:

    def test_missing_binary_raises(self):
        with patch("reasonsforge.llm.shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="not found in PATH"):
                invoke_model("hello", model="claude")

    def test_invokes_subprocess(self):
        json_out = json.dumps({"result": "response", "usage": {}, "total_cost_usd": 0.0})
        mock_result = type("Result", (), {"returncode": 0, "stdout": json_out, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run:
            result = invoke_model("hello", model="claude", timeout=60)
            assert result == "response"
            mock_run.assert_called_once()
            args = mock_run.call_args
            assert args[0][0] == ["claude", "-p", "--output-format", "json"]
            assert args[1]["input"] == "hello"
            assert args[1]["timeout"] == 60

    def test_nonzero_exit_raises(self):
        mock_result = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "error msg"})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="claude failed"):
                invoke_model("hello", model="claude")

    def test_ollama_command(self):
        mock_result = type("Result", (), {"returncode": 0, "stdout": "ollama response", "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/ollama"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run:
            result = invoke_model("hello", model="ollama:gemma3:4b")
            assert result == "ollama response"
            args = mock_run.call_args
            assert args[0][0] == ["ollama", "run", "gemma3:4b"]

    def test_ollama_strips_thinking_output(self):
        thinking = "Thinking...\nsome internal reasoning\n...done thinking.\nThe actual answer."
        mock_result = type("Result", (), {"returncode": 0, "stdout": thinking, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/ollama"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = invoke_model("hello", model="ollama:qwen3:4b")
            assert result == "The actual answer."

    def test_ollama_no_thinking_markers_unchanged(self):
        output = "Just a normal response."
        mock_result = type("Result", (), {"returncode": 0, "stdout": output, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/ollama"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = invoke_model("hello", model="ollama:qwen3:4b")
            assert result == "Just a normal response."

    def test_ollama_incomplete_thinking_unchanged(self):
        output = "Thinking...\nsome reasoning but no end marker"
        mock_result = type("Result", (), {"returncode": 0, "stdout": output, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/ollama"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = invoke_model("hello", model="ollama:qwen3:4b")
            assert result == output

    def test_claude_does_not_strip_thinking(self):
        output = "Thinking...\nsome reasoning\n...done thinking.\nAnswer."
        json_out = json.dumps({"result": output, "usage": {}, "total_cost_usd": 0.0})
        mock_result = type("Result", (), {"returncode": 0, "stdout": json_out, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = invoke_model("hello", model="claude")
            assert result == output

    def test_strips_claudecode_env(self):
        json_out = json.dumps({"result": "ok", "usage": {}, "total_cost_usd": 0.0})
        mock_result = type("Result", (), {"returncode": 0, "stdout": json_out, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run, \
             patch.dict("os.environ", {"CLAUDECODE": "1", "HOME": "/home/test"}):
            invoke_model("hello", model="claude")
            env = mock_run.call_args[1]["env"]
            assert "CLAUDECODE" not in env
            assert "HOME" in env


class TestResolveModelCmdApiModels:

    def test_api_prefix_not_in_resolve(self):
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_model_cmd("api:claude-sonnet-4-20250514")

    def test_vertex_prefix_not_in_resolve(self):
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_model_cmd("vertex:claude-sonnet-4-20250514")

    def test_error_message_lists_api_models(self):
        with pytest.raises(ValueError, match="api:<model>"):
            resolve_model_cmd("gpt-4")
        with pytest.raises(ValueError, match="vertex:<model>"):
            resolve_model_cmd("gpt-4")


class TestInvokeModelApiDispatch:

    def test_api_prefix_dispatches(self):
        with patch("reasonsforge.llm._invoke_api", return_value="api response") as mock:
            result = invoke_model("hello", model="api:claude-sonnet-4-20250514")
            assert result == "api response"
            mock.assert_called_once_with("hello", "api:claude-sonnet-4-20250514", 300)

    def test_vertex_prefix_dispatches(self):
        with patch("reasonsforge.llm._invoke_api", return_value="vertex response") as mock:
            result = invoke_model("hello", model="vertex:claude-sonnet-4-20250514", timeout=60)
            assert result == "vertex response"
            mock.assert_called_once_with("hello", "vertex:claude-sonnet-4-20250514", 60)

    def test_claude_still_uses_subprocess(self):
        json_out = json.dumps({"result": "cli response", "usage": {}, "total_cost_usd": 0.0})
        mock_result = type("Result", (), {"returncode": 0, "stdout": json_out, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = invoke_model("hello", model="claude")
            assert result == "cli response"


class TestInvokeApi:

    def test_api_invokes_chat_anthropic(self):
        mock_response = MagicMock()
        mock_response.content = "test response"
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response

        with patch.object(llm_module, "_HAS_LANGCHAIN_ANTHROPIC", True), \
             patch.object(llm_module, "ChatAnthropic", return_value=mock_model) as MockChat, \
             patch.object(llm_module, "_langfuse_checked", True), \
             patch.object(llm_module, "_langfuse_handler", None):
            result = _invoke_api("hello", "api:claude-sonnet-4-20250514", timeout=120)

        assert result == "test response"
        MockChat.assert_called_once_with(model="claude-sonnet-4-20250514", timeout=120.0)
        mock_model.invoke.assert_called_once_with("hello", config={})

    def test_vertex_invokes_chat_anthropic_vertex(self):
        mock_response = MagicMock()
        mock_response.content = "vertex response"
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response

        with patch.object(llm_module, "_HAS_LANGCHAIN_VERTEX", True), \
             patch.object(llm_module, "ChatAnthropicVertex", return_value=mock_model) as MockChat, \
             patch.object(llm_module, "_langfuse_checked", True), \
             patch.object(llm_module, "_langfuse_handler", None), \
             patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "my-proj", "GOOGLE_CLOUD_REGION": "us-west1"}):
            result = _invoke_api("hello", "vertex:claude-sonnet-4-20250514", timeout=60)

        assert result == "vertex response"
        MockChat.assert_called_once_with(
            model_name="claude-sonnet-4-20250514",
            project="my-proj", location="us-west1",
            request_timeout=60.0,
        )

    def test_vertex_default_region(self):
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response

        with patch.object(llm_module, "_HAS_LANGCHAIN_VERTEX", True), \
             patch.object(llm_module, "ChatAnthropicVertex", return_value=mock_model) as MockChat, \
             patch.object(llm_module, "_langfuse_checked", True), \
             patch.object(llm_module, "_langfuse_handler", None), \
             patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "my-proj"}, clear=False):
            os_env = dict(**os.environ)
            os_env.pop("GOOGLE_CLOUD_REGION", None)
            with patch.dict("os.environ", os_env, clear=True):
                _invoke_api("hello", "vertex:claude-sonnet-4-20250514")

        assert MockChat.call_args[1]["location"] == "us-east5"

    def test_api_missing_dep_raises(self):
        with patch.object(llm_module, "_HAS_LANGCHAIN_ANTHROPIC", False):
            with pytest.raises(ImportError, match="langchain-anthropic"):
                _invoke_api("hello", "api:claude-sonnet-4-20250514")

    def test_vertex_missing_dep_raises(self):
        with patch.object(llm_module, "_HAS_LANGCHAIN_VERTEX", False):
            with pytest.raises(ImportError, match="langchain-google-vertexai"):
                _invoke_api("hello", "vertex:claude-sonnet-4-20250514")

    def test_vertex_missing_project_raises(self):
        with patch.object(llm_module, "_HAS_LANGCHAIN_VERTEX", True), \
             patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
                _invoke_api("hello", "vertex:claude-sonnet-4-20250514")

    def test_timeout_reraises_as_timeout_expired(self):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Request timed out timeout")

        with patch.object(llm_module, "_HAS_LANGCHAIN_ANTHROPIC", True), \
             patch.object(llm_module, "ChatAnthropic", return_value=mock_model), \
             patch.object(llm_module, "_langfuse_checked", True), \
             patch.object(llm_module, "_langfuse_handler", None):
            with pytest.raises(subprocess.TimeoutExpired):
                _invoke_api("hello", "api:claude-sonnet-4-20250514", timeout=30)

    def test_api_error_reraises_as_runtime_error(self):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Authentication failed")

        with patch.object(llm_module, "_HAS_LANGCHAIN_ANTHROPIC", True), \
             patch.object(llm_module, "ChatAnthropic", return_value=mock_model), \
             patch.object(llm_module, "_langfuse_checked", True), \
             patch.object(llm_module, "_langfuse_handler", None):
            with pytest.raises(RuntimeError, match="api:claude-sonnet-4-20250514 failed"):
                _invoke_api("hello", "api:claude-sonnet-4-20250514")

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown API prefix"):
            _invoke_api("hello", "openai:gpt-4")


class TestLangfuseHandler:

    def setup_method(self):
        llm_module._langfuse_handler = None
        llm_module._langfuse_checked = False

    def teardown_method(self):
        llm_module._langfuse_handler = None
        llm_module._langfuse_checked = False

    def test_handler_created_when_env_set(self):
        mock_handler = MagicMock()
        with patch.object(llm_module, "_HAS_LANGFUSE", True), \
             patch.object(llm_module, "LangfuseCallbackHandler", return_value=mock_handler), \
             patch.dict("os.environ", {"LANGFUSE_PUBLIC_KEY": "pk-test"}):
            result = _get_langfuse_handler()
            assert result is mock_handler

    def test_handler_none_when_env_missing(self):
        with patch.object(llm_module, "_HAS_LANGFUSE", True), \
             patch.dict("os.environ", {}, clear=True):
            result = _get_langfuse_handler()
            assert result is None

    def test_handler_none_when_dep_missing(self):
        with patch.object(llm_module, "_HAS_LANGFUSE", False), \
             patch.dict("os.environ", {"LANGFUSE_PUBLIC_KEY": "pk-test"}):
            result = _get_langfuse_handler()
            assert result is None

    def test_handler_singleton(self):
        mock_handler = MagicMock()
        with patch.object(llm_module, "_HAS_LANGFUSE", True), \
             patch.object(llm_module, "LangfuseCallbackHandler", return_value=mock_handler) as MockCB, \
             patch.dict("os.environ", {"LANGFUSE_PUBLIC_KEY": "pk-test"}):
            first = _get_langfuse_handler()
            second = _get_langfuse_handler()
            assert first is second
            MockCB.assert_called_once()

    def test_langfuse_callback_passed_to_invoke(self):
        mock_handler = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "traced response"
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response

        with patch.object(llm_module, "_HAS_LANGCHAIN_ANTHROPIC", True), \
             patch.object(llm_module, "ChatAnthropic", return_value=mock_model), \
             patch.object(llm_module, "_langfuse_checked", True), \
             patch.object(llm_module, "_langfuse_handler", mock_handler):
            result = _invoke_api("hello", "api:claude-sonnet-4-20250514")

        assert result == "traced response"
        call_config = mock_model.invoke.call_args[1]["config"]
        assert call_config["callbacks"] == [mock_handler]


class TestParseCliJson:

    def test_claude_json(self):
        data = {
            "result": "The answer is 42.",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "total_cost_usd": 0.0021,
        }
        reset_cost_tracker()
        text = _parse_cli_json(json.dumps(data), "claude")
        assert text == "The answer is 42."
        s = get_cost_summary()
        assert s["input_tokens"] == 100
        assert s["output_tokens"] == 50
        assert s["total_cost_usd"] == pytest.approx(0.0021)
        assert s["calls"] == 1

    def test_claude_json_with_cache_tokens(self):
        data = {
            "result": "cached",
            "usage": {
                "input_tokens": 50,
                "output_tokens": 30,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 100,
            },
            "total_cost_usd": 0.01,
        }
        reset_cost_tracker()
        text = _parse_cli_json(json.dumps(data), "claude:sonnet")
        assert text == "cached"
        s = get_cost_summary()
        assert s["input_tokens"] == 350  # 50 + 200 + 100

    def test_gemini_json(self):
        data = {
            "response": "Gemini says hi.",
            "stats": {
                "models": {
                    "gemini-2.5-flash": {
                        "tokens": {"input": 80, "candidates": 40}
                    }
                }
            },
        }
        reset_cost_tracker()
        text = _parse_cli_json(json.dumps(data), "gemini")
        assert text == "Gemini says hi."
        s = get_cost_summary()
        assert s["input_tokens"] == 80
        assert s["output_tokens"] == 40
        assert s["total_cost_usd"] == 0.0

    def test_gemini_submodel_json(self):
        data = {
            "response": "Flash response.",
            "stats": {"models": {"flash": {"tokens": {"input": 10, "candidates": 5}}}},
        }
        reset_cost_tracker()
        text = _parse_cli_json(json.dumps(data), "gemini:flash")
        assert text == "Flash response."
        s = get_cost_summary()
        assert s["input_tokens"] == 10
        assert s["output_tokens"] == 5

    def test_invalid_json_returns_raw(self):
        reset_cost_tracker()
        text = _parse_cli_json("not json at all", "claude")
        assert text == "not json at all"
        assert get_cost_summary()["calls"] == 0

    def test_json_missing_result_returns_raw(self):
        reset_cost_tracker()
        text = _parse_cli_json(json.dumps({"other": "data"}), "claude")
        assert text == json.dumps({"other": "data"})

    def test_json_array_returns_raw(self):
        raw = json.dumps([{"id": "a", "valid": True}])
        reset_cost_tracker()
        text = _parse_cli_json(raw, "claude")
        assert text == raw
        assert get_cost_summary()["calls"] == 0


class TestCostTracker:

    def setup_method(self):
        reset_cost_tracker()

    def test_reset(self):
        _record_cost("claude", 100, 50, 0.01)
        reset_cost_tracker()
        s = get_cost_summary()
        assert s["calls"] == 0
        assert s["input_tokens"] == 0
        assert s["output_tokens"] == 0
        assert s["total_cost_usd"] == 0.0
        assert s["by_model"] == {}

    def test_record_accumulates(self):
        _record_cost("claude", 100, 50, 0.01)
        _record_cost("claude", 200, 100, 0.02)
        s = get_cost_summary()
        assert s["calls"] == 2
        assert s["input_tokens"] == 300
        assert s["output_tokens"] == 150
        assert s["total_cost_usd"] == pytest.approx(0.03)

    def test_by_model_tracking(self):
        _record_cost("claude", 100, 50, 0.01)
        _record_cost("gemini", 80, 40, 0.0)
        s = get_cost_summary()
        assert "claude" in s["by_model"]
        assert "gemini" in s["by_model"]
        assert s["by_model"]["claude"]["calls"] == 1
        assert s["by_model"]["gemini"]["input_tokens"] == 80

    def test_format_empty(self):
        assert format_cost_summary() == ""

    def test_format_with_cost(self):
        _record_cost("claude", 1000, 500, 0.0123)
        result = format_cost_summary()
        assert "$0.0123" in result
        assert "1,000 input" in result
        assert "500 output" in result
        assert "1 call(s)" in result

    def test_format_without_cost(self):
        _record_cost("gemini", 100, 50, 0.0)
        result = format_cost_summary()
        assert "$" not in result
        assert "100 input" in result

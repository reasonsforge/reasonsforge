"""Tests for import-api and export-api — agentic-mind-service sync."""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reasonsforge import api
from reasonsforge.mind_service import _resolve_config, fetch_export, push_belief


SAMPLE_EXPORT = {
    "meta": {"schema_version": "1.0", "project_name": "agent-beliefs"},
    "nodes": {
        "a": {"text": "Node A", "truth_value": "IN", "justifications": [],
               "source": "", "source_hash": "", "date": "", "metadata": {}},
        "b": {"text": "Node B", "truth_value": "IN",
               "justifications": [{"type": "SL", "antecedents": ["a"], "outlist": [], "label": ""}],
               "source": "obs", "source_hash": "", "date": "", "metadata": {}},
    },
    "nogoods": [],
    "repos": {},
}


def _mock_response(data):
    body = json.dumps(data).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestResolveConfig:

    def test_explicit_params(self):
        url, aid, key = _resolve_config("http://localhost", "agent-1", "key-1")
        assert url == "http://localhost"
        assert aid == "agent-1"
        assert key == "key-1"

    def test_env_vars(self, monkeypatch):
        monkeypatch.setenv("MIND_SERVICE_URL", "http://env-url")
        monkeypatch.setenv("MIND_AGENT_ID", "env-agent")
        monkeypatch.setenv("MIND_API_KEY", "env-key")
        url, aid, key = _resolve_config()
        assert url == "http://env-url"
        assert aid == "env-agent"
        assert key == "env-key"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MIND_SERVICE_URL", "http://env")
        url, _, _ = _resolve_config(url="http://explicit", agent_id="a", api_key="k")
        assert url == "http://explicit"

    def test_missing_url_raises(self, monkeypatch):
        monkeypatch.delenv("MIND_SERVICE_URL", raising=False)
        monkeypatch.delenv("MIND_AGENT_ID", raising=False)
        monkeypatch.delenv("MIND_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="url"):
            _resolve_config()

    def test_missing_agent_id_raises(self, monkeypatch):
        monkeypatch.delenv("MIND_AGENT_ID", raising=False)
        monkeypatch.delenv("MIND_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="agent-id"):
            _resolve_config(url="http://x")

    def test_trailing_slash_stripped(self):
        url, _, _ = _resolve_config("http://localhost/", "a", "k")
        assert url == "http://localhost"


class TestFetchExport:

    @patch("reasonsforge.mind_service.urlopen")
    def test_correct_url(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(SAMPLE_EXPORT)
        fetch_export("http://localhost", "agent-1", "key-1")
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost/api/agents/agent-1/export"

    @patch("reasonsforge.mind_service.urlopen")
    def test_auth_header(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(SAMPLE_EXPORT)
        fetch_export("http://localhost", "agent-1", "my-key")
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer my-key"

    @patch("reasonsforge.mind_service.urlopen")
    def test_returns_json_string(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(SAMPLE_EXPORT)
        result = fetch_export("http://localhost", "a", "k")
        data = json.loads(result)
        assert "nodes" in data

    @patch("reasonsforge.mind_service.urlopen")
    def test_401_raises(self, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError("url", 401, "Unauthorized", {}, BytesIO(b""))
        with pytest.raises(RuntimeError, match="Authentication failed"):
            fetch_export("http://localhost", "a", "k")

    @patch("reasonsforge.mind_service.urlopen")
    def test_404_raises(self, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError("url", 404, "Not Found", {}, BytesIO(b""))
        with pytest.raises(RuntimeError, match="Agent not found"):
            fetch_export("http://localhost", "agent-1", "k")


class TestPushBelief:

    @patch("reasonsforge.mind_service.urlopen")
    def test_correct_url(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"node_id": "a", "truth_value": "IN"})
        push_belief("http://localhost", "agent-1", "key", "a", "text A")
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost/api/agents/agent-1/beliefs"

    @patch("reasonsforge.mind_service.urlopen")
    def test_post_method(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"node_id": "a"})
        push_belief("http://localhost", "a", "k", "node-1", "text")
        req = mock_urlopen.call_args[0][0]
        assert req.method == "POST"

    @patch("reasonsforge.mind_service.urlopen")
    def test_json_body(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"node_id": "a"})
        push_belief("http://localhost", "a", "k", "node-1", "some text", sl="x,y", source="obs")
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["node_id"] == "node-1"
        assert body["text"] == "some text"
        assert body["sl"] == "x,y"
        assert body["source"] == "obs"

    @patch("reasonsforge.mind_service.urlopen")
    def test_omits_empty_sl(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"node_id": "a"})
        push_belief("http://localhost", "a", "k", "node-1", "text")
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert "sl" not in body

    @patch("reasonsforge.mind_service.urlopen")
    def test_auth_header(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"node_id": "a"})
        push_belief("http://localhost", "a", "my-key", "node-1", "text")
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer my-key"


class TestImportApi:

    @patch("reasonsforge.mind_service.urlopen")
    def test_import_with_init(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response(SAMPLE_EXPORT)
        db = str(tmp_path / "test.db")
        result = api.import_api(
            url="http://localhost", agent_id="a", api_key="k",
            init=True, db_path=db)
        assert result["nodes_imported"] == 2
        assert result["nogoods_imported"] == 0
        assert Path(db).exists()

    @patch("reasonsforge.mind_service.urlopen")
    def test_import_into_existing_db(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response(SAMPLE_EXPORT)
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        result = api.import_api(
            url="http://localhost", agent_id="a", api_key="k", db_path=db)
        assert result["nodes_imported"] == 2

    @patch("reasonsforge.mind_service.urlopen")
    def test_missing_config_raises(self, mock_urlopen, monkeypatch):
        monkeypatch.delenv("MIND_SERVICE_URL", raising=False)
        monkeypatch.delenv("MIND_AGENT_ID", raising=False)
        monkeypatch.delenv("MIND_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="Missing required config"):
            api.import_api()


class TestExportApi:

    @patch("reasonsforge.mind_service.urlopen")
    def test_export_pushes_all_nodes(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response({"node_id": "a", "truth_value": "IN"})
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Node A", db_path=db)
        api.add_node("b", "Node B", db_path=db)
        result = api.export_api(
            url="http://localhost", agent_id="agent-1", api_key="k", db_path=db)
        assert result["nodes_exported"] == 2
        assert result["errors"] == 0

    @patch("reasonsforge.mind_service.urlopen")
    def test_export_sends_sl_justifications(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response({"node_id": "d", "truth_value": "IN"})
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Premise B", db_path=db)
        api.add_node("d", "Derived", sl="a,b", db_path=db)
        result = api.export_api(
            url="http://localhost", agent_id="agent-1", api_key="k", db_path=db)
        assert result["nodes_exported"] == 3
        # Check the derived node's POST body has sl
        calls = mock_urlopen.call_args_list
        bodies = [json.loads(c[0][0].data.decode("utf-8")) for c in calls]
        derived_body = [b for b in bodies if b["node_id"] == "d"][0]
        assert "a" in derived_body["sl"]
        assert "b" in derived_body["sl"]

    @patch("reasonsforge.mind_service.urlopen")
    def test_export_counts_errors(self, mock_urlopen, tmp_path):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = [
            _mock_response({"node_id": "a"}),
            HTTPError("url", 500, "Server Error", {}, BytesIO(b"")),
        ]
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Node A", db_path=db)
        api.add_node("b", "Node B", db_path=db)
        result = api.export_api(
            url="http://localhost", agent_id="agent-1", api_key="k", db_path=db)
        assert result["nodes_exported"] == 1
        assert result["errors"] == 1

    @patch("reasonsforge.mind_service.urlopen")
    def test_missing_config_raises(self, mock_urlopen, monkeypatch):
        monkeypatch.delenv("MIND_SERVICE_URL", raising=False)
        monkeypatch.delenv("MIND_AGENT_ID", raising=False)
        monkeypatch.delenv("MIND_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="Missing required config"):
            api.export_api()

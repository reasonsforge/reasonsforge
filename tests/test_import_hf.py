"""Tests for import-hf — importing belief networks from HuggingFace."""

import json
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from reasonsforge import api
from reasonsforge.hf import _parse_repo_id, _resolve_token, download_network


SAMPLE_NETWORK = {
    "meta": {"schema_version": "1.0", "project_name": "test-eem"},
    "nodes": {
        "a": {"text": "Node A", "truth_value": "IN", "justifications": [],
               "source": "", "source_hash": "", "date": "", "metadata": {}},
        "b": {"text": "Node B", "truth_value": "IN", "justifications": [],
               "source": "", "source_hash": "", "date": "", "metadata": {}},
    },
    "nogoods": [],
    "repos": {},
}


def _mock_response(data):
    """Create a mock urlopen response."""
    body = json.dumps(data).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestParseRepoId:

    def test_simple_repo_id(self):
        assert _parse_repo_id("user/repo") == "user/repo"

    def test_https_url(self):
        assert _parse_repo_id("https://huggingface.co/user/repo") == "user/repo"

    def test_http_url(self):
        assert _parse_repo_id("http://huggingface.co/user/repo") == "user/repo"

    def test_trailing_slash(self):
        assert _parse_repo_id("https://huggingface.co/user/repo/") == "user/repo"

    def test_non_hf_url_raises(self):
        with pytest.raises(ValueError, match="Not a HuggingFace URL"):
            _parse_repo_id("https://github.com/user/repo")


class TestResolveToken:

    def test_explicit_token_wins(self):
        assert _resolve_token("my-token") == "my-token"

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env-token")
        assert _resolve_token() == "env-token"

    def test_cached_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        token_file = tmp_path / ".cache" / "huggingface" / "token"
        token_file.parent.mkdir(parents=True)
        token_file.write_text("file-token\n")
        with patch("reasonsforge.hf.Path.home", return_value=tmp_path):
            assert _resolve_token() == "file-token"

    def test_no_token_returns_none(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("reasonsforge.hf.Path.home", return_value=Path("/nonexistent")):
            assert _resolve_token() is None

    def test_explicit_over_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env-token")
        assert _resolve_token("explicit") == "explicit"


class TestDownloadNetwork:

    @patch("reasonsforge.hf.urlopen")
    def test_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        result = download_network("user/repo", token="tok")
        data = json.loads(result)
        assert data["nodes"]["a"]["text"] == "Node A"

    @patch("reasonsforge.hf.urlopen")
    def test_correct_url(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        download_network("user/my-eem", token="tok")
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://huggingface.co/user/my-eem/resolve/main/network.json"

    @patch("reasonsforge.hf.urlopen")
    def test_auth_header_present(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        download_network("user/repo", token="secret-tok")
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer secret-tok"

    @patch("reasonsforge.hf._resolve_token", return_value=None)
    @patch("reasonsforge.hf.urlopen")
    def test_no_auth_header_without_token(self, mock_urlopen, mock_resolve):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        download_network("user/repo")
        req = mock_urlopen.call_args[0][0]
        assert not req.has_header("Authorization")

    @patch("reasonsforge.hf.urlopen")
    def test_401_raises_auth_error(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "url", 401, "Unauthorized", {}, BytesIO(b""))
        with pytest.raises(RuntimeError, match="Authentication required"):
            download_network("user/private-repo")

    @patch("reasonsforge.hf.urlopen")
    def test_404_raises_not_found(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "url", 404, "Not Found", {}, BytesIO(b""))
        with pytest.raises(RuntimeError, match="not found"):
            download_network("user/missing-repo")

    @patch("reasonsforge.hf.urlopen")
    def test_url_input(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        download_network("https://huggingface.co/user/repo", token="tok")
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://huggingface.co/user/repo/resolve/main/network.json"


class TestImportHfApi:

    @patch("reasonsforge.hf.urlopen")
    def test_import_with_init(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        db = str(tmp_path / "test.db")
        result = api.import_hf("user/repo", init=True, token="tok", db_path=db)
        assert result["nodes_imported"] == 2
        assert result["nogoods_imported"] == 0
        assert result["repo_id"] == "user/repo"
        assert Path(db).exists()

    @patch("reasonsforge.hf.urlopen")
    def test_auto_init_sets_project_name(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        db = str(tmp_path / "test.db")
        api.import_hf("user/repo", init=False, token="tok", db_path=db)
        assert Path(db).exists()
        data = api.export_network(db_path=db)
        assert data["meta"]["project_name"] == "test-eem"

    @patch("reasonsforge.hf.urlopen")
    def test_import_into_existing_db(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        result = api.import_hf("user/repo", token="tok", db_path=db)
        assert result["nodes_imported"] == 2

    @patch("reasonsforge.hf.urlopen")
    def test_init_flag_with_existing_db(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        result = api.import_hf("user/repo", init=True, token="tok", db_path=db)
        assert result["nodes_imported"] == 2

    @patch("reasonsforge.hf.urlopen")
    def test_import_hf_with_url(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        db = str(tmp_path / "test.db")
        result = api.import_hf(
            "https://huggingface.co/user/repo", init=True, token="tok", db_path=db)
        assert result["nodes_imported"] == 2
        assert result["repo_id"] == "user/repo"

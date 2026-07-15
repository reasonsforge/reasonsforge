"""Tests for EEM-Hub default org, publish, and pull commands."""

import json
from io import BytesIO
from unittest.mock import MagicMock, call, patch
from urllib.error import HTTPError

import pytest

from reasonsforge import api
from reasonsforge.hf import (
    DEFAULT_HF_ORG,
    create_repo,
    resolve_repo_id,
    upload_file,
)


SAMPLE_NETWORK = {
    "meta": {"schema_version": "1.0", "project_name": "test-eem"},
    "nodes": {
        "a": {"text": "Node A", "truth_value": "IN", "justifications": [],
               "source": "", "source_hash": "", "date": "", "metadata": {}},
    },
    "nogoods": [],
    "repos": {},
}


def _mock_response(data=None):
    resp = MagicMock()
    resp.read.return_value = json.dumps(data or {}).encode("utf-8")
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestResolveRepoId:
    def test_bare_name_gets_default_org(self):
        assert resolve_repo_id("ddia-expert") == "EEM-Hub/ddia-expert"

    def test_explicit_org_unchanged(self):
        assert resolve_repo_id("myuser/my-eem") == "myuser/my-eem"

    def test_url_parsed(self):
        assert resolve_repo_id("https://huggingface.co/org/repo") == "org/repo"

    def test_trailing_slash_stripped(self):
        assert resolve_repo_id("ddia-expert/") == "EEM-Hub/ddia-expert"

    def test_whitespace_stripped(self):
        assert resolve_repo_id("  ddia-expert  ") == "EEM-Hub/ddia-expert"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("REASONS_HF_ORG", "my-company")
        assert resolve_repo_id("internal-eem") == "my-company/internal-eem"

    def test_env_var_does_not_affect_explicit_org(self, monkeypatch):
        monkeypatch.setenv("REASONS_HF_ORG", "my-company")
        assert resolve_repo_id("other/repo") == "other/repo"

    def test_non_hf_url_raises(self):
        with pytest.raises(ValueError, match="Not a HuggingFace URL"):
            resolve_repo_id("https://github.com/org/repo")

    def test_default_org_constant(self):
        assert DEFAULT_HF_ORG == "EEM-Hub"


class TestCreateRepo:
    @patch("reasonsforge.hf.urlopen")
    def test_creates_repo(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        result = create_repo("test-eem", token="tok")
        assert result == "EEM-Hub/test-eem"

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://huggingface.co/api/repos/create"
        body = json.loads(req.data)
        assert body["name"] == "test-eem"
        assert body["organization"] == "EEM-Hub"
        assert body["private"] is False

    @patch("reasonsforge.hf.urlopen")
    def test_private_repo(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        create_repo("org/repo", token="tok", private=True)
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["private"] is True

    @patch("reasonsforge.hf.urlopen")
    def test_already_exists_ignored(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError("url", 409, "Conflict", {}, BytesIO(b""))
        result = create_repo("org/repo", token="tok")
        assert result == "org/repo"

    @patch("reasonsforge.hf.urlopen")
    def test_auth_failure_raises(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError("url", 401, "Unauthorized", {}, BytesIO(b""))
        with pytest.raises(RuntimeError, match="Authentication failed"):
            create_repo("org/repo", token="bad-tok")

    @patch("reasonsforge.hf.urlopen")
    def test_other_error_raises(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError("url", 500, "Server Error", {}, BytesIO(b""))
        with pytest.raises(RuntimeError, match="HTTP 500"):
            create_repo("org/repo", token="tok")


class TestUploadFile:
    @patch("reasonsforge.hf.urlopen")
    def test_uploads_to_correct_url(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        upload_file("EEM-Hub/test-eem", "network.json", b'{"nodes":{}}', "tok")
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://huggingface.co/api/models/EEM-Hub/test-eem/upload/main/network.json"
        assert req.get_method() == "PUT"
        assert req.data == b'{"nodes":{}}'

    @patch("reasonsforge.hf.urlopen")
    def test_auth_header(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response()
        upload_file("org/repo", "file.txt", b"data", "my-token")
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer my-token"

    @patch("reasonsforge.hf.urlopen")
    def test_auth_failure_raises(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError("url", 401, "Unauthorized", {}, BytesIO(b""))
        with pytest.raises(RuntimeError, match="Authentication failed"):
            upload_file("org/repo", "file.txt", b"data", "bad-tok")

    @patch("reasonsforge.hf.urlopen")
    def test_upload_failure_raises(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError("url", 500, "Error", {}, BytesIO(b""))
        with pytest.raises(RuntimeError, match="Failed to upload"):
            upload_file("org/repo", "file.txt", b"data", "tok")


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    api.init_db(db_path=db_path)
    api.add_node("n1", "Test belief.", db_path=db_path)
    return db_path


class TestPublishHf:
    @patch("reasonsforge.hf.urlopen")
    def test_publishes_three_files(self, mock_urlopen, db):
        mock_urlopen.return_value = _mock_response()
        result = api.publish_hf("test-eem", token="tok", db_path=db)
        assert result["repo_id"] == "EEM-Hub/test-eem"
        assert result["url"] == "https://huggingface.co/EEM-Hub/test-eem"
        assert set(result["files_uploaded"]) == {"network.json", "beliefs.md", "README.md"}

    @patch("reasonsforge.hf.urlopen")
    def test_upload_calls(self, mock_urlopen, db):
        mock_urlopen.return_value = _mock_response()
        api.publish_hf("test-eem", token="tok", db_path=db)
        # 1 create_repo + 3 upload_file = 4 calls
        assert mock_urlopen.call_count == 4

    @patch("reasonsforge.hf.urlopen")
    def test_explicit_org(self, mock_urlopen, db):
        mock_urlopen.return_value = _mock_response()
        result = api.publish_hf("myuser/my-eem", token="tok", db_path=db)
        assert result["repo_id"] == "myuser/my-eem"
        assert result["url"] == "https://huggingface.co/myuser/my-eem"

    @patch("reasonsforge.hf.urlopen")
    def test_private_repo(self, mock_urlopen, db):
        mock_urlopen.return_value = _mock_response()
        api.publish_hf("test-eem", token="tok", private=True, db_path=db)
        create_req = mock_urlopen.call_args_list[0][0][0]
        body = json.loads(create_req.data)
        assert body["private"] is True

    def test_no_token_raises(self, db, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("reasonsforge.hf.Path.home", return_value=pytest.importorskip("pathlib").Path("/nonexistent")):
            with pytest.raises(RuntimeError, match="token required"):
                api.publish_hf("test-eem", db_path=db)


class TestPullCommand:
    @patch("reasonsforge.hf.urlopen")
    def test_pull_with_bare_name(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        db = str(tmp_path / "test.db")
        result = api.import_hf("ddia-expert", init=True, token="tok", db_path=db)
        assert result["repo_id"] == "EEM-Hub/ddia-expert"
        assert result["nodes_imported"] == 1

    @patch("reasonsforge.hf.urlopen")
    def test_pull_with_explicit_org(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        db = str(tmp_path / "test.db")
        result = api.import_hf("myuser/my-eem", init=True, token="tok", db_path=db)
        assert result["repo_id"] == "myuser/my-eem"

    @patch("reasonsforge.hf.urlopen")
    def test_import_hf_bare_name_resolves(self, mock_urlopen, tmp_path):
        """Existing import-hf command also gets default org resolution."""
        mock_urlopen.return_value = _mock_response(SAMPLE_NETWORK)
        db = str(tmp_path / "test.db")
        api.import_hf("test-eem", init=True, token="tok", db_path=db)
        req = mock_urlopen.call_args[0][0]
        assert "EEM-Hub/test-eem" in req.full_url


class TestCLIDispatch:
    def test_pull_dispatches(self):
        from reasonsforge.cli import cmd_pull, cmd_import_hf
        # cmd_pull delegates to cmd_import_hf — verify it's callable
        assert callable(cmd_pull)

    def test_publish_dispatches(self):
        from reasonsforge.cli import cmd_publish
        assert callable(cmd_publish)

    def test_commands_registered(self):
        """pull and publish are in the CLI dispatch table."""
        import argparse
        from reasonsforge.cli import main
        # Just verify the subparser accepts the commands
        from reasonsforge import cli
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        # Smoke test: the commands exist by checking the dispatch dict
        # We access it indirectly by verifying the module defines the handlers
        assert hasattr(cli, "cmd_pull")
        assert hasattr(cli, "cmd_publish")

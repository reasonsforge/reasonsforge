"""Tests for the PostgreSQL dispatch layer in api.py and cli.py."""

import argparse
import json
from unittest.mock import patch, MagicMock

import pytest

from reasonsforge.api import (
    _pg_dispatch, export_markdown,
    import_json, import_beliefs, import_agent, sync_agent,
    hash_sources, check_stale, lookup,
    add_repo, list_repos, list_negative,
)
from reasonsforge.cli import _backend_kwargs, _require_sqlite


class TestPgDispatch:

    def test_dispatch_calls_pgapi_method(self):
        mock_pg = MagicMock()
        mock_pg.get_status.return_value = {"total": 5}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = _pg_dispatch("postgresql://...", "proj-1", "get_status")
        assert result == {"total": 5}

    def test_dispatch_passes_kwargs(self):
        mock_pg = MagicMock()
        mock_pg.show_node.return_value = {"id": "a", "text": "Alpha"}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = _pg_dispatch("postgresql://...", "proj-1", "show_node", node_id="a")
        mock_pg.show_node.assert_called_once_with(node_id="a")


class TestBackendKwargs:

    def test_sqlite_default(self):
        args = argparse.Namespace(db="reasons.db", pg=None, project_id=None)
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("REASONSFORGE_PG_CONNINFO", "REASONSFORGE_PROJECT_ID")}
        with patch.dict("os.environ", env, clear=True):
            result = _backend_kwargs(args)
        assert result == {"db_path": "reasons.db"}

    def test_pg_with_project_id(self):
        args = argparse.Namespace(db="reasons.db", pg="postgresql://localhost/test",
                                  project_id="abc-123")
        result = _backend_kwargs(args)
        assert result == {"pg_conninfo": "postgresql://localhost/test",
                          "project_id": "abc-123"}

    def test_pg_missing_project_id_exits(self):
        args = argparse.Namespace(db="reasons.db", pg="postgresql://localhost/test",
                                  project_id=None)
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("REASONSFORGE_PG_CONNINFO", "REASONSFORGE_PROJECT_ID")}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(SystemExit):
                _backend_kwargs(args)

    def test_env_var_fallback(self):
        args = argparse.Namespace(db="reasons.db", pg=None, project_id=None)
        with patch.dict("os.environ", {"REASONSFORGE_PG_CONNINFO": "postgresql://env",
                                        "REASONSFORGE_PROJECT_ID": "env-proj"}):
            result = _backend_kwargs(args)
        assert result == {"pg_conninfo": "postgresql://env", "project_id": "env-proj"}

    def test_cli_overrides_env(self):
        args = argparse.Namespace(db="reasons.db", pg="postgresql://cli",
                                  project_id="cli-proj")
        with patch.dict("os.environ", {"REASONSFORGE_PG_CONNINFO": "postgresql://env",
                                        "REASONSFORGE_PROJECT_ID": "env-proj"}):
            result = _backend_kwargs(args)
        assert result == {"pg_conninfo": "postgresql://cli", "project_id": "cli-proj"}


class TestRequireSqlite:

    def test_no_pg_passes(self):
        args = argparse.Namespace(pg=None)
        env = {k: v for k, v in __import__("os").environ.items()
               if k != "REASONSFORGE_PG_CONNINFO"}
        with patch.dict("os.environ", env, clear=True):
            _require_sqlite(args, "hash-sources")

    def test_pg_set_exits(self):
        args = argparse.Namespace(pg="postgresql://localhost/test")
        with pytest.raises(SystemExit):
            _require_sqlite(args, "hash-sources")

    def test_env_pg_exits(self):
        args = argparse.Namespace(pg=None)
        with patch.dict("os.environ", {"REASONSFORGE_PG_CONNINFO": "postgresql://env"}):
            with pytest.raises(SystemExit):
                _require_sqlite(args, "hash-sources")


class TestExportMarkdownPgPath:

    EXPORT_DATA = {
        "meta": {
            "schema_version": "1.0",
            "project_name": "test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "node_count": 4,
            "generator": "reasonsforge/test",
        },
        "nodes": {
            "premise-a": {
                "text": "Alpha premise",
                "truth_value": "IN",
                "justifications": [],
                "source": "test.py",
                "source_url": "https://example.com",
                "source_hash": "abc123",
                "date": "2026-01-01",
                "metadata": {"domain": "test"},
            },
            "derived-b": {
                "text": "Beta derived from alpha",
                "truth_value": "IN",
                "justifications": [
                    {"type": "SL", "antecedents": ["premise-a"], "outlist": [], "label": ""}
                ],
                "source": "",
                "source_url": "",
                "source_hash": "",
                "date": "",
                "metadata": {},
            },
            "gated-c": {
                "text": "Gamma gated on blocker",
                "truth_value": "OUT",
                "justifications": [
                    {"type": "SL", "antecedents": ["premise-a"],
                     "outlist": ["blocker"], "label": "gated"}
                ],
                "source": "",
                "source_url": "",
                "source_hash": "",
                "date": "",
                "metadata": {},
            },
            "blocker": {
                "text": "Blocker node",
                "truth_value": "IN",
                "justifications": [],
                "source": "",
                "source_url": "",
                "source_hash": "",
                "date": "",
                "metadata": {},
            },
        },
        "nogoods": [
            {"id": "nogood-001", "nodes": ["premise-a", "blocker"],
             "discovered": "2026-01-01", "resolution": ""},
        ],
        "repos": {},
    }

    def test_produces_markdown(self):
        with patch("reasonsforge.api.export_network", return_value=self.EXPORT_DATA):
            md = export_markdown(pg_conninfo="postgresql://...", project_id="test")
        assert isinstance(md, str)
        assert len(md) > 0

    def test_contains_node_ids(self):
        with patch("reasonsforge.api.export_network", return_value=self.EXPORT_DATA):
            md = export_markdown(pg_conninfo="postgresql://...", project_id="test")
        assert "premise-a" in md
        assert "derived-b" in md
        assert "gated-c" in md

    def test_contains_node_text(self):
        with patch("reasonsforge.api.export_network", return_value=self.EXPORT_DATA):
            md = export_markdown(pg_conninfo="postgresql://...", project_id="test")
        assert "Alpha premise" in md
        assert "Beta derived from alpha" in md

    def test_contains_justification_refs(self):
        with patch("reasonsforge.api.export_network", return_value=self.EXPORT_DATA):
            md = export_markdown(pg_conninfo="postgresql://...", project_id="test")
        assert "premise-a" in md

    def test_contains_source_info(self):
        with patch("reasonsforge.api.export_network", return_value=self.EXPORT_DATA):
            md = export_markdown(pg_conninfo="postgresql://...", project_id="test")
        assert "test.py" in md

    def test_contains_nogoods(self):
        with patch("reasonsforge.api.export_network", return_value=self.EXPORT_DATA):
            md = export_markdown(pg_conninfo="postgresql://...", project_id="test")
        assert "nogood" in md.lower()

    def test_dependents_reconstructed(self):
        with patch("reasonsforge.api.export_network", return_value=self.EXPORT_DATA):
            md = export_markdown(pg_conninfo="postgresql://...", project_id="test")
        # premise-a is an antecedent of derived-b and gated-c,
        # so it should show dependents in the markdown
        assert "derived-b" in md
        assert "gated-c" in md

    def test_empty_network(self):
        empty = {"meta": {}, "nodes": {}, "nogoods": [], "repos": {}}
        with patch("reasonsforge.api.export_network", return_value=empty):
            md = export_markdown(pg_conninfo="postgresql://...", project_id="test")
        assert isinstance(md, str)


class TestImportJsonDispatch:

    def test_dispatches_parsed_data(self, tmp_path):
        json_file = tmp_path / "network.json"
        data = {"nodes": {"a": {"text": "Alpha", "truth_value": "IN",
                "justifications": []}}, "nogoods": []}
        json_file.write_text(json.dumps(data))

        mock_pg = MagicMock()
        mock_pg.import_json.return_value = {"nodes_imported": 1, "nogoods_imported": 0}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = import_json(str(json_file),
                                 pg_conninfo="postgresql://...", project_id="test")
        mock_pg.import_json.assert_called_once_with(data=data)
        assert result["nodes_imported"] == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            import_json("/nonexistent.json",
                        pg_conninfo="postgresql://...", project_id="test")


class TestImportBeliefsDispatch:

    def test_dispatches_text(self, tmp_path):
        beliefs_file = tmp_path / "beliefs.md"
        beliefs_file.write_text("### alpha [IN] premise\nAlpha belief\n")

        mock_pg = MagicMock()
        mock_pg.import_beliefs.return_value = {
            "claims_imported": 1, "claims_skipped": 0,
            "claims_retracted": 0, "nogoods_imported": 0,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = import_beliefs(str(beliefs_file),
                                    pg_conninfo="postgresql://...", project_id="test")
        mock_pg.import_beliefs.assert_called_once()
        call_kwargs = mock_pg.import_beliefs.call_args[1]
        assert "alpha" in call_kwargs["beliefs_text"].lower()
        assert result["claims_imported"] == 1


class TestImportAgentDispatch:

    def test_dispatches_markdown(self, tmp_path):
        beliefs_file = tmp_path / "beliefs.md"
        beliefs_file.write_text("### alpha [IN] premise\nAlpha belief\n")

        mock_pg = MagicMock()
        mock_pg.import_agent.return_value = {
            "agent": "remote", "prefix": "remote:",
            "active_node": "remote:active", "created_premise": True,
            "claims_imported": 1, "claims_skipped": 0,
            "claims_retracted": 0, "claims_propagated": 0,
            "nogoods_imported": 0,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = import_agent("remote", str(beliefs_file),
                                  pg_conninfo="postgresql://...", project_id="test")
        mock_pg.import_agent.assert_called_once()
        call_kwargs = mock_pg.import_agent.call_args[1]
        assert call_kwargs["agent_name"] == "remote"
        assert len(call_kwargs["claims"]) == 1
        assert result["agent"] == "remote"

    def test_dispatches_json(self, tmp_path):
        json_file = tmp_path / "network.json"
        data = {"nodes": {"a": {"text": "Alpha", "truth_value": "IN",
                "justifications": []}}, "nogoods": []}
        json_file.write_text(json.dumps(data))

        mock_pg = MagicMock()
        mock_pg.import_agent.return_value = {
            "agent": "remote", "prefix": "remote:",
            "active_node": "remote:active", "created_premise": True,
            "claims_imported": 1, "claims_skipped": 0,
            "claims_retracted": 0, "claims_propagated": 0,
            "nogoods_imported": 0,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = import_agent("remote", str(json_file),
                                  pg_conninfo="postgresql://...", project_id="test")
        mock_pg.import_agent.assert_called_once()
        call_kwargs = mock_pg.import_agent.call_args[1]
        assert call_kwargs["agent_name"] == "remote"


class TestSyncAgentDispatch:

    def test_dispatches_markdown(self, tmp_path):
        beliefs_file = tmp_path / "beliefs.md"
        beliefs_file.write_text("### alpha [IN] premise\nAlpha belief\n")

        mock_pg = MagicMock()
        mock_pg.sync_agent.return_value = {
            "agent": "remote", "prefix": "remote:",
            "active_node": "remote:active", "created_premise": True,
            "beliefs_added": 1, "beliefs_updated": 0,
            "beliefs_removed": 0, "beliefs_retracted": 0,
            "beliefs_unchanged": 0, "beliefs_propagated": 0,
            "nogoods_imported": 0,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = sync_agent("remote", str(beliefs_file),
                                pg_conninfo="postgresql://...", project_id="test")
        mock_pg.sync_agent.assert_called_once()
        assert result["beliefs_added"] == 1


class TestImportCliNoLongerBlocked:

    def test_import_json_accepts_pg(self, tmp_path):
        json_file = tmp_path / "network.json"
        json_file.write_text('{"nodes": {}, "nogoods": []}')
        mock_pg = MagicMock()
        mock_pg.import_json.return_value = {"nodes_imported": 0, "nogoods_imported": 0}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = import_json(str(json_file),
                                 pg_conninfo="postgresql://...", project_id="test")
        assert result["nodes_imported"] == 0

    def test_import_beliefs_accepts_pg(self, tmp_path):
        f = tmp_path / "beliefs.md"
        f.write_text("")
        mock_pg = MagicMock()
        mock_pg.import_beliefs.return_value = {
            "claims_imported": 0, "claims_skipped": 0,
            "claims_retracted": 0, "nogoods_imported": 0,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = import_beliefs(str(f),
                                    pg_conninfo="postgresql://...", project_id="test")
        assert result["claims_imported"] == 0


class TestHashSourcesDispatch:

    def test_dispatches_to_pg(self):
        mock_pg = MagicMock()
        mock_pg.hash_sources.return_value = {"hashed": [], "count": 0}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = hash_sources(pg_conninfo="postgresql://...", project_id="test")
        mock_pg.hash_sources.assert_called_once_with(force=False, repos=None)
        assert result == {"hashed": [], "count": 0}

    def test_passes_force_and_repos(self):
        mock_pg = MagicMock()
        mock_pg.hash_sources.return_value = {"hashed": [{"node_id": "a"}], "count": 1}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = hash_sources(force=True, repos={"myrepo": "/tmp/repo"},
                                  pg_conninfo="postgresql://...", project_id="test")
        mock_pg.hash_sources.assert_called_once_with(
            force=True, repos={"myrepo": "/tmp/repo"})
        assert result["count"] == 1


class TestCheckStaleDispatch:

    def test_dispatches_to_pg(self):
        mock_pg = MagicMock()
        mock_pg.check_stale.return_value = {
            "stale": [], "checked": 5, "stale_count": 0, "upgraded": 0,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = check_stale(pg_conninfo="postgresql://...", project_id="test")
        mock_pg.check_stale.assert_called_once_with(repos=None, upgrade_hashes=False, git_aware=False)
        assert result["checked"] == 5

    def test_passes_upgrade_hashes(self):
        mock_pg = MagicMock()
        mock_pg.check_stale.return_value = {
            "stale": [], "checked": 3, "stale_count": 0, "upgraded": 2,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = check_stale(upgrade_hashes=True,
                                 pg_conninfo="postgresql://...", project_id="test")
        mock_pg.check_stale.assert_called_once_with(repos=None, upgrade_hashes=True, git_aware=False)
        assert result["upgraded"] == 2


class TestLookupDispatch:

    def test_dispatches_to_pg(self):
        mock_pg = MagicMock()
        mock_pg.lookup.return_value = "Found 1 matching belief(s):\n\n### a [IN]\nAlpha\n"
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = lookup("alpha", pg_conninfo="postgresql://...", project_id="test")
        mock_pg.lookup.assert_called_once_with(query="alpha", visible_to=None)
        assert "alpha" in result.lower()

    def test_passes_visible_to(self):
        mock_pg = MagicMock()
        mock_pg.lookup.return_value = "No beliefs found matching 'secret'"
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = lookup("secret", visible_to=["admin"],
                            pg_conninfo="postgresql://...", project_id="test")
        mock_pg.lookup.assert_called_once_with(query="secret", visible_to=["admin"])


class TestMaintenanceCliNoLongerBlocked:

    def test_hash_sources_accepts_pg(self):
        mock_pg = MagicMock()
        mock_pg.hash_sources.return_value = {"hashed": [], "count": 0}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = hash_sources(pg_conninfo="postgresql://...", project_id="test")
        assert result["count"] == 0

    def test_check_stale_accepts_pg(self):
        mock_pg = MagicMock()
        mock_pg.check_stale.return_value = {
            "stale": [], "checked": 0, "stale_count": 0, "upgraded": 0,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = check_stale(pg_conninfo="postgresql://...", project_id="test")
        assert result["stale_count"] == 0

    def test_lookup_accepts_pg(self):
        mock_pg = MagicMock()
        mock_pg.lookup.return_value = "No beliefs found matching 'test'"
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = lookup("test", pg_conninfo="postgresql://...", project_id="test")
        assert "No beliefs" in result


class TestAddRepoDispatch:

    def test_dispatches_to_pg(self):
        mock_pg = MagicMock()
        mock_pg.add_repo.return_value = {"name": "myrepo", "path": "/tmp/repo"}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = add_repo("myrepo", "/tmp/repo",
                              pg_conninfo="postgresql://...", project_id="test")
        mock_pg.add_repo.assert_called_once_with(name="myrepo", path="/tmp/repo")
        assert result == {"name": "myrepo", "path": "/tmp/repo"}


class TestListReposDispatch:

    def test_dispatches_to_pg(self):
        mock_pg = MagicMock()
        mock_pg.list_repos.return_value = {"repos": {"myrepo": "/tmp/repo"}}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = list_repos(pg_conninfo="postgresql://...", project_id="test")
        mock_pg.list_repos.assert_called_once_with()
        assert result["repos"] == {"myrepo": "/tmp/repo"}

    def test_empty_repos(self):
        mock_pg = MagicMock()
        mock_pg.list_repos.return_value = {"repos": {}}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = list_repos(pg_conninfo="postgresql://...", project_id="test")
        assert result["repos"] == {}


class TestListNegativeDispatch:

    def test_dispatches_to_pg(self):
        mock_pg = MagicMock()
        mock_pg.list_negative.return_value = {
            "negative": [{"id": "a", "text": "A bug"}],
            "count": 1, "candidates": 3, "total": 10,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = list_negative(pg_conninfo="postgresql://...", project_id="test")
        mock_pg.list_negative.assert_called_once_with(visible_to=None, model="claude",
                                                          skip_llm=False)
        assert result["count"] == 1

    def test_passes_visible_to_and_model(self):
        mock_pg = MagicMock()
        mock_pg.list_negative.return_value = {
            "negative": [], "count": 0, "candidates": 0, "total": 5,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = list_negative(visible_to=["admin"], model="gemini",
                                   pg_conninfo="postgresql://...", project_id="test")
        mock_pg.list_negative.assert_called_once_with(
            visible_to=["admin"], model="gemini", skip_llm=False)
        assert result["total"] == 5


class TestRepoAndNegativeCliNoLongerBlocked:

    def test_add_repo_accepts_pg(self):
        mock_pg = MagicMock()
        mock_pg.add_repo.return_value = {"name": "r", "path": "/p"}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = add_repo("r", "/p",
                              pg_conninfo="postgresql://...", project_id="test")
        assert result["name"] == "r"

    def test_list_repos_accepts_pg(self):
        mock_pg = MagicMock()
        mock_pg.list_repos.return_value = {"repos": {}}
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = list_repos(pg_conninfo="postgresql://...", project_id="test")
        assert result["repos"] == {}

    def test_list_negative_accepts_pg(self):
        mock_pg = MagicMock()
        mock_pg.list_negative.return_value = {
            "negative": [], "count": 0, "candidates": 0, "total": 0,
        }
        with patch("reasonsforge.pg.PgApi") as MockPgApi:
            MockPgApi.return_value.__enter__ = MagicMock(return_value=mock_pg)
            MockPgApi.return_value.__exit__ = MagicMock(return_value=False)
            result = list_negative(pg_conninfo="postgresql://...", project_id="test")
        assert result["count"] == 0

"""Tests for metadata across storage formats."""

import json
from pathlib import Path

import pytest

from reasonsforge import api, Justification
from reasonsforge.metadata import build_meta, SCHEMA_VERSION
from reasonsforge.network import Network
from reasonsforge.export_markdown import export_markdown
from reasonsforge.import_beliefs import strip_frontmatter, import_into_network


class TestBuildMeta:

    def test_basic_fields(self):
        meta = build_meta("my-project", node_count=42)
        assert meta["schema_version"] == "1.0"
        assert meta["project_name"] == "my-project"
        assert meta["node_count"] == 42
        assert "reasonsforge/" in meta["generator"]
        assert meta["created_at"]
        assert meta["updated_at"]

    def test_preserves_created_at(self):
        meta = build_meta("p", created_at="2026-01-01T00:00:00+00:00")
        assert meta["created_at"] == "2026-01-01T00:00:00+00:00"

    def test_empty_project_name(self):
        meta = build_meta()
        assert meta["project_name"] == ""


class TestJsonExportMeta:

    def test_export_includes_meta(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        data = api.export_network(db_path=db)
        assert "meta" in data
        assert data["meta"]["schema_version"] == SCHEMA_VERSION
        assert data["meta"]["node_count"] == 1
        assert "reasonsforge/" in data["meta"]["generator"]

    def test_export_preserves_project_name(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db, project_name="my-beliefs")
        data = api.export_network(db_path=db)
        assert data["meta"]["project_name"] == "my-beliefs"

    def test_export_default_project_name(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        data = api.export_network(db_path=db)
        assert data["meta"]["project_name"] == "test"


class TestJsonImportMeta:

    def test_import_preserves_meta(self, tmp_path):
        db1 = str(tmp_path / "src.db")
        db2 = str(tmp_path / "dst.db")
        json_file = str(tmp_path / "export.json")

        api.init_db(db_path=db1, project_name="source-project")
        api.add_node("a", "Premise A", db_path=db1)
        data = api.export_network(db_path=db1)
        Path(json_file).write_text(json.dumps(data))

        api.init_db(db_path=db2)
        api.import_json(json_file, db_path=db2)

        data2 = api.export_network(db_path=db2)
        assert data2["meta"]["project_name"] == "source-project"

    def test_import_without_meta_is_safe(self, tmp_path):
        """Backward compat: JSON without meta key imports fine."""
        db = str(tmp_path / "test.db")
        json_file = str(tmp_path / "old.json")

        data = {
            "nodes": {
                "a": {
                    "text": "Premise A",
                    "truth_value": "IN",
                    "justifications": [],
                    "source": "",
                    "source_url": "",
                    "source_hash": "",
                    "date": "",
                    "metadata": {},
                }
            },
            "nogoods": [],
            "repos": {},
        }
        Path(json_file).write_text(json.dumps(data))

        api.init_db(db_path=db)
        result = api.import_json(json_file, db_path=db)
        assert result["nodes_imported"] == 1


class TestMarkdownExportMeta:

    def test_frontmatter_present(self):
        net = Network()
        net.add_node("a", "Premise A")
        md = export_markdown(net)
        assert md.startswith("---\n")
        assert 'schema_version: "1.0"' in md
        assert "node_count: 1" in md
        assert "generator: reasonsforge/" in md

    def test_frontmatter_with_project_name(self):
        net = Network()
        net.meta["project_name"] = "test-proj"
        net.add_node("a", "Premise A")
        md = export_markdown(net)
        assert 'project_name: "test-proj"' in md


class TestStripFrontmatter:

    def test_no_frontmatter(self):
        text = "# Beliefs\n\nSome content"
        body, fm = strip_frontmatter(text)
        assert body == text
        assert fm == {}

    def test_with_frontmatter(self):
        text = '---\nschema_version: "1.0"\nproject_name: test\n---\n\n# Beliefs'
        body, fm = strip_frontmatter(text)
        assert fm["schema_version"] == "1.0"
        assert fm["project_name"] == "test"
        assert body.strip().startswith("# Beliefs")
        assert "---" not in body

    def test_quoted_values(self):
        text = '---\nupdated_at: "2026-05-23T14:00:00Z"\n---\nContent'
        body, fm = strip_frontmatter(text)
        assert fm["updated_at"] == "2026-05-23T14:00:00Z"


class TestMarkdownRoundTrip:

    def test_export_import_preserves_beliefs(self, tmp_path):
        """Export to markdown with frontmatter, import back — beliefs survive."""
        db1 = str(tmp_path / "src.db")
        db2 = str(tmp_path / "dst.db")
        md_file = str(tmp_path / "beliefs.md")

        api.init_db(db_path=db1, project_name="round-trip-test")
        api.add_node("a", "Premise A", source="repo/a.md", db_path=db1)
        api.add_node("b", "Derived B", sl="a", db_path=db1)

        md = api.export_markdown(db_path=db1)
        Path(md_file).write_text(md)

        api.init_db(db_path=db2)
        result = api.import_beliefs(md_file, db_path=db2)
        assert result["claims_imported"] == 2

    def test_frontmatter_ignored_gracefully(self, tmp_path):
        """Old beliefs.md without frontmatter still imports."""
        db = str(tmp_path / "test.db")
        md_file = str(tmp_path / "old.md")

        Path(md_file).write_text(
            "# Belief Registry\n\n## Claims\n\n"
            "### a [IN] OBSERVATION\nPremise A\n"
        )

        api.init_db(db_path=db)
        result = api.import_beliefs(md_file, db_path=db)
        assert result["claims_imported"] == 1


class TestSqliteMeta:

    def test_init_sets_metadata(self, tmp_path):
        db = str(tmp_path / "myproject.db")
        api.init_db(db_path=db)

        import sqlite3
        conn = sqlite3.connect(db)
        rows = dict(conn.execute("SELECT key, value FROM network_meta").fetchall())
        conn.close()

        assert rows["schema_version"] == SCHEMA_VERSION
        assert rows["project_name"] == "myproject"
        assert "created_at" in rows
        assert "updated_at" in rows

    def test_save_preserves_created_at(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)

        import sqlite3
        conn = sqlite3.connect(db)
        rows1 = dict(conn.execute("SELECT key, value FROM network_meta").fetchall())
        conn.close()

        api.add_node("a", "Premise A", db_path=db)

        conn = sqlite3.connect(db)
        rows2 = dict(conn.execute("SELECT key, value FROM network_meta").fetchall())
        conn.close()

        assert rows2["created_at"] == rows1["created_at"]

    def test_project_name_override(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db, project_name="custom-name")

        import sqlite3
        conn = sqlite3.connect(db)
        rows = dict(conn.execute("SELECT key, value FROM network_meta").fetchall())
        conn.close()

        assert rows["project_name"] == "custom-name"

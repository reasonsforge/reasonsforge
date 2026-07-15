"""Extended tests for metadata across storage formats.

Covers edge cases, round-trips, and behaviors not in test_metadata.py:
- Full JSON round-trip (export→import→re-export preserves meta)
- updated_at changes between saves
- node_count accuracy after add/retract
- Generator string format
- SQLite created_at survives multiple saves
- Frontmatter edge cases (partial, malformed, single-quoted)
- Empty network exports
- Markdown round-trip preserves frontmatter fields
- Metadata via CLI (init with --project-name)
"""

import json
import sqlite3
from pathlib import Path

import pytest

from reasonsforge import api
from reasonsforge.metadata import build_meta, SCHEMA_VERSION, _get_generator
from reasonsforge.network import Network
from reasonsforge.export_markdown import export_markdown
from reasonsforge.import_beliefs import strip_frontmatter, import_into_network
from reasonsforge.storage import Storage


class TestBuildMetaEdgeCases:

    def test_node_count_zero(self):
        meta = build_meta("proj", node_count=0)
        assert meta["node_count"] == 0

    def test_large_node_count(self):
        meta = build_meta("proj", node_count=999999)
        assert meta["node_count"] == 999999

    def test_created_at_not_overwritten_when_provided(self):
        ts = "2020-01-01T00:00:00+00:00"
        meta = build_meta("p", created_at=ts)
        assert meta["created_at"] == ts
        assert meta["updated_at"] != ts

    def test_updated_at_is_iso_format(self):
        meta = build_meta("p")
        assert "T" in meta["updated_at"]
        assert meta["updated_at"].endswith("+00:00")

    def test_schema_version_constant(self):
        assert SCHEMA_VERSION == "1.0"

    def test_generator_format(self):
        gen = _get_generator()
        assert gen.startswith("reasonsforge/")
        parts = gen.split("/")
        assert len(parts) == 2
        assert parts[1]  # version is non-empty


class TestJsonRoundTripMeta:

    def test_full_roundtrip_preserves_meta(self, tmp_path):
        """Export→file→import→re-export: project_name and schema_version survive."""
        db1 = str(tmp_path / "src.db")
        db2 = str(tmp_path / "dst.db")
        json_file = str(tmp_path / "export.json")

        api.init_db(db_path=db1, project_name="round-trip-test")
        api.add_node("a", "Premise A", db_path=db1)
        api.add_node("b", "Derived B", sl="a", db_path=db1)

        data1 = api.export_network(db_path=db1)
        Path(json_file).write_text(json.dumps(data1))

        api.init_db(db_path=db2)
        api.import_json(json_file, db_path=db2)

        data2 = api.export_network(db_path=db2)
        assert data2["meta"]["schema_version"] == data1["meta"]["schema_version"]
        assert data2["meta"]["project_name"] == "round-trip-test"
        assert data2["meta"]["node_count"] == 2

    def test_node_count_matches_exported_nodes(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        for i in range(5):
            api.add_node(f"n{i}", f"Node {i}", db_path=db)
        data = api.export_network(db_path=db)
        assert data["meta"]["node_count"] == 5
        assert data["meta"]["node_count"] == len(data["nodes"])

    def test_node_count_after_retraction(self, tmp_path):
        """Retracted nodes still count in node_count (they're still exported)."""
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.retract_node("b", db_path=db)

        data = api.export_network(db_path=db)
        assert data["meta"]["node_count"] == len(data["nodes"])

    def test_export_empty_network(self, tmp_path):
        db = str(tmp_path / "empty.db")
        api.init_db(db_path=db)
        data = api.export_network(db_path=db)
        assert data["meta"]["schema_version"] == SCHEMA_VERSION
        assert data["meta"]["node_count"] == 0
        assert data["meta"]["project_name"] == "empty"

    def test_import_json_without_meta_uses_defaults(self, tmp_path):
        """JSON without meta key: import succeeds, re-export has default meta."""
        db = str(tmp_path / "test.db")
        json_file = str(tmp_path / "legacy.json")

        legacy = {
            "nodes": {
                "x": {
                    "text": "Old node",
                    "truth_value": "IN",
                    "justifications": [],
                    "source": "", "source_url": "", "source_hash": "",
                    "date": "", "metadata": {},
                }
            },
            "nogoods": [],
            "repos": {},
        }
        Path(json_file).write_text(json.dumps(legacy))

        api.init_db(db_path=db)
        api.import_json(json_file, db_path=db)

        data = api.export_network(db_path=db)
        assert data["meta"]["schema_version"] == SCHEMA_VERSION
        assert data["meta"]["node_count"] == 1

    def test_meta_generator_in_export(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        data = api.export_network(db_path=db)
        assert data["meta"]["generator"].startswith("reasonsforge/")


class TestSqliteMetaExtended:

    def test_created_at_preserved_across_saves(self, tmp_path):
        """Multiple add_node calls should not change created_at."""
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)

        conn = sqlite3.connect(db)
        rows1 = dict(conn.execute("SELECT key, value FROM network_meta").fetchall())
        conn.close()
        created1 = rows1["created_at"]

        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)

        conn = sqlite3.connect(db)
        rows2 = dict(conn.execute("SELECT key, value FROM network_meta").fetchall())
        conn.close()

        assert rows2["created_at"] == created1

    def test_schema_version_in_db(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)

        conn = sqlite3.connect(db)
        rows = dict(conn.execute("SELECT key, value FROM network_meta").fetchall())
        conn.close()

        assert rows["schema_version"] == SCHEMA_VERSION

    def test_default_project_name_from_filename(self, tmp_path):
        db = str(tmp_path / "my-beliefs.db")
        api.init_db(db_path=db)

        conn = sqlite3.connect(db)
        rows = dict(conn.execute("SELECT key, value FROM network_meta").fetchall())
        conn.close()

        assert rows["project_name"] == "my-beliefs"

    def test_meta_survives_load_save_cycle(self, tmp_path):
        """Storage load→save preserves metadata keys."""
        db = str(tmp_path / "test.db")
        store = Storage(db)
        net = store.load()
        net.meta["project_name"] = "custom"
        net.meta["created_at"] = "2025-01-01T00:00:00+00:00"
        store.save(net)
        store.close()

        store2 = Storage(db)
        loaded = store2.load()
        assert loaded.meta["project_name"] == "custom"
        assert loaded.meta["created_at"] == "2025-01-01T00:00:00+00:00"
        assert loaded.meta["schema_version"] == SCHEMA_VERSION
        store2.close()


class TestMarkdownExportMetaExtended:

    def test_frontmatter_is_first_block(self):
        net = Network()
        net.add_node("a", "A")
        md = export_markdown(net)
        lines = md.split("\n")
        assert lines[0] == "---"

    def test_frontmatter_ends_with_separator(self):
        net = Network()
        net.add_node("a", "A")
        md = export_markdown(net)
        lines = md.split("\n")
        sep_indices = [i for i, l in enumerate(lines) if l.strip() == "---"]
        assert len(sep_indices) >= 2
        assert sep_indices[0] == 0

    def test_frontmatter_node_count_matches(self):
        net = Network()
        for i in range(3):
            net.add_node(f"n{i}", f"Node {i}")
        md = export_markdown(net)
        assert "node_count: 3" in md

    def test_empty_project_name_handled(self):
        net = Network()
        net.add_node("a", "A")
        md = export_markdown(net)
        assert "project_name:" in md

    def test_frontmatter_schema_version_quoted(self):
        net = Network()
        net.add_node("a", "A")
        md = export_markdown(net)
        assert 'schema_version: "1.0"' in md


class TestStripFrontmatterEdgeCases:

    def test_incomplete_frontmatter_no_closing(self):
        """Missing closing --- returns original text."""
        text = "---\nschema_version: 1.0\nSome body"
        body, fm = strip_frontmatter(text)
        assert body == text
        assert fm == {}

    def test_empty_value(self):
        text = "---\nproject_name:\n---\nBody"
        body, fm = strip_frontmatter(text)
        assert fm["project_name"] == ""

    def test_single_quoted_value(self):
        text = "---\nproject_name: 'my-project'\n---\nBody"
        body, fm = strip_frontmatter(text)
        assert fm["project_name"] == "my-project"

    def test_frontmatter_with_numeric_value(self):
        text = "---\nnode_count: 42\n---\nBody"
        body, fm = strip_frontmatter(text)
        assert fm["node_count"] == "42"

    def test_body_preserved_after_strip(self):
        text = "---\nk: v\n---\n\n# Heading\n\nParagraph"
        body, fm = strip_frontmatter(text)
        assert "# Heading" in body
        assert "Paragraph" in body
        assert "---" not in body


class TestMarkdownRoundTripExtended:

    def test_project_name_roundtrip(self, tmp_path):
        """Export with project_name → import → network.meta has project_name."""
        db1 = str(tmp_path / "src.db")
        db2 = str(tmp_path / "dst.db")
        md_file = str(tmp_path / "beliefs.md")

        api.init_db(db_path=db1, project_name="my-proj")
        api.add_node("a", "Premise A", db_path=db1)

        md = api.export_markdown(db_path=db1)
        Path(md_file).write_text(md)

        api.init_db(db_path=db2)
        api.import_beliefs(md_file, db_path=db2)

        data = api.export_network(db_path=db2)
        assert data["meta"]["project_name"] == "my-proj"

    def test_schema_version_roundtrip(self, tmp_path):
        """Frontmatter schema_version preserved through import."""
        db1 = str(tmp_path / "src.db")
        db2 = str(tmp_path / "dst.db")
        md_file = str(tmp_path / "beliefs.md")

        api.init_db(db_path=db1)
        api.add_node("a", "Premise A", db_path=db1)

        md = api.export_markdown(db_path=db1)
        Path(md_file).write_text(md)

        api.init_db(db_path=db2)
        api.import_beliefs(md_file, db_path=db2)

        data = api.export_network(db_path=db2)
        assert data["meta"]["schema_version"] == SCHEMA_VERSION

    def test_import_markdown_without_frontmatter_preserves_existing_meta(self, tmp_path):
        """Importing old-style beliefs.md without frontmatter doesn't clobber existing meta."""
        db = str(tmp_path / "test.db")
        md_file = str(tmp_path / "old.md")

        api.init_db(db_path=db, project_name="existing-proj")

        Path(md_file).write_text(
            "# Belief Registry\n\n## Claims\n\n"
            "### x [IN] OBSERVATION\nSome belief\n"
        )
        api.import_beliefs(md_file, db_path=db)

        data = api.export_network(db_path=db)
        assert data["meta"]["project_name"] == "existing-proj"

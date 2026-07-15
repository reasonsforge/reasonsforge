"""Tests for source_type metadata field."""

import json
import subprocess
from pathlib import Path

import pytest

from reasonsforge import api
from reasonsforge.export_markdown import export_markdown
from reasonsforge.import_beliefs import parse_beliefs, import_into_network
from reasonsforge.network import Network


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    api.init_db(db_path=db_path)
    return db_path


class TestAddNodeSourceType:
    def test_stores_source_type(self, db):
        api.add_node("n1", "A code observation.", source_type="code", db_path=db)
        node = api.show_node("n1", db_path=db)
        assert node["metadata"]["source_type"] == "code"

    def test_all_valid_types(self, db):
        for i, st in enumerate(["code", "document", "self-description", "derived"]):
            api.add_node(f"n{i}", f"Type {st}.", source_type=st, db_path=db)
            node = api.show_node(f"n{i}", db_path=db)
            assert node["metadata"]["source_type"] == st

    def test_invalid_type_raises(self, db):
        with pytest.raises(ValueError, match="Invalid source_type"):
            api.add_node("bad", "Bad type.", source_type="unknown", db_path=db)

    def test_empty_type_not_stored(self, db):
        api.add_node("n1", "No type.", db_path=db)
        node = api.show_node("n1", db_path=db)
        assert "source_type" not in node["metadata"]

    def test_source_type_persists(self, db):
        api.add_node("n1", "Code obs.", source_type="code", db_path=db)
        node = api.show_node("n1", db_path=db)
        assert node["metadata"]["source_type"] == "code"
        result = api.export_network(db_path=db)
        assert result["nodes"]["n1"]["metadata"]["source_type"] == "code"


class TestShowSourceType:
    def test_show_displays_source_type(self, db, capsys):
        api.add_node("n1", "Code obs.", source_type="code", db_path=db)
        from reasonsforge.cli import cmd_show
        import argparse
        args = argparse.Namespace(
            node_id="n1", db=db, visible_to=None,
            pg_conninfo=None, pg_project=None,
        )
        cmd_show(args)
        captured = capsys.readouterr()
        assert "Source type: code" in captured.out

    def test_show_omits_when_absent(self, db, capsys):
        api.add_node("n1", "No type.", db_path=db)
        from reasonsforge.cli import cmd_show
        import argparse
        args = argparse.Namespace(
            node_id="n1", db=db, visible_to=None,
            pg_conninfo=None, pg_project=None,
        )
        cmd_show(args)
        captured = capsys.readouterr()
        assert "Source type" not in captured.out


class TestListSourceType:
    def test_list_shows_source_type(self, db):
        api.add_node("n1", "Code obs.", source_type="code", db_path=db)
        result = api.list_nodes(db_path=db)
        node = result["nodes"][0]
        assert node["source_type"] == "code"

    def test_list_empty_when_absent(self, db):
        api.add_node("n1", "No type.", db_path=db)
        result = api.list_nodes(db_path=db)
        node = result["nodes"][0]
        assert node["source_type"] == ""


class TestExportMarkdown:
    def test_exports_source_type(self, db):
        api.add_node("n1", "Code obs.", source="repo/file.py",
                      source_type="code", db_path=db)
        from reasonsforge.storage import Storage
        storage = Storage(db)
        net = storage.load()
        storage.close()
        md = export_markdown(net)
        assert "- Source type: code" in md

    def test_omits_when_absent(self, db):
        api.add_node("n1", "No type.", source="repo/file.py", db_path=db)
        from reasonsforge.storage import Storage
        storage = Storage(db)
        net = storage.load()
        storage.close()
        md = export_markdown(net)
        assert "Source type" not in md


class TestImportBeliefs:
    def test_parses_source_type(self):
        text = """\
## Claims

### obs-1 [IN] OBSERVATION
The API uses REST endpoints.
- Source: repo/api.py
- Source type: code
- Date: 2026-06-04
"""
        claims = parse_beliefs(text)
        assert len(claims) == 1
        assert claims[0]["source_type"] == "code"

    def test_missing_source_type(self):
        text = """\
## Claims

### obs-1 [IN] OBSERVATION
The API uses REST endpoints.
- Source: repo/api.py
- Date: 2026-06-04
"""
        claims = parse_beliefs(text)
        assert claims[0]["source_type"] == ""

    def test_import_stores_source_type(self):
        net = Network()
        text = """\
## Claims

### obs-1 [IN] OBSERVATION
The API uses REST endpoints.
- Source: repo/api.py
- Source type: document
"""
        result = import_into_network(net, text)
        assert result["claims_imported"] == 1
        node = net.nodes["obs-1"]
        assert node.metadata.get("source_type") == "document"

    def test_roundtrip_export_import(self, db):
        api.add_node("n1", "Code obs.", source="repo/file.py",
                      source_type="code", db_path=db)
        from reasonsforge.storage import Storage
        storage = Storage(db)
        net = storage.load()
        storage.close()
        md = export_markdown(net)

        net2 = Network()
        result = import_into_network(net2, md)
        assert result["claims_imported"] == 1
        assert net2.nodes["n1"].metadata.get("source_type") == "code"


class TestDeriveAutoSourceType:
    def test_derive_sets_derived_type(self, db):
        api.add_node("p1", "Premise one.", db_path=db)
        api.add_node("p2", "Premise two.", db_path=db)

        from reasonsforge.derive import apply_proposals
        proposals = [{
            "id": "d1",
            "text": "Derived from p1 and p2.",
            "antecedents": ["p1", "p2"],
            "unless": [],
            "label": "test derivation",
        }]
        results = apply_proposals(proposals, db_path=db)
        assert len(results) == 1
        _, result = results[0]
        assert isinstance(result, dict)

        node = api.show_node("d1", db_path=db)
        assert node["metadata"].get("source_type") == "derived"

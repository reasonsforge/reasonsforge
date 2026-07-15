"""Tests for round-trip fidelity: export-markdown -> import-beliefs preserves all fields."""

import pytest

from reasonsforge import api
from reasonsforge.export_markdown import export_markdown
from reasonsforge.import_beliefs import import_into_network, parse_beliefs
from reasonsforge.network import Network
from reasonsforge.storage import Storage


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    api.init_db(db_path=db_path)
    return db_path


def _load_network(db_path):
    storage = Storage(db_path)
    net = storage.load()
    storage.close()
    return net


class TestSourceUrlRoundTrip:
    def test_export_includes_source_url(self, db):
        api.add_node("n1", "A belief.", source="repo/file.py",
                      source_url="https://github.com/org/repo/blob/main/file.py",
                      db_path=db)
        net = _load_network(db)
        md = export_markdown(net)
        assert "- Source URL: https://github.com/org/repo/blob/main/file.py" in md

    def test_source_url_round_trip(self, db):
        url = "https://github.com/org/repo/blob/main/file.py"
        api.add_node("n1", "A belief.", source="repo/file.py",
                      source_url=url, db_path=db)
        net = _load_network(db)
        md = export_markdown(net)

        net2 = Network()
        import_into_network(net2, md)
        assert net2.nodes["n1"].source_url == url

    def test_source_url_omitted_when_empty(self, db):
        api.add_node("n1", "A belief.", source="repo/file.py", db_path=db)
        net = _load_network(db)
        md = export_markdown(net)
        assert "Source URL" not in md

    def test_parse_source_url(self):
        text = """\
## Claims

### obs-1 [IN] OBSERVATION
The API uses REST.
- Source: repo/api.py
- Source URL: https://example.com/api.py
- Date: 2026-06-04
"""
        claims = parse_beliefs(text)
        assert claims[0]["source_url"] == "https://example.com/api.py"


class TestAcceptedPrRoundTrip:
    def test_add_with_accepted_pr(self, db):
        api.add_node("n1", "A belief.", accepted_pr="https://github.com/org/repo/pull/42",
                      db_path=db)
        node = api.show_node("n1", db_path=db)
        assert node["metadata"]["accepted_pr"] == "https://github.com/org/repo/pull/42"

    def test_export_includes_accepted_pr(self, db):
        api.add_node("n1", "A belief.", accepted_pr="https://github.com/org/repo/pull/42",
                      db_path=db)
        net = _load_network(db)
        md = export_markdown(net)
        assert "- Accepted PR: https://github.com/org/repo/pull/42" in md

    def test_accepted_pr_round_trip(self, db):
        pr_url = "https://github.com/org/repo/pull/42"
        api.add_node("n1", "A belief.", accepted_pr=pr_url, db_path=db)
        net = _load_network(db)
        md = export_markdown(net)

        net2 = Network()
        import_into_network(net2, md)
        assert net2.nodes["n1"].metadata.get("accepted_pr") == pr_url

    def test_accepted_pr_omitted_when_empty(self, db):
        api.add_node("n1", "A belief.", db_path=db)
        net = _load_network(db)
        md = export_markdown(net)
        assert "Accepted PR" not in md

    def test_parse_accepted_pr(self):
        text = """\
## Claims

### obs-1 [IN] OBSERVATION
The API uses REST.
- Source: repo/api.py
- Accepted PR: https://github.com/org/repo/pull/99
"""
        claims = parse_beliefs(text)
        assert claims[0]["accepted_pr"] == "https://github.com/org/repo/pull/99"

    def test_show_displays_accepted_pr(self, db, capsys):
        api.add_node("n1", "A belief.", accepted_pr="https://github.com/org/repo/pull/42",
                      db_path=db)
        from reasonsforge.cli import cmd_show
        import argparse
        args = argparse.Namespace(
            node_id="n1", db=db, visible_to=None,
            pg_conninfo=None, pg_project=None,
        )
        cmd_show(args)
        captured = capsys.readouterr()
        assert "Accepted PR: https://github.com/org/repo/pull/42" in captured.out

    def test_show_omits_when_absent(self, db, capsys):
        api.add_node("n1", "A belief.", db_path=db)
        from reasonsforge.cli import cmd_show
        import argparse
        args = argparse.Namespace(
            node_id="n1", db=db, visible_to=None,
            pg_conninfo=None, pg_project=None,
        )
        cmd_show(args)
        captured = capsys.readouterr()
        assert "Accepted PR" not in captured.out


class TestFullRoundTrip:
    """End-to-end: add nodes with all fields -> export markdown -> import -> verify."""

    def test_all_fields_survive(self, db):
        api.add_node(
            "obs-1", "The API uses REST endpoints.",
            source="repo/api.py",
            source_url="https://github.com/org/repo/blob/main/api.py",
            source_type="code",
            accepted_pr="https://github.com/org/repo/pull/10",
            db_path=db,
        )
        # source_hash and date aren't API params — set them directly on the network
        net = _load_network(db)
        net.nodes["obs-1"].source_hash = "abc123"
        net.nodes["obs-1"].date = "2026-06-04"
        storage = Storage(db)
        storage.save(net)
        storage.close()

        api.add_node(
            "obs-2", "The config is YAML-based.",
            source="repo/config.yaml",
            source_url="https://github.com/org/repo/blob/main/config.yaml",
            source_type="document",
            db_path=db,
        )
        api.add_node(
            "derived-1", "REST + YAML means declarative API.",
            sl="obs-1,obs-2",
            source_type="derived",
            accepted_pr="https://github.com/org/repo/pull/11",
            db_path=db,
        )

        net = _load_network(db)
        md = export_markdown(net)

        net2 = Network()
        result = import_into_network(net2, md)
        assert result["claims_imported"] == 3

        n1 = net2.nodes["obs-1"]
        assert n1.text == "The API uses REST endpoints."
        assert n1.source == "repo/api.py"
        assert n1.source_url == "https://github.com/org/repo/blob/main/api.py"
        assert n1.source_hash == "abc123"
        assert n1.date == "2026-06-04"
        assert n1.metadata["source_type"] == "code"
        assert n1.metadata["accepted_pr"] == "https://github.com/org/repo/pull/10"

        n2 = net2.nodes["obs-2"]
        assert n2.source_url == "https://github.com/org/repo/blob/main/config.yaml"
        assert n2.metadata["source_type"] == "document"
        assert "accepted_pr" not in n2.metadata

        d1 = net2.nodes["derived-1"]
        assert d1.metadata["source_type"] == "derived"
        assert d1.metadata["accepted_pr"] == "https://github.com/org/repo/pull/11"
        assert len(d1.justifications) == 1
        assert set(d1.justifications[0].antecedents) == {"obs-1", "obs-2"}

    def test_stale_node_round_trip(self, db):
        api.add_node("n1", "Old belief.", source="repo/old.py", db_path=db)
        api.retract_node("n1", reason="Superseded by new API", db_path=db)

        net = _load_network(db)
        md = export_markdown(net)
        assert "[STALE]" in md
        assert "Stale reason: Superseded by new API" in md

        net2 = Network()
        result = import_into_network(net2, md)
        assert result["claims_retracted"] == 1
        assert net2.nodes["n1"].truth_value == "OUT"

    def test_unless_round_trip(self, db):
        api.add_node("a", "Base fact.", db_path=db)
        api.add_node("b", "Blocking fact.", db_path=db)
        api.add_node("c", "Conditional belief.", sl="a", unless="b", db_path=db)

        net = _load_network(db)
        md = export_markdown(net)
        assert "- Unless: b" in md

        net2 = Network()
        import_into_network(net2, md)
        j = net2.nodes["c"].justifications[0]
        assert j.antecedents == ["a"]
        assert j.outlist == ["b"]

    def test_nogoods_export_format(self, db):
        """Verify nogoods are exported — note: parse_nogoods expects 'nogood-N: label'
        format but export writes 'nogood-NNN' without colon. Full nogood round-trip
        via markdown requires a format fix (pre-existing gap)."""
        api.add_node("x", "Fact X.", db_path=db)
        api.add_node("y", "Fact Y.", db_path=db)
        api.add_nogood(["x", "y"], db_path=db)

        net = _load_network(db)
        md = export_markdown(net)
        assert "## Nogoods" in md
        assert "Affects: x, y" in md

    def test_double_round_trip_stability(self, db):
        """Export -> import -> export should produce identical markdown."""
        api.add_node(
            "obs-1", "REST API.",
            source="repo/api.py",
            source_url="https://github.com/org/repo/blob/main/api.py",
            source_type="code",
            accepted_pr="https://github.com/org/repo/pull/10",
            db_path=db,
        )

        net1 = _load_network(db)
        md1 = export_markdown(net1)

        net2 = Network()
        import_into_network(net2, md1)
        md2 = export_markdown(net2)

        lines1 = [l for l in md1.split("\n") if not l.startswith("updated_at")]
        lines2 = [l for l in md2.split("\n") if not l.startswith("updated_at")]
        assert lines1 == lines2

    def test_repos_preserved(self, db):
        net = _load_network(db)
        repos = {"reasonsforge": "/Users/ben/git/reasonsforge"}
        md = export_markdown(net, repos=repos)
        assert "## Repos" in md
        assert "- reasonsforge: /Users/ben/git/reasonsforge" in md

    def test_frontmatter_round_trip(self, db):
        api.add_node("n1", "A belief.", db_path=db)
        net = _load_network(db)
        net.meta["project_name"] = "test-project"
        md = export_markdown(net)
        assert 'project_name: "test-project"' in md

        net2 = Network()
        import_into_network(net2, md)
        assert net2.meta.get("project_name") == "test-project"

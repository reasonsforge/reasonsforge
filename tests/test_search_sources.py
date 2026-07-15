"""Tests for search-sources command and search_source_chunks function."""

import json
import sqlite3

import pytest

from reasonsforge.ask import search_source_chunks


def _create_fts_db(path):
    """Create a minimal FTS5 chunks database for testing."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            text TEXT,
            cluster TEXT,
            filename TEXT,
            section TEXT
        )
    """)
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=id)")
    conn.execute("""
        INSERT INTO chunks (id, text, cluster, filename, section)
        VALUES (1, 'Retraction cascades propagate through the dependency graph',
                'tms', 'architecture.md', 'Retraction')
    """)
    conn.execute("""
        INSERT INTO chunks (id, text, cluster, filename, section)
        VALUES (2, 'Nodes are justified by SL justifications with support lists',
                'tms', 'architecture.md', 'Justifications')
    """)
    conn.execute("""
        INSERT INTO chunks (id, text, cluster, filename, section)
        VALUES (3, 'The belief network can be exported as JSON or Markdown',
                'export', 'usage.md', '')
    """)
    conn.execute("""
        INSERT INTO chunks_fts (rowid, text)
        VALUES (1, 'Retraction cascades propagate through the dependency graph')
    """)
    conn.execute("""
        INSERT INTO chunks_fts (rowid, text)
        VALUES (2, 'Nodes are justified by SL justifications with support lists')
    """)
    conn.execute("""
        INSERT INTO chunks_fts (rowid, text)
        VALUES (3, 'The belief network can be exported as JSON or Markdown')
    """)
    conn.commit()
    conn.close()
    return str(path)


class TestSearchSourceChunks:

    def test_basic_search(self, tmp_path):
        db = _create_fts_db(tmp_path / "chunks.db")
        results = search_source_chunks("retraction", db)
        assert len(results) >= 1
        assert results[0]["filename"] == "architecture.md"
        assert "Retraction" in results[0]["text"]

    def test_returns_list_of_dicts(self, tmp_path):
        db = _create_fts_db(tmp_path / "chunks.db")
        results = search_source_chunks("justifications", db)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert all(k in results[0] for k in ("text", "filename", "section"))

    def test_no_matches(self, tmp_path):
        db = _create_fts_db(tmp_path / "chunks.db")
        results = search_source_chunks("xyznonexistent", db)
        assert results == []

    def test_top_k_limits(self, tmp_path):
        db = _create_fts_db(tmp_path / "chunks.db")
        results = search_source_chunks("network belief retraction", db, top_k=1)
        assert len(results) <= 1

    def test_empty_query(self, tmp_path):
        db = _create_fts_db(tmp_path / "chunks.db")
        results = search_source_chunks("", db)
        assert results == []

    def test_stop_words_fallback_still_searches(self, tmp_path):
        db = _create_fts_db(tmp_path / "chunks.db")
        results = search_source_chunks("the a an", db)
        assert isinstance(results, list)

    def test_section_included(self, tmp_path):
        db = _create_fts_db(tmp_path / "chunks.db")
        results = search_source_chunks("retraction", db)
        assert results[0]["section"] == "Retraction"

    def test_empty_section(self, tmp_path):
        db = _create_fts_db(tmp_path / "chunks.db")
        results = search_source_chunks("exported JSON", db)
        assert len(results) >= 1
        assert results[0]["section"] == ""

    def test_bad_db_raises(self, tmp_path):
        with pytest.raises(sqlite3.OperationalError):
            search_source_chunks("test", str(tmp_path / "nonexistent.db"))


class TestSearchSourcesCli:

    def test_text_output(self, tmp_path, capsys):
        db = _create_fts_db(tmp_path / "chunks.db")
        from reasonsforge.cli import cmd_search_sources
        args = type("Args", (), {
            "query": "retraction",
            "db": db,
            "top_k": 10,
            "format": "text",
        })()
        cmd_search_sources(args)
        out = capsys.readouterr().out
        assert "architecture.md" in out
        assert "Retraction" in out

    def test_json_output(self, tmp_path, capsys):
        db = _create_fts_db(tmp_path / "chunks.db")
        from reasonsforge.cli import cmd_search_sources
        args = type("Args", (), {
            "query": "retraction",
            "db": db,
            "top_k": 10,
            "format": "json",
        })()
        cmd_search_sources(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert data[0]["filename"] == "architecture.md"

    def test_no_results_message(self, tmp_path, capsys):
        db = _create_fts_db(tmp_path / "chunks.db")
        from reasonsforge.cli import cmd_search_sources
        args = type("Args", (), {
            "query": "xyznonexistent",
            "db": db,
            "top_k": 10,
            "format": "text",
        })()
        cmd_search_sources(args)
        out = capsys.readouterr().out
        assert "No matching chunks" in out

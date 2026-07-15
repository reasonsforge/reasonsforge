"""Tests for reasonsforge.forge.index_sources — FTS5 chunk indexing."""

import sqlite3
import types
from pathlib import Path

import pytest

from reasonsforge.forge.index_sources import cmd_index_sources, _init_db, _insert_chunks


@pytest.fixture
def source_dir(tmp_path):
    d = tmp_path / "sources"
    d.mkdir()
    return d


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    wd = tmp_path / "work"
    wd.mkdir()
    monkeypatch.chdir(wd)
    return wd


def make_args(input_dir, db="rag_fts.db", rebuild=False, recursive=False,
              chunk_type="source", chunk_size=2000):
    return types.SimpleNamespace(
        input_dir=str(input_dir),
        db=db,
        rebuild=rebuild,
        recursive=recursive,
        type=chunk_type,
        chunk_size=chunk_size,
    )


# --- _init_db ---

def test_init_db_creates_tables(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = _init_db(db_path)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert "chunks" in tables
    assert "chunks_fts" in tables
    conn.close()


def test_init_db_rebuild_clears_data(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = _init_db(db_path)
    _insert_chunks(conn, ["test content"], "test.md")
    conn.commit()
    conn.close()

    conn = _init_db(db_path, rebuild=True)
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    assert cur.fetchone()[0] == 0
    conn.close()


# --- cmd_index_sources ---

def test_indexes_markdown_files(source_dir, work_dir):
    (source_dir / "doc.md").write_text("# Hello\nSome content about testing.")
    db_path = str(work_dir / "test.db")
    args = make_args(source_dir, db=db_path)
    cmd_index_sources(args)

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    assert cur.fetchone()[0] >= 1
    cur = conn.execute("SELECT filename FROM chunks")
    filenames = [row[0] for row in cur.fetchall()]
    assert any("doc.md" in f for f in filenames)
    conn.close()


def test_indexes_python_files(source_dir, work_dir):
    (source_dir / "module.py").write_text("import os\n\ndef hello():\n    pass\n")
    db_path = str(work_dir / "test.db")
    args = make_args(source_dir, db=db_path)
    cmd_index_sources(args)

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    assert cur.fetchone()[0] >= 1
    conn.close()


def test_skips_already_indexed(source_dir, work_dir, capsys):
    (source_dir / "doc.md").write_text("# Hello\nContent")
    db_path = str(work_dir / "test.db")
    args = make_args(source_dir, db=db_path)

    cmd_index_sources(args)
    cmd_index_sources(args)

    captured = capsys.readouterr()
    assert "already indexed" in captured.out


def test_rebuild_reindexes(source_dir, work_dir):
    (source_dir / "doc.md").write_text("# Hello\nContent")
    db_path = str(work_dir / "test.db")

    args = make_args(source_dir, db=db_path)
    cmd_index_sources(args)

    args = make_args(source_dir, db=db_path, rebuild=True)
    cmd_index_sources(args)

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    assert cur.fetchone()[0] >= 1
    conn.close()


def test_recursive_indexes_nested(source_dir, work_dir):
    subdir = source_dir / "sub"
    subdir.mkdir()
    (subdir / "nested.md").write_text("# Nested\nContent here")
    (source_dir / "top.md").write_text("# Top\nContent here")
    db_path = str(work_dir / "test.db")
    args = make_args(source_dir, db=db_path, recursive=True)
    cmd_index_sources(args)

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(DISTINCT filename) FROM chunks")
    assert cur.fetchone()[0] == 2
    conn.close()


def test_non_recursive_skips_nested(source_dir, work_dir):
    subdir = source_dir / "sub"
    subdir.mkdir()
    (subdir / "nested.md").write_text("# Nested\nContent here")
    (source_dir / "top.md").write_text("# Top\nContent here")
    db_path = str(work_dir / "test.db")
    args = make_args(source_dir, db=db_path, recursive=False)
    cmd_index_sources(args)

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(DISTINCT filename) FROM chunks")
    assert cur.fetchone()[0] == 1
    conn.close()


def test_chunk_type_stored(source_dir, work_dir):
    (source_dir / "doc.md").write_text("# Hello\nContent")
    db_path = str(work_dir / "test.db")
    args = make_args(source_dir, db=db_path, chunk_type="summary")
    cmd_index_sources(args)

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT chunk_type FROM chunks")
    types = [row[0] for row in cur.fetchall()]
    assert all(t == "summary" for t in types)
    conn.close()


def test_fts5_search_works(source_dir, work_dir):
    (source_dir / "doc.md").write_text("# Kubernetes\nPod scheduling and node affinity rules.")
    db_path = str(work_dir / "test.db")
    args = make_args(source_dir, db=db_path)
    cmd_index_sources(args)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("""
        SELECT c.text, c.filename
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH 'kubernetes'
    """)
    results = [dict(row) for row in cur.fetchall()]
    assert len(results) >= 1
    assert "Kubernetes" in results[0]["text"]
    conn.close()


def test_large_file_chunked(source_dir, work_dir):
    text = "# Section 1\n" + "x" * 3000 + "\n\n# Section 2\n" + "y" * 3000
    (source_dir / "big.md").write_text(text)
    db_path = str(work_dir / "test.db")
    args = make_args(source_dir, db=db_path, chunk_size=2000)
    cmd_index_sources(args)

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    assert cur.fetchone()[0] == 2
    conn.close()


def test_source_url_from_frontmatter(source_dir, work_dir):
    fm = "---\nsource_url: https://example.com/doc\n---\n\n# Hello\nContent"
    (source_dir / "doc.md").write_text(fm)
    db_path = str(work_dir / "test.db")
    args = make_args(source_dir, db=db_path)
    cmd_index_sources(args)

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT source_url FROM chunks")
    urls = [row[0] for row in cur.fetchall()]
    assert all(u == "https://example.com/doc" for u in urls)
    conn.close()

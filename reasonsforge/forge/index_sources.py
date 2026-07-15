"""Build FTS5 chunks database from source documents."""

import sqlite3
import sys
from pathlib import Path

from .chunk_docs import chunk_markdown, chunk_python, chunk_fixed, _strip_frontmatter


DEFAULT_DB = "rag_fts.db"
DEFAULT_CHUNK_SIZE = 2000


def _init_db(db_path, rebuild=False):
    """Create the chunks and FTS5 tables."""
    conn = sqlite3.connect(db_path)
    if rebuild:
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.execute("DROP TABLE IF EXISTS chunks")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            cluster TEXT DEFAULT '',
            filename TEXT NOT NULL,
            section TEXT DEFAULT '',
            chunk_type TEXT DEFAULT 'source',
            source_url TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(text, content=chunks, content_rowid=id,
                   tokenize="porter unicode61")
    """)
    conn.commit()
    return conn


def _insert_chunks(conn, chunks, filename, chunk_type="source", source_url=""):
    """Insert chunks into the database."""
    for i, chunk_text in enumerate(chunks):
        section = f"chunk {i + 1}/{len(chunks)}" if len(chunks) > 1 else ""
        conn.execute(
            "INSERT INTO chunks (text, filename, section, chunk_type, source_url) "
            "VALUES (?, ?, ?, ?, ?)",
            (chunk_text.strip(), str(filename), section, chunk_type, source_url),
        )


def cmd_index_sources(args):
    """Build FTS5 chunks database from source documents."""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Source directory not found: {input_dir}")
        sys.exit(1)

    db_path = args.db
    rebuild = args.rebuild
    chunk_type = args.type
    max_chars = args.chunk_size

    glob = input_dir.rglob if getattr(args, "recursive", False) else input_dir.glob
    sources = sorted(
        [*glob("*.md"), *glob("*.py"), *glob("*.txt")],
        key=lambda p: p.name,
    )
    if not sources:
        print(f"No .md, .py, or .txt files in {input_dir}")
        return

    print(f"Indexing {len(sources)} files into {db_path}")
    conn = _init_db(db_path, rebuild=rebuild)

    try:
        existing = set()
        if not rebuild:
            cur = conn.execute("SELECT DISTINCT filename FROM chunks")
            existing = {row[0] for row in cur.fetchall()}

        indexed = 0
        skipped = 0

        for source_path in sources:
            if str(source_path) in existing:
                skipped += 1
                continue

            raw = source_path.read_text()
            meta, content = _strip_frontmatter(raw)

            if not content.strip():
                continue

            source_url = meta.get("source_url") or meta.get("source", "")
            if source_url and not source_url.startswith(("http://", "https://")):
                source_url = ""

            if source_path.suffix == ".py":
                chunks = chunk_python(content, max_chars=max_chars)
            elif source_path.suffix == ".md":
                chunks = chunk_markdown(content, max_chars=max_chars)
            else:
                chunks = chunk_fixed(content, max_chars=max_chars)

            _insert_chunks(conn, chunks, source_path, chunk_type=chunk_type,
                           source_url=source_url)
            indexed += 1
            print(f"  {source_path.name} -> {len(chunks)} chunk(s)")

        if indexed:
            conn.commit()
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            conn.commit()

        print(f"\nIndexed {indexed} files ({skipped} already indexed)")
    finally:
        conn.close()

"""Tests for reasonsforge.forge.chunk_docs — document chunking."""

import types
from pathlib import Path

import pytest

from reasonsforge.forge.chunk_docs import (
    chunk_markdown,
    chunk_python,
    chunk_fixed,
    cmd_chunk_docs,
)


# --- chunk_markdown ---

def test_chunk_markdown_by_headings():
    text = "# Section 1\nContent one.\n\n# Section 2\nContent two.\n"
    chunks = chunk_markdown(text, max_chars=50)
    assert len(chunks) == 2
    assert "Section 1" in chunks[0]
    assert "Section 2" in chunks[1]


def test_chunk_markdown_merges_small_sections():
    text = "# A\nShort.\n\n# B\nAlso short.\n\n# C\nStill short.\n"
    chunks = chunk_markdown(text, max_chars=1000)
    assert len(chunks) == 1
    assert "A" in chunks[0]
    assert "C" in chunks[0]


def test_chunk_markdown_h2_headings():
    text = "## First\nContent.\n\n## Second\nMore content.\n"
    chunks = chunk_markdown(text, max_chars=30)
    assert len(chunks) == 2


def test_chunk_markdown_no_headings_falls_back():
    text = "Just a plain text document with no headings at all. " * 100
    chunks = chunk_markdown(text, max_chars=200)
    assert len(chunks) > 1


# --- chunk_python ---

def test_chunk_python_by_definitions():
    text = (
        "import os\n\n"
        "def foo():\n    pass\n\n"
        "def bar():\n    pass\n"
    )
    chunks = chunk_python(text, max_chars=40)
    assert len(chunks) == 2
    assert "def foo" in chunks[0]
    assert "def bar" in chunks[1]


def test_chunk_python_keeps_imports():
    text = (
        "import os\nimport sys\n\n"
        "def first():\n    pass\n\n"
        "def second():\n    pass\n"
    )
    chunks = chunk_python(text, max_chars=60)
    assert len(chunks) == 2
    for chunk in chunks:
        assert "import os" in chunk


def test_chunk_python_class_boundary():
    text = (
        "import x\n\n"
        "class Foo:\n    pass\n\n"
        "class Bar:\n    pass\n"
    )
    chunks = chunk_python(text, max_chars=40)
    assert len(chunks) == 2
    assert "class Foo" in chunks[0]
    assert "class Bar" in chunks[1]


def test_chunk_python_decorator_stays_with_function():
    text = (
        "import os\n\n"
        "@decorator\n"
        "def foo():\n    pass\n\n"
        "def bar():\n    pass\n"
    )
    chunks = chunk_python(text, max_chars=60)
    assert len(chunks) == 2
    assert "@decorator" in chunks[0]
    assert "def foo" in chunks[0]


def test_chunk_python_no_defs_falls_back():
    text = "x = 1\ny = 2\nz = 3\n" * 100
    chunks = chunk_python(text, max_chars=100)
    assert len(chunks) > 1


# --- chunk_fixed ---

def test_chunk_fixed_small_text_single_chunk():
    text = "Short text."
    chunks = chunk_fixed(text, max_chars=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_fixed_with_overlap():
    text = "a" * 1000
    chunks = chunk_fixed(text, max_chars=400, overlap=100)
    assert len(chunks) >= 3
    assert chunks[0][-100:] == chunks[1][:100]


# --- cmd_chunk_docs ---

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


def make_args(input_dir, threshold=100, dry_run=False, recursive=False):
    return types.SimpleNamespace(
        input_dir=str(input_dir),
        threshold=threshold,
        dry_run=dry_run,
        recursive=recursive,
    )


def test_recursive_discovers_nested_files(source_dir, work_dir):
    subdir = source_dir / "subdir"
    subdir.mkdir()
    text = "# A\n" + "x" * 200 + "\n\n# B\n" + "y" * 200
    (subdir / "nested.md").write_text(text)
    args = make_args(source_dir, threshold=100, recursive=True)
    cmd_chunk_docs(args)
    entries = list((work_dir / "sources" / "chunks").rglob("*.md"))
    assert len(entries) == 2


def test_non_recursive_skips_nested_files(source_dir, work_dir, capsys):
    subdir = source_dir / "subdir"
    subdir.mkdir()
    text = "# A\n" + "x" * 200 + "\n\n# B\n" + "y" * 200
    (subdir / "nested.md").write_text(text)
    args = make_args(source_dir, threshold=100, recursive=False)
    cmd_chunk_docs(args)
    entries = list((work_dir / "sources" / "chunks").rglob("*.md")) if (work_dir / "sources" / "chunks").exists() else []
    assert len(entries) == 0


def test_skips_small_files(source_dir, work_dir, capsys):
    (source_dir / "small.md").write_text("# Short\nContent")
    args = make_args(source_dir, threshold=30000)
    cmd_chunk_docs(args)
    captured = capsys.readouterr()
    assert "Chunked 0 files" in captured.out
    entries = list((work_dir / "sources" / "chunks").rglob("*.md")) if (work_dir / "sources" / "chunks").exists() else []
    assert len(entries) == 0


def test_chunks_large_markdown(source_dir, work_dir):
    text = "# Section 1\n" + "x" * 200 + "\n\n# Section 2\n" + "y" * 200
    (source_dir / "big.md").write_text(text)
    args = make_args(source_dir, threshold=100)
    cmd_chunk_docs(args)
    entries = list((work_dir / "sources" / "chunks").rglob("*.md"))
    assert len(entries) == 2
    contents = [e.read_text() for e in sorted(entries)]
    assert "Section 1" in contents[0]
    assert "Section 2" in contents[1]


def test_chunks_large_python(source_dir, work_dir):
    text = "import os\n\n" + "def foo():\n    " + "x = 1\n    " * 50 + "\n\ndef bar():\n    " + "y = 2\n    " * 50
    (source_dir / "big.py").write_text(text)
    args = make_args(source_dir, threshold=100)
    cmd_chunk_docs(args)
    entries = list((work_dir / "sources" / "chunks").rglob("*.md"))
    assert len(entries) == 2


def test_dry_run_no_files_created(source_dir, work_dir, capsys):
    text = "# A\n" + "x" * 200 + "\n\n# B\n" + "y" * 200
    (source_dir / "big.md").write_text(text)
    args = make_args(source_dir, threshold=100, dry_run=True)
    cmd_chunk_docs(args)
    entries_dir = work_dir / "sources" / "chunks"
    assert not entries_dir.exists() or len(list(entries_dir.rglob("*.md"))) == 0
    captured = capsys.readouterr()
    assert "chunk 1" in captured.out

    manifest = work_dir / ".chunked-docs"
    assert not manifest.exists()


def test_dry_run_does_not_poison_manifest(source_dir, work_dir):
    """Dry-run should not prevent subsequent real runs."""
    text = "# A\n" + "x" * 200 + "\n\n# B\n" + "y" * 200
    (source_dir / "big.md").write_text(text)

    dry_args = make_args(source_dir, threshold=100, dry_run=True)
    cmd_chunk_docs(dry_args)

    real_args = make_args(source_dir, threshold=100, dry_run=False)
    cmd_chunk_docs(real_args)

    entries = list((work_dir / "sources" / "chunks").rglob("*.md"))
    assert len(entries) == 2


def test_provenance_frontmatter(source_dir, work_dir):
    fm = "---\nsource_url: https://example.com/doc\nsource_id: abc\n---\n\n"
    text = fm + "# Part 1\n" + "x" * 200 + "\n\n# Part 2\n" + "y" * 200
    (source_dir / "doc.md").write_text(text)
    args = make_args(source_dir, threshold=100)
    cmd_chunk_docs(args)
    entries = sorted((work_dir / "sources" / "chunks").rglob("*.md"))
    content = entries[0].read_text()
    assert "source_url: https://example.com/doc" in content
    assert "source_id: abc" in content
    assert "chunk: 1/" in content


def test_manifest_tracking(source_dir, work_dir):
    text = "# A\n" + "x" * 200 + "\n\n# B\n" + "y" * 200
    (source_dir / "big.md").write_text(text)
    args = make_args(source_dir, threshold=100)
    cmd_chunk_docs(args)
    entries_count_1 = len(list((work_dir / "sources" / "chunks").rglob("*.md")))

    cmd_chunk_docs(args)
    entries_count_2 = len(list((work_dir / "sources" / "chunks").rglob("*.md")))
    assert entries_count_1 == entries_count_2


def test_chunk_names_include_stem(source_dir, work_dir):
    text = "# A\n" + "x" * 200 + "\n\n# B\n" + "y" * 200
    (source_dir / "my-doc.md").write_text(text)
    args = make_args(source_dir, threshold=100)
    cmd_chunk_docs(args)
    entries = list((work_dir / "sources" / "chunks").rglob("*.md"))
    names = [e.name for e in entries]
    assert any("my-doc-chunk-1" in n for n in names)
    assert any("my-doc-chunk-2" in n for n in names)

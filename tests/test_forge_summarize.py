"""Tests for reasonsforge.forge.summarize."""

import asyncio
import types
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from reasonsforge.forge.summarize import cmd_summarize
from reasonsforge.forge.prompts import SUMMARIZE, SUMMARIZE_CODE


# --- Fixtures ---

@pytest.fixture
def source_dir(tmp_path):
    """Create a temp directory with sample source files."""
    src = tmp_path / "sources"
    src.mkdir()
    return src


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    """Set working directory to tmp_path so .summarized manifest is isolated."""
    wd = tmp_path / "work"
    wd.mkdir()
    monkeypatch.chdir(wd)
    return wd


def make_args(input_dir, model="test-model", limit=None, recursive=False, parallel=1):
    return types.SimpleNamespace(input_dir=str(input_dir), model=model, limit=limit, recursive=recursive, parallel=parallel)


def _find_entry(work_dir):
    """Find the generated entry file under entries/."""
    entries = list((work_dir / "entries").rglob("*.md"))
    assert len(entries) == 1, f"Expected 1 entry, found {len(entries)}: {entries}"
    return entries[0]


# --- File discovery tests ---

def test_discovers_md_files(source_dir, work_dir):
    (source_dir / "doc.md").write_text("# Hello\nSome content")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Topic Title\nSummary"):
        cmd_summarize(args)

    entry = _find_entry(work_dir)
    assert "Summary" in entry.read_text()


def test_discovers_py_files(source_dir, work_dir):
    (source_dir / "module.py").write_text("def hello(): pass")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Module\nSummary"):
        cmd_summarize(args)

    entry = _find_entry(work_dir)
    assert "Summary" in entry.read_text()


def test_discovers_both_md_and_py(source_dir, work_dir):
    (source_dir / "alpha.md").write_text("# Alpha\nContent")
    (source_dir / "beta.py").write_text("x = 1")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary") as mock_llm:
        cmd_summarize(args)

    assert mock_llm.call_count == 2
    entries = list((work_dir / "entries").rglob("*.md"))
    assert len(entries) == 2


def test_recursive_discovers_nested_files(source_dir, work_dir):
    subdir = source_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.md").write_text("# Nested\nContent")
    (source_dir / "top.md").write_text("# Top\nContent")
    args = make_args(source_dir, recursive=True)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary") as mock_llm:
        cmd_summarize(args)

    assert mock_llm.call_count == 2
    entries = list((work_dir / "entries").rglob("*.md"))
    assert len(entries) == 2


def test_non_recursive_skips_nested_files(source_dir, work_dir):
    subdir = source_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.md").write_text("# Nested\nContent")
    (source_dir / "top.md").write_text("# Top\nContent")
    args = make_args(source_dir, recursive=False)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary") as mock_llm:
        cmd_summarize(args)

    assert mock_llm.call_count == 1


def test_ignores_other_extensions(source_dir, work_dir):
    (source_dir / "data.json").write_text("{}")
    (source_dir / "image.png").write_bytes(b"\x89PNG")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock) as mock_llm:
        cmd_summarize(args)

    assert not mock_llm.called


def test_discovers_txt_files(source_dir, work_dir):
    (source_dir / "notes.txt").write_text("Some plain text notes")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Notes\nSummary"):
        cmd_summarize(args)

    entry = _find_entry(work_dir)
    assert "Summary" in entry.read_text()


# --- Template selection tests ---

def test_uses_summarize_code_for_py(source_dir, work_dir):
    (source_dir / "module.py").write_text("def hello(): pass")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Module\nSummary") as mock_llm:
        cmd_summarize(args)

    prompt = mock_llm.call_args[0][0]
    assert "source code" in prompt.lower()


def test_uses_summarize_for_md(source_dir, work_dir):
    (source_dir / "doc.md").write_text("# Hello\nSome content")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Doc Title\nSummary") as mock_llm:
        cmd_summarize(args)

    prompt = mock_llm.call_args[0][0]
    assert "documentation page" in prompt.lower()


# --- Truncation tests ---

def test_truncation_warning_for_large_file(source_dir, work_dir, capsys):
    (source_dir / "big.md").write_text("x" * 50000)
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Big Doc\nSummary"):
        cmd_summarize(args)

    captured = capsys.readouterr()
    assert "WARN: truncated from 50000 to 30000 chars" in captured.out
    assert "Consider: reasonsforge forge chunk-docs" in captured.out


def test_truncation_content_is_capped(source_dir, work_dir):
    (source_dir / "big.md").write_text("x" * 50000)
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Big\nSummary") as mock_llm:
        cmd_summarize(args)

    prompt = mock_llm.call_args[0][0]
    assert "[Truncated" in prompt
    assert len(prompt) < 50000


def test_no_truncation_warning_for_small_file(source_dir, work_dir, capsys):
    (source_dir / "small.md").write_text("Short content")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Small\nSummary"):
        cmd_summarize(args)

    captured = capsys.readouterr()
    assert "WARN" not in captured.out


# --- Manifest / idempotency tests ---

def test_skips_already_summarized(source_dir, work_dir):
    (source_dir / "doc.md").write_text("# Hello\nContent")
    manifest = work_dir / ".summarized"
    manifest.write_text(f"{source_dir / 'doc.md'}\n")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock) as mock_llm:
        cmd_summarize(args)

    assert not mock_llm.called


def test_manifest_records_processed_file(source_dir, work_dir):
    (source_dir / "doc.md").write_text("# Hello\nContent")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary"):
        cmd_summarize(args)

    manifest = work_dir / ".summarized"
    assert manifest.exists()
    assert str(source_dir / "doc.md") in manifest.read_text()


# --- Frontmatter stripping tests ---

def test_strips_frontmatter_before_summarizing(source_dir, work_dir):
    content = "---\nsource: https://example.com\n---\n\nActual content here"
    (source_dir / "doc.md").write_text(content)
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary") as mock_llm:
        cmd_summarize(args)

    prompt = mock_llm.call_args[0][0]
    assert "source:" not in prompt
    assert "Actual content here" in prompt


def test_strips_source_url_frontmatter(source_dir, work_dir):
    content = "---\nsource_url: https://example.com\n---\n\nActual content here"
    (source_dir / "doc.md").write_text(content)
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary") as mock_llm:
        cmd_summarize(args)

    prompt = mock_llm.call_args[0][0]
    assert "source_url" not in prompt
    assert "Actual content here" in prompt


def test_skips_empty_content_after_frontmatter(source_dir, work_dir, capsys):
    (source_dir / "empty.md").write_text("---\nsource: https://example.com\n---\n\n")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock) as mock_llm:
        cmd_summarize(args)

    assert not mock_llm.called
    captured = capsys.readouterr()
    assert "SKIP" in captured.out


# --- Provenance frontmatter tests ---

def test_entry_has_source_frontmatter(source_dir, work_dir):
    """Generated entry includes source path in frontmatter."""
    (source_dir / "doc.md").write_text("# Hello\nContent")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary"):
        cmd_summarize(args)

    entry = _find_entry(work_dir)
    content = entry.read_text()
    assert content.startswith("---\n")
    assert f"source: {source_dir}/doc.md" in content


def test_entry_has_source_url_from_fetch_frontmatter(source_dir, work_dir):
    """source: URL from fetch-docs frontmatter propagates as source_url."""
    fm = "---\nsource: https://example.com/docs/page\nfetched: 2026-06-04\n---\n\nDoc content"
    (source_dir / "page.md").write_text(fm)
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Page\nSummary"):
        cmd_summarize(args)

    entry = _find_entry(work_dir)
    content = entry.read_text()
    assert "source_url: https://example.com/docs/page" in content


def test_entry_has_source_id_when_present(source_dir, work_dir):
    """source_id propagates from source frontmatter to entry."""
    fm = "---\nsource_url: https://example.com\nsource_id: abc123\n---\n\nContent"
    (source_dir / "doc.md").write_text(fm)
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary"):
        cmd_summarize(args)

    entry = _find_entry(work_dir)
    content = entry.read_text()
    assert "source_url: https://example.com" in content
    assert "source_id: abc123" in content


def test_entry_contains_llm_summary(source_dir, work_dir):
    """The LLM summary is written as the entry body."""
    (source_dir / "doc.md").write_text("# Hello\nContent")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## My Title\nDetailed summary here"):
        cmd_summarize(args)

    entry = _find_entry(work_dir)
    content = entry.read_text()
    assert "Detailed summary here" in content


def test_entry_directory_structure(source_dir, work_dir):
    """Entries are written to entries/YYYY/MM/DD/topic.md."""
    (source_dir / "my-topic.md").write_text("# Topic\nContent")
    args = make_args(source_dir)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary"):
        cmd_summarize(args)

    entry = _find_entry(work_dir)
    assert entry.name == "my-topic.md"
    parts = entry.relative_to(work_dir).parts
    assert parts[0] == "entries"


# --- Prompt template tests ---

def test_summarize_template_requests_descriptive_title():
    assert "<Descriptive Title>" in SUMMARIZE


def test_summarize_code_template_requests_descriptive_title():
    assert "<Descriptive Title>" in SUMMARIZE_CODE


def test_summarize_template_has_content_placeholder():
    assert "{content}" in SUMMARIZE


def test_summarize_code_template_has_content_placeholder():
    assert "{content}" in SUMMARIZE_CODE


# --- Parallel tests ---

def test_parallel_summarizes_multiple_files(source_dir, work_dir):
    for i in range(4):
        (source_dir / f"doc{i}.md").write_text(f"# Doc {i}\nContent {i}")
    args = make_args(source_dir, parallel=2)

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary"):
        cmd_summarize(args)

    entries = list((work_dir / "entries").rglob("*.md"))
    assert len(entries) == 4


def test_parallel_default_is_sequential(source_dir, work_dir):
    (source_dir / "doc.md").write_text("# Hello\nContent")
    args = make_args(source_dir)
    assert args.parallel == 1

    with patch("reasonsforge.forge.summarize.check_model_available", return_value=True), \
         patch("reasonsforge.forge.summarize.invoke", new_callable=AsyncMock, return_value="## Title\nSummary"):
        cmd_summarize(args)

    entry = _find_entry(work_dir)
    assert "Summary" in entry.read_text()

"""Tests for reasonsforge.forge.propose — JSON belief parsing and incremental batch writing."""

import json
import types
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from reasonsforge.forge.propose import cmd_propose_beliefs


@pytest.fixture
def entries_dir(tmp_path):
    d = tmp_path / "entries"
    d.mkdir()
    return d


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    wd = tmp_path / "work"
    wd.mkdir()
    (wd / ".forge").mkdir()
    monkeypatch.chdir(wd)
    return wd


def make_args(input_dir, output="proposed-beliefs.md", batch_size=2, model="test", parallel=1):
    return types.SimpleNamespace(
        input_dir=str(input_dir),
        output=output,
        batch_size=batch_size,
        model=model,
        all=False,
        parallel=parallel,
    )


def _json_beliefs(*beliefs, accept=True):
    """Helper to build a JSON response from (id, claim) tuples."""
    return json.dumps([
        {"id": b[0], "claim": b[1], "accept": accept, "source": "entry.md", "source_url": ""}
        for b in beliefs
    ])


def test_proposals_written_after_each_batch(entries_dir, work_dir):
    """Proposals from completed batches survive a crash in a later batch."""
    for i in range(4):
        (entries_dir / f"entry{i}.md").write_text(f"# Entry {i}\nContent {i}")

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=2)

    call_count = 0
    def invoke_side_effect(prompt, model=None, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated crash")
        return _json_beliefs((f"belief-from-batch-{call_count}", "A belief."))

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    content = output.read_text()
    assert "belief-from-batch-1" in content
    assert "belief-from-batch-2" not in content


def test_all_batches_written_on_success(entries_dir, work_dir):
    for i in range(4):
        (entries_dir / f"entry{i}.md").write_text(f"# Entry {i}\nContent {i}")

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=2)

    call_count = 0
    def invoke_side_effect(prompt, model=None, timeout=None):
        nonlocal call_count
        call_count += 1
        return _json_beliefs((f"belief-{call_count}", "A belief."))

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    content = output.read_text()
    assert "belief-1" in content
    assert "belief-2" in content


def test_existing_beliefs_filtered_per_batch(entries_dir, work_dir):
    (entries_dir / "entry0.md").write_text("# Entry\nContent")

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=5)

    existing = [{"id": "already-exists", "text": "old belief", "source": ""}]

    def invoke_side_effect(prompt, model=None, timeout=None):
        return _json_beliefs(
            ("already-exists", "Duplicate."),
            ("new-belief", "Fresh."),
        )

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=existing), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    content = output.read_text()
    assert "new-belief" in content
    assert "already-exists" not in content


def test_failed_batch_entries_not_marked_processed(entries_dir, work_dir):
    """Entries from failed batches are not marked as processed."""
    for i in range(4):
        (entries_dir / f"entry{i}.md").write_text(f"# Entry {i}\nContent {i}")

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=2)

    call_count = 0
    def invoke_side_effect(prompt, model=None, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated crash")
        return _json_beliefs((f"belief-{call_count}", "A belief."))

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    processed_path = work_dir / ".forge" / "proposed-entries.json"
    processed = json.loads(processed_path.read_text())
    # Batch 1 (entry0, entry1) succeeded — should be marked processed
    assert any("entry0" in k for k in processed)
    assert any("entry1" in k for k in processed)
    # Batch 2 (entry2, entry3) failed — should NOT be marked processed
    assert not any("entry2" in k for k in processed)
    assert not any("entry3" in k for k in processed)


def test_json_retry_on_bad_response(entries_dir, work_dir):
    """When LLM returns non-JSON, retry and parse the retry response."""
    (entries_dir / "entry0.md").write_text("# Entry\nContent")

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=5)

    call_count = 0
    def invoke_side_effect(prompt, model=None, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "Here are some beliefs about the code..."
        return _json_beliefs(("retried-belief", "A belief from retry."))

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    content = output.read_text()
    assert "retried-belief" in content
    assert call_count == 2


def test_json_with_code_fence(entries_dir, work_dir):
    """LLM response wrapped in code fences is parsed correctly."""
    (entries_dir / "entry0.md").write_text("# Entry\nContent")

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=5)

    def invoke_side_effect(prompt, model=None, timeout=None):
        return '```json\n' + _json_beliefs(("fenced-belief", "A belief.")) + '\n```'

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    content = output.read_text()
    assert "fenced-belief" in content


def test_source_url_extracted_from_entry_frontmatter(entries_dir, work_dir):
    """source_url from entry frontmatter is passed in batch header."""
    fm = "---\nsource_url: https://example.com/doc\n---\n\n# Entry\nContent"
    (entries_dir / "entry0.md").write_text(fm)

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=5)

    captured_prompt = None
    def invoke_side_effect(prompt, model=None, timeout=None):
        nonlocal captured_prompt
        captured_prompt = prompt
        return _json_beliefs(("test-belief", "A belief."))

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    assert "SOURCE_URL: https://example.com/doc" in captured_prompt


def test_source_key_url_extracted_from_entry_frontmatter(entries_dir, work_dir):
    """source: with URL value is used as SOURCE_URL in batch header."""
    fm = "---\nsource: https://example.com/page\n---\n\n# Entry\nContent"
    (entries_dir / "entry0.md").write_text(fm)

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=5)

    captured_prompt = None
    def invoke_side_effect(prompt, model=None, timeout=None):
        nonlocal captured_prompt
        captured_prompt = prompt
        return _json_beliefs(("test-belief", "A belief."))

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    assert "SOURCE_URL: https://example.com/page" in captured_prompt


def test_source_key_local_path_not_used_as_url(entries_dir, work_dir):
    """source: with local file path is NOT used as SOURCE_URL."""
    fm = "---\nsource: sources/doc.md\n---\n\n# Entry\nContent"
    (entries_dir / "entry0.md").write_text(fm)

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=5)

    captured_prompt = None
    def invoke_side_effect(prompt, model=None, timeout=None):
        nonlocal captured_prompt
        captured_prompt = prompt
        return _json_beliefs(("test-belief", "A belief."))

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    assert "| SOURCE_URL:" not in captured_prompt


def test_appends_to_existing_output_file(entries_dir, work_dir):
    """When output file already exists, new proposals are appended."""
    (entries_dir / "entry0.md").write_text("# Entry\nContent")

    output = work_dir / "proposed-beliefs.md"
    output.write_text("# Proposed Beliefs\n\n### [ACCEPT] prior-belief\nOld.\n\n")
    args = make_args(entries_dir, output=str(output), batch_size=5)

    def invoke_side_effect(prompt, model=None, timeout=None):
        return _json_beliefs(("new-belief", "Fresh."))

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    content = output.read_text()
    assert "prior-belief" in content
    assert "new-belief" in content


def test_parallel_processes_all_batches(entries_dir, work_dir):
    """With parallel=2, all batches are processed and results written."""
    for i in range(6):
        (entries_dir / f"entry{i}.md").write_text(f"# Entry {i}\nContent {i}")

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=2, parallel=2)

    call_count = 0
    def invoke_side_effect(prompt, model=None, timeout=None):
        nonlocal call_count
        call_count += 1
        return _json_beliefs((f"belief-{call_count}", f"A belief from batch {call_count}."))

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    content = output.read_text()
    assert call_count == 3
    assert "belief-1" in content
    assert "belief-2" in content
    assert "belief-3" in content


def test_accept_verdict_from_llm(entries_dir, work_dir):
    """LLM's accept=true produces [ACCEPT] in output."""
    (entries_dir / "entry0.md").write_text("# Entry\nContent")

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=5)

    def invoke_side_effect(prompt, model=None, timeout=None):
        return _json_beliefs(("good-belief", "A solid claim."), accept=True)

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    content = output.read_text()
    assert "### [ACCEPT] good-belief" in content
    assert "### [REJECT]" not in content


def test_reject_verdict_from_llm(entries_dir, work_dir):
    """LLM's accept=false produces [REJECT] in output."""
    (entries_dir / "entry0.md").write_text("# Entry\nContent")

    output = work_dir / "proposed-beliefs.md"
    args = make_args(entries_dir, output=str(output), batch_size=5)

    def invoke_side_effect(prompt, model=None, timeout=None):
        return _json_beliefs(("weak-belief", "A vague claim."), accept=False)

    with patch("reasonsforge.forge.propose.check_model_available", return_value=True), \
         patch("reasonsforge.forge.propose.invoke", new_callable=AsyncMock, side_effect=invoke_side_effect), \
         patch("reasonsforge.forge.propose._load_existing_beliefs", return_value=[]), \
         patch("reasonsforge.forge.propose._has_embeddings", return_value=False):
        cmd_propose_beliefs(args)

    content = output.read_text()
    assert "### [REJECT] weak-belief" in content
    assert "### [ACCEPT]" not in content

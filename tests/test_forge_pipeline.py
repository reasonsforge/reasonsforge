"""Tests for the pipeline command."""

import types
from pathlib import Path
from unittest.mock import patch

import pytest

from reasonsforge.forge.pipeline import (
    cmd_pipeline,
    cmd_derive_review_repair,
    _run_convergence_loop,
    _stage_extract,
    _stage_derive,
    _stage_review,
    _stage_repair,
    _stage_deduplicate,
    _load_state,
    STATE_FILE,
)
from reasonsforge.forge.propose import auto_accept_proposals


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    """Set working directory to tmp_path for isolated pipeline runs."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sources").mkdir()
    (tmp_path / "entries").mkdir()
    (tmp_path / "reasons.db").touch()
    return tmp_path


def make_pipeline_args(**overrides):
    defaults = dict(
        pdf=None,
        sources_dir="sources",
        model="claude",
        rounds=3,
        max_derive_rounds=10,
        no_auto_accept=False,
        timeout=600,
        domain="Test domain",
        resume=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def make_drr_args(**overrides):
    defaults = dict(
        model="claude",
        rounds=3,
        max_derive_rounds=10,
        timeout=600,
        domain="Test domain",
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# --- auto_accept_proposals ---

class TestAutoAcceptProposals:
    def test_replaces_markers(self, tmp_path):
        f = tmp_path / "proposals.md"
        f.write_text(
            "### [ACCEPT/REJECT] belief-one\n"
            "Text one\n"
            "### [ACCEPT/REJECT] belief-two\n"
            "Text two\n"
        )
        auto_accept_proposals(str(f))
        text = f.read_text()
        assert "[ACCEPT/REJECT]" not in text
        assert text.count("[ACCEPT]") == 2

    def test_preserves_already_accepted(self, tmp_path):
        f = tmp_path / "proposals.md"
        f.write_text(
            "### [ACCEPT] already-good\n"
            "Text\n"
            "### [ACCEPT/REJECT] needs-accept\n"
            "Text\n"
        )
        auto_accept_proposals(str(f))
        text = f.read_text()
        assert text.count("[ACCEPT]") == 2
        assert "[ACCEPT/REJECT]" not in text

    def test_converts_reject_to_accept(self, tmp_path):
        f = tmp_path / "proposals.md"
        f.write_text(
            "### [ACCEPT] good-belief\n"
            "Text\n"
            "### [REJECT] weak-belief\n"
            "Text\n"
        )
        auto_accept_proposals(str(f))
        text = f.read_text()
        assert text.count("[ACCEPT]") == 2
        assert "[REJECT]" not in text

    def test_no_markers_is_noop(self, tmp_path):
        f = tmp_path / "proposals.md"
        original = "### ACCEPT belief-one\nText\n"
        f.write_text(original)
        auto_accept_proposals(str(f))
        assert f.read_text() == original


# --- Stage: Extract ---

class TestStageExtract:
    def test_stops_on_no_auto_accept(self, work_dir, capsys):
        args = make_pipeline_args(no_auto_accept=True)
        with patch("reasonsforge.forge.propose.cmd_propose_beliefs"):
            result = _stage_extract(args)
        assert result is False
        captured = capsys.readouterr()
        assert "--no-auto-accept" in captured.err

    def test_auto_accepts_and_imports(self, work_dir):
        proposals = work_dir / "proposed-beliefs.md"
        proposals.write_text("### [ACCEPT/REJECT] test-belief\nText\n- Source: test\n")
        args = make_pipeline_args()

        with patch("reasonsforge.forge.propose.cmd_propose_beliefs"), \
             patch("reasonsforge.forge.propose.cmd_accept_beliefs") as mock_accept:
            result = _stage_extract(args)

        assert result is True
        assert mock_accept.called
        text = proposals.read_text()
        assert "[ACCEPT]" in text
        assert "[ACCEPT/REJECT]" not in text


# --- Stage: Derive ---

class TestStageDerive:
    def test_returns_zero_on_empty_network(self, work_dir):
        args = make_pipeline_args()
        with patch("reasonsforge.api.export_network", return_value={"nodes": {}}):
            added = _stage_derive(args)
        assert added == 0

    def test_saturates_on_no_proposals(self, work_dir):
        args = make_pipeline_args()
        nodes = {"belief-1": {"text": "Test", "truth_value": "IN", "justifications": []}}
        stats = {"total_in": 1, "total_derived": 0, "max_depth": 0, "agents": 0}
        with patch("reasonsforge.api.export_network", return_value={"nodes": nodes}), \
             patch("reasonsforge.derive.build_prompt", return_value=("prompt", stats)), \
             patch("reasonsforge.forge.llm.invoke_sync", return_value="No proposals"), \
             patch("reasonsforge.derive.parse_proposals", return_value=[]):
            added = _stage_derive(args)
        assert added == 0

    def test_applies_valid_proposals(self, work_dir):
        args = make_pipeline_args(max_derive_rounds=1)
        nodes = {"belief-1": {"text": "Test", "truth_value": "IN", "justifications": []}}
        stats = {"total_in": 1, "total_derived": 0, "max_depth": 0, "agents": 0}
        proposal = {
            "id": "derived-1", "text": "Derived",
            "antecedents": ["belief-1"], "unless": [],
            "label": "test", "kind": "derive",
        }
        with patch("reasonsforge.api.export_network", return_value={"nodes": nodes}), \
             patch("reasonsforge.derive.build_prompt", return_value=("prompt", stats)), \
             patch("reasonsforge.forge.llm.invoke_sync", return_value="proposal text"), \
             patch("reasonsforge.derive.parse_proposals", return_value=[proposal]), \
             patch("reasonsforge.derive.validate_proposals", return_value=([proposal], [])), \
             patch("reasonsforge.derive.apply_proposals", return_value=[(proposal, {"truth_value": "IN"})]):
            added = _stage_derive(args)
        assert added == 1


# --- Stage: Review ---

class TestStageReview:
    def test_returns_review_result(self, work_dir):
        args = make_pipeline_args()
        result = {"reviewed": 5, "invalid": 2, "results": []}
        with patch("reasonsforge.api.review_beliefs", return_value=result):
            got = _stage_review(args)
        assert got["reviewed"] == 5
        assert got["invalid"] == 2


# --- Stage: Repair ---

class TestStageRepair:
    def test_skips_when_no_invalids(self, work_dir, capsys):
        args = make_pipeline_args()
        review_result = {"results": [{"belief_id": "b1", "valid": True}]}
        result = _stage_repair(args, review_result)
        assert result["total_invalid"] == 0
        captured = capsys.readouterr()
        assert "No invalid beliefs" in captured.err

    def test_calls_research_with_invalid_ids(self, work_dir):
        args = make_pipeline_args()
        review_result = {"results": [
            {"belief_id": "b1", "valid": False},
            {"belief_id": "b2", "valid": True},
            {"belief_id": "b3", "valid": False},
        ]}
        research_result = {
            "total_invalid": 2, "linked": 1,
            "softened": 1, "abandoned": 0,
        }
        with patch("reasonsforge.api.research", return_value=research_result) as mock_research:
            result = _stage_repair(args, review_result)
        assert mock_research.called
        call_kwargs = mock_research.call_args[1]
        assert set(call_kwargs["belief_ids"]) == {"b1", "b3"}

    def test_handles_id_key_from_review(self, work_dir):
        """review_beliefs returns 'id' not 'belief_id' — repair must handle both."""
        args = make_pipeline_args()
        review_result = {"results": [
            {"id": "b1", "valid": False},
            {"id": "b2", "valid": True},
            {"id": "b3", "valid": False},
        ]}
        research_result = {
            "total_invalid": 2, "linked": 1,
            "softened": 0, "abandoned": 1,
        }
        with patch("reasonsforge.api.research", return_value=research_result) as mock_research:
            result = _stage_repair(args, review_result)
        assert mock_research.called
        call_kwargs = mock_research.call_args[1]
        assert set(call_kwargs["belief_ids"]) == {"b1", "b3"}
        assert result["linked"] == 1
        assert result["linked"] == 1


# --- Stage: Deduplicate ---

class TestStageDeduplicate:
    def test_reports_no_duplicates(self, work_dir, capsys):
        args = make_pipeline_args()
        with patch("reasonsforge.api.deduplicate", return_value={"clusters": [], "retracted": []}):
            _stage_deduplicate(args)
        captured = capsys.readouterr()
        assert "No duplicates found" in captured.err


# --- Full Pipeline ---

class TestCmdPipeline:
    def test_model_not_available_exits(self, work_dir):
        args = make_pipeline_args(model="nonexistent")
        with patch("reasonsforge.forge.llm.check_model_available", return_value=False), \
             pytest.raises(SystemExit):
            cmd_pipeline(args)

    def test_converges_early_on_zero_invalids_and_zero_added(self, work_dir, capsys):
        args = make_pipeline_args(rounds=3, url=None, pdf=None)
        review_result = {"reviewed": 5, "invalid": 0, "results": []}

        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_summarize"), \
             patch("reasonsforge.forge.pipeline._stage_extract", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_derive", return_value=0), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"), \
             patch("reasonsforge.forge.pipeline._stage_export"), \
             patch("reasonsforge.forge.caffeinate.hold"):
            cmd_pipeline(args)

        captured = capsys.readouterr()
        assert "Converged after 1 cycle" in captured.err

    def test_runs_all_rounds_without_convergence(self, work_dir, capsys):
        args = make_pipeline_args(rounds=2, url=None, pdf=None)
        review_result = {"reviewed": 5, "invalid": 1, "results": [
            {"belief_id": "b1", "valid": False},
        ]}

        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_summarize"), \
             patch("reasonsforge.forge.pipeline._stage_extract", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_derive", return_value=1), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_repair"), \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"), \
             patch("reasonsforge.forge.pipeline._stage_export"), \
             patch("reasonsforge.forge.caffeinate.hold"):
            cmd_pipeline(args)

        captured = capsys.readouterr()
        assert "Converged" not in captured.err
        assert "Pipeline complete" in captured.err

    def test_no_auto_accept_stops_early(self, work_dir, capsys):
        args = make_pipeline_args(no_auto_accept=True, url=None, pdf=None)

        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_summarize"), \
             patch("reasonsforge.forge.pipeline._stage_extract", return_value=False), \
             patch("reasonsforge.forge.pipeline._stage_derive") as mock_derive, \
             patch("reasonsforge.forge.pipeline._stage_export") as mock_export, \
             patch("reasonsforge.forge.caffeinate.hold"):
            cmd_pipeline(args)

        assert not mock_derive.called
        assert not mock_export.called


# --- Pipeline State ---

class TestPipelineState:
    def test_state_file_created_on_run(self, work_dir):
        args = make_pipeline_args(rounds=1, url=None, pdf=None)
        review_result = {"reviewed": 0, "invalid": 0, "results": []}

        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_summarize"), \
             patch("reasonsforge.forge.pipeline._stage_extract", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_derive", return_value=0), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"), \
             patch("reasonsforge.forge.pipeline._stage_export"), \
             patch("reasonsforge.forge.caffeinate.hold"):
            cmd_pipeline(args)

        state = _load_state()
        assert state is not None
        assert state["status"] == "completed"
        assert state["stages"]["8_export"]["status"] == "completed"

    def test_state_records_failure(self, work_dir):
        args = make_pipeline_args(url=None, pdf=None)

        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_summarize",
                   side_effect=RuntimeError("LLM exploded")), \
             patch("reasonsforge.forge.caffeinate.hold"), \
             pytest.raises(RuntimeError, match="LLM exploded"):
            cmd_pipeline(args)

        state = _load_state()
        assert state["status"] == "failed"
        assert "LLM exploded" in state["error"]
        assert state["stages"]["2_summarize"]["status"] == "running"

    def test_resume_skips_completed_stages(self, work_dir, capsys):
        args = make_pipeline_args(rounds=1, url=None, pdf=None)
        review_result = {"reviewed": 0, "invalid": 0, "results": []}

        # First run: complete through summarize, then fail at extract
        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_summarize"), \
             patch("reasonsforge.forge.pipeline._stage_extract",
                   side_effect=RuntimeError("crash")), \
             patch("reasonsforge.forge.caffeinate.hold"), \
             pytest.raises(RuntimeError):
            cmd_pipeline(args)

        state = _load_state()
        assert state["stages"]["1_ingest"]["status"] == "completed"
        assert state["stages"]["2_summarize"]["status"] == "completed"
        assert state["stages"]["3_extract"]["status"] == "running"

        # Resume: should skip ingest and summarize
        resume_args = make_pipeline_args(rounds=1, url=None, pdf=None, resume=True)

        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_summarize") as mock_summarize, \
             patch("reasonsforge.forge.pipeline._stage_extract", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_derive", return_value=0), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"), \
             patch("reasonsforge.forge.pipeline._stage_export"), \
             patch("reasonsforge.forge.caffeinate.hold"):
            cmd_pipeline(resume_args)

        assert not mock_summarize.called
        captured = capsys.readouterr()
        assert "already completed, skipping" in captured.err

        state = _load_state()
        assert state["status"] == "completed"

    def test_resume_without_state_exits(self, work_dir):
        args = make_pipeline_args(resume=True)

        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.caffeinate.hold"), \
             pytest.raises(SystemExit):
            cmd_pipeline(args)

    def test_no_auto_accept_sets_paused(self, work_dir):
        args = make_pipeline_args(no_auto_accept=True, url=None, pdf=None)

        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_summarize"), \
             patch("reasonsforge.forge.pipeline._stage_extract", return_value=False), \
             patch("reasonsforge.forge.caffeinate.hold"):
            cmd_pipeline(args)

        state = _load_state()
        assert state["status"] == "paused"
        assert state["stages"]["3_extract"]["status"] == "completed"

    def test_resume_completed_pipeline_returns_early(self, work_dir, capsys):
        """Resuming an already-completed pipeline does nothing."""
        args = make_pipeline_args(rounds=1, url=None, pdf=None)
        review_result = {"reviewed": 0, "invalid": 0, "results": []}

        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_summarize"), \
             patch("reasonsforge.forge.pipeline._stage_extract", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_derive", return_value=0), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"), \
             patch("reasonsforge.forge.pipeline._stage_export"), \
             patch("reasonsforge.forge.caffeinate.hold"):
            cmd_pipeline(args)

        state = _load_state()
        assert state["status"] == "completed"

        # Now resume — should return early
        resume_args = make_pipeline_args(resume=True)
        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_export") as mock_export, \
             patch("reasonsforge.forge.caffeinate.hold"):
            cmd_pipeline(resume_args)

        assert not mock_export.called
        captured = capsys.readouterr()
        assert "already completed" in captured.err

    def test_corrupt_state_file_handled(self, work_dir):
        """Corrupt state file is treated as missing."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text("{truncated")

        args = make_pipeline_args(resume=True)
        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.caffeinate.hold"), \
             pytest.raises(SystemExit):
            cmd_pipeline(args)


# --- Derive-Review-Repair ---

class TestConvergenceLoop:
    def test_converges_on_zero_invalids_and_zero_derived(self, work_dir):
        args = make_drr_args(rounds=3)
        review_result = {"reviewed": 5, "invalid": 0, "results": []}

        with patch("reasonsforge.forge.pipeline._stage_derive", return_value=0), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"):
            summary = _run_convergence_loop(args, rounds=3)

        assert summary["converged"] is True
        assert summary["cycles"] == 1

    def test_runs_max_rounds_without_convergence(self, work_dir):
        args = make_drr_args(rounds=2)
        review_result = {"reviewed": 5, "invalid": 1, "results": [
            {"belief_id": "b1", "valid": False},
        ]}
        repair_result = {"linked": 1, "softened": 0, "abandoned": 0}

        with patch("reasonsforge.forge.pipeline._stage_derive", return_value=1), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_repair", return_value=repair_result), \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"):
            summary = _run_convergence_loop(args, rounds=2)

        assert summary["converged"] is False
        assert summary["cycles"] == 2
        assert summary["total_derived"] == 2
        assert summary["total_linked"] == 2

    def test_skips_repair_when_no_invalids(self, work_dir):
        args = make_drr_args(rounds=1)
        review_result = {"reviewed": 5, "invalid": 0, "results": []}

        with patch("reasonsforge.forge.pipeline._stage_derive", return_value=1), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_repair") as mock_repair, \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"):
            summary = _run_convergence_loop(args, rounds=1)

        assert not mock_repair.called
        assert summary["total_invalid"] == 0

    def test_summary_accumulates_across_cycles(self, work_dir):
        args = make_drr_args(rounds=2)
        review_result = {"reviewed": 3, "invalid": 1, "results": [
            {"belief_id": "b1", "valid": False},
        ]}
        repair_result = {"linked": 0, "softened": 1, "abandoned": 0}

        with patch("reasonsforge.forge.pipeline._stage_derive", return_value=2), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_repair", return_value=repair_result), \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"):
            summary = _run_convergence_loop(args, rounds=2)

        assert summary["total_derived"] == 4
        assert summary["total_reviewed"] == 6
        assert summary["total_invalid"] == 2
        assert summary["total_softened"] == 2

    def test_on_stage_callback_called(self, work_dir):
        args = make_drr_args(rounds=1)
        review_result = {"reviewed": 1, "invalid": 0, "results": []}
        events = []

        def on_stage(cycle, stage_num, event, **kwargs):
            events.append((cycle, stage_num, event))

        with patch("reasonsforge.forge.pipeline._stage_derive", return_value=0), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"):
            _run_convergence_loop(args, rounds=1, on_stage=on_stage)

        assert (1, 4, "start") in events
        assert (1, 4, "end") in events
        assert (1, 5, "start") in events
        assert (1, 5, "end") in events
        assert (1, 6, "end") in events
        assert (1, 7, "start") in events
        assert (1, 7, "end") in events


class TestCmdDeriveReviewRepair:
    def test_model_not_available_exits(self, work_dir):
        args = make_drr_args(model="nonexistent")
        with patch("reasonsforge.forge.llm.check_model_available", return_value=False), \
             pytest.raises(SystemExit):
            cmd_derive_review_repair(args)

    def test_missing_db_exits(self, work_dir):
        (work_dir / "reasons.db").unlink()
        args = make_drr_args()
        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.caffeinate.hold"), \
             pytest.raises(SystemExit):
            cmd_derive_review_repair(args)

    def test_prints_summary(self, work_dir, capsys):
        args = make_drr_args(rounds=1)
        review_result = {"reviewed": 5, "invalid": 0, "results": []}

        with patch("reasonsforge.forge.llm.check_model_available", return_value=True), \
             patch("reasonsforge.forge.pipeline._stage_derive", return_value=0), \
             patch("reasonsforge.forge.pipeline._stage_review", return_value=review_result), \
             patch("reasonsforge.forge.pipeline._stage_deduplicate"), \
             patch("reasonsforge.forge.caffeinate.hold"):
            cmd_derive_review_repair(args)

        captured = capsys.readouterr()
        assert "Summary" in captured.err
        assert "Derived: 0" in captured.err
        assert "Converged: yes" in captured.err

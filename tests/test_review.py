"""Tests for the review module (derived belief validation)."""

import json
import sys
from io import StringIO
from unittest.mock import patch, call

import pytest

from reasonsforge.review import (
    format_belief_for_review,
    parse_review_response,
    review_beliefs,
    REVIEW_BATCH_SIZE,
)
from reasonsforge import api
from reasonsforge.cli import main


def run_cli(*args, db_path=None):
    argv = ["reasons"]
    if db_path:
        argv += ["--db", db_path]
    argv += list(args)
    stdout, stderr = StringIO(), StringIO()
    with patch.object(sys, "argv", argv), \
         patch.object(sys, "stdout", stdout), \
         patch.object(sys, "stderr", stderr):
        try:
            main()
        except SystemExit as e:
            return stdout.getvalue(), stderr.getvalue(), e.code
    return stdout.getvalue(), stderr.getvalue(), 0


def _make_nodes():
    """Build a minimal exported nodes dict for testing."""
    return {
        "premise-a": {
            "text": "A is true",
            "truth_value": "IN",
            "justifications": [],
        },
        "premise-b": {
            "text": "B is true",
            "truth_value": "IN",
            "justifications": [],
        },
        "premise-c": {
            "text": "C is true",
            "truth_value": "IN",
            "justifications": [],
        },
        "derived-ab": {
            "text": "A and B together imply AB",
            "truth_value": "IN",
            "justifications": [{
                "type": "SL",
                "antecedents": ["premise-a", "premise-b"],
                "outlist": [],
                "label": "combined observation",
            }],
        },
        "gated-abc": {
            "text": "ABC holds unless C is present",
            "truth_value": "OUT",
            "justifications": [{
                "type": "SL",
                "antecedents": ["premise-a", "premise-b"],
                "outlist": ["premise-c"],
                "label": "gated on C",
            }],
        },
        "derived-deep": {
            "text": "Deep conclusion from AB",
            "truth_value": "IN",
            "justifications": [{
                "type": "SL",
                "antecedents": ["derived-ab", "premise-c"],
                "outlist": [],
                "label": "deeper reasoning",
            }],
        },
    }


class TestFormatBeliefForReview:

    def test_formats_belief_with_antecedents(self):
        nodes = _make_nodes()
        result = format_belief_for_review("derived-ab", nodes)
        assert "### derived-ab" in result
        assert "A and B together imply AB" in result
        assert "premise-a: A is true" in result
        assert "premise-b: B is true" in result
        assert "Label: combined observation" in result

    def test_includes_outlist(self):
        nodes = _make_nodes()
        result = format_belief_for_review("gated-abc", nodes)
        assert "Unless (must be OUT):" in result
        assert "premise-c: C is true" in result

    def test_missing_antecedent_graceful(self):
        nodes = _make_nodes()
        nodes["derived-ab"]["justifications"][0]["antecedents"] = [
            "premise-a", "nonexistent"
        ]
        result = format_belief_for_review("derived-ab", nodes)
        assert "nonexistent: (not found in network)" in result

    def test_missing_node_returns_empty(self):
        nodes = _make_nodes()
        result = format_belief_for_review("does-not-exist", nodes)
        assert result == ""

    def test_multiple_justifications(self):
        nodes = _make_nodes()
        nodes["multi-just"] = {
            "text": "Supported by two independent paths",
            "truth_value": "IN",
            "justifications": [
                {
                    "type": "SL",
                    "antecedents": ["premise-a", "premise-b"],
                    "outlist": [],
                    "label": "path one",
                },
                {
                    "type": "SL",
                    "antecedents": ["premise-c"],
                    "outlist": [],
                    "label": "path two",
                },
            ],
        }
        result = format_belief_for_review("multi-just", nodes)
        assert "Justification 1/2:" in result
        assert "Justification 2/2:" in result
        assert "premise-a: A is true" in result
        assert "premise-c: C is true" in result
        assert "Label: path one" in result
        assert "Label: path two" in result

    def test_single_justification_no_numbering(self):
        nodes = _make_nodes()
        result = format_belief_for_review("derived-ab", nodes)
        assert "Justification 1/" not in result

    def test_no_justifications(self):
        nodes = _make_nodes()
        result = format_belief_for_review("premise-a", nodes)
        assert "### premise-a" in result
        assert "Antecedents:" not in result


class TestParseReviewResponse:

    def test_parses_valid_json(self):
        response = json.dumps([{
            "id": "derived-ab",
            "valid": True,
            "sufficient": True,
            "necessary": False,
            "unnecessary_antecedents": ["premise-b"],
            "comment": "premise-b is redundant",
        }])
        results = parse_review_response(response)
        assert len(results) == 1
        assert results[0]["id"] == "derived-ab"
        assert results[0]["valid"] is True
        assert results[0]["necessary"] is False
        assert results[0]["unnecessary_antecedents"] == ["premise-b"]

    def test_extracts_from_surrounding_prose(self):
        response = (
            "Here are my findings:\n\n"
            '[{"id": "derived-ab", "valid": false, "sufficient": true, '
            '"necessary": true, "unnecessary_antecedents": [], '
            '"comment": "conclusion does not follow"}]\n\n'
            "Hope this helps!"
        )
        results = parse_review_response(response)
        assert len(results) == 1
        assert results[0]["valid"] is False

    def test_malformed_json_returns_empty(self):
        response = "This is not JSON at all."
        results = parse_review_response(response)
        assert results == []

    def test_missing_fields_get_defaults(self):
        response = json.dumps([{"id": "derived-ab"}])
        results = parse_review_response(response)
        assert len(results) == 1
        assert results[0]["valid"] is True
        assert results[0]["sufficient"] is True
        assert results[0]["necessary"] is True
        assert results[0]["unnecessary_antecedents"] == []
        assert results[0]["comment"] == ""
        assert results[0]["defeat_reason_type"] == ""

    def test_defeat_reason_type_parsed(self):
        response = json.dumps([{
            "id": "derived-ab",
            "valid": False,
            "sufficient": True,
            "necessary": True,
            "unnecessary_antecedents": [],
            "comment": "overclaims",
            "scope_findings": [{"antecedent": "p1", "establishes": "X"}],
            "missing_property": "Y",
            "defeat_reason_type": "unsupported-conjunct",
        }])
        results = parse_review_response(response)
        assert len(results) == 1
        assert results[0]["defeat_reason_type"] == "unsupported-conjunct"

    def test_skips_items_without_id(self):
        response = json.dumps([
            {"valid": True},
            {"id": "derived-ab", "valid": False},
        ])
        results = parse_review_response(response)
        assert len(results) == 1
        assert results[0]["id"] == "derived-ab"

    def test_prose_brackets_before_json(self):
        response = (
            "I checked [see antecedents] and [the outlist] carefully.\n\n"
            '[{"id": "derived-ab", "valid": false, "sufficient": true, '
            '"necessary": true, "unnecessary_antecedents": [], '
            '"comment": "does not follow"}]'
        )
        results = parse_review_response(response)
        assert len(results) == 1
        assert results[0]["id"] == "derived-ab"
        assert results[0]["valid"] is False

    def test_non_list_json_skipped(self):
        response = '{"id": "derived-ab", "valid": true}'
        results = parse_review_response(response)
        assert results == []


class TestReviewBeliefs:

    def test_reviews_batch(self):
        nodes = _make_nodes()
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            results = review_beliefs(nodes, belief_ids=["derived-ab"], model="claude")
        assert len(results) == 1
        assert results[0]["id"] == "derived-ab"

    def test_empty_derived_returns_empty(self):
        nodes = {
            "premise-a": {
                "text": "A is true",
                "truth_value": "IN",
                "justifications": [],
            },
        }
        results = review_beliefs(nodes)
        assert results == []

    def test_filters_to_existing_ids(self):
        nodes = _make_nodes()
        mock_response = json.dumps([])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            results = review_beliefs(nodes, belief_ids=["nonexistent"])
        assert results == []

    def test_batch_size_respected(self):
        nodes = _make_nodes()
        # derived-ab and derived-deep are the only derived IN beliefs
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run:
            results = review_beliefs(nodes, batch_size=1)
        # 2 derived IN beliefs (derived-ab, derived-deep) = 2 batches
        assert mock_run.call_count == 2

    def test_timeout_passed_through(self):
        nodes = _make_nodes()
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run:
            review_beliefs(nodes, belief_ids=["derived-ab"], timeout=600)
        assert mock_run.call_args[1]["timeout"] == 600

    def test_llm_error_continues(self):
        nodes = _make_nodes()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=RuntimeError("LLM failed")):
            results = review_beliefs(nodes, belief_ids=["derived-ab"])
        assert results == []


class TestReviewBeliefsApi:

    @pytest.fixture
    def db_path(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A is true", db_path=db)
        api.add_node("premise-b", "B is true", db_path=db)
        api.add_node("derived-ab", "AB combined", sl="premise-a,premise-b",
                      label="combined", db_path=db)
        api.add_node("premise-c", "C is true", db_path=db)
        api.add_node("derived-abc", "ABC combined",
                      sl="derived-ab,premise-c", label="deeper", db_path=db)
        return db

    def test_filters_to_derived_only(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
            {"id": "derived-abc", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_beliefs(db_path=db_path)
        assert result["reviewed"] == 2
        assert result["total_derived"] == 2

    def test_min_depth_filter(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-abc", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_beliefs(min_depth=2, db_path=db_path)
        # derived-abc is depth 2 (depends on derived-ab which is depth 1)
        assert result["reviewed"] == 1

    def test_sample_limits_count(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_beliefs(sample=1, db_path=db_path)
        assert result["reviewed"] == 1

    def test_depends_on_filter(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-abc", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_beliefs(depends_on="derived-ab", db_path=db_path)
        # Only derived-abc depends on derived-ab
        assert result["reviewed"] == 1

    def test_namespace_filter(self, db_path):
        api.add_node("eng:premise-x", "X is true", db_path=db_path)
        api.add_node("eng:derived-x", "X derived", sl="eng:premise-x",
                      label="eng", db_path=db_path)
        mock_response = json.dumps([
            {"id": "eng:derived-x", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_beliefs(namespace="eng", db_path=db_path)
        assert result["reviewed"] == 1
        assert result["results"][0]["id"] == "eng:derived-x"

    def test_empty_namespace_filters_local(self, db_path):
        api.add_node("eng:premise-x", "X is true", db_path=db_path)
        api.add_node("eng:derived-x", "X derived", sl="eng:premise-x",
                      label="eng", db_path=db_path)
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
            {"id": "derived-abc", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_beliefs(namespace="", db_path=db_path)
        # Only local derived beliefs (no colon in ID)
        assert result["reviewed"] == 2
        ids = [r["id"] for r in result["results"]]
        assert "eng:derived-x" not in ids

    def test_returns_summary_counts(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": False, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "invalid"},
            {"id": "derived-abc", "valid": True, "sufficient": False,
             "necessary": False, "unnecessary_antecedents": ["premise-c"],
             "comment": "insufficient and unnecessary"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_beliefs(db_path=db_path)
        assert result["invalid"] == 1
        assert result["insufficient"] == 1
        assert result["unnecessary"] == 1

    def test_visible_to_filter(self, db_path):
        api.add_node("tagged-derived", "tagged belief",
                      sl="premise-a,premise-b", label="tagged",
                      access_tags=["secret"],
                      db_path=db_path)
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
            {"id": "derived-abc", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_beliefs(visible_to=["public"], db_path=db_path)
        # tagged-derived requires "secret" tag, so excluded; only 2 untagged derived remain
        assert result["reviewed"] == 2


class TestCmdReviewBeliefs:

    @pytest.fixture
    def db_path(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A is true", db_path=db)
        api.add_node("premise-b", "B is true", db_path=db)
        api.add_node("derived-ab", "AB combined", sl="premise-a,premise-b",
                      label="combined", db_path=db)
        return db

    def _mock_review(self, response_data):
        mock_response = json.dumps(response_data)
        return type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()

    def test_auto_retract_retracts_invalid(self, db_path):
        mock_result = self._mock_review([
            {"id": "derived-ab", "valid": False, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [],
             "comment": "conclusion does not follow"},
        ])
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            stdout, stderr, code = run_cli(
                "review-beliefs", "--auto-retract", db_path=db_path)
        assert "RETRACTED derived-ab" in stdout
        # Verify it was actually retracted in the DB
        result = api.show_node("derived-ab", db_path=db_path)
        assert result["truth_value"] == "OUT"

    def test_dry_run_prevents_retraction(self, db_path):
        mock_result = self._mock_review([
            {"id": "derived-ab", "valid": False, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [],
             "comment": "conclusion does not follow"},
        ])
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            stdout, stderr, code = run_cli(
                "review-beliefs", "--auto-retract", "--dry-run", db_path=db_path)
        assert "RETRACTED" not in stdout
        # Verify it was NOT retracted
        result = api.show_node("derived-ab", db_path=db_path)
        assert result["truth_value"] == "IN"

    def test_output_writes_findings_file(self, db_path, tmp_path):
        output_file = str(tmp_path / "findings.md")
        mock_result = self._mock_review([
            {"id": "derived-ab", "valid": False, "sufficient": True,
             "necessary": False, "unnecessary_antecedents": ["premise-b"],
             "comment": "not valid and unnecessary antecedent"},
        ])
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            stdout, stderr, code = run_cli(
                "review-beliefs", "-o", output_file, db_path=db_path)
        assert f"Wrote findings to {output_file}" in stdout
        with open(output_file) as f:
            content = f.read()
        assert "# Belief Review Findings" in content
        assert "### derived-ab" in content
        assert "- Valid: FAIL" in content
        assert "- Sufficient: PASS" in content
        assert "- Necessary: FAIL" in content
        assert "- Unnecessary antecedents: premise-b" in content
        assert "- Comment: not valid and unnecessary antecedent" in content

    def test_displays_flags_for_issues(self, db_path):
        mock_result = self._mock_review([
            {"id": "derived-ab", "valid": False, "sufficient": False,
             "necessary": False, "unnecessary_antecedents": ["premise-a"],
             "comment": "all three axes fail"},
        ])
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            stdout, stderr, code = run_cli(
                "review-beliefs", "--dry-run", db_path=db_path)
        assert "INVALID" in stdout
        assert "INSUFFICIENT" in stdout
        assert "UNNECESSARY(premise-a)" in stdout
        assert "all three axes fail" in stdout

    def test_no_derived_beliefs_message(self, tmp_path):
        db = str(tmp_path / "empty.db")
        api.add_node("just-a-premise", "simple fact", db_path=db)
        stdout, stderr, code = run_cli("review-beliefs", db_path=db)
        assert "No derived beliefs to review." in stdout

    def test_json_report_written(self, db_path, tmp_path):
        report_dir = str(tmp_path / "reports")
        mock_result = self._mock_review([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            stdout, stderr, code = run_cli(
                "review-beliefs", "--report-dir", report_dir, db_path=db_path)
        assert "Report:" in stdout

        import os
        reports = [f for f in os.listdir(report_dir) if f.endswith(".json")]
        assert len(reports) == 1

        with open(os.path.join(report_dir, reports[0])) as f:
            report = json.load(f)
        assert "timestamp" in report
        assert report["status"] == "complete"
        assert report["reviewed"] == 1
        assert "results" in report
        assert len(report["results"]) == 1
        assert report["results"][0]["id"] == "derived-ab"

    def test_partial_report_on_batch(self, db_path, tmp_path):
        """on_batch callback writes partial report after each batch."""
        report_dir = str(tmp_path / "reports")
        batch_results = []

        def capture_on_batch(results):
            batch_results.append(list(results))

        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {
            "returncode": 0, "stdout": mock_response, "stderr": ""
        })()

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            api.review_beliefs(
                model="claude", on_batch=capture_on_batch, db_path=db_path)

        assert len(batch_results) == 1
        assert batch_results[0][0]["id"] == "derived-ab"

    def test_report_status_complete_after_cli_run(self, db_path, tmp_path):
        """CLI run produces final report with status=complete."""
        report_dir = str(tmp_path / "reports")
        mock_result = self._mock_review([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            stdout, stderr, code = run_cli(
                "review-beliefs", "--report-dir", report_dir, db_path=db_path)

        import os
        reports = [f for f in os.listdir(report_dir) if f.endswith(".json")]
        assert len(reports) == 1
        with open(os.path.join(report_dir, reports[0])) as f:
            final_report = json.load(f)
        assert final_report["status"] == "complete"
        assert "total_derived" in final_report

    def test_no_report_flag(self, db_path, tmp_path):
        report_dir = str(tmp_path / "reports")
        mock_result = self._mock_review([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            stdout, stderr, code = run_cli(
                "review-beliefs", "--no-report", "--report-dir", report_dir,
                db_path=db_path)
        assert "Report:" not in stdout
        import os
        assert not os.path.exists(report_dir)


class TestReviewBeliefsMetadata:

    @pytest.fixture
    def db_path(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A is true", db_path=db)
        api.add_node("premise-b", "B is true", db_path=db)
        api.add_node("derived-ab", "AB combined", sl="premise-a,premise-b",
                      label="combined", db_path=db)
        api.add_node("premise-c", "C is true", db_path=db)
        api.add_node("derived-abc", "ABC combined",
                      sl="derived-ab,premise-c", label="deeper", db_path=db)
        return db

    def test_metadata_written_after_review(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
            {"id": "derived-abc", "valid": False, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [],
             "comment": "does not follow"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            api.review_beliefs(db_path=db_path)

        node_ab = api.show_node("derived-ab", db_path=db_path)
        assert node_ab["reviewed_at"]
        assert node_ab["metadata"]["review_result"] == "pass"

        node_abc = api.show_node("derived-abc", db_path=db_path)
        assert node_abc["reviewed_at"]
        assert node_abc["metadata"]["review_result"] == "invalid"

    def test_dry_run_skips_metadata(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            api.review_beliefs(dry_run=True, db_path=db_path)

        node = api.show_node("derived-ab", db_path=db_path)
        assert not node["reviewed_at"]

    def test_review_result_classification_priority(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": False, "sufficient": False,
             "necessary": False, "unnecessary_antecedents": ["premise-b"],
             "comment": "all fail"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            api.review_beliefs(belief_ids=["derived-ab"], db_path=db_path)

        node = api.show_node("derived-ab", db_path=db_path)
        assert node["metadata"]["review_result"] == "invalid"

    def test_insufficient_result_classification(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": False,
             "necessary": True, "unnecessary_antecedents": [],
             "comment": "not enough"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            api.review_beliefs(belief_ids=["derived-ab"], db_path=db_path)

        node = api.show_node("derived-ab", db_path=db_path)
        assert node["metadata"]["review_result"] == "insufficient"


class TestListNodesReviewFilters:

    @pytest.fixture
    def db_path(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A is true", db_path=db)
        api.add_node("premise-b", "B is true", db_path=db)
        api.add_node("derived-ab", "AB combined", sl="premise-a,premise-b",
                      label="combined", db_path=db)
        api.add_node("derived-cd", "CD combined", sl="premise-a,premise-b",
                      label="combined2", db_path=db)
        return db

    def test_never_reviewed_filter(self, db_path):
        result = api.list_nodes(never_reviewed=True, db_path=db_path)
        ids = [n["id"] for n in result["nodes"]]
        assert "derived-ab" in ids
        assert "derived-cd" in ids
        assert "premise-a" not in ids
        assert "premise-b" not in ids

    def test_never_reviewed_excludes_reviewed(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            api.review_beliefs(belief_ids=["derived-ab"], db_path=db_path)

        result = api.list_nodes(never_reviewed=True, db_path=db_path)
        ids = [n["id"] for n in result["nodes"]]
        assert "derived-ab" not in ids
        assert "derived-cd" in ids

    def test_not_reviewed_since_includes_never_reviewed(self, db_path):
        mock_response = json.dumps([
            {"id": "derived-ab", "valid": True, "sufficient": True,
             "necessary": True, "unnecessary_antecedents": [], "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": mock_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            api.review_beliefs(belief_ids=["derived-ab"], db_path=db_path)

        result = api.list_nodes(not_reviewed_since=30, db_path=db_path)
        ids = [n["id"] for n in result["nodes"]]
        assert "derived-cd" in ids
        assert "derived-ab" not in ids

    def test_by_impact_sort(self, db_path):
        result = api.list_nodes(by_impact=True, db_path=db_path)
        counts = [n["dependent_count"] for n in result["nodes"]]
        assert counts == sorted(counts, reverse=True)

    def test_list_includes_review_fields(self, db_path):
        result = api.list_nodes(db_path=db_path)
        for node in result["nodes"]:
            assert "last_reviewed" in node
            assert "review_result" in node

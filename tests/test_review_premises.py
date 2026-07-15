"""Tests for review-premises command and supporting functions."""

import json
from unittest.mock import patch

import pytest

from reasonsforge import api
from reasonsforge.review_premises import (
    format_premise_for_review,
    parse_premise_review_response,
    review_premises,
)


class TestFormatPremiseForReview:

    def test_basic_formatting(self):
        nodes = {
            "obs-1": {
                "text": "Redis is used for session storage",
                "source": "docs/architecture.md",
            }
        }
        result = format_premise_for_review("obs-1", nodes, "Session data is stored in PostgreSQL.")
        assert "### obs-1" in result
        assert "Redis is used for session storage" in result
        assert "docs/architecture.md" in result
        assert "Session data is stored in PostgreSQL." in result

    def test_missing_node(self):
        assert format_premise_for_review("missing", {}, "content") == ""

    def test_no_source_field(self):
        nodes = {"obs-1": {"text": "A claim"}}
        result = format_premise_for_review("obs-1", nodes, "source text")
        assert "### obs-1" in result
        assert "A claim" in result
        assert "Source reference:" not in result


class TestParsePremiseReviewResponse:

    def test_valid_json(self):
        response = json.dumps([{
            "id": "obs-1",
            "accurate": False,
            "well_scoped": True,
            "error_type": "misread_source",
            "comment": "Source says PostgreSQL, not Redis",
        }])
        results = parse_premise_review_response(response)
        assert len(results) == 1
        assert results[0]["id"] == "obs-1"
        assert results[0]["accurate"] is False
        assert results[0]["error_type"] == "misread_source"

    def test_json_with_prose(self):
        response = "Here are my findings:\n" + json.dumps([{
            "id": "obs-1", "accurate": True, "well_scoped": True,
            "error_type": None, "comment": "Matches source",
        }]) + "\n\nHope this helps!"
        results = parse_premise_review_response(response)
        assert len(results) == 1
        assert results[0]["accurate"] is True

    def test_malformed_json(self):
        assert parse_premise_review_response("not json at all") == []

    def test_defaults_for_missing_fields(self):
        response = json.dumps([{"id": "obs-1"}])
        results = parse_premise_review_response(response)
        assert results[0]["accurate"] is True
        assert results[0]["well_scoped"] is True
        assert results[0]["error_type"] is None
        assert results[0]["comment"] == ""

    def test_multiple_results(self):
        response = json.dumps([
            {"id": "obs-1", "accurate": True, "well_scoped": True, "error_type": None, "comment": "ok"},
            {"id": "obs-2", "accurate": False, "well_scoped": False, "error_type": "fabricated", "comment": "made up"},
        ])
        results = parse_premise_review_response(response)
        assert len(results) == 2
        assert results[0]["id"] == "obs-1"
        assert results[1]["id"] == "obs-2"


class TestReviewPremisesBatchLoop:

    def test_calls_llm_and_parses(self):
        nodes = {
            "obs-1": {"text": "Claim A", "source": "doc.md", "truth_value": "IN"},
        }
        source_contents = {"doc.md": "Source content for doc."}
        llm_response = json.dumps([{
            "id": "obs-1", "accurate": True, "well_scoped": True,
            "error_type": None, "comment": "matches",
        }])
        mock_result = type("R", (), {"returncode": 0, "stdout": llm_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            results = review_premises(nodes, ["obs-1"], source_contents, model="claude")
        assert len(results) == 1
        assert results[0]["accurate"] is True

    def test_empty_ids_returns_empty(self):
        assert review_premises({}, [], {}) == []

    def test_on_batch_callback(self):
        nodes = {
            "obs-1": {"text": "Claim", "source": "doc.md", "truth_value": "IN"},
        }
        source_contents = {"doc.md": "Content."}
        llm_response = json.dumps([{
            "id": "obs-1", "accurate": True, "well_scoped": True,
            "error_type": None, "comment": "ok",
        }])
        mock_result = type("R", (), {"returncode": 0, "stdout": llm_response, "stderr": ""})()
        batches = []
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            review_premises(nodes, ["obs-1"], source_contents,
                           on_batch=lambda r: batches.append(len(r)))
        assert batches == [1]

    def test_failed_batch_continues(self):
        nodes = {
            "obs-1": {"text": "A", "source": "a.md", "truth_value": "IN"},
        }
        source_contents = {"a.md": "Content."}
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=Exception("LLM down")):
            results = review_premises(nodes, ["obs-1"], source_contents)
        assert results == []


class TestApiReviewPremises:

    @pytest.fixture
    def db_with_premises(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_file = tmp_path / "entry.md"
        source_file.write_text("The system uses PostgreSQL for storage.")

        api.init_db(db_path=db)
        api.add_node("obs-1", "PostgreSQL is used for storage",
                     source=str(source_file), db_path=db)
        api.add_node("obs-2", "Redis is used for caching",
                     source=str(source_file), db_path=db)
        api.add_node("derived-1", "Storage is reliable",
                     sl="obs-1", db_path=db)
        return db

    def test_filters_premises_only(self, db_with_premises):
        llm_response = json.dumps([
            {"id": "obs-1", "accurate": True, "well_scoped": True, "error_type": None, "comment": "ok"},
            {"id": "obs-2", "accurate": True, "well_scoped": True, "error_type": None, "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": llm_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_premises(dry_run=True, db_path=db_with_premises)
        reviewed_ids = {r["id"] for r in result["results"]}
        assert "derived-1" not in reviewed_ids
        assert result["total_premises"] == 2

    def test_skips_premises_without_source(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("no-src", "A claim with no source", db_path=db)
        result = api.review_premises(dry_run=True, db_path=db)
        assert result["skipped_no_source"] == 1
        assert result["reviewed"] == 0

    def test_specific_belief_ids(self, db_with_premises):
        llm_response = json.dumps([
            {"id": "obs-1", "accurate": True, "well_scoped": True, "error_type": None, "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": llm_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_premises(belief_ids=["obs-1"], dry_run=True,
                                         db_path=db_with_premises)
        assert result["reviewed"] == 1

    def test_sample_limits(self, db_with_premises):
        llm_response = json.dumps([
            {"id": "obs-1", "accurate": True, "well_scoped": True, "error_type": None, "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": llm_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_premises(sample=1, dry_run=True,
                                         db_path=db_with_premises)
        assert result["reviewed"] == 1

    def test_stores_metadata(self, db_with_premises):
        llm_response = json.dumps([
            {"id": "obs-1", "accurate": False, "well_scoped": True,
             "error_type": "misread_source", "comment": "wrong"},
            {"id": "obs-2", "accurate": True, "well_scoped": True,
             "error_type": None, "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": llm_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            api.review_premises(db_path=db_with_premises)

        net = api.export_network(db_path=db_with_premises)
        obs1 = net["nodes"]["obs-1"]
        obs2 = net["nodes"]["obs-2"]
        assert obs1["metadata"]["premise_review_result"] == "misread_source"
        assert obs2["metadata"]["premise_review_result"] == "pass"
        assert "last_premise_reviewed" in obs1["metadata"]

    def test_dry_run_no_metadata(self, db_with_premises):
        llm_response = json.dumps([
            {"id": "obs-1", "accurate": False, "well_scoped": True,
             "error_type": "fabricated", "comment": "made up"},
            {"id": "obs-2", "accurate": True, "well_scoped": True,
             "error_type": None, "comment": "ok"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": llm_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            api.review_premises(dry_run=True, db_path=db_with_premises)

        net = api.export_network(db_path=db_with_premises)
        assert "premise_review_result" not in net["nodes"]["obs-1"].get("metadata", {})

    def test_return_counts(self, db_with_premises):
        llm_response = json.dumps([
            {"id": "obs-1", "accurate": False, "well_scoped": True,
             "error_type": "misread_source", "comment": "wrong"},
            {"id": "obs-2", "accurate": True, "well_scoped": False,
             "error_type": None, "comment": "too broad"},
        ])
        mock_result = type("R", (), {"returncode": 0, "stdout": llm_response, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.review_premises(dry_run=True, db_path=db_with_premises)
        assert result["inaccurate"] == 1
        assert result["overgeneralized"] == 1
        assert result["reviewed"] == 2


class TestCliDispatch:

    def test_review_premises_registered(self):
        from reasonsforge import cli
        assert hasattr(cli, "cmd_review_premises")

    def test_dispatch_table(self):
        from reasonsforge.cli import main
        import argparse
        from reasonsforge import cli
        assert callable(cli.cmd_review_premises)

"""Tests for the verify command."""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from reasonsforge import api
from reasonsforge.verify import (
    format_belief_for_verify,
    parse_verify_response,
    read_source,
)
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


def _mock_result(stdout):
    return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()


class TestParseVerifyResponse:

    def test_valid_response(self):
        response = json.dumps({
            "p1": {"verdict": "CONFIRMED", "reason": "matches", "quote": "exact text"},
            "p2": {"verdict": "STALE", "reason": "not found", "quote": None},
        })
        results = parse_verify_response(response)
        assert results["p1"]["verdict"] == "CONFIRMED"
        assert results["p1"]["quote"] == "exact text"
        assert results["p2"]["verdict"] == "STALE"

    def test_json_in_prose(self):
        response = 'Here is my analysis:\n' + json.dumps({
            "p1": {"verdict": "PARTIAL", "reason": "differs", "quote": "close"},
        }) + '\nDone.'
        results = parse_verify_response(response)
        assert results["p1"]["verdict"] == "PARTIAL"

    def test_malformed(self):
        assert parse_verify_response("not json") == {}

    def test_empty(self):
        assert parse_verify_response("") == {}

    def test_missing_verdict_skipped(self):
        response = json.dumps({
            "p1": {"reason": "no verdict field"},
            "p2": {"verdict": "confirmed", "reason": "ok", "quote": None},
        })
        results = parse_verify_response(response)
        assert "p1" not in results
        assert results["p2"]["verdict"] == "CONFIRMED"

    def test_case_insensitive(self):
        response = json.dumps({
            "p1": {"verdict": "confirmed", "reason": "ok", "quote": None},
        })
        results = parse_verify_response(response)
        assert results["p1"]["verdict"] == "CONFIRMED"


class TestFormatBeliefForVerify:

    def test_basic_format(self):
        belief = {"id": "p1", "text": "A claim", "source": "doc.md",
                  "source_url": "https://example.com"}
        result = format_belief_for_verify(belief, "Source document content here.")
        assert "### `p1`" in result
        assert "**Claim:** A claim" in result
        assert "doc.md" in result
        assert "https://example.com" in result
        assert "Source document content here." in result

    def test_no_source(self):
        belief = {"id": "p1", "text": "A claim", "source": "", "source_url": ""}
        result = format_belief_for_verify(belief, None)
        assert "(source not available)" in result


class TestReadSource:

    def test_reads_file(self, tmp_path):
        source_file = tmp_path / "doc.md"
        source_file.write_text("Important evidence here.")
        db_path = str(tmp_path / "reasons.db")

        content = read_source("doc.md", db_path=db_path)
        assert content == "Important evidence here."

    def test_missing_file(self, tmp_path):
        db_path = str(tmp_path / "reasons.db")
        content = read_source("nonexistent.md", db_path=db_path)
        assert content is None

    def test_empty_source(self):
        content = read_source("")
        assert content is None

    def test_truncation(self, tmp_path):
        source_file = tmp_path / "big.md"
        source_file.write_text("x" * 40_000)
        db_path = str(tmp_path / "reasons.db")

        content = read_source("big.md", db_path=db_path)
        assert len(content) < 35_000
        assert "[... truncated ...]" in content


class TestApiVerifyBelief:

    def test_premise_verified(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_file = tmp_path / "doc.md"
        source_file.write_text("PostgreSQL is the primary database.")
        api.add_node("p1", "PostgreSQL is the primary database",
                     source=str(source_file), db_path=db)

        llm_resp = json.dumps({
            "p1": {"verdict": "CONFIRMED", "reason": "exact match",
                   "quote": "PostgreSQL is the primary database."},
        })
        mock = _mock_result(llm_resp)
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            result = api.verify_belief("p1", db_path=db)

        assert result["verified"] == ["p1"]
        assert result["stale"] == []
        assert result["results"]["p1"]["verdict"] == "CONFIRMED"

    def test_verified_at_stamped(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_file = tmp_path / "doc.md"
        source_file.write_text("PostgreSQL is the primary database.")
        api.add_node("p1", "PostgreSQL is the primary database",
                     source=str(source_file), db_path=db)

        llm_resp = json.dumps({
            "p1": {"verdict": "CONFIRMED", "reason": "exact match",
                   "quote": "PostgreSQL is the primary database."},
        })
        mock = _mock_result(llm_resp)
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            api.verify_belief("p1", db_path=db)

        node = api.show_node("p1", db_path=db)
        assert node["verified_at"] is not None

    def test_stale_detected(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_file = tmp_path / "doc.md"
        source_file.write_text("Redis is used for caching only.")
        api.add_node("p1", "Redis is the primary database",
                     source=str(source_file), db_path=db)

        llm_resp = json.dumps({
            "p1": {"verdict": "STALE", "reason": "source says caching only",
                   "quote": "Redis is used for caching only."},
        })
        mock = _mock_result(llm_resp)
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            result = api.verify_belief("p1", db_path=db)

        assert result["stale"] == ["p1"]
        assert result["verified"] == []

    def test_retract_stale(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_file = tmp_path / "doc.md"
        source_file.write_text("No mention of this claim.")
        api.add_node("p1", "Unsupported claim",
                     source=str(source_file), db_path=db)

        llm_resp = json.dumps({
            "p1": {"verdict": "STALE", "reason": "not in source", "quote": None},
        })
        mock = _mock_result(llm_resp)
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            result = api.verify_belief("p1", retract=True, db_path=db)

        assert result["stale"] == ["p1"]
        node = api.show_node("p1", db_path=db)
        assert node["truth_value"] == "OUT"

    def test_dry_run(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_file = tmp_path / "doc.md"
        source_file.write_text("Content.")
        api.add_node("p1", "A claim", source=str(source_file), db_path=db)

        result = api.verify_belief("p1", dry_run=True, db_path=db)

        assert result["dry_run"] is True
        assert len(result["beliefs_checked"]) == 1
        assert result["results"] == {}

    def test_trace_derived(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_a = tmp_path / "a.md"
        source_a.write_text("Evidence for A.")
        source_b = tmp_path / "b.md"
        source_b.write_text("Evidence for B.")

        api.add_node("p-a", "Premise A", source=str(source_a), db_path=db)
        api.add_node("p-b", "Premise B", source=str(source_b), db_path=db)
        api.add_node("d1", "Derived from A and B", sl="p-a,p-b", db_path=db)

        llm_resp = json.dumps({
            "p-a": {"verdict": "CONFIRMED", "reason": "matches", "quote": "Evidence for A."},
            "p-b": {"verdict": "CONFIRMED", "reason": "matches", "quote": "Evidence for B."},
        })
        mock = _mock_result(llm_resp)
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            result = api.verify_belief("d1", trace=True, db_path=db)

        assert len(result["beliefs_checked"]) == 2
        assert set(result["verified"]) == {"p-a", "p-b"}
        assert result["is_derived"] is True

    def test_derived_without_trace(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("p1", "Premise", db_path=db)
        api.add_node("d1", "Derived", sl="p1", db_path=db)

        llm_resp = json.dumps({
            "d1": {"verdict": "INCONCLUSIVE", "reason": "no source", "quote": None},
        })
        mock = _mock_result(llm_resp)
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            result = api.verify_belief("d1", trace=False, db_path=db)

        assert len(result["beliefs_checked"]) == 1
        assert result["beliefs_checked"][0]["id"] == "d1"

    def test_missing_node_raises(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)

        with pytest.raises(KeyError):
            api.verify_belief("nonexistent", db_path=db)


class TestCliVerify:

    def test_help(self):
        stdout, stderr, code = run_cli("verify", "--help")
        assert code == 0
        assert "verify" in stdout.lower()

    def test_dry_run(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_file = tmp_path / "doc.md"
        source_file.write_text("Content.")
        api.add_node("p1", "A claim", source=str(source_file), db_path=db)

        stdout, stderr, code = run_cli(
            "verify", "p1", "--dry-run", db_path=db,
        )

        assert code == 0
        assert "dry-run" in stdout
        assert "p1" in stdout

    def test_basic_verify(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_file = tmp_path / "doc.md"
        source_file.write_text("The system uses PostgreSQL.")
        api.add_node("p1", "The system uses PostgreSQL",
                     source=str(source_file), db_path=db)

        llm_resp = json.dumps({
            "p1": {"verdict": "CONFIRMED", "reason": "exact match",
                   "quote": "The system uses PostgreSQL."},
        })
        mock = _mock_result(llm_resp)
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            stdout, stderr, code = run_cli("verify", "p1", db_path=db)

        assert code == 0
        assert "CONFIRMED" in stdout
        assert "1 confirmed" in stdout

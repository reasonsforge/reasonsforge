"""Tests for the report-gated command."""

import os
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from reasonsforge import api
from reasonsforge.report_gated import (
    _build_structured_report, _format_data_for_prompt, report_gated,
)


def run_cli(*args, db_path=None):
    from reasonsforge.cli import main
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
            code = 0
        except SystemExit as e:
            code = e.code if e.code is not None else 0
    return stdout.getvalue(), stderr.getvalue(), code


@pytest.fixture
def db(tmp_path):
    """Network with a blocker, gated belief, retracted premise, and normal nodes."""
    db_path = str(tmp_path / "test.db")
    api.add_node("normal-premise-a", "A normal IN premise", db_path=db_path)
    api.add_node("normal-premise-b", "Another IN premise", db_path=db_path)
    api.add_node("blocker-x", "This blocks a conclusion", db_path=db_path)
    api.add_node("support-for-gated", "Support node", db_path=db_path)
    api.add_node("gated-conclusion", "Would be true if blocker removed",
                 sl="support-for-gated", unless="blocker-x", db_path=db_path)
    api.add_node("retracted-defect", "A bug that was fixed", db_path=db_path)
    api.retract_node("retracted-defect", reason="Fixed in PR #99",
                     db_path=db_path)
    return db_path


class TestBuildStructuredReport:

    def _make_gated_data(self, blockers=None):
        if blockers is None:
            blockers = {
                "blocker-a": {
                    "text": "Blocker A text",
                    "gated": [{"id": "gated-1", "text": "Gated node 1"}],
                },
            }
        gated_count = sum(len(b["gated"]) for b in blockers.values())
        return {
            "blockers": blockers,
            "blocker_count": len(blockers),
            "gated_count": gated_count,
        }

    def test_includes_title(self):
        gated = self._make_gated_data()
        result = _build_structured_report(10, 5, gated, [], {})
        assert "# Gated Beliefs Report" in result

    def test_includes_summary_stats(self):
        gated = self._make_gated_data()
        result = _build_structured_report(100, 50, gated, [], {})
        assert "100 IN / 50 OUT" in result
        assert "1 active blocker(s)" in result

    def test_includes_retracted_premises(self):
        gated = self._make_gated_data({})
        retracted = [{
            "id": "fixed-bug",
            "text": "A bug that existed",
            "retract_reason": "Fixed in PR #42",
            "dependent_count": 5,
        }]
        result = _build_structured_report(10, 5, gated, retracted, {})
        assert "Fixed Defects" in result
        assert "`fixed-bug`" in result
        assert "Fixed in PR #42" in result
        assert "5 dependent(s)" in result

    def test_includes_active_blockers(self):
        gated = self._make_gated_data()
        details = {"blocker-a": {"dependent_count": 3}}
        result = _build_structured_report(10, 5, gated, [], details)
        assert "`blocker-a`" in result
        assert "Blocker A text" in result
        assert "Gates 1 belief(s)" in result
        assert "`gated-1`" in result

    def test_empty_network(self):
        gated = self._make_gated_data({})
        result = _build_structured_report(0, 0, gated, [], {})
        assert "# Gated Beliefs Report" in result
        assert "0 IN / 0 OUT" in result
        assert "No blockers or retracted premises found." in result

    def test_no_retracted_premises(self):
        gated = self._make_gated_data()
        result = _build_structured_report(10, 5, gated, [], {})
        assert "Fixed Defects" not in result

    def test_blockers_sorted_by_gated_count(self):
        gated = self._make_gated_data({
            "blocker-few": {
                "text": "Few",
                "gated": [{"id": "g1", "text": "one"}],
            },
            "blocker-many": {
                "text": "Many",
                "gated": [
                    {"id": "g2", "text": "two"},
                    {"id": "g3", "text": "three"},
                    {"id": "g4", "text": "four"},
                ],
            },
        })
        result = _build_structured_report(10, 5, gated, [], {})
        pos_many = result.index("`blocker-many`")
        pos_few = result.index("`blocker-few`")
        assert pos_many < pos_few


class TestFormatDataForPrompt:

    def test_includes_counts(self):
        gated = {"blockers": {}, "blocker_count": 0, "gated_count": 0}
        result = _format_data_for_prompt(50, 20, gated, [], {})
        assert "50 IN" in result
        assert "20 OUT" in result

    def test_includes_retracted(self):
        gated = {"blockers": {}, "blocker_count": 0, "gated_count": 0}
        retracted = [{
            "id": "old-bug",
            "text": "Old bug text",
            "retract_reason": "Fixed",
            "dependent_count": 3,
        }]
        result = _format_data_for_prompt(10, 5, gated, retracted, {})
        assert "old-bug" in result
        assert "Old bug text" in result
        assert "Fixed" in result

    def test_includes_blockers(self):
        gated = {
            "blockers": {
                "b1": {"text": "Blocker text", "gated": [{"id": "g1", "text": "Gated"}]},
            },
            "blocker_count": 1,
            "gated_count": 1,
        }
        result = _format_data_for_prompt(10, 5, gated, [], {"b1": {"dependent_count": 2}})
        assert "b1" in result
        assert "Blocker text" in result
        assert "g1" in result


class TestReportGatedApi:

    def test_returns_report_string(self, db):
        result = api.report_gated(db_path=db)
        assert "# Gated Beliefs Report" in result["report"]
        assert isinstance(result["report"], str)

    def test_counts_match(self, db):
        result = api.report_gated(db_path=db)
        assert result["blocker_count"] == 1
        assert result["gated_count"] == 1
        assert result["retracted_count"] == 1

    def test_report_contains_blocker(self, db):
        result = api.report_gated(db_path=db)
        assert "blocker-x" in result["report"]
        assert "gated-conclusion" in result["report"]

    def test_report_contains_retracted(self, db):
        result = api.report_gated(db_path=db)
        assert "retracted-defect" in result["report"]
        assert "Fixed in PR #99" in result["report"]

    def test_empty_db(self, tmp_path):
        db = str(tmp_path / "empty.db")
        result = api.report_gated(db_path=db)
        assert result["blocker_count"] == 0
        assert result["gated_count"] == 0
        assert result["retracted_count"] == 0
        assert "# Gated Beliefs Report" in result["report"]


class TestReportGatedLlm:

    def test_llm_mode_calls_invoke_model(self):
        gated = {
            "blockers": {"b1": {"text": "T", "gated": [{"id": "g1", "text": "G"}]}},
            "blocker_count": 1, "gated_count": 1,
        }
        with patch("reasonsforge.llm.invoke_model",
                    return_value="LLM narrative") as mock:
            result = report_gated(gated, [], {}, 10, 5, model="claude")
            mock.assert_called_once()
            assert "LLM narrative" in result

    def test_no_model_no_invoke(self):
        gated = {"blockers": {}, "blocker_count": 0, "gated_count": 0}
        with patch("reasonsforge.llm.invoke_model") as mock:
            report_gated(gated, [], {}, 10, 5)
            mock.assert_not_called()

    def test_llm_includes_title_and_stats(self):
        gated = {"blockers": {}, "blocker_count": 0, "gated_count": 0}
        with patch("reasonsforge.llm.invoke_model",
                    return_value="Some content"):
            result = report_gated(gated, [], {}, 10, 5, model="claude")
            assert "# Gated Beliefs Report" in result
            assert "10 IN / 5 OUT" in result

    def test_api_passes_model_through(self, db):
        with patch("reasonsforge.llm.invoke_model",
                    return_value="Generated report") as mock:
            result = api.report_gated(model="claude", db_path=db)
        assert mock.call_count == 1
        assert "Generated report" in result["report"]


class TestReportGatedCli:

    def test_help(self):
        stdout, stderr, code = run_cli("report-gated", "--help")
        assert code == 0
        assert "report-gated" in stdout or "report" in stdout

    def test_stdout_output(self, db):
        stdout, stderr, code = run_cli("report-gated", db_path=db)
        assert code == 0
        assert "# Gated Beliefs Report" in stdout
        assert "blocker(s)" in stderr

    def test_file_output(self, db, tmp_path):
        out_file = str(tmp_path / "report.md")
        stdout, stderr, code = run_cli("report-gated", "-o", out_file,
                                       db_path=db)
        assert code == 0
        assert "Report written to" in stdout
        assert os.path.isfile(out_file)
        content = open(out_file).read()
        assert "# Gated Beliefs Report" in content

"""Tests for the report belief command."""

import json
import sqlite3
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from reasonsforge import api
from reasonsforge.report_belief import _build_structured_report
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


class TestBuildStructuredReport:

    def test_basic_report(self):
        node_detail = {"text": "Water boils at 100C", "truth_value": "IN"}
        explain_steps = [
            {"node": "derived-boil", "truth_value": "IN",
             "reason": "SL justification valid",
             "antecedents": ["premise-a", "premise-b"], "label": ""},
            {"node": "premise-a", "truth_value": "IN", "reason": "premise"},
            {"node": "premise-b", "truth_value": "IN", "reason": "premise"},
        ]
        premises_data = [
            {"id": "premise-a", "text": "Water is H2O", "truth_value": "IN",
             "source": "chem.md"},
            {"id": "premise-b", "text": "Pressure is 1 atm", "truth_value": "IN",
             "source": "env.md"},
        ]

        report = _build_structured_report(
            "derived-boil", node_detail, explain_steps, premises_data,
            source_chunks={},
        )

        assert "# Belief Report: `derived-boil`" in report
        assert "Water boils at 100C" in report
        assert "`premise-a`" in report
        assert "`premise-b`" in report
        assert "Water is H2O" in report
        assert "chem.md" in report
        assert "Root Premises (2)" in report

    def test_with_source_chunks(self):
        node_detail = {"text": "A claim", "truth_value": "IN"}
        explain_steps = [
            {"node": "p1", "truth_value": "IN", "reason": "premise"},
        ]
        premises_data = [
            {"id": "p1", "text": "Evidence from source", "truth_value": "IN",
             "source": "doc.md"},
        ]
        chunks = {
            "p1": [
                {"filename": "doc.md", "section": "Chapter 1",
                 "text": "The relevant evidence text.", "cluster": ""},
            ],
        }

        report = _build_structured_report(
            "test-belief", node_detail, explain_steps, premises_data,
            source_chunks=chunks,
        )

        assert "Source Evidence:" in report
        assert "doc.md > Chapter 1" in report
        assert "The relevant evidence text." in report

    def test_no_premises(self):
        node_detail = {"text": "A premise itself", "truth_value": "IN"}
        explain_steps = [
            {"node": "p1", "truth_value": "IN", "reason": "premise"},
        ]

        report = _build_structured_report(
            "p1", node_detail, explain_steps, premises_data=[],
            source_chunks={},
        )

        assert "Root Premises (0)" in report
        assert "No root premises found." in report

    def test_out_belief(self):
        node_detail = {"text": "Retracted claim", "truth_value": "OUT"}
        explain_steps = [
            {"node": "d1", "truth_value": "OUT",
             "reason": "all justifications invalid"},
        ]
        premises_data = [
            {"id": "p1", "text": "Retracted evidence", "truth_value": "OUT",
             "source": ""},
        ]

        report = _build_structured_report(
            "d1", node_detail, explain_steps, premises_data,
            source_chunks={},
        )

        assert "status: OUT" in report
        assert "[-]" in report

    def test_outlist_shown(self):
        node_detail = {"text": "Gated belief", "truth_value": "OUT"}
        explain_steps = [
            {"node": "g1", "truth_value": "OUT",
             "reason": "SL justification invalid",
             "antecedents": ["p1"], "outlist": ["blocker-1"],
             "label": "gate"},
        ]

        report = _build_structured_report(
            "g1", node_detail, explain_steps, premises_data=[],
            source_chunks={},
        )

        assert "`blocker-1`" in report
        assert "outlist" in report


class TestApiReportBelief:

    def test_basic_report(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "Water is H2O", db_path=db)
        api.add_node("premise-b", "Pressure is 1 atm", db_path=db)
        api.add_node("derived-boil", "Water boils at 100C",
                     sl="premise-a,premise-b", db_path=db)

        result = api.report_belief("derived-boil", db_path=db)

        assert result["premise_count"] == 2
        assert "derived-boil" in result["report"]
        assert "premise-a" in result["report"]
        assert "premise-b" in result["report"]

    def test_premise_report(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("p1", "A fact", db_path=db)

        result = api.report_belief("p1", db_path=db)

        assert result["premise_count"] == 1
        assert "p1" in result["report"]

    def test_with_sources_db(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("p1", "PostgreSQL handles storage", db_path=db)
        api.add_node("d1", "Storage is relational", sl="p1", db_path=db)

        sources_db = str(tmp_path / "sources.db")
        conn = sqlite3.connect(sources_db)
        conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT, cluster TEXT, filename TEXT, section TEXT)")
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=id)")
        conn.execute("INSERT INTO chunks VALUES (1, 'PostgreSQL is the primary database for persistent storage', '', 'arch.md', 'Storage Layer')")
        conn.execute("INSERT INTO chunks_fts(rowid, text) VALUES (1, 'PostgreSQL is the primary database for persistent storage')")
        conn.commit()
        conn.close()

        result = api.report_belief("d1", sources_db=sources_db, db_path=db)

        assert result["premise_count"] == 1
        assert "Source Evidence:" in result["report"] or "PostgreSQL" in result["report"]

    def test_missing_node_raises(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)

        with pytest.raises(KeyError):
            api.report_belief("nonexistent", db_path=db)


class TestCliReport:

    def test_help(self):
        stdout, stderr, code = run_cli("report", "--help")
        assert code == 0
        assert "report" in stdout.lower()

    def test_basic(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("p1", "A fact", db_path=db)
        api.add_node("d1", "A derived fact", sl="p1", db_path=db)

        stdout, stderr, code = run_cli("report", "d1", db_path=db)

        assert code == 0
        assert "Belief Report" in stdout
        assert "d1" in stdout
        assert "p1" in stdout

    def test_output_file(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("p1", "A fact", db_path=db)

        out_file = str(tmp_path / "report.md")
        stdout, stderr, code = run_cli(
            "report", "p1", "-o", out_file, db_path=db,
        )

        assert code == 0
        with open(out_file) as f:
            content = f.read()
        assert "Belief Report" in content

"""Tests for the contradictions module (LLM-powered nogood detection)."""

import sys
from io import StringIO
from unittest.mock import patch

import pytest

from reasonsforge.contradictions import (
    format_beliefs_for_contradiction_check,
    parse_contradiction_response,
    detect_contradictions,
    CONTRADICTION_BATCH_SIZE,
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
    return {
        "premise-a": {
            "text": "A is true",
            "truth_value": "IN",
            "justifications": [],
        },
        "premise-b": {
            "text": "B is false",
            "truth_value": "IN",
            "justifications": [],
        },
        "premise-c": {
            "text": "C is true",
            "truth_value": "IN",
            "justifications": [],
        },
        "out-node": {
            "text": "This is OUT",
            "truth_value": "OUT",
            "justifications": [],
        },
    }


class TestFormatBeliefsForContradictionCheck:

    def test_formats_beliefs_as_list(self):
        nodes = _make_nodes()
        result = format_beliefs_for_contradiction_check(
            ["premise-a", "premise-b"], nodes)
        assert "- `premise-a`: A is true" in result
        assert "- `premise-b`: B is false" in result

    def test_truncates_long_text(self):
        nodes = {"long": {
            "text": "x" * 300,
            "truth_value": "IN",
        }}
        result = format_beliefs_for_contradiction_check(["long"], nodes)
        assert len(result.split(": ", 1)[1]) == 200
        assert result.endswith("...")

    def test_skips_missing_nodes(self):
        nodes = _make_nodes()
        result = format_beliefs_for_contradiction_check(
            ["premise-a", "nonexistent"], nodes)
        assert "premise-a" in result
        assert "nonexistent" not in result


class TestParseContradictionResponse:

    def test_parses_nogood_sections(self):
        response = (
            "### NOGOOD scope-conflict\n"
            "- Claims: premise-a, premise-b\n"
            "- Analysis: A says true, B says false\n"
            "- Severity: High\n"
        )
        results = parse_contradiction_response(response)
        assert len(results) == 1
        assert results[0]["id"] == "scope-conflict"
        assert results[0]["claims"] == ["premise-a", "premise-b"]
        assert results[0]["analysis"] == "A says true, B says false"
        assert results[0]["severity"] == "High"

    def test_requires_minimum_two_claims(self):
        response = (
            "### NOGOOD single-claim\n"
            "- Claims: premise-a\n"
            "- Analysis: just one\n"
            "- Severity: Low\n"
        )
        results = parse_contradiction_response(response)
        assert results == []

    def test_filters_nonexistent_claim_ids(self):
        response = (
            "### NOGOOD bad-refs\n"
            "- Claims: premise-a, fake-node, premise-b\n"
            "- Analysis: mixed real and fake\n"
            "- Severity: Medium\n"
        )
        valid = {"premise-a", "premise-b"}
        results = parse_contradiction_response(response, valid_ids=valid)
        assert len(results) == 1
        assert results[0]["claims"] == ["premise-a", "premise-b"]

    def test_filters_to_below_two_claims_drops_nogood(self):
        response = (
            "### NOGOOD all-fake\n"
            "- Claims: fake-1, fake-2\n"
            "- Analysis: none real\n"
            "- Severity: Low\n"
        )
        valid = {"premise-a"}
        results = parse_contradiction_response(response, valid_ids=valid)
        assert results == []

    def test_malformed_response_returns_empty(self):
        response = "No contradictions detected."
        results = parse_contradiction_response(response)
        assert results == []

    def test_extracts_severity(self):
        response = (
            "### NOGOOD sev-test\n"
            "- Claims: premise-a, premise-b\n"
            "- Analysis: test\n"
            "- Severity: Medium\n"
        )
        results = parse_contradiction_response(response)
        assert results[0]["severity"] == "Medium"

    def test_multiple_nogoods(self):
        response = (
            "### NOGOOD first\n"
            "- Claims: premise-a, premise-b\n"
            "- Analysis: first conflict\n"
            "- Severity: High\n"
            "\n"
            "### NOGOOD second\n"
            "- Claims: premise-b, premise-c\n"
            "- Analysis: second conflict\n"
            "- Severity: Low\n"
        )
        valid = {"premise-a", "premise-b", "premise-c"}
        results = parse_contradiction_response(response, valid_ids=valid)
        assert len(results) == 2
        assert results[0]["id"] == "first"
        assert results[1]["id"] == "second"

    def test_three_claims(self):
        response = (
            "### NOGOOD triple\n"
            "- Claims: premise-a, premise-b, premise-c\n"
            "- Analysis: three-way conflict\n"
            "- Severity: High\n"
        )
        results = parse_contradiction_response(response)
        assert len(results) == 1
        assert len(results[0]["claims"]) == 3


class TestDetectContradictions:

    def test_batches_beliefs(self):
        nodes = _make_nodes()
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": "No contradictions detected.",
            "stderr": "",
        })()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run:
            detect_contradictions(nodes, batch_size=2)
        # 3 IN beliefs, batch_size=2 → 2 batches
        assert mock_run.call_count == 2

    def test_empty_network_returns_empty(self):
        nodes = {
            "out-only": {
                "text": "everything is OUT",
                "truth_value": "OUT",
                "justifications": [],
            },
        }
        results = detect_contradictions(nodes)
        assert results == []

    def test_timeout_passed_through(self):
        nodes = _make_nodes()
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": "No contradictions detected.",
            "stderr": "",
        })()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run:
            detect_contradictions(nodes, belief_ids=["premise-a", "premise-b"],
                                  timeout=600)
        assert mock_run.call_args[1]["timeout"] == 600

    def test_batch_failure_continues(self):
        nodes = _make_nodes()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=RuntimeError("LLM failed")):
            results = detect_contradictions(nodes)
        assert results == []

    def test_filters_to_in_only(self):
        nodes = _make_nodes()
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": "No contradictions detected.",
            "stderr": "",
        })()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run:
            detect_contradictions(nodes, batch_size=100)
        # Only 3 IN beliefs (premise-a, premise-b, premise-c), out-node excluded
        prompt = mock_run.call_args[1]["input"]
        assert "out-node" not in prompt

    def test_specific_ids_filtered(self):
        nodes = _make_nodes()
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": "No contradictions detected.",
            "stderr": "",
        })()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run:
            detect_contradictions(nodes,
                                  belief_ids=["premise-a", "out-node"],
                                  batch_size=100)
        # out-node is OUT, so only premise-a should be checked
        assert mock_run.call_count == 1
        prompt = mock_run.call_args[1]["input"]
        assert "premise-a" in prompt
        assert "out-node" not in prompt


class TestDetectContradictionsApi:

    @pytest.fixture
    def db_path(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("belief-a", "X is always true", db_path=db)
        api.add_node("belief-b", "X is never true", db_path=db)
        api.add_node("belief-c", "Y is sometimes true", db_path=db)
        return db

    def test_filters_to_in_only(self, db_path):
        api.retract_node("belief-c", reason="testing", db_path=db_path)
        nogood_response = (
            "### NOGOOD x-conflict\n"
            "- Claims: belief-a, belief-b\n"
            "- Analysis: direct contradiction\n"
            "- Severity: High\n"
        )
        mock_result = type("R", (), {
            "returncode": 0, "stdout": nogood_response, "stderr": ""
        })()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.detect_contradictions(db_path=db_path)
        assert result["checked"] == 2
        assert result["found"] == 1

    def test_sample_limits(self, db_path):
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": "No contradictions detected.",
            "stderr": "",
        })()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.detect_contradictions(sample=2, db_path=db_path)
        assert result["checked"] == 2

    def test_auto_apply_calls_add_nogood(self, db_path):
        nogood_response = (
            "### NOGOOD x-conflict\n"
            "- Claims: belief-a, belief-b\n"
            "- Analysis: direct contradiction\n"
            "- Severity: High\n"
        )
        mock_result = type("R", (), {
            "returncode": 0, "stdout": nogood_response, "stderr": ""
        })()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.detect_contradictions(auto_apply=True, db_path=db_path)
        assert result["applied"] >= 1



class TestCmdContradictions:

    @pytest.fixture
    def db_path(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("belief-a", "X is always true", db_path=db)
        api.add_node("belief-b", "X is never true", db_path=db)
        return db

    def _mock_response(self, text):
        return type("R", (), {
            "returncode": 0, "stdout": text, "stderr": ""
        })()

    def test_no_contradictions_message(self, db_path):
        mock_result = self._mock_response("No contradictions detected.")
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            stdout, stderr, code = run_cli(
                "contradictions", db_path=db_path)
        assert "No contradictions detected" in stdout

    def test_displays_found_contradictions(self, db_path, tmp_path):
        nogood_response = (
            "### NOGOOD x-conflict\n"
            "- Claims: belief-a, belief-b\n"
            "- Analysis: direct negation\n"
            "- Severity: High\n"
        )
        mock_result = self._mock_response(nogood_response)
        output_file = str(tmp_path / "plan.md")
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            stdout, stderr, code = run_cli(
                "contradictions", "-o", output_file, db_path=db_path)
        assert "[NOGOOD] x-conflict (High)" in stdout
        assert "belief-a, belief-b" in stdout
        assert "direct negation" in stdout

    def test_output_writes_plan(self, db_path, tmp_path):
        output_file = str(tmp_path / "plan.md")
        nogood_response = (
            "### NOGOOD x-conflict\n"
            "- Claims: belief-a, belief-b\n"
            "- Analysis: direct negation\n"
            "- Severity: High\n"
        )
        mock_result = self._mock_response(nogood_response)
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            stdout, stderr, code = run_cli(
                "contradictions", "-o", output_file, db_path=db_path)
        assert "--accept" in stdout
        with open(output_file) as f:
            content = f.read()
        assert "# Contradiction Plan" in content
        assert "### NOGOOD x-conflict [APPLY]" in content
        assert "belief-a, belief-b" in content


try:
    from reasonsforge.cluster import HAS_CLUSTER_DEPS
except ImportError:
    HAS_CLUSTER_DEPS = False

skip_no_cluster = pytest.mark.skipif(
    not HAS_CLUSTER_DEPS,
    reason="sentence-transformers and scikit-learn not installed"
)


@skip_no_cluster
class TestDetectContradictionsSemantic:

    def test_semantic_groups_by_cluster(self):
        nodes = {
            "auth-login-validates": {
                "text": "Login validates user credentials against the database",
                "truth_value": "IN",
                "justifications": [],
            },
            "auth-login-skips-validation": {
                "text": "Login skips credential validation for speed",
                "truth_value": "IN",
                "justifications": [],
            },
            "db-queries-are-fast": {
                "text": "Database queries are optimized for read-heavy workloads",
                "truth_value": "IN",
                "justifications": [],
            },
        }
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": "No contradictions detected.",
            "stderr": "",
        })()
        from reasonsforge.contradictions import detect_contradictions_semantic
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run:
            detect_contradictions_semantic(nodes)
        assert mock_run.call_count >= 1
        all_prompts = " ".join(
            call[1].get("input", "") for call in mock_run.call_args_list
        )
        assert "auth-login-validates" in all_prompts
        assert "auth-login-skips-validation" in all_prompts

    def test_semantic_returns_found_contradictions(self):
        nodes = {
            "auth-login-validates": {
                "text": "Login validates user credentials against the database",
                "truth_value": "IN",
                "justifications": [],
            },
            "auth-login-skips-validation": {
                "text": "Login skips credential validation for speed",
                "truth_value": "IN",
                "justifications": [],
            },
            "db-queries-are-fast": {
                "text": "Database queries are optimized for read-heavy workloads",
                "truth_value": "IN",
                "justifications": [],
            },
        }
        nogood_response = (
            "### NOGOOD auth-contradiction\n"
            "- Claims: auth-login-validates, auth-login-skips-validation\n"
            "- Analysis: Cannot both validate and skip validation\n"
            "- Severity: High\n"
        )
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": nogood_response,
            "stderr": "",
        })()
        from reasonsforge.contradictions import detect_contradictions_semantic
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            results = detect_contradictions_semantic(nodes)
        assert len(results) == 1
        assert results[0]["id"] == "auth-contradiction"
        assert set(results[0]["claims"]) == {"auth-login-validates", "auth-login-skips-validation"}

    def test_semantic_empty_network(self):
        from reasonsforge.contradictions import detect_contradictions_semantic
        nodes = {"out-only": {"text": "x", "truth_value": "OUT", "justifications": []}}
        results = detect_contradictions_semantic(nodes)
        assert results == []

    def test_semantic_single_belief_skips_llm(self):
        from reasonsforge.contradictions import detect_contradictions_semantic
        nodes = {"only-one": {"text": "Single belief", "truth_value": "IN", "justifications": []}}
        with patch("reasonsforge.llm.subprocess.run") as mock_run:
            results = detect_contradictions_semantic(nodes)
        assert results == []
        mock_run.assert_not_called()


class TestContradictionPlan:

    def test_write_contradiction_plan_format(self, tmp_path):
        contradictions = [
            {"id": "test-nogood", "claims": ["a", "b"],
             "analysis": "They conflict", "severity": "High"},
            {"id": "another-nogood", "claims": ["c", "d"],
             "analysis": "Also conflict", "severity": "Medium"},
        ]
        path = str(tmp_path / "plan.md")
        api.write_contradiction_plan(contradictions, path)
        text = open(path).read()
        assert "# Contradiction Plan" in text
        assert "### NOGOOD test-nogood [APPLY]" in text
        assert "### NOGOOD another-nogood [APPLY]" in text
        assert "- Claims: a, b" in text
        assert "- Analysis: They conflict" in text
        assert "- Severity: High" in text
        assert "--accept" in text

    def test_write_contradiction_plan_append(self, tmp_path):
        path = str(tmp_path / "plan.md")
        batch1 = [{"id": "first", "claims": ["a", "b"], "analysis": "", "severity": ""}]
        batch2 = [{"id": "second", "claims": ["c", "d"], "analysis": "", "severity": ""}]
        api.write_contradiction_plan(batch1, path, append=False)
        api.write_contradiction_plan(batch2, path, append=True)
        text = open(path).read()
        assert text.count("# Contradiction Plan") == 1
        assert "### NOGOOD first [APPLY]" in text
        assert "### NOGOOD second [APPLY]" in text

    def test_parse_contradiction_plan_apply(self):
        plan_text = (
            "# Contradiction Plan\n\n---\n\n"
            "### NOGOOD keep-this [APPLY]\n"
            "- Claims: a, b\n"
            "- Analysis: conflict\n\n"
            "### NOGOOD skip-this [SKIP]\n"
            "- Claims: c, d\n"
            "- Analysis: not really\n\n"
        )
        entries = api.parse_contradiction_plan(plan_text)
        assert len(entries) == 1
        assert entries[0]["id"] == "keep-this"
        assert entries[0]["claims"] == ["a", "b"]

    def test_parse_contradiction_plan_deleted(self):
        plan_text = (
            "# Contradiction Plan\n\n---\n\n"
            "### NOGOOD keep-this [APPLY]\n"
            "- Claims: a, b\n\n"
        )
        entries = api.parse_contradiction_plan(plan_text)
        assert len(entries) == 1

    def test_apply_contradiction_plan(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Belief A", db_path=db)
        api.add_node("b", "Belief B", db_path=db)
        plan = [{"id": "test-nogood", "claims": ["a", "b"]}]
        result = api.apply_contradiction_plan(plan, db_path=db)
        assert result["applied"] == 1
        assert len(result["nogoods"]) == 1
        assert result["nogoods"][0]["id"] == "test-nogood"

    def test_accept_cli_flow(self, tmp_path):
        db = str(tmp_path / "test.db")
        run_cli("init", db_path=db)
        run_cli("add", "x", "Belief X", db_path=db)
        run_cli("add", "y", "Belief Y", db_path=db)
        plan_file = tmp_path / "plan.md"
        plan_file.write_text(
            "### NOGOOD test [APPLY]\n"
            "- Claims: x, y\n"
        )
        out, err, code = run_cli("contradictions", "--accept",
                                  str(plan_file), db_path=db)
        assert code == 0
        assert "Applied 1 nogood" in out

    def test_accept_missing_file(self):
        out, err, code = run_cli("contradictions", "--accept",
                                  "/nonexistent.md", db_path="/tmp/dummy.db")
        assert code == 1

    def test_accept_empty_plan(self, tmp_path):
        db = str(tmp_path / "test.db")
        run_cli("init", db_path=db)
        plan_file = tmp_path / "empty.md"
        plan_file.write_text("")
        out, err, code = run_cli("contradictions", "--accept",
                                  str(plan_file), db_path=db)
        assert code == 0
        assert "No APPLY entries" in out

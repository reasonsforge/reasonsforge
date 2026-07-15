"""Tests for the repair module (smuggled premise search-and-link)."""

import json
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from reasonsforge.repair import (
    parse_extract_response,
    parse_match_response,
    extract_smuggled_claim,
    find_matching_premises,
    repair_smuggled_beliefs,
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


def _make_repair_nodes():
    """Build nodes with a smuggled-premise scenario."""
    return {
        "premise-a": {
            "text": "Water boils at 100C at sea level",
            "truth_value": "IN",
            "justifications": [],
        },
        "premise-b": {
            "text": "The experiment was conducted at sea level",
            "truth_value": "IN",
            "justifications": [],
        },
        "premise-c": {
            "text": "The sample reached 100C",
            "truth_value": "IN",
            "justifications": [],
        },
        "derived-boil": {
            "text": "The water in the sample boiled, releasing steam",
            "truth_value": "IN",
            "justifications": [{
                "type": "SL",
                "antecedents": ["premise-b", "premise-c"],
                "outlist": [],
                "label": "boiling inference",
            }],
        },
    }


# --- parse_extract_response ---

class TestParseExtractResponse:
    def test_plain_text(self):
        assert parse_extract_response("Water boils at 100C") == "Water boils at 100C"

    def test_strips_whitespace(self):
        assert parse_extract_response("  Water boils at 100C  \n") == "Water boils at 100C"

    def test_strips_quotes(self):
        assert parse_extract_response('"Water boils at 100C"') == "Water boils at 100C"
        assert parse_extract_response("'Water boils at 100C'") == "Water boils at 100C"

    def test_strips_prefix(self):
        assert parse_extract_response("Smuggled claim: Water boils") == "Water boils"

    def test_empty(self):
        assert parse_extract_response("") == ""
        assert parse_extract_response("   ") == ""

    def test_multiline_returns_full(self):
        resp = "Water boils at 100C.\nThis is a known physical fact."
        result = parse_extract_response(resp)
        assert "Water boils at 100C" in result


# --- parse_match_response ---

class TestParseMatchResponse:
    def test_valid_json(self):
        resp = '{"matched_ids": ["premise-a"], "rationale": "directly states it"}'
        result = parse_match_response(resp, {"premise-a", "premise-b"})
        assert result["matched_ids"] == ["premise-a"]
        assert result["rationale"] == "directly states it"

    def test_json_in_prose(self):
        resp = 'Here is my answer:\n{"matched_ids": ["premise-a"], "rationale": "match"}\nDone.'
        result = parse_match_response(resp, {"premise-a"})
        assert result["matched_ids"] == ["premise-a"]

    def test_filters_invalid_ids(self):
        resp = '{"matched_ids": ["premise-a", "nonexistent"], "rationale": "partial"}'
        result = parse_match_response(resp, {"premise-a"})
        assert result["matched_ids"] == ["premise-a"]

    def test_empty_match(self):
        resp = '{"matched_ids": [], "rationale": "no match found"}'
        result = parse_match_response(resp, {"premise-a"})
        assert result["matched_ids"] == []
        assert result["rationale"] == "no match found"

    def test_malformed_json(self):
        result = parse_match_response("no json here", {"premise-a"})
        assert result["matched_ids"] == []

    def test_no_matched_ids_key(self):
        resp = '{"rationale": "something"}'
        result = parse_match_response(resp, {"premise-a"})
        assert result["matched_ids"] == []


# --- extract_smuggled_claim ---

class TestExtractSmuggledClaim:
    def test_calls_llm_with_context(self):
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": "Water boils at 100C at sea level",
            "stderr": "",
        })()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result) as mock_run:
            claim = extract_smuggled_claim(
                "### derived-boil\nClaim: water boiled",
                "Conclusion assumes boiling point knowledge",
            )
            assert claim == "Water boils at 100C at sea level"
            prompt_sent = mock_run.call_args[1].get("input") or mock_run.call_args[0][0]
            if isinstance(prompt_sent, str):
                assert "derived-boil" in prompt_sent or "boiling" in prompt_sent

    def test_empty_response(self):
        mock_result = type("R", (), {
            "returncode": 0, "stdout": "  ", "stderr": "",
        })()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            claim = extract_smuggled_claim("context", "comment")
            assert claim == ""


# --- find_matching_premises ---

class TestFindMatchingPremises:
    def test_matches_premise(self):
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": '{"matched_ids": ["premise-a"], "rationale": "states the fact"}',
            "stderr": "",
        })()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = find_matching_premises(
                "Water boils at 100C",
                [{"id": "premise-a", "text": "Water boils at 100C at sea level"}],
            )
            assert result["matched_ids"] == ["premise-a"]

    def test_empty_candidates_skips_llm(self):
        with patch("reasonsforge.llm.subprocess.run") as mock_run:
            result = find_matching_premises("some claim", [])
            mock_run.assert_not_called()
            assert result["matched_ids"] == []
            assert "no candidates" in result["rationale"]


# --- repair_smuggled_beliefs ---

class TestRepairSmuggledBeliefs:
    def _mock_result(self, stdout):
        return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    def _mock_search(self, results):
        return lambda query, **kw: json.dumps(results)

    def test_full_repair_flow(self):
        nodes = _make_repair_nodes()
        review_results = [{
            "id": "derived-boil",
            "valid": False,
            "comment": "Assumes boiling point knowledge not in antecedents",
        }]
        search_results = [
            {"id": "premise-a", "text": "Water boils at 100C at sea level",
             "truth_value": "IN", "source": "", "match": True},
        ]
        extract_resp = self._mock_result("Water boils at 100C at sea level")
        match_resp = self._mock_result(
            '{"matched_ids": ["premise-a"], "rationale": "directly states it"}'
        )

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[extract_resp, match_resp]), \
             patch("reasonsforge.api.add_justification") as mock_add:
            repairs = repair_smuggled_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=self._mock_search(search_results),
            )

        assert len(repairs) == 1
        assert repairs[0]["status"] == "repaired"
        assert repairs[0]["matched_premises"] == ["premise-a"]
        mock_add.assert_called_once()
        assert "premise-a" in mock_add.call_args[1]["sl"]

    def test_dry_run_no_apply(self):
        nodes = _make_repair_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "smuggled",
        }]
        search_results = [
            {"id": "premise-a", "text": "Water boils at 100C at sea level",
             "truth_value": "IN", "source": "", "match": True},
        ]
        extract_resp = self._mock_result("Water boils at 100C")
        match_resp = self._mock_result(
            '{"matched_ids": ["premise-a"], "rationale": "match"}'
        )

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[extract_resp, match_resp]), \
             patch("reasonsforge.api.add_justification") as mock_add:
            repairs = repair_smuggled_beliefs(
                review_results, nodes, db_path="test.db", dry_run=True,
                search_fn=self._mock_search(search_results),
            )

        assert repairs[0]["status"] == "repaired"
        mock_add.assert_not_called()

    def test_no_candidates(self):
        nodes = _make_repair_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "smuggled",
        }]
        extract_resp = self._mock_result("Water boils at 100C")

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[extract_resp]):
            repairs = repair_smuggled_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=self._mock_search([]),
            )

        assert repairs[0]["status"] == "no_candidates"

    def test_no_match(self):
        nodes = _make_repair_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "smuggled",
        }]
        search_results = [
            {"id": "premise-a", "text": "Water boils at 100C at sea level",
             "truth_value": "IN", "source": "", "match": True},
        ]
        extract_resp = self._mock_result("Water boils at 100C")
        match_resp = self._mock_result(
            '{"matched_ids": [], "rationale": "no match"}'
        )

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[extract_resp, match_resp]):
            repairs = repair_smuggled_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=self._mock_search(search_results),
            )

        assert repairs[0]["status"] == "no_match"

    def test_extraction_failed(self):
        nodes = _make_repair_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "smuggled",
        }]
        extract_resp = self._mock_result("   ")

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[extract_resp]):
            repairs = repair_smuggled_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=self._mock_search([]),
            )

        assert repairs[0]["status"] == "extraction_failed"

    def test_skips_existing_antecedents(self):
        """Search results that are already antecedents should be filtered out."""
        nodes = _make_repair_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "smuggled",
        }]
        search_results = [
            {"id": "premise-b", "text": "The experiment was conducted at sea level",
             "truth_value": "IN", "source": "", "match": True},
        ]
        extract_resp = self._mock_result("Something about sea level")

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[extract_resp]):
            repairs = repair_smuggled_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=self._mock_search(search_results),
            )

        assert repairs[0]["status"] == "no_candidates"

    def test_skips_derived_beliefs_in_candidates(self):
        """Derived beliefs (with justifications) should not be candidates."""
        nodes = _make_repair_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "smuggled",
        }]
        search_results = [
            {"id": "derived-boil", "text": "The water boiled",
             "truth_value": "IN", "source": "", "match": True},
        ]
        extract_resp = self._mock_result("Boiling occurs")

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[extract_resp]):
            repairs = repair_smuggled_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=self._mock_search(search_results),
            )

        assert repairs[0]["status"] == "no_candidates"

    def test_belief_not_in_network(self):
        nodes = _make_repair_nodes()
        review_results = [{
            "id": "nonexistent", "valid": False, "comment": "smuggled",
        }]

        repairs = repair_smuggled_beliefs(
            review_results, nodes, db_path="test.db",
            search_fn=self._mock_search([]),
        )

        assert repairs[0]["status"] == "error"
        assert "not found" in repairs[0]["error"]

    def test_preserves_outlist(self):
        """Repaired justification should preserve the original outlist."""
        nodes = _make_repair_nodes()
        nodes["premise-d"] = {
            "text": "Pressure is normal", "truth_value": "IN", "justifications": [],
        }
        nodes["blocker-x"] = {
            "text": "Experiment was invalid", "truth_value": "OUT", "justifications": [],
        }
        nodes["derived-boil"]["justifications"][0]["outlist"] = ["blocker-x"]

        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "smuggled",
        }]
        search_results = [
            {"id": "premise-d", "text": "Pressure is normal",
             "truth_value": "IN", "source": "", "match": True},
        ]
        extract_resp = self._mock_result("Normal pressure assumed")
        match_resp = self._mock_result(
            '{"matched_ids": ["premise-d"], "rationale": "match"}'
        )

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[extract_resp, match_resp]), \
             patch("reasonsforge.api.add_justification") as mock_add:
            repairs = repair_smuggled_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=self._mock_search(search_results),
            )

        assert repairs[0]["status"] == "repaired"
        mock_add.assert_called_once()
        assert mock_add.call_args[1]["unless"] == "blocker-x"

    def test_multiple_invalid_beliefs(self):
        """Processes multiple invalid beliefs in one batch."""
        nodes = _make_repair_nodes()
        nodes["premise-d"] = {
            "text": "Steam is gaseous water", "truth_value": "IN", "justifications": [],
        }
        nodes["derived-steam"] = {
            "text": "Steam was produced",
            "truth_value": "IN",
            "justifications": [{
                "type": "SL",
                "antecedents": ["premise-c"],
                "outlist": [],
                "label": "",
            }],
        }
        review_results = [
            {"id": "derived-boil", "valid": False, "comment": "smuggled boiling"},
            {"id": "derived-steam", "valid": False, "comment": "smuggled steam"},
        ]
        extract_resp1 = self._mock_result("Water boils at 100C")
        match_resp1 = self._mock_result(
            '{"matched_ids": ["premise-a"], "rationale": "match"}'
        )
        extract_resp2 = self._mock_result("Steam is gaseous water")
        match_resp2 = self._mock_result(
            '{"matched_ids": ["premise-d"], "rationale": "match"}'
        )
        search1 = [
            {"id": "premise-a", "text": "Water boils at 100C at sea level",
             "truth_value": "IN", "source": "", "match": True},
        ]
        search2 = [
            {"id": "premise-d", "text": "Steam is gaseous water",
             "truth_value": "IN", "source": "", "match": True},
        ]
        call_count = [0]
        def mock_search(query, **kw):
            call_count[0] += 1
            return json.dumps(search1 if call_count[0] == 1 else search2)

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[
                 extract_resp1, match_resp1, extract_resp2, match_resp2,
             ]), \
             patch("reasonsforge.api.add_justification"):
            repairs = repair_smuggled_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=mock_search,
            )

        assert len(repairs) == 2
        assert repairs[0]["status"] == "repaired"
        assert repairs[1]["status"] == "repaired"

    def test_skips_valid_beliefs(self):
        """Only processes beliefs where valid=False."""
        nodes = _make_repair_nodes()
        review_results = [
            {"id": "derived-boil", "valid": True, "comment": "looks good"},
        ]

        repairs = repair_smuggled_beliefs(
            review_results, nodes, db_path="test.db",
            search_fn=self._mock_search([]),
        )

        assert len(repairs) == 0


# --- CLI ---

class TestCmdRepairSmuggled:
    def test_help(self):
        stdout, stderr, code = run_cli("repair-smuggled", "--help")
        assert code == 0
        assert "repair" in stdout.lower() or "smuggled" in stdout.lower()

    def test_no_args_error(self):
        stdout, stderr, code = run_cli("repair-smuggled", db_path="test.db")
        assert code == 1
        assert "provide" in stderr.lower() or "provide" in stdout.lower()

    def test_from_review_file(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A is true", db_path=db)
        api.add_node("premise-b", "B is true", db_path=db)
        api.add_node("derived-ab", "AB follows", sl="premise-a", db_path=db)

        review_file = tmp_path / "review.json"
        review_file.write_text(json.dumps({
            "results": [{
                "id": "derived-ab",
                "valid": False,
                "comment": "smuggles B",
            }],
        }))

        extract_resp = type("R", (), {
            "returncode": 0, "stdout": "B is true", "stderr": "",
        })()
        match_resp = type("R", (), {
            "returncode": 0,
            "stdout": '{"matched_ids": ["premise-b"], "rationale": "match"}',
            "stderr": "",
        })()

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[extract_resp, match_resp]):
            stdout, stderr, code = run_cli(
                "repair-smuggled",
                "--review-file", str(review_file),
                db_path=db,
            )

        assert code == 0
        assert "REPAIRED" in stdout
        assert "premise-b" in stdout

    def test_dry_run(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A is true", db_path=db)
        api.add_node("derived-a", "A extended", sl="premise-a", db_path=db)

        review_file = tmp_path / "review.json"
        review_file.write_text(json.dumps({
            "results": [{
                "id": "derived-a",
                "valid": False,
                "comment": "smuggled",
            }],
        }))

        extract_resp = type("R", (), {
            "returncode": 0, "stdout": "some claim", "stderr": "",
        })()

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=[extract_resp]):
            stdout, stderr, code = run_cli(
                "repair-smuggled",
                "--review-file", str(review_file),
                "--dry-run",
                db_path=db,
            )

        assert code == 0
        assert "dry run" in stdout.lower()

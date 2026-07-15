"""Tests for the research module (triage + soften + abandon)."""

import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from reasonsforge.repair import (
    parse_triage_response,
    parse_soften_response,
    triage_belief,
    soften_belief,
    repair_beliefs,
    research_beliefs,
    _compute_depth,
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
                "label": "",
            }],
        },
    }


def _mock_result(stdout):
    return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()


def _mock_search(results):
    return lambda query, **kw: json.dumps(results)


# --- parse_triage_response ---

class TestParseTriageResponse:
    def test_valid_search_and_link(self):
        resp = '{"pattern": "search_and_link", "rationale": "missing antecedent"}'
        result = parse_triage_response(resp)
        assert result["pattern"] == "search_and_link"
        assert result["rationale"] == "missing antecedent"

    def test_valid_soften(self):
        resp = '{"pattern": "soften", "rationale": "overstated"}'
        result = parse_triage_response(resp)
        assert result["pattern"] == "soften"

    def test_valid_abandon(self):
        resp = '{"pattern": "abandon", "rationale": "too broken"}'
        result = parse_triage_response(resp)
        assert result["pattern"] == "abandon"

    def test_valid_research(self):
        resp = '{"pattern": "research", "rationale": "needs investigation"}'
        result = parse_triage_response(resp)
        assert result["pattern"] == "research"
        assert result["rationale"] == "needs investigation"

    def test_invalid_pattern_ignored(self):
        resp = '{"pattern": "unknown", "rationale": "hmm"}'
        result = parse_triage_response(resp)
        assert result["pattern"] == ""

    def test_json_in_prose(self):
        resp = 'I think:\n{"pattern": "soften", "rationale": "x"}\nDone.'
        result = parse_triage_response(resp)
        assert result["pattern"] == "soften"

    def test_malformed(self):
        result = parse_triage_response("no json here")
        assert result["pattern"] == ""


# --- parse_soften_response ---

class TestParseSoftenResponse:
    def test_valid(self):
        resp = '{"softened_text": "weaker claim", "rationale": "removed absolute"}'
        result = parse_soften_response(resp)
        assert result["softened_text"] == "weaker claim"
        assert result["rationale"] == "removed absolute"

    def test_empty_text(self):
        resp = '{"softened_text": "", "rationale": "nothing"}'
        result = parse_soften_response(resp)
        assert result["softened_text"] == ""

    def test_malformed(self):
        result = parse_soften_response("not json")
        assert result["softened_text"] == ""

    def test_json_in_prose(self):
        resp = 'Here: {"softened_text": "x", "rationale": "y"} done'
        result = parse_soften_response(resp)
        assert result["softened_text"] == "x"


# --- _compute_depth ---

class TestComputeDepth:
    def test_premise_depth_zero(self):
        nodes = _make_nodes()
        assert _compute_depth("premise-a", nodes) == 0

    def test_derived_depth_one(self):
        nodes = _make_nodes()
        assert _compute_depth("derived-boil", nodes) == 1

    def test_deep_chain(self):
        nodes = _make_nodes()
        nodes["derived-deep"] = {
            "text": "deep",
            "truth_value": "IN",
            "justifications": [{
                "type": "SL",
                "antecedents": ["derived-boil"],
                "outlist": [],
                "label": "",
            }],
        }
        assert _compute_depth("derived-deep", nodes) == 2

    def test_missing_node(self):
        assert _compute_depth("nonexistent", {}) == 0


# --- triage_belief ---

class TestTriageBelief:
    def test_returns_pattern(self):
        resp = _mock_result('{"pattern": "soften", "rationale": "overstated"}')
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=resp):
            result = triage_belief("belief context", "review comment",
                                   depth=3, flagged_ancestors=1)
        assert result["pattern"] == "soften"

    def test_prompt_includes_depth(self):
        resp = _mock_result('{"pattern": "abandon", "rationale": "x"}')
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=resp) as mock_run:
            triage_belief("ctx", "cmt", depth=5, flagged_ancestors=3)
            prompt_sent = mock_run.call_args[1].get("input", "")
            assert "5" in prompt_sent
            assert "3" in prompt_sent


# --- soften_belief ---

class TestSoftenBelief:
    def test_returns_softened_text(self):
        resp = _mock_result('{"softened_text": "weaker", "rationale": "toned down"}')
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=resp):
            result = soften_belief("strong claim", "- ant: evidence")
        assert result["softened_text"] == "weaker"


# --- repair_beliefs ---

class TestResearchBeliefs:
    def test_search_and_link(self):
        nodes = _make_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False,
            "comment": "missing antecedent",
        }]
        search_results = [
            {"id": "premise-a", "text": "Water boils at 100C at sea level",
             "truth_value": "IN", "source": "", "match": True},
        ]
        triage_resp = _mock_result('{"pattern": "search_and_link", "rationale": "gap"}')
        extract_resp = _mock_result("Water boils at 100C")
        match_resp = _mock_result(
            '{"matched_ids": ["premise-a"], "rationale": "match"}'
        )

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp, extract_resp, match_resp]), \
             patch("reasonsforge.api.add_justification"), \
             patch("reasonsforge.api.set_metadata") as mock_meta:
            results = repair_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=_mock_search(search_results),
            )

        assert len(results) == 1
        assert results[0]["pattern"] == "search_and_link"
        assert results[0]["status"] == "linked"
        assert results[0]["matched_premises"] == ["premise-a"]
        mock_meta.assert_called_once_with(
            "derived-boil", "repair_action", "search_and_link", db_path="test.db",
        )

    def test_soften(self):
        nodes = _make_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "overstated",
        }]
        triage_resp = _mock_result('{"pattern": "soften", "rationale": "too strong"}')
        soften_resp = _mock_result(
            '{"softened_text": "The water likely boiled", "rationale": "weakened"}'
        )

        mock_sup_result = {"old_id": "derived-boil", "new_id": "derived-boil-v2", "changed": []}
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp, soften_resp]), \
             patch("reasonsforge.api.supersede_with_text", return_value=mock_sup_result) as mock_sup, \
             patch("reasonsforge.api.set_metadata") as mock_meta:
            results = repair_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=_mock_search([]),
            )

        assert results[0]["pattern"] == "soften"
        assert results[0]["status"] == "softened"
        assert results[0]["softened_text"] == "The water likely boiled"
        mock_sup.assert_called_once_with(
            "derived-boil", "The water likely boiled", db_path="test.db",
        )
        mock_meta.assert_called_once_with(
            "derived-boil-v2", "repair_action", "softened", db_path="test.db",
        )

    def test_abandon(self):
        nodes = _make_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "broken chain",
        }]
        triage_resp = _mock_result('{"pattern": "abandon", "rationale": "too deep"}')

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp]), \
             patch("reasonsforge.api.retract_node") as mock_retract, \
             patch("reasonsforge.api.set_metadata") as mock_meta:
            results = repair_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=_mock_search([]),
            )

        assert results[0]["pattern"] == "abandon"
        assert results[0]["status"] == "abandoned"
        mock_retract.assert_called_once()
        assert "abandoned" in mock_retract.call_args[1]["reason"]
        mock_meta.assert_called_once_with(
            "derived-boil", "repair_action", "abandoned", db_path="test.db",
        )

    def test_research_pattern(self):
        nodes = _make_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "needs more evidence",
        }]
        triage_resp = _mock_result('{"pattern": "research", "rationale": "needs code review"}')

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp]), \
             patch("reasonsforge.api.set_metadata") as mock_meta:
            results = repair_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=_mock_search([]),
            )

        assert results[0]["pattern"] == "research"
        assert results[0]["status"] == "needs_research"
        assert mock_meta.call_count == 2
        mock_meta.assert_any_call(
            "derived-boil", "repair_research", "needs code review",
            db_path="test.db",
        )
        mock_meta.assert_any_call(
            "derived-boil", "repair_action", "research",
            db_path="test.db",
        )

    def test_research_dry_run_no_metadata(self):
        nodes = _make_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "needs more evidence",
        }]
        triage_resp = _mock_result('{"pattern": "research", "rationale": "needs investigation"}')

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp]), \
             patch("reasonsforge.api.set_metadata") as mock_meta:
            results = repair_beliefs(
                review_results, nodes, db_path="test.db", dry_run=True,
                search_fn=_mock_search([]),
            )

        assert results[0]["pattern"] == "research"
        assert results[0]["status"] == "needs_research"
        mock_meta.assert_not_called()

    def test_triage_failed(self):
        nodes = _make_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "unclear",
        }]
        triage_resp = _mock_result("I cannot decide")

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp]):
            results = repair_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=_mock_search([]),
            )

        assert results[0]["status"] == "triage_failed"

    def test_soften_failed(self):
        nodes = _make_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "overstated",
        }]
        triage_resp = _mock_result('{"pattern": "soften", "rationale": "x"}')
        soften_resp = _mock_result('{"softened_text": "", "rationale": "cant"}')

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp, soften_resp]):
            results = repair_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=_mock_search([]),
            )

        assert results[0]["status"] == "soften_failed"

    def test_dry_run_no_mutations(self):
        nodes = _make_nodes()
        review_results = [
            {"id": "derived-boil", "valid": False, "comment": "overstated"},
        ]
        triage_resp = _mock_result('{"pattern": "soften", "rationale": "x"}')
        soften_resp = _mock_result(
            '{"softened_text": "weaker", "rationale": "y"}'
        )

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp, soften_resp]), \
             patch("reasonsforge.api.update_node") as mock_update, \
             patch("reasonsforge.api.retract_node") as mock_retract, \
             patch("reasonsforge.api.add_justification") as mock_add, \
             patch("reasonsforge.api.set_metadata") as mock_meta:
            results = repair_beliefs(
                review_results, nodes, db_path="test.db", dry_run=True,
                search_fn=_mock_search([]),
            )

        assert results[0]["status"] == "softened"
        mock_update.assert_not_called()
        mock_retract.assert_not_called()
        mock_add.assert_not_called()
        mock_meta.assert_not_called()

    def test_multiple_beliefs_different_patterns(self):
        nodes = _make_nodes()
        nodes["premise-d"] = {
            "text": "Extra evidence", "truth_value": "IN", "justifications": [],
        }
        nodes["derived-steam"] = {
            "text": "Steam was produced for sure",
            "truth_value": "IN",
            "justifications": [{
                "type": "SL",
                "antecedents": ["premise-c"],
                "outlist": [],
                "label": "",
            }],
        }
        review_results = [
            {"id": "derived-boil", "valid": False, "comment": "overstated"},
            {"id": "derived-steam", "valid": False, "comment": "broken"},
        ]
        triage1 = _mock_result('{"pattern": "soften", "rationale": "x"}')
        soften1 = _mock_result('{"softened_text": "weaker", "rationale": "y"}')
        triage2 = _mock_result('{"pattern": "abandon", "rationale": "z"}')

        mock_sup_result = {"old_id": "derived-boil", "new_id": "derived-boil-v2", "changed": []}
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage1, soften1, triage2]), \
             patch("reasonsforge.api.supersede_with_text", return_value=mock_sup_result), \
             patch("reasonsforge.api.retract_node"), \
             patch("reasonsforge.api.set_metadata"):
            results = repair_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=_mock_search([]),
            )

        assert len(results) == 2
        assert results[0]["status"] == "softened"
        assert results[1]["status"] == "abandoned"

    def test_belief_not_found(self):
        nodes = _make_nodes()
        review_results = [{
            "id": "nonexistent", "valid": False, "comment": "x",
        }]

        results = repair_beliefs(
            review_results, nodes, db_path="test.db",
            search_fn=_mock_search([]),
        )

        assert results[0]["status"] == "error"
        assert "not found" in results[0]["error"]

    def test_search_and_link_extraction_failed(self):
        """When extract returns empty, status should be extraction_failed."""
        nodes = _make_nodes()
        review_results = [{
            "id": "derived-boil", "valid": False, "comment": "smuggled",
        }]
        triage_resp = _mock_result('{"pattern": "search_and_link", "rationale": "gap"}')
        extract_resp = _mock_result("   ")

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp, extract_resp]):
            results = repair_beliefs(
                review_results, nodes, db_path="test.db",
                search_fn=_mock_search([]),
            )

        assert results[0]["status"] == "extraction_failed"
        assert results[0]["pattern"] == "search_and_link"

    def test_skips_valid_beliefs(self):
        nodes = _make_nodes()
        review_results = [
            {"id": "derived-boil", "valid": True, "comment": "fine"},
        ]

        results = repair_beliefs(
            review_results, nodes, db_path="test.db",
            search_fn=_mock_search([]),
        )

        assert len(results) == 0


# --- API wrapper ---

class TestApiResearch:
    def test_from_belief_ids(self, tmp_path):
        """Test the belief_ids input path through api.research."""
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A is true", db_path=db)
        api.add_node("derived-a", "A extended", sl="premise-a", db_path=db)

        review_resp = _mock_result(json.dumps([{
            "id": "derived-a", "valid": False, "sufficient": True,
            "necessary": True, "unnecessary_antecedents": [],
            "comment": "overstated",
        }]))
        triage_resp = _mock_result('{"pattern": "abandon", "rationale": "broken"}')

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[review_resp, triage_resp]):
            result = api.research(
                belief_ids=["derived-a"],
                model="claude",
                dry_run=True,
                db_path=db,
            )

        assert result["total_invalid"] == 1
        assert result["abandoned"] == 1
        assert len(result["results"]) == 1

    def test_counts_add_up(self, tmp_path):
        """Verify total_invalid == linked + softened + abandoned + failed + errors."""
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A", db_path=db)
        api.add_node("derived-a", "A ext", sl="premise-a", db_path=db)

        review_file = tmp_path / "review.json"
        review_file.write_text(json.dumps({
            "results": [{
                "id": "derived-a", "valid": False, "comment": "smuggled",
            }],
        }))

        triage_resp = _mock_result(
            '{"pattern": "search_and_link", "rationale": "gap"}'
        )
        extract_resp = _mock_result("   ")

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp, extract_resp]):
            result = api.research(
                review_file=str(review_file),
                model="claude",
                dry_run=True,
                db_path=db,
            )

        total = (result["linked"] + result["softened"] + result["abandoned"]
                 + result["failed"] + result["errors"])
        assert total == result["total_invalid"]
        assert result["failed"] == 1

    def test_no_invalid(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A", db_path=db)
        api.add_node("derived-a", "A ext", sl="premise-a", db_path=db)

        review_resp = _mock_result(json.dumps([{
            "id": "derived-a", "valid": True, "sufficient": True,
            "necessary": True, "unnecessary_antecedents": [],
            "comment": "fine",
        }]))

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[review_resp]):
            result = api.research(
                belief_ids=["derived-a"],
                model="claude",
                db_path=db,
            )

        assert result["total_invalid"] == 0
        assert result["results"] == []


# --- CLI ---

class TestCmdResearch:
    def test_help(self):
        stdout, stderr, code = run_cli("research", "--help")
        assert code == 0
        assert "research" in stdout.lower()

    def test_no_args_error(self):
        stdout, stderr, code = run_cli("research", db_path="test.db")
        assert code == 1

    def test_from_review_file(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A is true", db_path=db)
        api.add_node("derived-a", "A extended", sl="premise-a", db_path=db)

        review_file = tmp_path / "review.json"
        review_file.write_text(json.dumps({
            "results": [{
                "id": "derived-a",
                "valid": False,
                "comment": "overstated",
            }],
        }))

        triage_resp = _mock_result('{"pattern": "soften", "rationale": "x"}')
        soften_resp = _mock_result(
            '{"softened_text": "A weakly extended", "rationale": "weakened"}'
        )

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp, soften_resp]):
            stdout, stderr, code = run_cli(
                "research",
                "--review-file", str(review_file),
                db_path=db,
            )

        assert code == 0
        assert "SOFTENED" in stdout

    def test_dry_run(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A is true", db_path=db)
        api.add_node("derived-a", "A extended", sl="premise-a", db_path=db)

        review_file = tmp_path / "review.json"
        review_file.write_text(json.dumps({
            "results": [{
                "id": "derived-a",
                "valid": False,
                "comment": "broken",
            }],
        }))

        triage_resp = _mock_result('{"pattern": "abandon", "rationale": "x"}')

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp]):
            stdout, stderr, code = run_cli(
                "research",
                "--review-file", str(review_file),
                "--dry-run",
                db_path=db,
            )

        assert code == 0
        assert "ABANDONED" in stdout
        assert "dry run" in stdout.lower()

        node = api.show_node("derived-a", db_path=db)
        assert node["truth_value"] == "IN"


class TestBackwardCompatAlias:
    def test_research_beliefs_alias(self):
        assert research_beliefs is repair_beliefs

    def test_api_research_alias(self):
        assert api.research is api.repair


class TestCmdRepair:
    def test_help(self):
        stdout, stderr, code = run_cli("repair", "--help")
        assert code == 0
        assert "repair" in stdout.lower()

    def test_from_review_file(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("premise-a", "A is true", db_path=db)
        api.add_node("derived-a", "A extended", sl="premise-a", db_path=db)

        review_file = tmp_path / "review.json"
        review_file.write_text(json.dumps({
            "results": [{
                "id": "derived-a",
                "valid": False,
                "comment": "overstated",
            }],
        }))

        triage_resp = _mock_result('{"pattern": "soften", "rationale": "x"}')
        soften_resp = _mock_result(
            '{"softened_text": "A weakly extended", "rationale": "weakened"}'
        )

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run",
                   side_effect=[triage_resp, soften_resp]):
            stdout, stderr, code = run_cli(
                "repair",
                "--review-file", str(review_file),
                db_path=db,
            )

        assert code == 0
        assert "SOFTENED" in stdout

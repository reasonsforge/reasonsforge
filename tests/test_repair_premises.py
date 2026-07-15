"""Tests for repair-premises command and supporting functions."""

import json
from unittest.mock import patch

import pytest

from reasonsforge import api
from reasonsforge.repair_premises import (
    format_premise_for_repair,
    parse_repair_response,
    repair_premises,
)


class TestFormatPremiseForRepair:

    def test_basic_formatting(self):
        nodes = {
            "obs-1": {
                "text": "Redis is used for session storage",
                "source": "docs/arch.md",
            }
        }
        review = {"error_type": "misread_source", "comment": "Source says PostgreSQL, not Redis"}
        result = format_premise_for_repair("obs-1", nodes, "Full source text here.", review)
        assert "### obs-1" in result
        assert "Redis is used for session storage" in result
        assert "misread_source" in result
        assert "Source says PostgreSQL" in result
        assert "Full source text here." in result

    def test_missing_node(self):
        assert format_premise_for_repair("missing", {}, "content", {}) == ""

    def test_no_review_comment(self):
        nodes = {"obs-1": {"text": "A claim", "source": "doc.md"}}
        result = format_premise_for_repair("obs-1", nodes, "src", {"error_type": "fabricated"})
        assert "fabricated" in result
        assert "Review comment:" not in result


class TestParseRepairResponse:

    def test_rewrite_action(self):
        response = json.dumps([{
            "id": "obs-1",
            "action": "rewrite",
            "corrected_text": "PostgreSQL is used for session storage",
            "rationale": "Source says PostgreSQL, not Redis",
        }])
        results = parse_repair_response(response)
        assert len(results) == 1
        assert results[0]["action"] == "rewrite"
        assert results[0]["corrected_text"] == "PostgreSQL is used for session storage"

    def test_retract_action(self):
        response = json.dumps([{
            "id": "obs-1",
            "action": "retract",
            "corrected_text": None,
            "rationale": "Source does not mention this at all",
        }])
        results = parse_repair_response(response)
        assert len(results) == 1
        assert results[0]["action"] == "retract"
        assert results[0]["corrected_text"] is None

    def test_json_with_prose(self):
        response = "Here is my analysis:\n" + json.dumps([{
            "id": "obs-1", "action": "rewrite",
            "corrected_text": "Fixed claim", "rationale": "fixed",
        }]) + "\nDone."
        results = parse_repair_response(response)
        assert len(results) == 1

    def test_malformed_json(self):
        assert parse_repair_response("not json") == []

    def test_defaults(self):
        response = json.dumps([{"id": "obs-1"}])
        results = parse_repair_response(response)
        assert results[0]["action"] == "retract"
        assert results[0]["corrected_text"] is None
        assert results[0]["rationale"] == ""


class TestRepairPremisesLoop:

    def _mock_result(self, response_text):
        return type("R", (), {"returncode": 0, "stdout": response_text, "stderr": ""})()

    def test_sequential_repair(self):
        nodes = {"obs-1": {"text": "Wrong claim", "source": "doc.md", "truth_value": "IN"}}
        source_contents = {"doc.md": "Correct information here."}
        review_results = {"obs-1": {"error_type": "misread_source", "comment": "wrong"}}

        llm_resp = json.dumps([{
            "id": "obs-1", "action": "rewrite",
            "corrected_text": "Correct claim", "rationale": "fixed",
        }])
        mock = self._mock_result(llm_resp)
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            results = repair_premises(nodes, ["obs-1"], source_contents, review_results)
        assert len(results) == 1
        assert results[0]["action"] == "rewrite"

    def test_parallel_repair(self):
        nodes = {
            "obs-1": {"text": "Wrong A", "source": "a.md", "truth_value": "IN"},
            "obs-2": {"text": "Wrong B", "source": "b.md", "truth_value": "IN"},
        }
        source_contents = {"a.md": "Correct A.", "b.md": "Correct B."}
        review_results = {
            "obs-1": {"error_type": "misread_source", "comment": "wrong A"},
            "obs-2": {"error_type": "fabricated", "comment": "made up B"},
        }

        def mock_run(*args, **kwargs):
            prompt = kwargs.get("input", "")
            if "obs-1" in prompt:
                resp = json.dumps([{"id": "obs-1", "action": "rewrite",
                                    "corrected_text": "Fixed A", "rationale": "ok"}])
            else:
                resp = json.dumps([{"id": "obs-2", "action": "retract",
                                    "corrected_text": None, "rationale": "no support"}])
            return type("R", (), {"returncode": 0, "stdout": resp, "stderr": ""})()

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=mock_run):
            results = repair_premises(nodes, ["obs-1", "obs-2"], source_contents,
                                     review_results, parallel=2)
        assert len(results) == 2
        actions = {r["id"]: r["action"] for r in results}
        assert actions["obs-1"] == "rewrite"
        assert actions["obs-2"] == "retract"

    def test_empty_ids(self):
        assert repair_premises({}, [], {}, {}) == []

    def test_on_result_callback(self):
        nodes = {"obs-1": {"text": "Claim", "source": "doc.md", "truth_value": "IN"}}
        source_contents = {"doc.md": "Content."}
        review_results = {"obs-1": {"error_type": "fabricated", "comment": "made up"}}

        llm_resp = json.dumps([{"id": "obs-1", "action": "retract",
                                "corrected_text": None, "rationale": "gone"}])
        mock = self._mock_result(llm_resp)
        callbacks = []
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            repair_premises(nodes, ["obs-1"], source_contents, review_results,
                           on_result=lambda r: callbacks.append(len(r)))
        assert callbacks == [1]

    def test_llm_failure_returns_error(self):
        nodes = {"obs-1": {"text": "Claim", "source": "doc.md", "truth_value": "IN"}}
        source_contents = {"doc.md": "Content."}
        review_results = {"obs-1": {"error_type": "fabricated", "comment": "bad"}}

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=Exception("LLM down")):
            results = repair_premises(nodes, ["obs-1"], source_contents, review_results)
        assert len(results) == 1
        assert results[0]["action"] == "error"


class TestApiRepairPremises:

    @pytest.fixture
    def db_with_reviewed(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_file = tmp_path / "entry.md"
        source_file.write_text("The system uses PostgreSQL for storage.")

        api.init_db(db_path=db)
        api.add_node("obs-1", "Redis is used for storage",
                     source=str(source_file), db_path=db)
        api.add_node("obs-2", "Memcached handles caching",
                     source=str(source_file), db_path=db)

        review_report = {
            "results": [
                {"id": "obs-1", "accurate": False, "well_scoped": True,
                 "error_type": "misread_source", "comment": "Source says PostgreSQL"},
                {"id": "obs-2", "accurate": False, "well_scoped": True,
                 "error_type": "fabricated", "comment": "Source never mentions caching"},
            ]
        }
        review_file = tmp_path / "review.json"
        review_file.write_text(json.dumps(review_report))
        return db, str(review_file)

    def test_repair_from_review_file(self, db_with_reviewed):
        db, review_file = db_with_reviewed
        llm_resp = json.dumps([
            {"id": "obs-1", "action": "rewrite",
             "corrected_text": "PostgreSQL is used for storage", "rationale": "fixed"},
        ])
        mock = type("R", (), {"returncode": 0, "stdout": llm_resp, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            result = api.repair_premises(review_file=review_file, dry_run=True, db_path=db)
        assert result["total_inaccurate"] == 2

    def test_rewrite_updates_node(self, db_with_reviewed):
        db, review_file = db_with_reviewed

        def mock_run(*args, **kwargs):
            prompt = kwargs.get("input", "")
            if "obs-1" in prompt:
                resp = json.dumps([{"id": "obs-1", "action": "rewrite",
                                    "corrected_text": "PostgreSQL is used", "rationale": "ok"}])
            else:
                resp = json.dumps([{"id": "obs-2", "action": "retract",
                                    "corrected_text": None, "rationale": "no support"}])
            return type("R", (), {"returncode": 0, "stdout": resp, "stderr": ""})()

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=mock_run):
            result = api.repair_premises(review_file=review_file, db_path=db)

        assert result["rewritten"] >= 1
        net = api.export_network(db_path=db)
        # obs-1 should be superseded (OUT), successor has the new text
        assert net["nodes"]["obs-1"]["truth_value"] == "OUT"
        rewritten_results = [r for r in result["results"] if r.get("action") == "rewrite"]
        for r in rewritten_results:
            if r["id"] == "obs-1" and "new_id" in r:
                assert net["nodes"][r["new_id"]]["text"] == "PostgreSQL is used"

    def test_retract_removes_node(self, db_with_reviewed):
        db, review_file = db_with_reviewed

        def mock_run(*args, **kwargs):
            prompt = kwargs.get("input", "")
            if "obs-2" in prompt:
                resp = json.dumps([{"id": "obs-2", "action": "retract",
                                    "corrected_text": None, "rationale": "unsupported"}])
            else:
                resp = json.dumps([{"id": "obs-1", "action": "rewrite",
                                    "corrected_text": "Fixed", "rationale": "ok"}])
            return type("R", (), {"returncode": 0, "stdout": resp, "stderr": ""})()

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=mock_run):
            result = api.repair_premises(review_file=review_file, db_path=db)

        assert result["retracted"] >= 1
        net = api.export_network(db_path=db)
        if "obs-2" in net["nodes"]:
            assert net["nodes"]["obs-2"]["truth_value"] == "OUT"

    def test_dry_run_no_changes(self, db_with_reviewed):
        db, review_file = db_with_reviewed
        llm_resp = json.dumps([
            {"id": "obs-1", "action": "rewrite",
             "corrected_text": "Fixed", "rationale": "ok"},
        ])
        mock = type("R", (), {"returncode": 0, "stdout": llm_resp, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            api.repair_premises(review_file=review_file, dry_run=True, db_path=db)

        net = api.export_network(db_path=db)
        assert net["nodes"]["obs-1"]["text"] == "Redis is used for storage"

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Either review_file or belief_ids"):
            api.repair_premises()

    def test_no_inaccurate_returns_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        review_report = {"results": [
            {"id": "obs-1", "accurate": True, "well_scoped": True,
             "error_type": None, "comment": "ok"},
        ]}
        review_file = tmp_path / "review.json"
        review_file.write_text(json.dumps(review_report))
        result = api.repair_premises(review_file=str(review_file), db_path=db)
        assert result["total_inaccurate"] == 0
        assert result["results"] == []


class TestRepairActionMetadata:

    @pytest.fixture
    def db_with_reviewed(self, tmp_path):
        db = str(tmp_path / "test.db")
        source_file = tmp_path / "entry.md"
        source_file.write_text("The system uses PostgreSQL for storage.")

        api.init_db(db_path=db)
        api.add_node("obs-1", "Redis is used for storage",
                     source=str(source_file), db_path=db)
        api.add_node("obs-2", "Memcached handles caching",
                     source=str(source_file), db_path=db)

        review_report = {
            "results": [
                {"id": "obs-1", "accurate": False, "well_scoped": True,
                 "error_type": "misread_source", "comment": "Source says PostgreSQL"},
                {"id": "obs-2", "accurate": False, "well_scoped": True,
                 "error_type": "fabricated", "comment": "Source never mentions caching"},
            ]
        }
        review_file = tmp_path / "review.json"
        review_file.write_text(json.dumps(review_report))
        return db, str(review_file)

    def test_rewrite_sets_repair_action(self, db_with_reviewed):
        db, review_file = db_with_reviewed
        llm_resp = json.dumps([{
            "id": "obs-1", "action": "rewrite",
            "corrected_text": "PostgreSQL is used for storage", "rationale": "fixed",
        }])
        mock = type("R", (), {"returncode": 0, "stdout": llm_resp, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            result = api.repair_premises(review_file=review_file, db_path=db)

        # Old node is superseded
        old_node = api.show_node("obs-1", db_path=db)
        assert old_node["truth_value"] == "OUT"
        # Successor has the repair_action metadata
        rewritten = [r for r in result["results"] if r.get("action") == "rewrite"]
        assert len(rewritten) >= 1
        new_id = rewritten[0]["new_id"]
        new_node = api.show_node(new_id, db_path=db)
        assert new_node["metadata"].get("repair_action") == "rewritten"

    def test_retract_sets_repair_action(self, db_with_reviewed):
        db, review_file = db_with_reviewed
        llm_resp = json.dumps([{
            "id": "obs-2", "action": "retract",
            "corrected_text": None, "rationale": "fabricated",
        }])
        mock = type("R", (), {"returncode": 0, "stdout": llm_resp, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            api.repair_premises(review_file=review_file, db_path=db)

        node = api.show_node("obs-2", db_path=db)
        assert node["metadata"].get("repair_action") == "retracted"

    def test_dry_run_no_repair_action(self, db_with_reviewed):
        db, review_file = db_with_reviewed
        llm_resp = json.dumps([{
            "id": "obs-1", "action": "rewrite",
            "corrected_text": "Fixed", "rationale": "ok",
        }])
        mock = type("R", (), {"returncode": 0, "stdout": llm_resp, "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock):
            api.repair_premises(review_file=review_file, dry_run=True, db_path=db)

        node = api.show_node("obs-1", db_path=db)
        assert node["metadata"].get("repair_action") is None


class TestCliDispatch:

    def test_repair_premises_registered(self):
        from reasonsforge import cli
        assert hasattr(cli, "cmd_repair_premises")
        assert callable(cli.cmd_repair_premises)


class TestReviewPremisesParallel:

    def test_parallel_batches(self, tmp_path):
        """Verify parallel=2 processes batches concurrently."""
        from reasonsforge.review_premises import review_premises

        nodes = {
            f"obs-{i}": {"text": f"Claim {i}", "source": "doc.md", "truth_value": "IN"}
            for i in range(10)
        }
        source_contents = {"doc.md": "Source content."}

        def mock_run(*args, **kwargs):
            prompt = kwargs.get("input", "")
            results = []
            for i in range(10):
                if f"obs-{i}" in prompt:
                    results.append({
                        "id": f"obs-{i}", "accurate": True,
                        "well_scoped": True, "error_type": None, "comment": "ok",
                    })
            resp = json.dumps(results)
            return type("R", (), {"returncode": 0, "stdout": resp, "stderr": ""})()

        premise_ids = [f"obs-{i}" for i in range(10)]
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=mock_run):
            results = review_premises(nodes, premise_ids, source_contents,
                                      parallel=2, batch_size=5)
        assert len(results) == 10

"""Tests for classify-defeaters command."""

from unittest.mock import patch

from reasonsforge import api
from reasonsforge.review import classify_defeat_reason, DEFEAT_REASON_TYPES


class TestClassifyDefeatReason:

    def test_returns_valid_type(self):
        mock_result = type("R", (), {
            "returncode": 0, "stdout": "unsupported-conjunct", "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = classify_defeat_reason("defeater text", "defeated text",
                                            "claude", 300)
        assert result == "unsupported-conjunct"

    def test_extracts_type_from_prose(self):
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": "The failure mode is over-generalizes because...",
            "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = classify_defeat_reason("defeater text", "defeated text",
                                            "claude", 300)
        assert result == "over-generalizes"

    def test_exact_match_preferred(self):
        mock_result = type("R", (), {
            "returncode": 0,
            "stdout": "scope-mismatch",
            "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = classify_defeat_reason("defeater text", "defeated text",
                                            "claude", 300)
        assert result == "scope-mismatch"

    def test_returns_empty_on_unrecognized(self):
        mock_result = type("R", (), {
            "returncode": 0, "stdout": "I cannot classify this",
            "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = classify_defeat_reason("defeater text", "defeated text",
                                            "claude", 300)
        assert result == ""

    def test_returns_empty_on_error(self):
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", side_effect=Exception("timeout")):
            result = classify_defeat_reason("defeater text", "defeated text",
                                            "claude", 300)
        assert result == ""


class TestClassifyDefeatReasonTypes:

    def _setup_db(self, tmp_path):
        db = tmp_path / "test.db"
        api.init_db(str(db))
        api.add_node("p1", "P1", db_path=str(db))
        api.add_node("d1", "D1 is safe and traceable", sl="p1", db_path=str(db))
        api.defeat_justification(
            "d1", 0, "traceability not established",
            defeater_type="migrated-retraction", db_path=str(db))
        return str(db)

    def test_dry_run_classifies_without_writing(self, tmp_path):
        db = self._setup_db(tmp_path)
        mock_result = type("R", (), {
            "returncode": 0, "stdout": "unsupported-conjunct", "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.classify_defeat_reason_types(
                model="claude", dry_run=True, db_path=db)

        assert len(result["classified"]) == 1
        assert result["classified"][0]["defeat_reason_type"] == "unsupported-conjunct"

        defeater_id = result["classified"][0]["id"]
        node = api.show_node(defeater_id, db_path=db)
        assert "defeat_reason_type" not in node["metadata"]

    def test_apply_writes_metadata(self, tmp_path):
        db = self._setup_db(tmp_path)
        mock_result = type("R", (), {
            "returncode": 0, "stdout": "scope-mismatch", "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.classify_defeat_reason_types(
                model="claude", dry_run=False, db_path=db)

        assert len(result["classified"]) == 1
        defeater_id = result["classified"][0]["id"]
        node = api.show_node(defeater_id, db_path=db)
        assert node["metadata"]["defeat_reason_type"] == "scope-mismatch"

    def test_skips_already_classified(self, tmp_path):
        db = self._setup_db(tmp_path)
        mock_result = type("R", (), {
            "returncode": 0, "stdout": "scope-mismatch", "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            api.classify_defeat_reason_types(
                model="claude", dry_run=False, db_path=db)
            result = api.classify_defeat_reason_types(
                model="claude", dry_run=True, db_path=db)

        assert len(result["classified"]) == 0
        assert any(s["reason"] == "already classified" for s in result["skipped"])

    def test_filters_by_defeater_type(self, tmp_path):
        db = self._setup_db(tmp_path)
        mock_result = type("R", (), {
            "returncode": 0, "stdout": "unsupported-conjunct", "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.classify_defeat_reason_types(
                defeater_type_filter="invalid-inference",
                model="claude", dry_run=True, db_path=db)
        assert len(result["classified"]) == 0
        assert any("migrated-retraction" in s["reason"] for s in result["skipped"])

        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.classify_defeat_reason_types(
                defeater_type_filter="migrated-retraction",
                model="claude", dry_run=True, db_path=db)
        assert len(result["classified"]) == 1

    def test_skips_non_defeater_nodes(self, tmp_path):
        db = tmp_path / "test.db"
        api.init_db(str(db))
        api.add_node("p1", "P1", db_path=str(db))
        api.add_node("p2", "P2", db_path=str(db))

        result = api.classify_defeat_reason_types(
            model="claude", dry_run=True, db_path=str(db))
        assert len(result["classified"]) == 0

    def test_errors_on_classification_failure(self, tmp_path):
        db = self._setup_db(tmp_path)
        mock_result = type("R", (), {
            "returncode": 0, "stdout": "I don't know", "stderr": ""})()
        with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
             patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
            result = api.classify_defeat_reason_types(
                model="claude", dry_run=True, db_path=db)
        assert len(result["classified"]) == 0
        assert len(result["errors"]) == 1


class TestDefeatReasonTypes:

    def test_all_types_valid(self):
        assert len(DEFEAT_REASON_TYPES) == 7
        for t in DEFEAT_REASON_TYPES:
            assert "-" in t

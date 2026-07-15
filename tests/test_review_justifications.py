"""Tests for review-justifications command."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from reasonsforge import api
from reasonsforge.review_justifications import (
    parse_justification_review,
    review_justifications,
    _has_multi_antecedent_sl,
)


@pytest.fixture
def network_db():
    path = os.path.join(tempfile.mkdtemp(), "test.db")
    api.add_node("fact-a", "Observable fact A", db_path=path)
    api.add_node("fact-b", "Observable fact B", db_path=path)
    api.add_node("fact-c", "Observable fact C", db_path=path)
    api.add_node("derived-1", "Derived from A and B",
                 sl="fact-a,fact-b", label="test", db_path=path)
    api.add_node("derived-2", "Derived from A, B, and C",
                 sl="fact-a,fact-b,fact-c", label="test", db_path=path)
    api.add_node("single-ant", "Derived from A only",
                 sl="fact-a", label="test", db_path=path)
    return path


class TestParseJustificationReview:

    def test_parse_valid_response(self):
        response = json.dumps([
            {
                "id": "derived-1",
                "classification": "ANY",
                "required_antecedents": [],
                "independent_antecedents": ["fact-a", "fact-b"],
                "comment": "each independently supports"
            }
        ])
        results = parse_justification_review(response)
        assert len(results) == 1
        assert results[0]["classification"] == "ANY"
        assert results[0]["independent_antecedents"] == ["fact-a", "fact-b"]

    def test_parse_with_surrounding_prose(self):
        response = "Here are my findings:\n" + json.dumps([
            {"id": "x", "classification": "ALL", "comment": "needs both"}
        ]) + "\nThat's my analysis."
        results = parse_justification_review(response)
        assert len(results) == 1
        assert results[0]["classification"] == "ALL"

    def test_parse_defaults_classification_to_all(self):
        response = json.dumps([{"id": "x"}])
        results = parse_justification_review(response)
        assert results[0]["classification"] == "ALL"
        assert results[0]["required_antecedents"] == []
        assert results[0]["independent_antecedents"] == []
        assert results[0]["comment"] == ""

    def test_parse_mixed(self):
        response = json.dumps([{
            "id": "m1",
            "classification": "MIXED",
            "required_antecedents": ["a", "b"],
            "independent_antecedents": ["c"],
            "comment": "a and b are linked, c is independent"
        }])
        results = parse_justification_review(response)
        assert results[0]["classification"] == "MIXED"
        assert results[0]["required_antecedents"] == ["a", "b"]
        assert results[0]["independent_antecedents"] == ["c"]

    def test_parse_skips_items_without_id(self):
        response = json.dumps([
            {"classification": "ANY"},
            {"id": "good", "classification": "ANY"}
        ])
        results = parse_justification_review(response)
        assert len(results) == 1
        assert results[0]["id"] == "good"

    def test_parse_empty_response(self):
        assert parse_justification_review("no json here") == []

    def test_parse_normalizes_case(self):
        response = json.dumps([{"id": "x", "classification": "any"}])
        results = parse_justification_review(response)
        assert results[0]["classification"] == "ANY"

    def test_parse_null_fields(self):
        response = json.dumps([{
            "id": "x",
            "classification": None,
            "required_antecedents": None,
            "independent_antecedents": None,
            "comment": None,
        }])
        results = parse_justification_review(response)
        assert results[0]["classification"] == "ALL"
        assert results[0]["required_antecedents"] == []
        assert results[0]["independent_antecedents"] == []
        assert results[0]["comment"] == ""


class TestHasMultiAntecedentSl:

    def test_multi_antecedent(self):
        node = {"justifications": [
            {"type": "SL", "antecedents": ["a", "b"]}
        ]}
        assert _has_multi_antecedent_sl(node, 2)

    def test_single_antecedent(self):
        node = {"justifications": [
            {"type": "SL", "antecedents": ["a"]}
        ]}
        assert not _has_multi_antecedent_sl(node, 2)

    def test_no_justifications(self):
        assert not _has_multi_antecedent_sl({}, 2)

    def test_min_antecedents_3(self):
        node = {"justifications": [
            {"type": "SL", "antecedents": ["a", "b"]}
        ]}
        assert not _has_multi_antecedent_sl(node, 3)
        node2 = {"justifications": [
            {"type": "SL", "antecedents": ["a", "b", "c"]}
        ]}
        assert _has_multi_antecedent_sl(node2, 3)


class TestReviewJustifications:

    def test_filters_to_multi_antecedent(self):
        nodes = {
            "single": {
                "truth_value": "IN",
                "justifications": [{"type": "SL", "antecedents": ["a"]}],
                "text": "single",
            },
            "multi": {
                "truth_value": "IN",
                "justifications": [{"type": "SL", "antecedents": ["a", "b"]}],
                "text": "multi",
            },
            "premise": {
                "truth_value": "IN",
                "justifications": [],
                "text": "premise",
            },
            "a": {"truth_value": "IN", "text": "A", "justifications": []},
            "b": {"truth_value": "IN", "text": "B", "justifications": []},
        }
        mock_response = json.dumps([
            {"id": "multi", "classification": "ANY", "comment": "independent"}
        ])
        with patch("reasonsforge.review_justifications.invoke_model",
                   return_value=mock_response):
            results = review_justifications(nodes)
        assert len(results) == 1
        assert results[0]["id"] == "multi"

    def test_parallel(self):
        nodes = {
            "d1": {
                "truth_value": "IN",
                "justifications": [{"type": "SL", "antecedents": ["a", "b"]}],
                "text": "derived 1",
            },
            "d2": {
                "truth_value": "IN",
                "justifications": [{"type": "SL", "antecedents": ["a", "c"]}],
                "text": "derived 2",
            },
            "a": {"truth_value": "IN", "text": "A", "justifications": []},
            "b": {"truth_value": "IN", "text": "B", "justifications": []},
            "c": {"truth_value": "IN", "text": "C", "justifications": []},
        }

        def mock_invoke(prompt, model=None, timeout=None):
            if "derived 1" in prompt:
                return json.dumps([{"id": "d1", "classification": "ANY"}])
            return json.dumps([{"id": "d2", "classification": "ALL"}])

        with patch("reasonsforge.review_justifications.invoke_model",
                   side_effect=mock_invoke):
            results = review_justifications(nodes, batch_size=1, parallel=2)
        ids = {r["id"] for r in results}
        assert ids == {"d1", "d2"}
        assert len(results) == 2

    def test_respects_min_antecedents(self):
        nodes = {
            "two-ant": {
                "truth_value": "IN",
                "justifications": [{"type": "SL", "antecedents": ["a", "b"]}],
                "text": "two",
            },
            "three-ant": {
                "truth_value": "IN",
                "justifications": [{"type": "SL", "antecedents": ["a", "b", "c"]}],
                "text": "three",
            },
            "a": {"truth_value": "IN", "text": "A", "justifications": []},
            "b": {"truth_value": "IN", "text": "B", "justifications": []},
            "c": {"truth_value": "IN", "text": "C", "justifications": []},
        }
        mock_response = json.dumps([
            {"id": "three-ant", "classification": "ANY"}
        ])
        with patch("reasonsforge.review_justifications.invoke_model",
                   return_value=mock_response):
            results = review_justifications(nodes, min_antecedents=3)
        assert len(results) == 1
        assert results[0]["id"] == "three-ant"


class TestApiReviewJustifications:

    def test_returns_summary(self, network_db):
        mock_response = json.dumps([
            {"id": "derived-1", "classification": "ANY", "comment": "independent"},
            {"id": "derived-2", "classification": "ALL", "comment": "requires all"},
        ])
        with patch("reasonsforge.review_justifications.invoke_model",
                   return_value=mock_response):
            result = api.review_justifications(db_path=network_db)

        assert result["reviewed"] == 2
        assert result["convert_any"] == 1
        assert result["keep_all"] == 1
        assert result["convert_mixed"] == 0

    def test_specific_node(self, network_db):
        mock_response = json.dumps([
            {"id": "derived-1", "classification": "ANY"},
        ])
        with patch("reasonsforge.review_justifications.invoke_model",
                   return_value=mock_response):
            result = api.review_justifications(
                belief_ids=["derived-1"], db_path=network_db
            )
        assert result["reviewed"] == 1

    def test_does_not_modify_db(self, network_db):
        mock_response = json.dumps([
            {"id": "derived-1", "classification": "ANY"},
        ])
        with patch("reasonsforge.review_justifications.invoke_model",
                   return_value=mock_response):
            api.review_justifications(db_path=network_db)

        node = api.show_node("derived-1", db_path=network_db)
        assert len(node["justifications"]) == 1
        assert len(node["justifications"][0]["antecedents"]) == 2

"""Tests for propose-update command."""

import json
from unittest.mock import patch

import pytest

from reasonsforge.propose_update import (
    format_belief_for_update,
    format_proposal_markdown,
    format_proposals_file,
    parse_update_proposals,
    propose_updates,
)


@pytest.fixture
def sample_nodes():
    return {
        "premise-a": {
            "text": "The sky is blue.",
            "truth_value": "IN",
            "source": "observations/sky.md",
            "source_url": "https://example.com/sky.md",
            "justifications": [],
            "dependents": ["derived-b"],
            "metadata": {},
        },
        "premise-stale": {
            "text": "The API uses REST.",
            "truth_value": "IN",
            "source": "docs/api.md",
            "source_url": "https://example.com/api.md",
            "justifications": [],
            "dependents": [],
            "metadata": {"stale_reason": "content_changed"},
        },
        "derived-b": {
            "text": "Blue skies indicate clear weather.",
            "truth_value": "IN",
            "source": "",
            "source_url": "",
            "justifications": [
                {
                    "antecedents": ["premise-a"],
                    "outlist": [],
                    "label": "weather inference",
                }
            ],
            "dependents": ["derived-c"],
            "metadata": {},
        },
        "derived-c": {
            "text": "Clear weather is good for flying.",
            "truth_value": "IN",
            "source": "",
            "source_url": "",
            "justifications": [
                {
                    "antecedents": ["derived-b"],
                    "outlist": ["bad-weather-warning"],
                    "label": "aviation safety",
                }
            ],
            "dependents": [],
            "metadata": {},
        },
        "bad-weather-warning": {
            "text": "Storm warning issued.",
            "truth_value": "OUT",
            "source": "",
            "source_url": "",
            "justifications": [],
            "dependents": ["derived-c"],
            "metadata": {},
        },
    }


class TestFormatBeliefForUpdate:
    def test_premise_with_source(self, sample_nodes):
        result = format_belief_for_update("premise-a", sample_nodes)
        assert "### premise-a" in result
        assert "Text: The sky is blue." in result
        assert "Source: observations/sky.md" in result
        assert "Source URL: https://example.com/sky.md" in result
        assert "Type: premise" in result
        assert "Dependents (1):" in result
        assert "derived-b" in result

    def test_derived_with_justification(self, sample_nodes):
        result = format_belief_for_update("derived-b", sample_nodes)
        assert "### derived-b" in result
        assert "Antecedents:" in result
        assert "premise-a: The sky is blue." in result
        assert "Label: weather inference" in result

    def test_derived_with_outlist(self, sample_nodes):
        result = format_belief_for_update("derived-c", sample_nodes)
        assert "Unless (must be OUT):" in result
        assert "bad-weather-warning" in result

    def test_stale_node(self, sample_nodes):
        result = format_belief_for_update("premise-stale", sample_nodes)
        assert "Stale reason: content_changed" in result

    def test_missing_node(self, sample_nodes):
        result = format_belief_for_update("nonexistent", sample_nodes)
        assert result == ""

    def test_no_dependents(self, sample_nodes):
        result = format_belief_for_update("derived-c", sample_nodes)
        assert "Dependents" not in result

    def test_multiple_justifications(self):
        nodes = {
            "multi-just": {
                "text": "Has two justifications.",
                "truth_value": "IN",
                "source": "",
                "source_url": "",
                "justifications": [
                    {"antecedents": ["a"], "outlist": [], "label": "first"},
                    {"antecedents": ["b"], "outlist": [], "label": "second"},
                ],
                "dependents": [],
                "metadata": {},
            },
            "a": {"text": "Ant A.", "truth_value": "IN", "justifications": [],
                   "source": "", "source_url": "", "dependents": [], "metadata": {}},
            "b": {"text": "Ant B.", "truth_value": "IN", "justifications": [],
                   "source": "", "source_url": "", "dependents": [], "metadata": {}},
        }
        result = format_belief_for_update("multi-just", nodes)
        assert "Justification 1/2:" in result
        assert "Justification 2/2:" in result
        assert "Label: first" in result
        assert "Label: second" in result


class TestParseUpdateProposals:
    def test_valid_json(self):
        response = """Here are my proposals:

```json
[
  {
    "id": "premise-a",
    "action": "update",
    "proposed_text": "The sky appears blue due to Rayleigh scattering.",
    "failure_mode": "smuggled-premise",
    "basis": "prior-knowledge",
    "evidence": "The original claim is simplistic.",
    "comment": "Adds mechanistic explanation."
  }
]
```"""
        results = parse_update_proposals(response)
        assert len(results) == 1
        assert results[0]["id"] == "premise-a"
        assert results[0]["action"] == "update"
        assert "Rayleigh" in results[0]["proposed_text"]
        assert results[0]["failure_mode"] == "smuggled-premise"
        assert results[0]["basis"] == "prior-knowledge"

    def test_retract_proposal(self):
        response = json.dumps([{
            "id": "bad-belief",
            "action": "retract",
            "proposed_text": None,
            "failure_mode": "contradicted-by-source",
            "basis": "source-divergence",
            "evidence": "Source changed.",
            "comment": "No longer supported.",
        }])
        results = parse_update_proposals(response)
        assert len(results) == 1
        assert results[0]["action"] == "retract"
        assert results[0]["proposed_text"] is None

    def test_multiple_proposals(self):
        response = json.dumps([
            {"id": "a", "action": "update", "proposed_text": "new a",
             "failure_mode": "stale", "basis": "source-divergence",
             "evidence": "", "comment": "stale"},
            {"id": "b", "action": "retract", "proposed_text": None,
             "failure_mode": "contradicted-by-source", "basis": "detected-contradiction",
             "evidence": "", "comment": "wrong"},
        ])
        results = parse_update_proposals(response)
        assert len(results) == 2
        assert results[0]["id"] == "a"
        assert results[1]["id"] == "b"

    def test_empty_array(self):
        results = parse_update_proposals("[]")
        assert results == []

    def test_malformed_json(self):
        results = parse_update_proposals("not json at all")
        assert results == []

    def test_invalid_basis_defaults(self):
        response = json.dumps([{
            "id": "x", "action": "update", "proposed_text": "new",
            "failure_mode": "stale", "basis": "invalid-basis",
            "evidence": "", "comment": "",
        }])
        results = parse_update_proposals(response)
        assert results[0]["basis"] == "prior-knowledge"

    def test_invalid_failure_mode_defaults(self):
        response = json.dumps([{
            "id": "x", "action": "update", "proposed_text": "new",
            "failure_mode": "made-up-mode", "basis": "source-divergence",
            "evidence": "", "comment": "",
        }])
        results = parse_update_proposals(response)
        assert results[0]["failure_mode"] == ""

    def test_invalid_action_defaults(self):
        response = json.dumps([{
            "id": "x", "action": "invalid",
            "proposed_text": "new", "failure_mode": "stale",
            "basis": "source-divergence", "evidence": "", "comment": "",
        }])
        results = parse_update_proposals(response)
        assert results[0]["action"] == "update"

    def test_prose_around_json(self):
        response = "I found issues with these beliefs:\n" + json.dumps([{
            "id": "x", "action": "retract", "proposed_text": None,
            "failure_mode": "stale", "basis": "source-divergence",
            "evidence": "", "comment": "stale",
        }]) + "\n\nThat's all I found."
        results = parse_update_proposals(response)
        assert len(results) == 1
        assert results[0]["id"] == "x"

    def test_skips_items_without_id(self):
        response = json.dumps([
            {"action": "update", "proposed_text": "missing id"},
            {"id": "x", "action": "update", "proposed_text": "has id",
             "failure_mode": "stale", "basis": "source-divergence",
             "evidence": "", "comment": ""},
        ])
        results = parse_update_proposals(response)
        assert len(results) == 1
        assert results[0]["id"] == "x"


class TestFormatProposalMarkdown:
    def test_update_proposal(self, sample_nodes):
        proposal = {
            "id": "premise-a",
            "action": "update",
            "proposed_text": "The sky appears blue.",
            "failure_mode": "smuggled-premise",
            "basis": "prior-knowledge",
            "evidence": "Simplistic claim.",
            "comment": "More precise.",
        }
        result = format_proposal_markdown(proposal, nodes=sample_nodes)
        assert "## UPDATE: premise-a" in result
        assert "**Current text:** The sky is blue." in result
        assert "**Proposed text:** The sky appears blue." in result
        assert "**Failure mode:** smuggled-premise" in result
        assert "**Basis:** prior-knowledge" in result
        assert "**Evidence:** Simplistic claim." in result
        assert "reasons update premise-a" in result

    def test_retract_proposal(self, sample_nodes):
        proposal = {
            "id": "premise-a",
            "action": "retract",
            "proposed_text": None,
            "failure_mode": "contradicted-by-source",
            "basis": "source-divergence",
            "evidence": "Source changed.",
            "comment": "No longer valid.",
        }
        result = format_proposal_markdown(proposal, nodes=sample_nodes)
        assert "## RETRACT: premise-a" in result
        assert "reasons retract premise-a" in result
        assert "Proposed text" not in result

    def test_with_cascade(self, sample_nodes):
        proposal = {
            "id": "premise-a",
            "action": "retract",
            "proposed_text": None,
            "failure_mode": "stale",
            "basis": "source-divergence",
            "evidence": "",
            "comment": "Stale.",
        }
        cascade = {
            "retracted": [
                {"id": "derived-b", "text": "Blue skies indicate clear weather.", "depth": 1, "dependents": 1},
            ],
            "restored": [],
            "total_affected": 1,
        }
        result = format_proposal_markdown(proposal, nodes=sample_nodes, cascade=cascade)
        assert "**Cascade impact:** 1 dependent(s) affected" in result
        assert 'derived-b: "Blue skies indicate clear weather." — will go OUT' in result

    def test_update_cascade_says_re_review(self, sample_nodes):
        proposal = {
            "id": "premise-a",
            "action": "update",
            "proposed_text": "Updated sky claim.",
            "failure_mode": "smuggled-premise",
            "basis": "prior-knowledge",
            "evidence": "",
            "comment": "",
        }
        cascade = {
            "retracted": [
                {"id": "derived-b", "text": "Blue skies indicate clear weather.", "depth": 1, "dependents": 1},
            ],
            "restored": [],
            "total_affected": 1,
        }
        result = format_proposal_markdown(proposal, nodes=sample_nodes, cascade=cascade)
        assert "need re-review" in result
        assert "will go OUT" not in result

    def test_no_cascade(self, sample_nodes):
        proposal = {
            "id": "premise-a",
            "action": "update",
            "proposed_text": "Updated.",
            "failure_mode": "stale",
            "basis": "source-divergence",
            "evidence": "",
            "comment": "",
        }
        result = format_proposal_markdown(proposal, nodes=sample_nodes)
        assert "**Cascade impact:** not computed" in result

    def test_empty_cascade(self, sample_nodes):
        proposal = {
            "id": "derived-c",
            "action": "update",
            "proposed_text": "Updated.",
            "failure_mode": "stale",
            "basis": "source-divergence",
            "evidence": "",
            "comment": "",
        }
        cascade = {"retracted": [], "restored": [], "total_affected": 0}
        result = format_proposal_markdown(proposal, nodes=sample_nodes, cascade=cascade)
        assert "**Cascade impact:** none" in result


class TestFormatProposalsFile:
    def test_no_proposals(self):
        result = format_proposals_file([], nodes={})
        assert "No updates proposed." in result

    def test_with_proposals(self, sample_nodes):
        proposals = [
            {
                "id": "premise-a",
                "action": "update",
                "proposed_text": "Updated.",
                "failure_mode": "stale",
                "basis": "source-divergence",
                "evidence": "",
                "comment": "",
            }
        ]
        result = format_proposals_file(proposals, nodes=sample_nodes)
        assert "# Proposed Updates" in result
        assert "## UPDATE: premise-a" in result
        assert "---" in result


class TestProposeUpdates:
    @patch("reasonsforge.propose_update.invoke_model")
    def test_basic_flow(self, mock_invoke, sample_nodes):
        mock_invoke.return_value = json.dumps([{
            "id": "premise-a",
            "action": "update",
            "proposed_text": "Updated sky claim.",
            "failure_mode": "smuggled-premise",
            "basis": "prior-knowledge",
            "evidence": "Too simple.",
            "comment": "Improved.",
        }])
        results = propose_updates(
            sample_nodes,
            belief_ids=["premise-a"],
            model="claude",
            timeout=300,
        )
        assert len(results) == 1
        assert results[0]["id"] == "premise-a"
        mock_invoke.assert_called_once()

    @patch("reasonsforge.propose_update.invoke_model")
    def test_no_proposals_returned(self, mock_invoke, sample_nodes):
        mock_invoke.return_value = "[]"
        results = propose_updates(
            sample_nodes,
            belief_ids=["premise-a"],
        )
        assert results == []

    @patch("reasonsforge.propose_update.invoke_model")
    def test_batching(self, mock_invoke, sample_nodes):
        mock_invoke.return_value = "[]"
        propose_updates(
            sample_nodes,
            belief_ids=["premise-a", "derived-b", "derived-c"],
            batch_size=2,
        )
        assert mock_invoke.call_count == 2

    @patch("reasonsforge.propose_update.invoke_model")
    def test_default_all_in_beliefs(self, mock_invoke, sample_nodes):
        mock_invoke.return_value = "[]"
        propose_updates(sample_nodes)
        call_prompt = mock_invoke.call_args[0][0]
        assert "premise-a" in call_prompt
        assert "derived-b" in call_prompt

    @patch("reasonsforge.propose_update.invoke_model")
    def test_handles_llm_error(self, mock_invoke, sample_nodes):
        mock_invoke.side_effect = RuntimeError("model failed")
        results = propose_updates(
            sample_nodes,
            belief_ids=["premise-a"],
        )
        assert results == []

    @patch("reasonsforge.propose_update.invoke_model")
    def test_on_batch_callback(self, mock_invoke, sample_nodes):
        mock_invoke.return_value = json.dumps([{
            "id": "premise-a", "action": "update",
            "proposed_text": "new", "failure_mode": "stale",
            "basis": "source-divergence", "evidence": "", "comment": "",
        }])
        batches_seen = []
        propose_updates(
            sample_nodes,
            belief_ids=["premise-a"],
            on_batch=lambda results: batches_seen.append(len(results)),
        )
        assert batches_seen == [1]

    def test_empty_belief_ids(self, sample_nodes):
        results = propose_updates(sample_nodes, belief_ids=[])
        assert results == []

    def test_nonexistent_belief_ids(self, sample_nodes):
        results = propose_updates(sample_nodes, belief_ids=["nonexistent"])
        assert results == []

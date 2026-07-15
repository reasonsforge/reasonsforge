"""Tests for sync-agent: updating beliefs after initial import (remote wins)."""

import json

import pytest

from reasonsforge import api


INITIAL_BELIEFS = """\
## Beliefs

### alpha-fact [IN] OBSERVATION
Alpha is the first letter of the Greek alphabet
- Source: alphabet.md
- Date: 2026-03-28

### beta-depends-alpha [IN] DERIVED
Beta follows alpha in the alphabet
- Source: alphabet.md
- Date: 2026-03-28
- Depends on: alpha-fact

### gamma-stale [STALE] OBSERVATION
Gamma is the fourth letter
- Source: old.md
- Stale reason: gamma is actually third
"""


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "reasons.db")
    api.init_db(db_path=db_path)
    return db_path


@pytest.fixture
def initial_import(db, tmp_path):
    """Import initial beliefs and return (db_path, tmp_path)."""
    p = tmp_path / "beliefs.md"
    p.write_text(INITIAL_BELIEFS)
    api.import_agent("test-agent", str(p), db_path=db)
    return db, tmp_path


class TestSyncNoChanges:
    def test_sync_unchanged(self, initial_import):
        db, tmp_path = initial_import
        p = tmp_path / "beliefs.md"
        p.write_text(INITIAL_BELIEFS)

        result = api.sync_agent("test-agent", str(p), db_path=db)

        assert result["beliefs_added"] == 0
        assert result["beliefs_removed"] == 0
        assert result["beliefs_retracted"] == 1  # gamma-stale still OUT


class TestSyncNewBeliefs:
    def test_sync_adds_new_belief(self, initial_import):
        db, tmp_path = initial_import

        updated = INITIAL_BELIEFS + """
### delta-new [IN] OBSERVATION
Delta is the fourth letter
- Source: alphabet.md
- Date: 2026-04-18
"""
        p = tmp_path / "beliefs.md"
        p.write_text(updated)

        result = api.sync_agent("test-agent", str(p), db_path=db)
        assert result["beliefs_added"] == 1

        node = api.show_node("test-agent:delta-new", db_path=db)
        assert node["truth_value"] == "IN"
        assert node["metadata"]["agent"] == "test-agent"


class TestSyncRemovedBeliefs:
    def test_sync_retracts_removed_belief(self, initial_import):
        db, tmp_path = initial_import

        # Remove beta-depends-alpha from the remote
        reduced = """\
## Beliefs

### alpha-fact [IN] OBSERVATION
Alpha is the first letter of the Greek alphabet
- Source: alphabet.md
- Date: 2026-03-28

### gamma-stale [STALE] OBSERVATION
Gamma is the fourth letter
- Source: old.md
- Stale reason: gamma is actually third
"""
        p = tmp_path / "beliefs.md"
        p.write_text(reduced)

        result = api.sync_agent("test-agent", str(p), db_path=db)
        assert result["beliefs_removed"] == 1

        node = api.show_node("test-agent:beta-depends-alpha", db_path=db)
        assert node["truth_value"] == "OUT"


class TestSyncUpdatedText:
    def test_sync_updates_text(self, initial_import):
        db, tmp_path = initial_import

        updated = INITIAL_BELIEFS.replace(
            "Alpha is the first letter of the Greek alphabet",
            "Alpha (Α) is the first letter of the Greek alphabet",
        )
        p = tmp_path / "beliefs.md"
        p.write_text(updated)

        result = api.sync_agent("test-agent", str(p), db_path=db)
        assert result["beliefs_updated"] >= 1

        node = api.show_node("test-agent:alpha-fact", db_path=db)
        assert "Α" in node["text"]


class TestSyncTruthValueChanges:
    def test_sync_in_to_out(self, initial_import):
        """Remote changes a belief from IN to OUT."""
        db, tmp_path = initial_import

        updated = INITIAL_BELIEFS.replace(
            "### alpha-fact [IN] OBSERVATION",
            "### alpha-fact [OUT] OBSERVATION",
        )
        p = tmp_path / "beliefs.md"
        p.write_text(updated)

        result = api.sync_agent("test-agent", str(p), db_path=db)

        node = api.show_node("test-agent:alpha-fact", db_path=db)
        assert node["truth_value"] == "OUT"

        # Dependent should also go OUT
        beta = api.show_node("test-agent:beta-depends-alpha", db_path=db)
        assert beta["truth_value"] == "OUT"

    def test_sync_out_to_in(self, initial_import):
        """Remote changes a belief from STALE to IN (un-retract)."""
        db, tmp_path = initial_import

        # Verify gamma is OUT initially
        node = api.show_node("test-agent:gamma-stale", db_path=db)
        assert node["truth_value"] == "OUT"

        updated = INITIAL_BELIEFS.replace(
            "### gamma-stale [STALE] OBSERVATION",
            "### gamma-stale [IN] OBSERVATION",
        )
        p = tmp_path / "beliefs.md"
        p.write_text(updated)

        result = api.sync_agent("test-agent", str(p), db_path=db)

        node = api.show_node("test-agent:gamma-stale", db_path=db)
        assert node["truth_value"] == "IN"


class TestSyncRetractedCleared:
    def test_sync_clears_local_retraction(self, initial_import):
        """If user retracted a belief locally but remote says IN, remote wins."""
        db, tmp_path = initial_import

        # Locally retract alpha-fact
        api.retract_node("test-agent:alpha-fact", db_path=db)
        node = api.show_node("test-agent:alpha-fact", db_path=db)
        assert node["truth_value"] == "OUT"

        # Sync with unchanged remote (alpha-fact still IN)
        p = tmp_path / "beliefs.md"
        p.write_text(INITIAL_BELIEFS)

        result = api.sync_agent("test-agent", str(p), db_path=db)

        node = api.show_node("test-agent:alpha-fact", db_path=db)
        assert node["truth_value"] == "IN"

        # Dependent should also come back
        beta = api.show_node("test-agent:beta-depends-alpha", db_path=db)
        assert beta["truth_value"] == "IN"


class TestSyncUpdatedDependencies:
    def test_sync_changes_dependency(self, initial_import):
        """Remote changes which node a belief depends on."""
        db, tmp_path = initial_import

        # Change beta to depend on gamma-stale instead of alpha-fact
        updated = """\
## Beliefs

### alpha-fact [IN] OBSERVATION
Alpha is the first letter of the Greek alphabet
- Source: alphabet.md
- Date: 2026-03-28

### beta-depends-alpha [IN] DERIVED
Beta follows gamma (updated)
- Source: alphabet.md
- Date: 2026-04-18
- Depends on: gamma-stale

### gamma-stale [IN] OBSERVATION
Gamma is the third letter (corrected)
- Source: new.md
- Date: 2026-04-18
"""
        p = tmp_path / "beliefs.md"
        p.write_text(updated)

        result = api.sync_agent("test-agent", str(p), db_path=db)

        node = api.show_node("test-agent:beta-depends-alpha", db_path=db)
        j = node["justifications"][0]
        assert "test-agent:gamma-stale" in j["antecedents"]
        assert "test-agent:alpha-fact" not in j["antecedents"]
        assert node["truth_value"] == "IN"


class TestSyncJson:
    def test_sync_json_adds_and_updates(self, db, tmp_path):
        """JSON sync: initial import then sync with changes."""
        initial_data = {
            "nodes": {
                "premise-a": {
                    "text": "A premise",
                    "truth_value": "IN",
                    "justifications": [],
                    "source": "test.md",
                },
                "derived-b": {
                    "text": "Derived from A",
                    "truth_value": "IN",
                    "justifications": [{
                        "type": "SL",
                        "antecedents": ["premise-a"],
                    }],
                    "source": "test.md",
                },
            },
            "nogoods": [],
        }

        p = tmp_path / "network.json"
        p.write_text(json.dumps(initial_data))
        api.import_agent("json-agent", str(p), db_path=db)

        # Update: change text and add a new node
        updated_data = {
            "nodes": {
                "premise-a": {
                    "text": "A premise (updated)",
                    "truth_value": "IN",
                    "justifications": [],
                    "source": "test.md",
                },
                "derived-b": {
                    "text": "Derived from A",
                    "truth_value": "IN",
                    "justifications": [{
                        "type": "SL",
                        "antecedents": ["premise-a"],
                    }],
                    "source": "test.md",
                },
                "new-c": {
                    "text": "A new belief",
                    "truth_value": "IN",
                    "justifications": [],
                    "source": "test.md",
                },
            },
            "nogoods": [],
        }
        p.write_text(json.dumps(updated_data))

        result = api.sync_agent("json-agent", str(p), db_path=db)
        assert result["beliefs_added"] == 1
        assert result["beliefs_updated"] >= 1

        node = api.show_node("json-agent:premise-a", db_path=db)
        assert "(updated)" in node["text"]

        new_node = api.show_node("json-agent:new-c", db_path=db)
        assert new_node["truth_value"] == "IN"

    def test_sync_json_removes_belief(self, db, tmp_path):
        """JSON sync: removing a belief from remote retracts it locally."""
        initial_data = {
            "nodes": {
                "fact-a": {
                    "text": "Fact A",
                    "truth_value": "IN",
                    "justifications": [],
                },
                "fact-b": {
                    "text": "Fact B",
                    "truth_value": "IN",
                    "justifications": [],
                },
            },
            "nogoods": [],
        }

        p = tmp_path / "network.json"
        p.write_text(json.dumps(initial_data))
        api.import_agent("rm-agent", str(p), db_path=db)

        # Remove fact-b from remote
        updated_data = {
            "nodes": {
                "fact-a": {
                    "text": "Fact A",
                    "truth_value": "IN",
                    "justifications": [],
                },
            },
            "nogoods": [],
        }
        p.write_text(json.dumps(updated_data))

        result = api.sync_agent("rm-agent", str(p), db_path=db)
        assert result["beliefs_removed"] == 1

        node = api.show_node("rm-agent:fact-b", db_path=db)
        assert node["truth_value"] == "OUT"

    def test_sync_json_clears_retraction(self, db, tmp_path):
        """JSON sync: remote IN overrides local retraction."""
        data = {
            "nodes": {
                "fact-x": {
                    "text": "Fact X",
                    "truth_value": "IN",
                    "justifications": [],
                },
            },
            "nogoods": [],
        }

        p = tmp_path / "network.json"
        p.write_text(json.dumps(data))
        api.import_agent("clr-agent", str(p), db_path=db)

        # Locally retract
        api.retract_node("clr-agent:fact-x", db_path=db)
        node = api.show_node("clr-agent:fact-x", db_path=db)
        assert node["truth_value"] == "OUT"

        # Sync with remote still saying IN
        result = api.sync_agent("clr-agent", str(p), db_path=db)

        node = api.show_node("clr-agent:fact-x", db_path=db)
        assert node["truth_value"] == "IN"


class TestSyncCountingAccuracy:
    def test_unchanged_beliefs_counted_correctly(self, initial_import):
        """When nothing changes, beliefs_updated should be 0."""
        db, tmp_path = initial_import
        p = tmp_path / "beliefs.md"
        p.write_text(INITIAL_BELIEFS)

        result = api.sync_agent("test-agent", str(p), db_path=db)
        assert result["beliefs_updated"] == 0
        assert result["beliefs_unchanged"] >= 2  # alpha-fact, beta-depends-alpha

    def test_justification_only_change_counted_as_update(self, initial_import):
        """Changing only dependencies (not text) should count as an update."""
        db, tmp_path = initial_import

        # Add a new premise, then change beta to depend on it instead of alpha
        # (text stays the same, only depends_on changes)
        updated = """\
## Beliefs

### alpha-fact [IN] OBSERVATION
Alpha is the first letter of the Greek alphabet
- Source: alphabet.md
- Date: 2026-03-28

### new-premise [IN] OBSERVATION
A new premise
- Source: alphabet.md
- Date: 2026-04-18

### beta-depends-alpha [IN] DERIVED
Beta follows alpha in the alphabet
- Source: alphabet.md
- Date: 2026-03-28
- Depends on: new-premise

### gamma-stale [STALE] OBSERVATION
Gamma is the fourth letter
- Source: old.md
- Stale reason: gamma is actually third
"""
        p = tmp_path / "beliefs.md"
        p.write_text(updated)

        result = api.sync_agent("test-agent", str(p), db_path=db)
        assert result["beliefs_added"] == 1  # new-premise
        assert result["beliefs_updated"] >= 1  # beta's justification changed


class TestSyncFirstTime:
    def test_sync_works_as_initial_import(self, db, tmp_path):
        """Sync on a fresh agent (no prior import) should work like import."""
        p = tmp_path / "beliefs.md"
        p.write_text(INITIAL_BELIEFS)

        result = api.sync_agent("fresh-agent", str(p), db_path=db)
        assert result["created_premise"] is True
        assert result["beliefs_added"] == 3
        assert result["beliefs_removed"] == 0

        node = api.show_node("fresh-agent:alpha-fact", db_path=db)
        assert node["truth_value"] == "IN"


class TestSyncIdempotency:
    def test_resync_removed_no_double_count(self, initial_import):
        """Re-syncing after a removal should not re-count removed beliefs."""
        db, tmp_path = initial_import

        reduced = """\
## Beliefs

### alpha-fact [IN] OBSERVATION
Alpha is the first letter of the Greek alphabet
- Source: alphabet.md
- Date: 2026-03-28

### gamma-stale [STALE] OBSERVATION
Gamma is the fourth letter
- Source: old.md
- Stale reason: gamma is actually third
"""
        p = tmp_path / "beliefs.md"
        p.write_text(reduced)

        result1 = api.sync_agent("test-agent", str(p), db_path=db)
        assert result1["beliefs_removed"] == 1

        # Second sync — removal already happened
        result2 = api.sync_agent("test-agent", str(p), db_path=db)
        assert result2["beliefs_removed"] == 0

    def test_sync_twice_unchanged_is_stable(self, initial_import):
        """Two consecutive syncs with the same data should both be no-ops."""
        db, tmp_path = initial_import
        p = tmp_path / "beliefs.md"
        p.write_text(INITIAL_BELIEFS)

        api.sync_agent("test-agent", str(p), db_path=db)
        result = api.sync_agent("test-agent", str(p), db_path=db)

        assert result["beliefs_added"] == 0
        assert result["beliefs_removed"] == 0


class TestSyncAgentRevocation:
    def test_synced_beliefs_still_cascade_on_agent_revocation(self, initial_import):
        """After sync, revoking agent:active should still cascade OUT all beliefs.

        Regression: assert_node during sync must not detach beliefs from
        their justifications (which include inactive_id in outlist).
        """
        db, tmp_path = initial_import
        p = tmp_path / "beliefs.md"
        p.write_text(INITIAL_BELIEFS)

        # Sync (no-op but exercises the update path)
        api.sync_agent("test-agent", str(p), db_path=db)

        # Verify justifications still have inactive in outlist
        node = api.show_node("test-agent:alpha-fact", db_path=db)
        assert any("test-agent:inactive" in j["outlist"] for j in node["justifications"])

        # Revoke the agent
        api.retract_node("test-agent:active", db_path=db)

        node = api.show_node("test-agent:alpha-fact", db_path=db)
        assert node["truth_value"] == "OUT"

        beta = api.show_node("test-agent:beta-depends-alpha", db_path=db)
        assert beta["truth_value"] == "OUT"

    def test_synced_restored_belief_still_cascades(self, initial_import):
        """After sync restores a locally-retracted belief, agent revocation still works."""
        db, tmp_path = initial_import

        # Locally retract alpha
        api.retract_node("test-agent:alpha-fact", db_path=db)

        # Sync restores it (remote says IN)
        p = tmp_path / "beliefs.md"
        p.write_text(INITIAL_BELIEFS)
        api.sync_agent("test-agent", str(p), db_path=db)

        node = api.show_node("test-agent:alpha-fact", db_path=db)
        assert node["truth_value"] == "IN"

        # Now revoke the agent — everything should go OUT
        api.retract_node("test-agent:active", db_path=db)

        node = api.show_node("test-agent:alpha-fact", db_path=db)
        assert node["truth_value"] == "OUT"


class TestSyncOutWithJustifications:
    """Sync tests for OUT nodes that preserve justifications (JSON path)."""

    def test_sync_in_to_out_preserves_justifications(self, db, tmp_path):
        """JSON sync: IN→OUT transition preserves justifications and forces OUT."""
        initial = {
            "nodes": {
                "fact": {"text": "A fact", "truth_value": "IN", "justifications": []},
                "derived": {
                    "text": "Derived from fact",
                    "truth_value": "IN",
                    "justifications": [{"type": "SL", "antecedents": ["fact"]}],
                },
            },
            "nogoods": [],
        }
        p = tmp_path / "network.json"
        p.write_text(json.dumps(initial))
        api.import_agent("owj-agent", str(p), db_path=db)

        node = api.show_node("owj-agent:derived", db_path=db)
        assert node["truth_value"] == "IN"

        updated = {
            "nodes": {
                "fact": {"text": "A fact", "truth_value": "IN", "justifications": []},
                "derived": {
                    "text": "Derived from fact",
                    "truth_value": "OUT",
                    "justifications": [{"type": "SL", "antecedents": ["fact"]}],
                },
            },
            "nogoods": [],
        }
        p.write_text(json.dumps(updated))
        result = api.sync_agent("owj-agent", str(p), db_path=db)

        node = api.show_node("owj-agent:derived", db_path=db)
        assert node["truth_value"] == "OUT"
        assert len(node["justifications"]) >= 1

    def test_sync_out_to_in_clears_retracted(self, db, tmp_path):
        """JSON sync: OUT→IN transition clears _retracted and resurrects."""
        initial = {
            "nodes": {
                "fact": {"text": "A fact", "truth_value": "IN", "justifications": []},
                "gated": {
                    "text": "Gated belief",
                    "truth_value": "OUT",
                    "justifications": [{"type": "SL", "antecedents": ["fact"]}],
                },
            },
            "nogoods": [],
        }
        p = tmp_path / "network.json"
        p.write_text(json.dumps(initial))
        api.import_agent("res-agent", str(p), db_path=db)

        node = api.show_node("res-agent:gated", db_path=db)
        assert node["truth_value"] == "OUT"

        updated = {
            "nodes": {
                "fact": {"text": "A fact", "truth_value": "IN", "justifications": []},
                "gated": {
                    "text": "Gated belief",
                    "truth_value": "IN",
                    "justifications": [{"type": "SL", "antecedents": ["fact"]}],
                },
            },
            "nogoods": [],
        }
        p.write_text(json.dumps(updated))
        result = api.sync_agent("res-agent", str(p), db_path=db)

        node = api.show_node("res-agent:gated", db_path=db)
        assert node["truth_value"] == "IN"

    def test_sync_out_with_justifications_idempotent(self, db, tmp_path):
        """JSON sync: re-syncing OUT-with-justifications is a no-op."""
        data = {
            "nodes": {
                "fact": {"text": "A fact", "truth_value": "IN", "justifications": []},
                "derived": {
                    "text": "Derived from fact",
                    "truth_value": "OUT",
                    "justifications": [{"type": "SL", "antecedents": ["fact"]}],
                },
            },
            "nogoods": [],
        }
        p = tmp_path / "network.json"
        p.write_text(json.dumps(data))
        api.import_agent("idem-agent", str(p), db_path=db)

        result1 = api.sync_agent("idem-agent", str(p), db_path=db)
        result2 = api.sync_agent("idem-agent", str(p), db_path=db)

        assert result2["beliefs_updated"] == 0
        assert result2["beliefs_added"] == 0
        assert result2["beliefs_removed"] == 0


class TestSyncRegistersRepo:

    def test_sync_registers_repo(self, initial_import):
        db, tmp_path = initial_import
        p = tmp_path / "beliefs.md"
        p.write_text(INITIAL_BELIEFS)
        api.sync_agent("test-agent", str(p), db_path=db)

        repos = api.list_repos(db_path=db)["repos"]
        assert "test-agent" in repos
        from pathlib import Path
        assert repos["test-agent"] == str(Path(str(p)).resolve().parent)

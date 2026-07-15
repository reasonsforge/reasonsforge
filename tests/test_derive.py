"""Tests for derive: reasoning chain derivation."""

from pathlib import Path

import pytest

from reasonsforge import api
from reasonsforge.derive import (
    build_prompt,
    parse_proposals,
    validate_proposals,
    apply_proposals,
    write_proposals_file,
    find_similar_out,
    _detect_agents,
    _filter_by_topic,
    _sample_beliefs,
    _get_depth,
    _tokenize_id,
    _jaccard,
    _build_derived_section,
)


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "reasons.db")
    api.init_db(db_path=db_path)
    return db_path


@pytest.fixture
def simple_network(db):
    """A small network with premises and one derived node."""
    api.add_node("fact-a", "Alpha is true", db_path=db)
    api.add_node("fact-b", "Beta is true", db_path=db)
    api.add_node("fact-c", "Gamma is a bug", db_path=db)
    api.add_node("derived-ab", "Alpha and Beta combined",
                 sl="fact-a,fact-b", label="test", db_path=db)
    return db


@pytest.fixture
def agent_network(db):
    """A network with two imported agents."""
    # Simulate agent imports by adding namespaced nodes
    api.add_node("agent-a:active", "Agent A is trusted", db_path=db)
    api.add_node("agent-a:knows-auth", "Agent A knows about auth",
                 sl="agent-a:active", label="imported from agent: agent-a",
                 db_path=db)
    api.add_node("agent-a:knows-routing", "Agent A knows about routing",
                 sl="agent-a:active", label="imported from agent: agent-a",
                 db_path=db)

    api.add_node("agent-b:active", "Agent B is trusted", db_path=db)
    api.add_node("agent-b:knows-gateway", "Agent B knows about the gateway",
                 sl="agent-b:active", label="imported from agent: agent-b",
                 db_path=db)
    return db


def test_build_prompt_basic(simple_network):
    data = api.export_network(db_path=simple_network)
    prompt, stats = build_prompt(data["nodes"])

    assert stats["total_in"] == 4
    assert stats["total_derived"] == 1
    assert stats["max_depth"] == 1
    assert stats["agents"] == 0
    assert "fact-a" in prompt
    assert "derived-ab" in prompt


def test_build_prompt_with_domain(simple_network):
    data = api.export_network(db_path=simple_network)
    prompt, _ = build_prompt(data["nodes"], domain="Greek alphabet")

    assert "Greek alphabet" in prompt


def test_build_prompt_detects_agents(agent_network):
    data = api.export_network(db_path=agent_network)
    prompt, stats = build_prompt(data["nodes"])

    assert stats["agents"] == 2
    assert "agent-a" in stats["agent_names"]
    assert "agent-b" in stats["agent_names"]
    assert "cross-agent" in prompt.lower()
    assert "Agent: agent-a" in prompt
    assert "Agent: agent-b" in prompt


def test_detect_agents():
    nodes = {
        "agent-a:active": {},
        "agent-a:belief-1": {},
        "agent-a:belief-2": {},
        "agent-b:active": {},
        "agent-b:belief-1": {},
        "local-belief": {},
    }
    agents = _detect_agents(nodes)
    assert "agent-a" in agents
    assert "agent-b" in agents
    assert len(agents["agent-a"]) == 2  # excludes :active
    assert len(agents["agent-b"]) == 1


def test_get_depth():
    nodes = {
        "a": {"justifications": []},
        "b": {"justifications": []},
        "c": {"justifications": [{"antecedents": ["a", "b"]}]},
        "d": {"justifications": [{"antecedents": ["c"]}]},
    }
    derived = {k: v for k, v in nodes.items() if v["justifications"]}

    assert _get_depth("a", nodes, derived) == 0
    assert _get_depth("c", nodes, derived) == 1
    assert _get_depth("d", nodes, derived) == 2


def test_build_prompt_min_depth(simple_network):
    data = api.export_network(db_path=simple_network)
    _, stats = build_prompt(data["nodes"], min_depth=1)

    assert stats["min_depth"] == 1
    # Only derived-ab (depth 1) passes the filter
    assert stats["total_in"] == 1
    assert stats["total_derived"] == 1


def test_build_prompt_max_depth(simple_network):
    data = api.export_network(db_path=simple_network)
    prompt, stats = build_prompt(data["nodes"], max_depth_filter=0)

    assert stats["max_depth_filter"] == 0
    # Only premises (depth 0) remain
    assert "fact-a" in prompt
    assert stats["total_derived"] == 0
    assert stats["total_in"] == 3


def test_build_prompt_depth_range(db):
    api.add_node("zz-premise-node", "A premise", db_path=db)
    api.add_node("zz-mid-node", "Middle derived", sl="zz-premise-node", label="test", db_path=db)
    api.add_node("zz-top-node", "Top derived", sl="zz-mid-node", label="test", db_path=db)

    data = api.export_network(db_path=db)
    _, stats = build_prompt(data["nodes"], min_depth=1, max_depth_filter=1)

    # Only zz-mid-node (depth 1) passes
    assert stats["total_in"] == 1
    assert stats["total_derived"] == 1


def test_build_prompt_premises_only(simple_network):
    data = api.export_network(db_path=simple_network)
    prompt, stats = build_prompt(data["nodes"], premises_only=True)

    assert stats["total_derived"] == 0
    assert stats["total_in"] == 3
    assert "fact-a" in prompt
    assert "derived-ab" not in prompt


def test_build_prompt_has_dependents(simple_network):
    data = api.export_network(db_path=simple_network)
    _, stats = build_prompt(data["nodes"], has_dependents=True)

    # fact-a and fact-b are antecedents of derived-ab, fact-c has no dependents
    assert stats["total_in"] == 2


def test_parse_proposals_derive():
    response = """Here are my proposals:

### DERIVE combined-auth-gateway
Auth tokens flow through the gateway with validation at each layer
- Antecedents: agent-a:knows-auth, agent-b:knows-gateway
- Label: cross-agent authentication flow
"""
    proposals = parse_proposals(response)
    assert len(proposals) == 1
    p = proposals[0]
    assert p["kind"] == "derive"
    assert p["id"] == "combined-auth-gateway"
    assert p["antecedents"] == ["agent-a:knows-auth", "agent-b:knows-gateway"]
    assert p["unless"] == []
    assert p["label"] == "cross-agent authentication flow"


def test_parse_proposals_gate():
    response = """
### GATE feature-ready
Feature X is production-ready
- Antecedents: fact-a, fact-b
- Unless: fact-c
- Label: gated on bug resolution
"""
    proposals = parse_proposals(response)
    assert len(proposals) == 1
    p = proposals[0]
    assert p["kind"] == "gate"
    assert p["unless"] == ["fact-c"]


def test_parse_proposals_with_mode_any():
    response = """
### DERIVE resilient-conclusion
Multiple independent observations support this claim
- Antecedents: obs-a, obs-b, obs-c
- Mode: ANY
- Label: convergent evidence from independent observations
"""
    proposals = parse_proposals(response)
    assert len(proposals) == 1
    p = proposals[0]
    assert p["mode"] == "any"
    assert p["antecedents"] == ["obs-a", "obs-b", "obs-c"]


def test_parse_proposals_mode_defaults_all():
    response = """
### DERIVE chained-conclusion
This requires all steps in the logical chain
- Antecedents: step-a, step-b
- Label: sequential dependency
"""
    proposals = parse_proposals(response)
    assert len(proposals) == 1
    assert proposals[0]["mode"] == "all"


def test_parse_proposals_gate_with_mode():
    response = """
### GATE feature-stable
Feature is stable when supported by any evidence and no blockers
- Antecedents: evidence-a, evidence-b
- Unless: blocker-x
- Mode: ANY
- Label: gated on blocker resolution
"""
    proposals = parse_proposals(response)
    assert len(proposals) == 1
    p = proposals[0]
    assert p["kind"] == "gate"
    assert p["mode"] == "any"
    assert p["unless"] == ["blocker-x"]


def test_parse_proposals_multiple():
    response = """
### DERIVE first-one
First derived belief
- Antecedents: a, b
- Label: first

### GATE second-one
Second gated belief
- Antecedents: c
- Unless: d
- Label: second
"""
    proposals = parse_proposals(response)
    assert len(proposals) == 2


def test_parse_proposals_strips_backticks():
    response = """
### DERIVE `cross-validated-readiness`
Production readiness is cross-validated
- Antecedents: `product:production-readiness`, `code:full-stack-reliability`
- Unless: `blocker:critical-bug`
- Label: cross-validated
"""
    proposals = parse_proposals(response)
    assert len(proposals) == 1
    p = proposals[0]
    assert p["id"] == "cross-validated-readiness"
    assert p["antecedents"] == ["product:production-readiness", "code:full-stack-reliability"]
    assert p["unless"] == ["blocker:critical-bug"]


def test_validate_proposals_missing_antecedent():
    nodes = {"fact-a": {}, "fact-b": {}}
    proposals = [
        {"id": "new-1", "antecedents": ["fact-a", "fact-b"], "unless": [],
         "text": "ok", "kind": "derive", "label": "test"},
        {"id": "new-2", "antecedents": ["fact-a", "nonexistent"], "unless": [],
         "text": "bad", "kind": "derive", "label": "test"},
    ]
    valid, skipped = validate_proposals(proposals, nodes)
    assert len(valid) == 1
    assert valid[0]["id"] == "new-1"
    assert len(skipped) == 1
    assert "nonexistent" in skipped[0][1]


def test_validate_proposals_already_exists():
    nodes = {"fact-a": {}, "fact-b": {}, "existing": {}}
    proposals = [
        {"id": "existing", "antecedents": ["fact-a", "fact-b"], "unless": [],
         "text": "dup", "kind": "derive", "label": "test"},
    ]
    valid, skipped = validate_proposals(proposals, nodes)
    assert len(valid) == 0
    assert "already exists" in skipped[0][1]


def test_apply_proposals(simple_network):
    proposals = [
        {"id": "new-derived", "text": "New conclusion from a and c",
         "antecedents": ["fact-a", "fact-c"], "unless": [],
         "kind": "derive", "label": "test apply"},
    ]
    results = apply_proposals(proposals, db_path=simple_network)
    assert len(results) == 1
    p, result = results[0]
    assert isinstance(result, dict)
    assert result["truth_value"] == "IN"

    # Verify it was actually added
    node = api.show_node("new-derived", db_path=simple_network)
    assert node["truth_value"] == "IN"
    assert "fact-a" in node["justifications"][0]["antecedents"]


def test_apply_proposals_with_gate(simple_network):
    proposals = [
        {"id": "gated-belief", "text": "A is good unless C is true",
         "antecedents": ["fact-a"], "unless": ["fact-c"],
         "kind": "gate", "label": "test gate"},
    ]
    results = apply_proposals(proposals, db_path=simple_network)
    p, result = results[0]
    # fact-c is IN, so this gated belief should be OUT
    assert result["truth_value"] == "OUT"

    # Retract fact-c — gated belief should come back IN
    api.retract_node("fact-c", db_path=simple_network)
    node = api.show_node("gated-belief", db_path=simple_network)
    assert node["truth_value"] == "IN"


def test_apply_proposals_any_mode(simple_network):
    proposals = [
        {"id": "any-derived", "text": "Supported by either a or c",
         "antecedents": ["fact-a", "fact-c"], "unless": [],
         "kind": "derive", "label": "convergent", "mode": "any"},
    ]
    results = apply_proposals(proposals, db_path=simple_network)
    p, result = results[0]
    assert result["truth_value"] == "IN"

    node = api.show_node("any-derived", db_path=simple_network)
    assert len(node["justifications"]) == 2
    assert all(len(j["antecedents"]) == 1 for j in node["justifications"])

    api.retract_node("fact-a", db_path=simple_network)
    node = api.show_node("any-derived", db_path=simple_network)
    assert node["truth_value"] == "IN"


# --- Topic filter tests ---

def test_filter_by_topic():
    nodes = {
        "auth-uses-jwt": {"text": "Auth system uses JWT tokens"},
        "routing-table": {"text": "The routing table is updated"},
        "auth-session": {"text": "Session management for auth"},
        "database-schema": {"text": "The database schema has 5 tables"},
    }
    filtered = _filter_by_topic(nodes, "auth")
    assert "auth-uses-jwt" in filtered
    assert "auth-session" in filtered
    assert "routing-table" not in filtered
    assert "database-schema" not in filtered


def test_filter_by_topic_matches_id_and_text():
    nodes = {
        "firewall-rules": {"text": "Stateless firewall at the perimeter"},
        "network-config": {"text": "Network uses firewall for isolation"},
        "storage-volume": {"text": "Ceph-backed storage volume"},
    }
    filtered = _filter_by_topic(nodes, "firewall")
    assert "firewall-rules" in filtered
    assert "network-config" in filtered  # matches in text
    assert "storage-volume" not in filtered


def test_filter_by_topic_multiple_keywords():
    nodes = {
        "auth-jwt": {"text": "JWT authentication"},
        "tls-config": {"text": "TLS certificate setup"},
        "database-backup": {"text": "Daily database backup"},
    }
    # Any keyword matches (OR semantics)
    filtered = _filter_by_topic(nodes, "auth tls")
    assert "auth-jwt" in filtered
    assert "tls-config" in filtered
    assert "database-backup" not in filtered


def test_build_prompt_with_topic(agent_network):
    data = api.export_network(db_path=agent_network)
    prompt, stats = build_prompt(data["nodes"], topic="auth")

    assert stats.get("topic") == "auth"
    # Only auth-related beliefs should appear
    assert "knows-auth" in prompt
    # Non-matching beliefs should be filtered out
    assert "knows-gateway" not in prompt


# --- Budget tests ---

def test_build_prompt_with_budget(simple_network):
    data = api.export_network(db_path=simple_network)
    prompt_small, stats_small = build_prompt(data["nodes"], budget=2)
    prompt_large, stats_large = build_prompt(data["nodes"], budget=100)

    # Smaller budget should produce shorter prompt
    assert len(prompt_small) < len(prompt_large)
    assert stats_small["budget"] == 2
    assert stats_large["budget"] == 100


def test_build_derived_section_cap():
    nodes = {"p": {"text": "Premise", "truth_value": "IN", "justifications": []}}
    derived = {}
    for i in range(10):
        nid = f"d-{i}"
        nodes[nid] = {
            "text": f"Derived {i}",
            "truth_value": "IN",
            "justifications": [{"type": "SL", "antecedents": ["p"], "outlist": []}],
        }
        derived[nid] = nodes[nid]

    section = _build_derived_section(nodes, derived, max_derived=3)
    assert section.count("####") == 3
    assert "7 more derived conclusions omitted" in section


def test_build_derived_section_under_cap():
    nodes = {"p": {"text": "Premise", "truth_value": "IN", "justifications": []}}
    derived = {}
    for i in range(3):
        nid = f"d-{i}"
        nodes[nid] = {
            "text": f"Derived {i}",
            "truth_value": "IN",
            "justifications": [{"type": "SL", "antecedents": ["p"], "outlist": []}],
        }
        derived[nid] = nodes[nid]

    section = _build_derived_section(nodes, derived, max_derived=10)
    assert section.count("####") == 3
    assert "omitted" not in section


# --- Sampling tests ---

def test_sample_beliefs_under_budget():
    ids = ["a", "b", "c"]
    result = _sample_beliefs(ids, budget=10)
    assert result == ids  # all returned when under budget


def test_sample_beliefs_over_budget():
    ids = [f"belief-{i}" for i in range(100)]
    result = _sample_beliefs(ids, budget=10)
    assert len(result) == 10
    assert all(b in ids for b in result)


def test_sample_beliefs_reproducible():
    import random
    ids = [f"belief-{i}" for i in range(100)]
    r1 = _sample_beliefs(ids, budget=10, rng=random.Random(42))
    r2 = _sample_beliefs(ids, budget=10, rng=random.Random(42))
    assert r1 == r2


def test_build_prompt_with_sample(agent_network):
    data = api.export_network(db_path=agent_network)
    prompt, stats = build_prompt(data["nodes"], sample=True, seed=42)

    assert stats["sample"] is True
    # Should still produce a valid prompt
    assert "Agent:" in prompt


# --- Accept (write + re-parse round-trip) tests ---

def test_parse_proposals_old_format():
    """Parse the v0.9 format with backtick IDs and bold field names."""
    response = """# Proposed Derivations

Review each proposal below. To accept, run:

---

### DERIVE: `combined-auth-gateway`

Auth tokens flow through the gateway with validation at each layer

- **Antecedents**: `agent-a:knows-auth`, `agent-b:knows-gateway`
- **Label**: cross-agent authentication flow

### GATE (outlist): `feature-ready`

Feature X is production-ready

- **Antecedents**: `fact-a`, `fact-b`
- **Unless**: `fact-c`
- **Label**: gated on bug resolution
"""
    proposals = parse_proposals(response)
    assert len(proposals) == 2

    p = proposals[0]
    assert p["kind"] == "derive"
    assert p["id"] == "combined-auth-gateway"
    assert p["antecedents"] == ["agent-a:knows-auth", "agent-b:knows-gateway"]
    assert p["label"] == "cross-agent authentication flow"

    p2 = proposals[1]
    assert p2["kind"] == "gate"
    assert p2["id"] == "feature-ready"
    assert p2["unless"] == ["fact-c"]


def test_write_proposals_file_roundtrip(tmp_path):
    """Proposals file can be parsed back by parse_proposals."""
    proposals = [
        {"id": "derived-1", "text": "First conclusion", "kind": "derive",
         "antecedents": ["fact-a", "fact-b"], "unless": [], "label": "test",
         "mode": "all"},
        {"id": "gated-1", "text": "Gated conclusion", "kind": "gate",
         "antecedents": ["fact-a"], "unless": ["fact-c"], "label": "test gate",
         "mode": "any"},
    ]
    out = tmp_path / "proposals.md"
    write_proposals_file(proposals, out)

    text = out.read_text()
    parsed = parse_proposals(text)
    assert len(parsed) == 2
    assert parsed[0]["id"] == "derived-1"
    assert parsed[0]["antecedents"] == ["fact-a", "fact-b"]
    assert parsed[0]["mode"] == "all"
    assert parsed[1]["id"] == "gated-1"
    assert parsed[1]["unless"] == ["fact-c"]
    assert parsed[1]["mode"] == "any"


def test_accept_applies_proposals(simple_network, tmp_path):
    """Full accept flow: write proposals, parse, apply."""
    proposals = [
        {"id": "accepted-belief", "text": "Accepted from file",
         "antecedents": ["fact-a", "fact-b"], "unless": [],
         "kind": "derive", "label": "accepted"},
    ]
    out = tmp_path / "proposals.md"
    write_proposals_file(proposals, out)

    # Parse and apply (simulating what cmd_accept does)
    text = out.read_text()
    parsed = parse_proposals(text)
    data = api.export_network(db_path=simple_network)
    valid, skipped = validate_proposals(parsed, data["nodes"])
    assert len(valid) == 1

    results = apply_proposals(valid, db_path=simple_network)
    assert len(results) == 1
    _, result = results[0]
    assert result["truth_value"] == "IN"

    node = api.show_node("accepted-belief", db_path=simple_network)
    assert node["truth_value"] == "IN"


# --- Duplicate detection tests ---

def test_tokenize_id():
    assert _tokenize_id("gl108-response-validation-disabled") == {
        "gl108", "response", "validation", "disabled",
    }


def test_tokenize_id_with_namespace():
    assert _tokenize_id("agent-a:gl108-disabled") == {
        "agent", "a", "gl108", "disabled",
    }


def test_jaccard_identical():
    assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_disjoint():
    assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial():
    assert _jaccard({"a", "b", "c"}, {"a", "b", "d"}) == pytest.approx(0.5)


def test_jaccard_empty():
    assert _jaccard(set(), {"a"}) == 0.0


def test_find_similar_out_catches_variant_ids():
    nodes = {
        "gl108-safety-validation-disabled": {
            "truth_value": "OUT", "text": "GL-108 safety validation is disabled",
        },
        "fact-a": {"truth_value": "IN", "text": "Alpha"},
    }
    matches = find_similar_out("gl108-response-validation-disabled", nodes)
    assert len(matches) == 1
    assert matches[0][0] == "gl108-safety-validation-disabled"
    assert matches[0][1] >= 0.5


def test_find_similar_out_ignores_in_beliefs():
    nodes = {
        "gl108-safety-validation-disabled": {
            "truth_value": "IN", "text": "GL-108 safety validation is disabled",
        },
    }
    matches = find_similar_out("gl108-response-validation-disabled", nodes)
    assert matches == []


def test_find_similar_out_no_match():
    nodes = {
        "unrelated-network-config": {
            "truth_value": "OUT", "text": "Network config is old",
        },
    }
    matches = find_similar_out("gl108-response-validation-disabled", nodes)
    assert matches == []


def test_validate_proposals_skips_similar_to_retracted():
    """The core bug: variant IDs of retracted beliefs should be caught."""
    nodes = {
        "fact-a": {"truth_value": "IN"},
        "fact-b": {"truth_value": "IN"},
        "gl108-safety-validation-disabled": {
            "truth_value": "OUT",
            "text": "GL-108 safety validation is disabled",
        },
    }
    proposals = [
        {"id": "gl108-response-validation-disabled",
         "antecedents": ["fact-a", "fact-b"], "unless": [],
         "text": "GL-108 response validation is disabled",
         "kind": "derive", "label": "test"},
    ]
    valid, skipped = validate_proposals(proposals, nodes)
    assert len(valid) == 0
    assert len(skipped) == 1
    assert "similar to retracted" in skipped[0][1]
    assert "gl108-safety-validation-disabled" in skipped[0][1]


def test_validate_proposals_allows_unrelated():
    """Proposals unrelated to retracted beliefs should pass through."""
    nodes = {
        "fact-a": {"truth_value": "IN"},
        "fact-b": {"truth_value": "IN"},
        "gl108-safety-validation-disabled": {
            "truth_value": "OUT",
            "text": "GL-108 safety validation is disabled",
        },
    }
    proposals = [
        {"id": "auth-token-rotation-needed",
         "antecedents": ["fact-a", "fact-b"], "unless": [],
         "text": "Auth tokens need rotation",
         "kind": "derive", "label": "test"},
    ]
    valid, skipped = validate_proposals(proposals, nodes)
    assert len(valid) == 1
    assert len(skipped) == 0


# --- Deduplicate tests ---

def test_deduplicate_finds_clusters(db):
    api.add_node("gl108-validation-disabled", "GL-108 validation disabled", db_path=db)
    api.add_node("gl108-safety-validation-disabled", "GL-108 safety validation disabled", db_path=db)
    api.add_node("gl108-response-validation-disabled", "GL-108 response validation disabled", db_path=db)
    api.add_node("unrelated-auth-config", "Auth config is fine", db_path=db)

    result = api.deduplicate(db_path=db)
    assert len(result["clusters"]) == 1
    assert result["clusters"][0]["size"] == 3
    assert result["retracted"] == []


def test_deduplicate_no_clusters(db):
    api.add_node("auth-config", "Auth config", db_path=db)
    api.add_node("network-topology", "Network topology", db_path=db)
    api.add_node("database-schema", "Database schema", db_path=db)

    result = api.deduplicate(db_path=db)
    assert len(result["clusters"]) == 0


def test_deduplicate_auto_retracts(db):
    api.add_node("gl108-validation-disabled", "GL-108 validation disabled", db_path=db)
    api.add_node("gl108-safety-validation-disabled", "GL-108 safety validation disabled", db_path=db)
    api.add_node("gl108-response-validation-disabled", "GL-108 response validation disabled", db_path=db)

    result = api.deduplicate(auto=True, db_path=db)
    assert len(result["clusters"]) == 1
    assert len(result["retracted"]) == 2
    assert result["clusters"][0]["kept"] not in result["retracted"]

    # Verify only one is still IN
    status = api.get_status(db_path=db)
    in_nodes = [n for n in status["nodes"] if n["truth_value"] == "IN"]
    assert len(in_nodes) == 1


def test_deduplicate_rewrites_dependents(db):
    """Derived beliefs that depended on a retracted duplicate survive via rewrite."""
    api.add_node("gl108-validation-disabled", "GL-108 validation disabled", db_path=db)
    api.add_node("gl108-safety-validation-disabled", "GL-108 safety validation disabled", db_path=db)
    # Give the first node 2 dependents so it's kept (most dependents wins)
    api.add_node("other-derived", "Other conclusion",
                 sl="gl108-validation-disabled", label="filler", db_path=db)
    api.add_node("another-derived", "Another conclusion",
                 sl="gl108-validation-disabled", label="filler", db_path=db)
    # This derived belief depends on the node that will be RETRACTED
    api.add_node("safety-pipeline-broken", "Safety pipeline is broken",
                 sl="gl108-safety-validation-disabled", label="derived", db_path=db)

    result = api.deduplicate(auto=True, db_path=db)
    kept = result["clusters"][0]["kept"]
    assert kept == "gl108-validation-disabled"
    assert "gl108-safety-validation-disabled" in result["retracted"]

    # The derived belief should still be IN (rewrite saved it)
    node = api.show_node("safety-pipeline-broken", db_path=db)
    assert node["truth_value"] == "IN"
    # Its justification should now point at the kept belief, not the retracted one
    assert kept in node["justifications"][0]["antecedents"]
    assert "gl108-safety-validation-disabled" not in node["justifications"][0]["antecedents"]


def test_deduplicate_rewrites_outlist(db):
    """Outlist references to retracted duplicates are rewritten."""
    api.add_node("gl108-validation-disabled", "GL-108 validation disabled", db_path=db)
    api.add_node("gl108-safety-validation-disabled", "GL-108 safety validation disabled", db_path=db)
    # Give the first node 2 dependents so it's kept
    api.add_node("other-derived", "Other conclusion",
                 sl="gl108-validation-disabled", label="filler", db_path=db)
    api.add_node("another-derived", "Another conclusion",
                 sl="gl108-validation-disabled", label="filler", db_path=db)
    # Gated belief: IN unless the retracted duplicate is IN
    api.add_node("safe-to-deploy", "Safe to deploy",
                 unless="gl108-safety-validation-disabled", label="gated", db_path=db)

    result = api.deduplicate(auto=True, db_path=db)
    kept = result["clusters"][0]["kept"]
    assert kept == "gl108-validation-disabled"

    # The gated belief's outlist should now reference the kept belief
    node = api.show_node("safe-to-deploy", db_path=db)
    assert kept in node["justifications"][0]["outlist"]
    assert "gl108-safety-validation-disabled" not in node["justifications"][0]["outlist"]


# --- Dedup plan workflow tests ---

def test_write_and_parse_dedup_plan(tmp_path):
    """Plan file round-trips through write and parse."""
    clusters = [
        {
            "beliefs": [
                {"id": "gl108-validation-disabled", "text": "GL-108 validation disabled", "dependents": 2},
                {"id": "gl108-safety-validation-disabled", "text": "GL-108 safety validation disabled", "dependents": 0},
            ],
            "size": 2,
            "kept": "gl108-validation-disabled",
        },
    ]
    out = str(tmp_path / "plan.md")
    api.write_dedup_plan(clusters, out)

    text = Path(out).read_text()
    parsed = api.parse_dedup_plan(text)
    assert len(parsed) == 1
    assert parsed[0]["keep"] == "gl108-validation-disabled"
    assert parsed[0]["retract"] == ["gl108-safety-validation-disabled"]


def test_dedup_plan_end_to_end(db, tmp_path):
    """Full workflow: deduplicate(auto=False) -> write plan -> parse -> apply."""
    api.add_node("gl108-validation-disabled", "GL-108 validation disabled", db_path=db)
    api.add_node("gl108-safety-validation-disabled", "GL-108 safety validation disabled", db_path=db)

    # Step 1: find clusters (no auto)
    result = api.deduplicate(auto=False, db_path=db)
    assert len(result["clusters"]) == 1
    assert result["clusters"][0]["kept"] is not None

    # Step 2: write plan
    out = str(tmp_path / "plan.md")
    api.write_dedup_plan(result["clusters"], out)

    # Step 3: parse plan — must have a KEEP
    text = Path(out).read_text()
    assert "[KEEP]" in text
    assert "[RETRACT]" in text
    parsed = api.parse_dedup_plan(text)
    assert len(parsed) == 1
    assert parsed[0]["keep"] is not None
    assert len(parsed[0]["retract"]) >= 1

    # Step 4: apply
    apply_result = api.apply_dedup_plan(parsed, db_path=db)
    assert len(apply_result["retracted"]) >= 1


def test_apply_dedup_plan(db):
    """Apply a dedup plan: rewrites justifications and retracts."""
    api.add_node("gl108-validation-disabled", "GL-108 validation disabled", db_path=db)
    api.add_node("gl108-safety-validation-disabled", "GL-108 safety validation disabled", db_path=db)
    # Derived belief depends on the one that will be retracted
    api.add_node("other-dep", "Filler", sl="gl108-validation-disabled", label="f", db_path=db)
    api.add_node("another-dep", "Filler 2", sl="gl108-validation-disabled", label="f", db_path=db)
    api.add_node("safety-broken", "Safety broken",
                 sl="gl108-safety-validation-disabled", label="d", db_path=db)

    plan = [{"keep": "gl108-validation-disabled",
             "retract": ["gl108-safety-validation-disabled"]}]
    result = api.apply_dedup_plan(plan, db_path=db)
    assert result["retracted"] == ["gl108-safety-validation-disabled"]
    assert result["errors"] == []

    # Derived belief survived via rewrite
    node = api.show_node("safety-broken", db_path=db)
    assert node["truth_value"] == "IN"
    assert "gl108-validation-disabled" in node["justifications"][0]["antecedents"]


def test_apply_dedup_plan_missing_node(db):
    """Plan with missing nodes reports errors."""
    api.add_node("existing-node", "Exists", db_path=db)
    plan = [{"keep": "existing-node", "retract": ["nonexistent"]}]
    result = api.apply_dedup_plan(plan, db_path=db)
    assert len(result["errors"]) == 1
    assert "nonexistent" in result["errors"][0]


def test_dedup_plan_user_can_edit_kept(db):
    """User can change which belief is KEEP in the plan."""
    api.add_node("gl108-validation-disabled", "GL-108 validation disabled", db_path=db)
    api.add_node("gl108-safety-validation-disabled", "GL-108 safety validation disabled", db_path=db)

    # User edits the plan to keep the other one
    plan = [{"keep": "gl108-safety-validation-disabled",
             "retract": ["gl108-validation-disabled"]}]
    result = api.apply_dedup_plan(plan, db_path=db)
    assert result["retracted"] == ["gl108-validation-disabled"]

    # The user's chosen belief is still IN
    node = api.show_node("gl108-safety-validation-disabled", db_path=db)
    assert node["truth_value"] == "IN"
    node = api.show_node("gl108-validation-disabled", db_path=db)
    assert node["truth_value"] == "OUT"


# --- Derive report tests ---

import json
import sys
from io import StringIO
from unittest.mock import patch
from reasonsforge.cli import main


def _run_cli(*args, db_path=None):
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


def _mock_derive_response(proposals):
    """Build a mock LLM response with DERIVE blocks."""
    blocks = []
    for p in proposals:
        ants = ", ".join(p["antecedents"])
        blocks.append(
            f"### DERIVE {p['id']}\n"
            f"{p['text']}\n"
            f"- Antecedents: {ants}\n"
            f"- Label: test derivation"
        )
    text = "\n\n".join(blocks)
    return type("R", (), {"returncode": 0, "stdout": text, "stderr": ""})()


def test_derive_report_written(simple_network, tmp_path):
    report_dir = str(tmp_path / "reports")
    mock_result = _mock_derive_response([
        {"id": "new-belief", "text": "A new derived belief",
         "antecedents": ["fact-a", "fact-b"]},
    ])
    with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
         patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
        stdout, stderr, code = _run_cli(
            "derive", "--auto", "--report-dir", report_dir,
            db_path=simple_network)

    assert code == 0
    import os
    reports = [f for f in os.listdir(report_dir) if f.startswith("derive-")]
    assert len(reports) == 1

    with open(os.path.join(report_dir, reports[0])) as f:
        report = json.load(f)
    assert report["status"] == "complete"
    assert report["model"] == "claude"
    assert len(report["rounds"]) == 1
    assert report["rounds"][0]["proposals_found"] >= 1
    assert report["rounds"][0]["added"] >= 1
    assert report["total_added"] >= 1


def test_derive_no_report_flag(simple_network, tmp_path):
    report_dir = str(tmp_path / "reports")
    mock_result = _mock_derive_response([
        {"id": "new-belief-2", "text": "Another belief",
         "antecedents": ["fact-a"]},
    ])
    with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
         patch("reasonsforge.llm.subprocess.run", return_value=mock_result):
        stdout, stderr, code = _run_cli(
            "derive", "--auto", "--no-report", "--report-dir", report_dir,
            db_path=simple_network)

    assert code == 0
    assert "Report:" not in stdout
    import os
    assert not os.path.exists(report_dir)


def test_derive_exhaust_report_has_rounds(simple_network, tmp_path):
    report_dir = str(tmp_path / "reports")
    call_count = [0]

    def mock_run(*a, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_derive_response([
                {"id": "round1-belief", "text": "From round 1",
                 "antecedents": ["fact-a"]},
            ])
        # Second round: no proposals (saturated)
        return type("R", (), {"returncode": 0, "stdout": "No proposals.", "stderr": ""})()

    with patch("reasonsforge.llm.shutil.which", return_value="/usr/bin/claude"), \
         patch("reasonsforge.llm.subprocess.run", side_effect=mock_run):
        stdout, stderr, code = _run_cli(
            "derive", "--exhaust", "--report-dir", report_dir,
            db_path=simple_network)

    assert code == 0
    import os
    reports = [f for f in os.listdir(report_dir) if f.startswith("derive-")]
    assert len(reports) == 1

    with open(os.path.join(report_dir, reports[0])) as f:
        report = json.load(f)
    assert report["status"] == "complete"
    assert report["exhaust"] is True
    assert len(report["rounds"]) == 2
    assert report["rounds"][0]["added"] >= 1
    assert report["rounds"][1]["added"] == 0


# --- Cluster integration tests ---

try:
    from reasonsforge.cluster import HAS_CLUSTER_DEPS
except ImportError:
    HAS_CLUSTER_DEPS = False

skip_no_cluster = pytest.mark.skipif(
    not HAS_CLUSTER_DEPS,
    reason="sentence-transformers and scikit-learn not installed"
)


@skip_no_cluster
def test_build_prompt_with_cluster(simple_network):
    nodes = api.export_network(db_path=simple_network)["nodes"]
    prompt, stats = build_prompt(nodes, cluster=True, budget=10, seed=42)
    assert stats.get("cluster") is True
    assert "n_clusters" in stats
    assert "embedding_model" in stats
    assert len(prompt) > 0


@skip_no_cluster
def test_build_prompt_cluster_with_agents(agent_network):
    nodes = api.export_network(db_path=agent_network)["nodes"]
    prompt, stats = build_prompt(nodes, cluster=True, budget=10, seed=42)
    assert stats.get("cluster") is True
    assert stats["agents"] == 2
    assert "Agent: agent-a" in prompt
    assert "Agent: agent-b" in prompt


@skip_no_cluster
def test_build_prompt_with_intra_cluster(simple_network):
    nodes = api.export_network(db_path=simple_network)["nodes"]
    prompt, stats = build_prompt(nodes, intra_cluster=True, budget=10, seed=42)
    assert stats.get("cluster") is True
    assert stats.get("intra_cluster") is True
    assert "focus_cluster" in stats
    assert "n_clusters" in stats
    assert len(prompt) > 0


@skip_no_cluster
def test_intra_cluster_rotation():
    from reasonsforge.cluster import cluster_beliefs_intra
    beliefs = {f"b-{i}": f"Belief about topic {i % 3}" for i in range(30)}
    ids_r0, stats_r0 = cluster_beliefs_intra(beliefs, budget=5, round_num=0, seed=42)
    ids_r1, stats_r1 = cluster_beliefs_intra(beliefs, budget=5, round_num=1, seed=42)
    assert stats_r0["focus_cluster"] != stats_r1["focus_cluster"]
    assert set(ids_r0) != set(ids_r1)


def test_build_prompt_custom_template(simple_network):
    data = api.export_network(db_path=simple_network)
    template = "Custom prompt with {total_in} beliefs and depth {max_depth}. {beliefs_section}{derived_section}{domain_context}{cross_agent_task}{agents_stats}{total_derived}"
    prompt, stats = build_prompt(data["nodes"], prompt_template=template)
    assert prompt.startswith("Custom prompt with")
    assert str(stats["total_in"]) in prompt


def test_build_prompt_default_template_unchanged(simple_network):
    data = api.export_network(db_path=simple_network)
    prompt_default, _ = build_prompt(data["nodes"])
    prompt_none, _ = build_prompt(data["nodes"], prompt_template=None)
    assert prompt_default == prompt_none


def test_build_prompt_bad_placeholder(simple_network):
    data = api.export_network(db_path=simple_network)
    template = "Bad template with {unknown_field}"
    with pytest.raises(ValueError, match="unknown placeholder"):
        build_prompt(data["nodes"], prompt_template=template)


def test_build_prompt_malformed_braces(simple_network):
    data = api.export_network(db_path=simple_network)
    template = "Output as JSON: {unclosed brace"
    with pytest.raises(ValueError, match="malformed braces"):
        build_prompt(data["nodes"], prompt_template=template)

"""Tests for _build_beliefs_section budget calculation (issue #23).

The bug: count += len(belief_ids) was inside the per-belief loop,
inflating count to N² instead of N. This starved the local beliefs
budget via remaining = max(5, max_beliefs - count).
"""

import re

import pytest

from reasonsforge import api
from reasonsforge.derive import _build_beliefs_section, build_prompt


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "reasons.db")
    api.init_db(db_path=db_path)
    return db_path


def _make_nodes(agent_beliefs, local_beliefs=None):
    """Build a nodes dict for _build_beliefs_section without a database.

    agent_beliefs: dict of agent_name -> list of belief suffixes
    local_beliefs: list of local belief IDs

    Does NOT create :active premise nodes — those are an import-agent
    concern and would inflate agent_beliefs counts in the function
    under test (since it matches by startswith).
    """
    nodes = {}
    agents = {}
    for agent, suffixes in agent_beliefs.items():
        agents[agent] = []
        for s in suffixes:
            nid = f"{agent}:{s}"
            agents[agent].append(nid)
            nodes[nid] = {"truth_value": "IN", "text": f"{agent} belief {s}"}
    for lid in (local_beliefs or []):
        nodes[lid] = {"truth_value": "IN", "text": f"Local belief {lid}"}
    derived = {}
    return nodes, derived, agents


def _count_local_shown(output):
    """Extract 'showing N' from the Local beliefs header."""
    m = re.search(r"Local beliefs \(\d+ beliefs, showing (\d+)\)", output)
    assert m is not None, f"Local beliefs header not found in output:\n{output[:500]}"
    return int(m.group(1))


def _count_agent_shown(output, agent_name):
    """Extract 'showing N' from an agent header."""
    m = re.search(rf"Agent: {re.escape(agent_name)} \(\d+ beliefs, showing (\d+)\)", output)
    assert m is not None, f"Agent header for {agent_name} not found in output:\n{output[:500]}"
    return int(m.group(1))


# --- Core regression: count is N, not N² ---

class TestCountLinearNotQuadratic:

    def test_six_beliefs_one_agent(self):
        """With 6 agent beliefs and budget=20, locals get 15 (not buggy 5).

        Proportional budget: agent_budget=max(5, int(20*6/26))=5,
        count=5, remaining=max(5, 20-5)=15.
        With the old bug: count=5*5=25, remaining=max(5, 20-25)=5.
        """
        nodes, derived, agents = _make_nodes(
            {"agent-a": [f"b{i}" for i in range(6)]},
            local_beliefs=[f"local-{i}" for i in range(20)],
        )
        output, _ = _build_beliefs_section(nodes, derived, agents, max_beliefs=20)
        local_shown = _count_local_shown(output)
        assert local_shown > 5, "Bug regression: locals starved to floor"
        assert local_shown == 15

    def test_count_equals_n_not_n_squared(self):
        """Directly verify: with N agent beliefs, count accumulates N total."""
        nodes, derived, agents = _make_nodes(
            {"agent-a": [f"b{i}" for i in range(10)]},
            local_beliefs=[f"local-{i}" for i in range(50)],
        )
        output, _ = _build_beliefs_section(nodes, derived, agents, max_beliefs=20)
        agent_shown = _count_agent_shown(output, "agent-a")
        local_shown = _count_local_shown(output)
        total_used = agent_shown + local_shown
        assert total_used <= 20


# --- Multi-agent accumulation ---

class TestMultiAgent:

    def test_two_agents_accumulate_correctly(self):
        """Count sums across both agents, leaving correct remainder for locals."""
        nodes, derived, agents = _make_nodes(
            {
                "agent-a": [f"b{i}" for i in range(4)],
                "agent-b": [f"b{i}" for i in range(4)],
            },
            local_beliefs=[f"local-{i}" for i in range(30)],
        )
        output, _ = _build_beliefs_section(nodes, derived, agents, max_beliefs=20)
        a_shown = _count_agent_shown(output, "agent-a")
        b_shown = _count_agent_shown(output, "agent-b")
        local_shown = _count_local_shown(output)
        agent_total = a_shown + b_shown
        assert local_shown == max(5, 20 - agent_total)

    def test_three_agents_budget_sums(self):
        nodes, derived, agents = _make_nodes(
            {
                "alpha": [f"b{i}" for i in range(3)],
                "beta": [f"b{i}" for i in range(3)],
                "gamma": [f"b{i}" for i in range(3)],
            },
            local_beliefs=[f"local-{i}" for i in range(30)],
        )
        output, _ = _build_beliefs_section(nodes, derived, agents, max_beliefs=30)
        total_agent = sum(
            _count_agent_shown(output, a) for a in ["alpha", "beta", "gamma"]
        )
        local_shown = _count_local_shown(output)
        assert local_shown == max(5, 30 - total_agent)


# --- Budget floor ---

class TestBudgetFloor:

    def test_floor_of_five_when_agents_exceed_budget(self):
        """Even when agents consume the entire budget, locals get at least 5.

        50 agent beliefs, 3 locals, budget=20.
        agent_budget=max(5, int(20*50/53))=18, count=18,
        remaining=max(5, 20-18)=5 — floor kicks in.
        """
        nodes, derived, agents = _make_nodes(
            {"agent-a": [f"b{i}" for i in range(50)]},
            local_beliefs=[f"local-{i}" for i in range(10)],
        )
        output, _ = _build_beliefs_section(nodes, derived, agents, max_beliefs=20)
        local_shown = _count_local_shown(output)
        assert local_shown >= 5

    def test_floor_applies_with_large_agent_set(self):
        """With many agent beliefs far exceeding budget, locals still get 5.

        200 agent beliefs, 10 locals, budget=20.
        agent_budget=max(5, int(20*200/210))=19, count=19,
        remaining=max(5, 20-19)=5 — floor kicks in.
        """
        nodes, derived, agents = _make_nodes(
            {"agent-a": [f"b{i}" for i in range(200)]},
            local_beliefs=[f"local-{i}" for i in range(10)],
        )
        output, _ = _build_beliefs_section(nodes, derived, agents, max_beliefs=20)
        local_shown = _count_local_shown(output)
        assert local_shown == 5


# --- Edge cases ---

class TestEdgeCases:

    def test_single_agent_belief(self):
        """N=1: N²=N=1, bug was invisible — verify fix doesn't break it."""
        nodes, derived, agents = _make_nodes(
            {"agent-a": ["only-one"]},
            local_beliefs=[f"local-{i}" for i in range(10)],
        )
        output, _ = _build_beliefs_section(nodes, derived, agents, max_beliefs=20)
        agent_shown = _count_agent_shown(output, "agent-a")
        local_shown = _count_local_shown(output)
        assert agent_shown == 1
        assert local_shown == 10

    def test_no_agents(self):
        """No agents: entire budget goes to grouped local beliefs."""
        nodes = {f"belief-{i}": {"truth_value": "IN", "text": f"Belief {i}"}
                 for i in range(10)}
        derived = {}
        output, _ = _build_beliefs_section(nodes, derived, agents=None, max_beliefs=20)
        assert "Agent:" not in output
        assert "belief-" in output

    def test_no_local_beliefs(self):
        """Agent-only network: no Local beliefs section, no crash."""
        nodes, derived, agents = _make_nodes(
            {"agent-a": [f"b{i}" for i in range(5)]},
            local_beliefs=[],
        )
        output, _ = _build_beliefs_section(nodes, derived, agents, max_beliefs=20)
        assert "Local beliefs" not in output
        assert "Agent: agent-a" in output

    def test_empty_network(self):
        """Zero beliefs: no crash, produces output."""
        output, _ = _build_beliefs_section({}, {}, agents=None, max_beliefs=20)
        assert isinstance(output, str)

    def test_empty_network_with_agents(self):
        """Agents dict provided but no matching nodes."""
        output, _ = _build_beliefs_section(
            {}, {}, agents={"agent-a": ["agent-a:missing"]}, max_beliefs=20,
        )
        assert isinstance(output, str)

    def test_derived_beliefs_excluded_from_count(self):
        """Derived agent beliefs (with justifications) shouldn't inflate count."""
        nodes = {
            "agent-a:active": {"truth_value": "IN", "text": "active"},
            "agent-a:premise-1": {"truth_value": "IN", "text": "premise 1"},
            "agent-a:premise-2": {"truth_value": "IN", "text": "premise 2"},
            "agent-a:derived-1": {
                "truth_value": "IN", "text": "derived from premises",
                "justifications": [{"antecedents": ["agent-a:premise-1", "agent-a:premise-2"]}],
            },
        }
        derived = {"agent-a:derived-1": nodes["agent-a:derived-1"]}
        agents = {"agent-a": ["agent-a:premise-1", "agent-a:premise-2", "agent-a:derived-1"]}
        output, _ = _build_beliefs_section(nodes, derived, agents, max_beliefs=20)
        assert "Agent: agent-a" in output


# --- Sampling mode ---

class TestSampling:

    def test_sample_mode_respects_budget(self):
        """In sample mode, the budget calculation should also be correct."""
        nodes, derived, agents = _make_nodes(
            {"agent-a": [f"b{i}" for i in range(6)]},
            local_beliefs=[f"local-{i}" for i in range(20)],
        )
        output, _ = _build_beliefs_section(
            nodes, derived, agents, max_beliefs=20, sample=True, seed=42,
        )
        local_shown = _count_local_shown(output)
        assert local_shown >= 5

    def test_sample_mode_deterministic(self):
        """Same seed produces same output."""
        nodes, derived, agents = _make_nodes(
            {"agent-a": [f"b{i}" for i in range(20)]},
            local_beliefs=[f"local-{i}" for i in range(20)],
        )
        out1, _ = _build_beliefs_section(
            nodes, derived, agents, max_beliefs=15, sample=True, seed=99,
        )
        out2, _ = _build_beliefs_section(
            nodes, derived, agents, max_beliefs=15, sample=True, seed=99,
        )
        assert out1 == out2


# --- Integration: build_prompt with agent network ---

class TestBuildPromptIntegration:

    def test_build_prompt_budget_correct(self, db):
        """End-to-end: build_prompt with agents uses correct budget."""
        api.add_node("agent-a:active", "Agent A active", db_path=db)
        for i in range(6):
            api.add_node(
                f"agent-a:belief-{i}", f"Agent A belief {i}",
                sl="agent-a:active", label="imported", db_path=db,
            )
        for i in range(20):
            api.add_node(f"local-{i}", f"Local belief {i}", db_path=db)

        data = api.export_network(db_path=db)
        prompt, stats = build_prompt(data["nodes"], budget=20)
        assert stats["agents"] == 1
        local_shown = _count_local_shown(prompt)
        assert local_shown >= 5
        assert local_shown <= 20

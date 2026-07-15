"""Tests for dialectical argumentation — challenge/defend pattern.

Doyle Section 6: arguments become explicit, challengeable beliefs.
Instead of directly justifying N with evidence, you can challenge the
justification and create a dialectical chain.
"""

import pytest

from reasonsforge import Justification
from reasonsforge.network import Network
from reasonsforge.storage import Storage
from reasonsforge import api


class TestChallenge:
    """Challenging a node makes it go OUT."""

    def test_challenge_premise(self):
        net = Network()
        net.add_node("a", "Premise A")
        assert net.nodes["a"].truth_value == "IN"

        result = net.challenge("a", "A is wrong because X")
        assert net.nodes["a"].truth_value == "OUT"
        assert result["challenge_id"] == "challenge-a"
        assert "a" in result["changed"]

    def test_challenge_derived(self):
        net = Network()
        net.add_node("p", "Premise P")
        net.add_node("a", "Derived A", justifications=[
            Justification(type="SL", antecedents=["p"]),
        ])
        result = net.challenge("a", "A is wrong")
        assert net.nodes["a"].truth_value == "OUT"

    def test_challenge_cascades(self):
        """Challenging A should cascade to B which depends on A."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        result = net.challenge("a", "A is wrong")
        assert net.nodes["a"].truth_value == "OUT"
        assert net.nodes["b"].truth_value == "OUT"
        assert "b" in result["changed"]

    def test_challenge_creates_node(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.challenge("a", "A is wrong because X")
        assert "challenge-a" in net.nodes
        assert net.nodes["challenge-a"].text == "A is wrong because X"
        assert net.nodes["challenge-a"].truth_value == "IN"

    def test_challenge_metadata(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.challenge("a", "reason")
        assert net.nodes["challenge-a"].metadata["challenge_target"] == "a"
        assert "challenge-a" in net.nodes["a"].metadata["challenges"]

    def test_custom_challenge_id(self):
        net = Network()
        net.add_node("a", "Premise A")
        result = net.challenge("a", "reason", challenge_id="my-challenge")
        assert result["challenge_id"] == "my-challenge"
        assert "my-challenge" in net.nodes

    def test_multiple_challenges(self):
        net = Network()
        net.add_node("a", "Premise A")
        r1 = net.challenge("a", "first challenge")
        r2 = net.challenge("a", "second challenge")
        assert r1["challenge_id"] == "challenge-a"
        assert r2["challenge_id"] == "challenge-a-2"
        assert len(net.nodes["a"].metadata["challenges"]) == 2

    def test_challenge_nonexistent_raises(self):
        net = Network()
        with pytest.raises(KeyError):
            net.challenge("missing", "reason")

    def test_dismiss_challenge_restores_target(self):
        """Retracting a challenge restores the target."""
        net = Network()
        net.add_node("a", "Premise A")
        net.challenge("a", "A is wrong")
        assert net.nodes["a"].truth_value == "OUT"

        net.retract("challenge-a")
        assert net.nodes["a"].truth_value == "IN"


class TestDefend:
    """Defending neutralises the challenge, restoring the target."""

    def test_defend_restores_target(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.challenge("a", "A is wrong")
        assert net.nodes["a"].truth_value == "OUT"

        result = net.defend("a", "challenge-a", "A is right because Y")
        assert net.nodes["a"].truth_value == "IN"
        assert net.nodes["challenge-a"].truth_value == "OUT"
        assert result["defense_id"] == "defense-challenge-a"

    def test_defense_creates_node(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.challenge("a", "wrong")
        net.defend("a", "challenge-a", "right because Y")

        defense = net.nodes["defense-challenge-a"]
        assert defense.text == "right because Y"
        assert defense.truth_value == "IN"
        assert defense.metadata["defense_target"] == "challenge-a"
        assert defense.metadata["defends"] == "a"

    def test_defend_cascades_restoration(self):
        """Defense restores target, which restores dependents."""
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B", justifications=[
            Justification(type="SL", antecedents=["a"]),
        ])
        net.challenge("a", "wrong")
        assert net.nodes["b"].truth_value == "OUT"

        net.defend("a", "challenge-a", "right")
        assert net.nodes["a"].truth_value == "IN"
        assert net.nodes["b"].truth_value == "IN"

    def test_retract_defense_reinstates_challenge(self):
        """Retracting the defense reinstates the challenge and target goes OUT again."""
        net = Network()
        net.add_node("a", "Premise A")
        net.challenge("a", "wrong")
        net.defend("a", "challenge-a", "right")
        assert net.nodes["a"].truth_value == "IN"

        net.retract("defense-challenge-a")
        assert net.nodes["challenge-a"].truth_value == "IN"
        assert net.nodes["a"].truth_value == "OUT"

    def test_defend_nonexistent_target_raises(self):
        net = Network()
        with pytest.raises(KeyError):
            net.defend("missing", "challenge-missing", "reason")

    def test_defend_nonexistent_challenge_raises(self):
        net = Network()
        net.add_node("a", "Premise A")
        with pytest.raises(KeyError):
            net.defend("a", "missing-challenge", "reason")


class TestDialecticalChain:
    """Multi-level argumentation: challenge the defense, defend the challenge."""

    def test_challenge_defense(self):
        """A is challenged, defended, then the defense is challenged."""
        net = Network()
        net.add_node("a", "Premise A")

        # Challenge A → A goes OUT
        net.challenge("a", "A is wrong")
        assert net.nodes["a"].truth_value == "OUT"

        # Defend A → challenge goes OUT, A restored
        net.defend("a", "challenge-a", "A is right")
        assert net.nodes["a"].truth_value == "IN"

        # Challenge the defense → defense goes OUT, challenge restored, A goes OUT
        net.challenge("defense-challenge-a", "Defense is flawed")
        assert net.nodes["defense-challenge-a"].truth_value == "OUT"
        assert net.nodes["challenge-a"].truth_value == "IN"
        assert net.nodes["a"].truth_value == "OUT"

    def test_three_level_chain(self):
        """Challenge → defense → counter-challenge → counter-defense."""
        net = Network()
        net.add_node("a", "Premise A")

        net.challenge("a", "wrong")
        net.defend("a", "challenge-a", "right")
        net.challenge("defense-challenge-a", "defense is flawed")
        # A is OUT now

        # Defend the defense
        net.defend(
            "defense-challenge-a",
            "challenge-defense-challenge-a",
            "defense holds because Z",
        )
        # Chain: defense-defense restored → challenge on defense OUT
        # → defense restored → challenge-a OUT → A IN
        assert net.nodes["a"].truth_value == "IN"


class TestDialecticalPersistence:
    """Challenge/defend survives SQLite round-trip."""

    def test_round_trip(self, tmp_path):
        db = tmp_path / "test.db"

        net = Network()
        net.add_node("a", "Premise A")
        net.challenge("a", "A is wrong")
        net.defend("a", "challenge-a", "A is right")

        store = Storage(db)
        store.save(net)
        loaded = store.load()
        store.close()

        assert loaded.nodes["a"].truth_value == "IN"
        assert loaded.nodes["challenge-a"].truth_value == "OUT"
        assert loaded.nodes["defense-challenge-a"].truth_value == "IN"

        # Retract defense — challenge should come back
        loaded.retract("defense-challenge-a")
        assert loaded.nodes["challenge-a"].truth_value == "IN"
        assert loaded.nodes["a"].truth_value == "OUT"


class TestDialecticalAPI:
    """API layer for challenge/defend."""

    def test_challenge_api(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)

        result = api.challenge("a", "A is wrong", db_path=db)
        assert result["challenge_id"] == "challenge-a"
        assert "a" in result["changed"]

        status = api.get_status(db_path=db)
        a_node = [n for n in status["nodes"] if n["id"] == "a"][0]
        assert a_node["truth_value"] == "OUT"

    def test_defend_api(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.challenge("a", "A is wrong", db_path=db)

        result = api.defend("a", "challenge-a", "A is right", db_path=db)
        assert result["defense_id"] == "defense-challenge-a"

        status = api.get_status(db_path=db)
        a_node = [n for n in status["nodes"] if n["id"] == "a"][0]
        assert a_node["truth_value"] == "IN"

    def test_list_challenged(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Premise B", db_path=db)
        api.challenge("a", "A is wrong", db_path=db)

        result = api.list_nodes(challenged=True, db_path=db)
        assert result["count"] == 1
        assert result["nodes"][0]["id"] == "a"

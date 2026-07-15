"""Tests for --any flag (#20), 3+ premise warning (#19), and restoration hints (#21)."""

import pytest

from reasonsforge import api


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "reasons.db")
    api.init_db(db_path=db_path)
    return db_path


class TestAnyModeAdd:
    """#20 — --any flag on add expands SL into per-premise justifications."""

    def test_any_creates_separate_justifications(self, db):
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("c", "C", db_path=db)
        api.add_node("x", "Conclusion", sl="a,b,c", any_mode=True, db_path=db)

        node = api.show_node("x", db_path=db)
        assert len(node["justifications"]) == 3
        for j in node["justifications"]:
            assert len(j["antecedents"]) == 1

    def test_any_survives_single_retraction(self, db):
        """With --any, retracting one premise leaves node IN via the others."""
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("c", "C", db_path=db)
        api.add_node("x", "Conclusion", sl="a,b,c", any_mode=True, db_path=db)

        api.retract_node("a", db_path=db)
        node = api.show_node("x", db_path=db)
        assert node["truth_value"] == "IN"

    def test_any_goes_out_when_all_retracted(self, db):
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("x", "Conclusion", sl="a,b", any_mode=True, db_path=db)

        api.retract_node("a", db_path=db)
        api.retract_node("b", db_path=db)
        node = api.show_node("x", db_path=db)
        assert node["truth_value"] == "OUT"

    def test_any_with_single_premise_no_expansion(self, db):
        """--any with a single premise behaves normally."""
        api.add_node("a", "A", db_path=db)
        api.add_node("x", "Conclusion", sl="a", any_mode=True, db_path=db)

        node = api.show_node("x", db_path=db)
        assert len(node["justifications"]) == 1

    def test_any_preserves_outlist(self, db):
        """Each expanded justification gets the outlist."""
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("enemy", "Enemy", db_path=db)
        api.add_node("x", "Conclusion", sl="a,b", unless="enemy", any_mode=True, db_path=db)

        node = api.show_node("x", db_path=db)
        assert len(node["justifications"]) == 2
        for j in node["justifications"]:
            assert "enemy" in j["outlist"]

    def test_without_any_single_justification(self, db):
        """Without --any, multiple premises create one conjunctive SL."""
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("c", "C", db_path=db)
        api.add_node("x", "Conclusion", sl="a,b,c", db_path=db)

        node = api.show_node("x", db_path=db)
        assert len(node["justifications"]) == 1
        assert len(node["justifications"][0]["antecedents"]) == 3


class TestAnyModeAddJustification:
    """#20 — --any flag on add-justification."""

    def test_any_expands_add_justification(self, db):
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("c", "C", db_path=db)
        api.add_node("x", "Conclusion", sl="a", db_path=db)

        api.add_justification("x", sl="b,c", any_mode=True, db_path=db)

        node = api.show_node("x", db_path=db)
        # Original SL(a) + SL(b) + SL(c) = 3
        assert len(node["justifications"]) == 3

    def test_any_add_justification_restores_out_node(self, db):
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("c", "C", db_path=db)
        api.add_node("x", "Conclusion", sl="a", db_path=db)
        api.retract_node("a", db_path=db)

        node = api.show_node("x", db_path=db)
        assert node["truth_value"] == "OUT"

        result = api.add_justification("x", sl="b,c", any_mode=True, db_path=db)
        assert result["new_truth_value"] == "IN"


class TestPremiseCountWarning:
    """#19 — premise_count in return value enables CLI warning."""

    def test_add_returns_premise_count(self, db):
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("c", "C", db_path=db)

        result = api.add_node("x", "Conclusion", sl="a,b,c", db_path=db)
        assert result["premise_count"] == 3

    def test_add_any_returns_low_premise_count(self, db):
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("c", "C", db_path=db)

        result = api.add_node("x", "Conclusion", sl="a,b,c", any_mode=True, db_path=db)
        # With --any, each justification has 1 premise
        assert result["premise_count"] == 1

    def test_add_justification_returns_premise_count(self, db):
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("c", "C", db_path=db)
        api.add_node("x", "Conclusion", db_path=db)

        result = api.add_justification("x", sl="a,b,c", db_path=db)
        assert result["premise_count"] == 3

    def test_premise_returns_zero_count(self, db):
        result = api.add_node("p", "Premise", db_path=db)
        assert result["premise_count"] == 0


class TestRestorationHints:
    """#21 — retraction cascade shows hints for multi-premise SL nodes."""

    def test_hint_when_multi_premise_cascades(self, db):
        """Retracting one premise of a 3-premise SL produces a hint."""
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("c", "C", db_path=db)
        api.add_node("x", "Conclusion", sl="a,b,c", db_path=db)

        result = api.retract_node("a", db_path=db)

        assert len(result["restoration_hints"]) == 1
        hint = result["restoration_hints"][0]
        assert hint["node_id"] == "x"
        assert "a" in hint["all_premises"]
        assert "b" in hint["surviving_premises"]
        assert "c" in hint["surviving_premises"]
        assert "a" not in hint["surviving_premises"]

    def test_no_hint_when_all_premises_out(self, db):
        """If all premises go OUT, no surviving premises → no hint."""
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", sl="a", db_path=db)
        api.add_node("x", "Conclusion", sl="a,b", db_path=db)

        result = api.retract_node("a", db_path=db)

        # Both a and b go OUT, so x has no surviving premises
        hints = [h for h in result["restoration_hints"] if h["node_id"] == "x"]
        assert len(hints) == 0

    def test_no_hint_for_single_premise_sl(self, db):
        """Single-premise SL nodes don't produce hints."""
        api.add_node("a", "A", db_path=db)
        api.add_node("x", "Conclusion", sl="a", db_path=db)

        result = api.retract_node("a", db_path=db)
        assert result["restoration_hints"] == []

    def test_no_hint_for_directly_retracted_node(self, db):
        """The directly retracted node itself doesn't get a hint."""
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)

        result = api.retract_node("a", db_path=db)
        assert result["restoration_hints"] == []

    def test_hint_with_any_flag_no_false_alarm(self, db):
        """Nodes created with --any don't produce hints (each SL has 1 premise)."""
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)
        api.add_node("c", "C", db_path=db)
        api.add_node("x", "Conclusion", sl="a,b,c", any_mode=True, db_path=db)

        result = api.retract_node("a", db_path=db)

        # x stays IN (via b and c), so no hints needed
        assert result["restoration_hints"] == []
        node = api.show_node("x", db_path=db)
        assert node["truth_value"] == "IN"

    def test_hint_includes_correct_surviving_premises(self, db):
        """Only premises that are still IN appear in surviving_premises."""
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", sl="a", db_path=db)
        api.add_node("c", "C", db_path=db)
        api.add_node("x", "Conclusion", sl="a,b,c", db_path=db)

        # Retracting a also cascades b OUT
        result = api.retract_node("a", db_path=db)

        hints = [h for h in result["restoration_hints"] if h["node_id"] == "x"]
        assert len(hints) == 1
        # Only c survives (a retracted, b cascaded OUT)
        assert hints[0]["surviving_premises"] == ["c"]

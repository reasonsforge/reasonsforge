"""Tests for compact summary."""

from reasonsforge import Justification
from reasonsforge.network import Network
from reasonsforge.compact import compact, estimate_tokens


class TestEstimateTokens:

    def test_uses_char_count_not_word_count(self):
        text = "a longish sentence with several words in it here"
        assert estimate_tokens(text) == len(text) // 4
        assert estimate_tokens(text) != len(text.split())

    def test_minimum_one_token(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("hi") == 1

    def test_long_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100


class TestCompact:

    def test_empty_network(self):
        net = Network()
        result = compact(net)
        assert "Belief State Summary" in result
        assert "0 nodes tracked" in result

    def test_includes_nogoods(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_nogood(["a", "b"])
        result = compact(net)
        assert "## Nogoods" in result
        assert "nogood-001" in result

    def test_includes_out_nodes(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.retract("a")
        result = compact(net)
        assert "## OUT (retracted)" in result
        assert "a: Premise A" in result

    def test_includes_in_nodes(self):
        net = Network()
        net.add_node("a", "Premise A")
        result = compact(net)
        assert "## IN (active)" in result
        assert "a: Premise A" in result

    def test_truncates_long_text(self):
        net = Network()
        long_text = "A" * 200
        net.add_node("a", long_text)
        result = compact(net, truncate=True)
        assert "..." in result
        assert "A" * 200 not in result

    def test_no_truncate(self):
        net = Network()
        long_text = "A" * 200
        net.add_node("a", long_text)
        result = compact(net, truncate=False)
        assert "A" * 200 in result

    def test_budget_limits_in_nodes(self):
        net = Network()
        for i in range(50):
            net.add_node(f"node-{i:03d}", f"This is node number {i} with some text")
        result = compact(net, budget=100)
        assert "more IN nodes omitted" in result

    def test_budget_limits_out_nodes(self):
        net = Network()
        for i in range(50):
            net.add_node(f"out-{i:03d}", f"This is retracted node number {i} with text")
            net.retract(f"out-{i:03d}")
        result = compact(net, budget=100)
        assert "more OUT nodes omitted" in result

    def test_budget_limits_nogoods(self):
        net = Network()
        for i in range(50):
            a = f"a{i:03d}"
            b = f"b{i:03d}"
            net.add_node(a, f"Node {a}")
            net.add_node(b, f"Node {b}")
            net.add_nogood([a, b])
        result = compact(net, budget=100)
        assert "more nogoods omitted" in result

    def test_budget_respected_across_all_sections(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Premise B")
        net.add_nogood(["a", "b"])
        for i in range(20):
            net.add_node(f"out-{i:03d}", f"Retracted node {i}")
            net.retract(f"out-{i:03d}")
        for i in range(20):
            net.add_node(f"in-{i:03d}", f"Active node {i}")
        result = compact(net, budget=200)
        actual_tokens = estimate_tokens(result)
        # Budget is approximate (chars/4 heuristic); structural lines
        # (headers, truncation messages) may cause minor overshoot
        assert actual_tokens < 250, f"output ({actual_tokens}) far exceeds budget 200"
        assert "more IN nodes omitted" in result

    def test_most_depended_on_first(self):
        net = Network()
        net.add_node("root", "Root premise")
        net.add_node("leaf", "Leaf node")
        net.add_node("dep", "Depends on root", justifications=[
            Justification(type="SL", antecedents=["root"])
        ])
        result = compact(net, budget=5000)
        # root has 1 dependent (dep), leaf has 0
        root_pos = result.index("root:")
        leaf_pos = result.index("leaf:")
        assert root_pos < leaf_pos

    def test_shows_dependencies(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B", justifications=[
            Justification(type="SL", antecedents=["a"])
        ])
        result = compact(net, budget=5000)
        assert "<- a" in result

    def test_shows_dependent_count(self):
        net = Network()
        net.add_node("a", "Premise A")
        net.add_node("b", "Derived B", justifications=[
            Justification(type="SL", antecedents=["a"])
        ])
        net.add_node("c", "Derived C", justifications=[
            Justification(type="SL", antecedents=["a"])
        ])
        result = compact(net, budget=5000)
        assert "(2 dependents)" in result

    def test_stale_reason_in_out(self):
        net = Network()
        net.add_node("a", "Old belief", metadata={"stale_reason": "new data"})
        net.retract("a")
        result = compact(net)
        assert "stale: new data" in result

    def test_token_count_line(self):
        net = Network()
        net.add_node("a", "Premise A")
        result = compact(net, budget=500)
        assert "Token count:" in result
        assert "/ 500 budget" in result

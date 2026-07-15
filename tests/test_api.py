"""Tests for the functional Python API."""

from unittest.mock import patch

import pytest

from reasonsforge import api


@pytest.fixture
def db_path(tmp_path):
    p = str(tmp_path / "test_reasons.db")
    api.init_db(db_path=p)
    return p


class TestInitDb:

    def test_creates_db(self, tmp_path):
        p = str(tmp_path / "new.db")
        result = api.init_db(db_path=p)
        assert result["created"] is True

    def test_refuses_existing(self, db_path):
        with pytest.raises(FileExistsError):
            api.init_db(db_path=db_path)

    def test_force_overwrites(self, db_path):
        result = api.init_db(db_path=db_path, force=True)
        assert result["created"] is True


class TestAddNode:

    def test_add_premise(self, db_path):
        result = api.add_node("a", "Premise A", db_path=db_path)
        assert result["node_id"] == "a"
        assert result["truth_value"] == "IN"
        assert result["type"] == "premise"

    def test_add_with_sl(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        result = api.add_node("b", "Derived B", sl="a", db_path=db_path)
        assert result["truth_value"] == "IN"
        assert result["type"] == "SL"

    def test_add_duplicate_raises(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        with pytest.raises(ValueError):
            api.add_node("a", "Duplicate", db_path=db_path)


class TestRetractNode:

    def test_retract(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        result = api.retract_node("a", db_path=db_path)
        assert "a" in result["changed"]

    def test_retract_cascades(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        api.add_node("b", "Derived B", sl="a", db_path=db_path)
        result = api.retract_node("a", db_path=db_path)
        assert set(result["changed"]) == {"a", "b"}

    def test_retract_missing_raises(self, db_path):
        with pytest.raises(KeyError):
            api.retract_node("missing", db_path=db_path)

    def test_retract_already_out(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        api.retract_node("a", db_path=db_path)
        result = api.retract_node("a", db_path=db_path)
        assert result["changed"] == []


class TestAssertNode:

    def test_assert_restores(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        api.add_node("b", "Derived B", sl="a", db_path=db_path)
        api.retract_node("a", db_path=db_path)
        result = api.assert_node("a", db_path=db_path)
        assert set(result["changed"]) == {"a", "b"}

    def test_assert_already_in(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        result = api.assert_node("a", db_path=db_path)
        assert result["changed"] == []


class TestPropagate:

    def test_no_changes(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        result = api.propagate(db_path=db_path)
        assert result["changed"] == []

    def test_with_changes(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        api.add_node("b", "Derived B", sl="a", db_path=db_path)
        assert api.show_node("b", db_path=db_path)["truth_value"] == "IN"
        api.retract_node("a", reason="test", db_path=db_path)
        api.assert_node("a", db_path=db_path)
        from reasonsforge.storage import Storage
        store = Storage(db_path)
        net = store.load()
        net.nodes["b"].truth_value = "OUT"
        store.save(net)
        store.close()
        result = api.propagate(db_path=db_path)
        assert "b" in result["changed"]
        assert api.show_node("b", db_path=db_path)["truth_value"] == "IN"


class TestGetStatus:

    def test_empty(self, db_path):
        result = api.get_status(db_path=db_path)
        assert result["nodes"] == []
        assert result["total"] == 0

    def test_with_nodes(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        api.add_node("b", "Premise B", db_path=db_path)
        result = api.get_status(db_path=db_path)
        assert result["total"] == 2
        assert result["in_count"] == 2
        ids = [n["id"] for n in result["nodes"]]
        assert "a" in ids and "b" in ids


class TestShowNode:

    def test_show(self, db_path):
        api.add_node("a", "Premise A", source="repo:file.py", db_path=db_path)
        result = api.show_node("a", db_path=db_path)
        assert result["id"] == "a"
        assert result["text"] == "Premise A"
        assert result["source"] == "repo:file.py"
        assert result["justifications"] == []
        assert result["dependents"] == []

    def test_show_missing_raises(self, db_path):
        with pytest.raises(KeyError):
            api.show_node("missing", db_path=db_path)


class TestExplainNode:

    def test_explain_premise(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        result = api.explain_node("a", db_path=db_path)
        assert result["steps"][0]["reason"] == "premise"

    def test_explain_chain(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        api.add_node("b", "Derived B", sl="a", db_path=db_path)
        result = api.explain_node("b", db_path=db_path)
        nodes_in_trace = [s["node"] for s in result["steps"]]
        assert "b" in nodes_in_trace
        assert "a" in nodes_in_trace


    def test_explain_circular_dependency(self, db_path):
        api.add_node("p", "Premise", db_path=db_path)
        api.add_node("x", "Derived X", sl="p", db_path=db_path)
        api.add_justification("x", sl="y", db_path=db_path)
        api.add_node("y", "Derived Y", sl="x", db_path=db_path)
        api.retract_node("p", db_path=db_path)
        result = api.explain_node("x", db_path=db_path)
        reasons = [s["reason"] for s in result["steps"]]
        assert any("circular" in r for r in reasons)

    def test_explain_diamond_no_false_circular(self, db_path):
        api.add_node("root", "Root premise", db_path=db_path)
        api.add_node("left", "Left", sl="root", db_path=db_path)
        api.add_node("right", "Right", sl="root", db_path=db_path)
        api.add_node("top", "Top", sl="left,right", db_path=db_path)
        result = api.explain_node("top", db_path=db_path)
        reasons = [s["reason"] for s in result["steps"]]
        assert not any("circular" in r for r in reasons)


class TestAddNogood:

    def test_nogood(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        api.add_node("b", "Premise B", db_path=db_path)
        result = api.add_nogood(["a", "b"], db_path=db_path)
        assert result["nogood_id"] == "nogood-001"
        assert result["nodes"] == ["a", "b"]
        assert len(result["changed"]) > 0


class TestGetBeliefSet:

    def test_belief_set(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        api.add_node("b", "Premise B", db_path=db_path)
        api.retract_node("b", db_path=db_path)
        result = api.get_belief_set(db_path=db_path)
        assert result == ["a"]


class TestGetLog:

    def test_log(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        result = api.get_log(db_path=db_path)
        assert len(result["entries"]) > 0

    def test_log_last(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        api.add_node("b", "Premise B", db_path=db_path)
        result = api.get_log(last=1, db_path=db_path)
        assert len(result["entries"]) == 1


class TestExportNetwork:

    def test_export(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        result = api.export_network(db_path=db_path)
        assert "a" in result["nodes"]
        assert result["nodes"]["a"]["truth_value"] == "IN"


class TestEndToEnd:
    """Full workflow through the API — same scenarios as test_network.py."""

    def test_retract_and_restore_chain(self, db_path):
        api.add_node("a", "Premise A", db_path=db_path)
        api.add_node("b", "Derived B", sl="a", db_path=db_path)
        api.add_node("c", "Derived C", sl="b", db_path=db_path)

        # All IN
        status = api.get_status(db_path=db_path)
        assert status["in_count"] == 3

        # Retract A → cascade
        result = api.retract_node("a", db_path=db_path)
        assert set(result["changed"]) == {"a", "b", "c"}

        status = api.get_status(db_path=db_path)
        assert status["in_count"] == 0

        # Assert A → restore
        result = api.assert_node("a", db_path=db_path)
        assert set(result["changed"]) == {"a", "b", "c"}

        status = api.get_status(db_path=db_path)
        assert status["in_count"] == 3


class TestListNodesDepth:

    def test_list_min_depth(self, db_path):
        api.add_node("p1", "premise", db_path=db_path)
        api.add_node("d1", "derived", sl="p1", label="t", db_path=db_path)

        result = api.list_nodes(min_depth=1, db_path=db_path)
        ids = [n["id"] for n in result["nodes"]]
        assert "d1" in ids
        assert "p1" not in ids

    def test_list_max_depth(self, db_path):
        api.add_node("p1", "premise", db_path=db_path)
        api.add_node("d1", "derived", sl="p1", label="t", db_path=db_path)

        result = api.list_nodes(max_depth=0, db_path=db_path)
        ids = [n["id"] for n in result["nodes"]]
        assert "p1" in ids
        assert "d1" not in ids

    def test_list_depth_range(self, db_path):
        api.add_node("p", "premise", db_path=db_path)
        api.add_node("mid", "mid", sl="p", label="t", db_path=db_path)
        api.add_node("top", "top", sl="mid", label="t", db_path=db_path)

        result = api.list_nodes(min_depth=1, max_depth=1, db_path=db_path)
        ids = [n["id"] for n in result["nodes"]]
        assert ids == ["mid"]

    def test_list_by_label(self, db_path):
        api.add_node("a", "Node A", db_path=db_path)
        api.add_node("b", "Node B", sl="a", label="WARNING", db_path=db_path)
        api.add_node("c", "Node C", sl="a", label="INFO", db_path=db_path)

        result = api.list_nodes(label="WARNING", db_path=db_path)
        ids = [n["id"] for n in result["nodes"]]
        assert ids == ["b"]

    def test_list_by_label_no_match(self, db_path):
        api.add_node("a", "Node A", db_path=db_path)
        api.add_node("b", "Node B", sl="a", label="INFO", db_path=db_path)

        result = api.list_nodes(label="WARNING", db_path=db_path)
        assert result["count"] == 0

    def test_list_by_label_premise_excluded(self, db_path):
        api.add_node("a", "Premise", db_path=db_path)
        api.add_node("b", "Derived", sl="a", label="WARNING", db_path=db_path)

        result = api.list_nodes(label="WARNING", db_path=db_path)
        ids = [n["id"] for n in result["nodes"]]
        assert "a" not in ids
        assert "b" in ids


class TestFtsSearch:

    def test_porter_stemming(self, db_path):
        api.add_node("a", "sandbox access is auto-deactivated after 21 days", db_path=db_path)
        result = api.search("deactivation", db_path=db_path)
        assert "a" in result

    def test_porter_stemming_plural(self, db_path):
        from reasonsforge.api import _fts_search
        api.add_node("a", "max 250 jobs per pipeline", db_path=db_path)
        results = _fts_search("job", db_path)
        assert "a" in results

    def test_progressive_relaxation(self, db_path):
        api.add_node("a", "sandbox access expires after 21 days", db_path=db_path)
        result = api.search("sandbox access duration expiration", db_path=db_path)
        assert "a" in result

    def test_two_term_no_relaxation(self, db_path):
        api.add_node("a", "sandbox access expires", db_path=db_path)
        result = api.search("sandbox quantum", db_path=db_path, format="compact")
        assert "a" not in result

    def test_no_false_positives(self, db_path):
        api.add_node("a", "the quick brown fox", db_path=db_path)
        result = api.search("quantum computing blockchain", db_path=db_path, format="compact")
        assert "a" not in result

    def test_stop_words_filtered(self, db_path):
        from reasonsforge.api import _fts_search
        api.add_node("a", "propagation uses BFS algorithm", db_path=db_path)
        results = _fts_search("What is the propagation algorithm?", db_path)
        assert "a" in results

    def test_all_stop_words_falls_back_to_raw(self, db_path):
        from reasonsforge.api import _fts_search
        api.add_node("a", "the system is working", db_path=db_path)
        results = _fts_search("what is the", db_path)
        assert "a" in results

    def test_single_char_words_only_returns_empty(self, db_path):
        from reasonsforge.api import _fts_search
        api.add_node("a", "some content", db_path=db_path)
        results = _fts_search("a b c", db_path)
        assert results == []

    def test_natural_language_question(self, db_path):
        api.add_node("a", "retraction cascades through dependent nodes", db_path=db_path)
        result = api.search("How does retraction work in the system?",
                            db_path=db_path, format="compact")
        assert "a" in result

    def test_punctuation_in_query(self, db_path):
        from reasonsforge.api import _fts_search
        api.add_node("a", "propagation uses BFS", db_path=db_path)
        results = _fts_search("propagation? (BFS)", db_path)
        assert "a" in results

    def test_long_query_does_not_explode(self, db_path):
        from reasonsforge.api import _fts_search, _fts_query
        from unittest.mock import patch as mock_patch
        api.add_node("a", "alpha beta gamma delta", db_path=db_path)
        query = " ".join(f"term{i}" for i in range(20))
        call_count = [0]
        original_fts_query = _fts_query

        def counting_fts_query(conn, terms):
            call_count[0] += 1
            return original_fts_query(conn, terms)

        with mock_patch("reasonsforge.api._fts_query", side_effect=counting_fts_query):
            _fts_search(query, db_path)
        assert call_count[0] <= 51

    def test_depth_1_includes_direct_antecedents(self, db_path):
        api.add_node("premise", "Propagation uses BFS", db_path=db_path)
        api.add_node("derived", "Propagation is safe", sl="premise", db_path=db_path)
        result = api.search("safe", db_path=db_path, format="compact", depth=1)
        assert "premise" in result

    def test_depth_2_includes_transitive_antecedents(self, db_path):
        api.add_node("root", "BFS traversal algorithm", db_path=db_path)
        api.add_node("mid", "Propagation uses BFS", sl="root", db_path=db_path)
        api.add_node("leaf", "Propagation is safe", sl="mid", db_path=db_path)
        result_d1 = api.search("safe", db_path=db_path, format="compact", depth=1)
        result_d2 = api.search("safe", db_path=db_path, format="compact", depth=2)
        assert "root" not in result_d1
        assert "root" in result_d2

    def test_depth_0_no_expansion(self, db_path):
        api.add_node("premise", "Propagation uses BFS", db_path=db_path)
        api.add_node("derived", "Propagation is safe", sl="premise", db_path=db_path)
        result = api.search("safe", db_path=db_path, format="compact", depth=0)
        assert "premise" not in result
        assert "derived" in result


class TestListGated:

    def test_no_gates(self, db_path):
        api.add_node("a", "Alpha", db_path=db_path)
        result = api.list_gated(db_path=db_path)
        assert result["blockers"] == {}
        assert result["gated_count"] == 0

    def test_active_gate(self, db_path):
        api.add_node("premise", "Supporting premise", db_path=db_path)
        api.add_node("blocker", "Defect premise", db_path=db_path)
        api.add_node("gated", "Conclusion unless blocker", sl="premise", unless="blocker", db_path=db_path)
        result = api.list_gated(db_path=db_path)
        assert result["blocker_count"] == 1
        assert result["gated_count"] == 1
        assert "blocker" in result["blockers"]
        assert result["blockers"]["blocker"]["gated"][0]["id"] == "gated"

    def test_satisfied_gate(self, db_path):
        api.add_node("premise", "Supporting premise", db_path=db_path)
        api.add_node("blocker", "Defect premise", db_path=db_path)
        api.add_node("gated", "Conclusion unless blocker", sl="premise", unless="blocker", db_path=db_path)
        api.retract_node("blocker", db_path=db_path)
        result = api.list_gated(db_path=db_path)
        assert result["blockers"] == {}

    def test_multiple_gated_per_blocker(self, db_path):
        api.add_node("premise", "Supporting premise", db_path=db_path)
        api.add_node("blocker", "Defect", db_path=db_path)
        api.add_node("g1", "Gated 1", sl="premise", unless="blocker", db_path=db_path)
        api.add_node("g2", "Gated 2", sl="premise", unless="blocker", db_path=db_path)
        result = api.list_gated(db_path=db_path)
        assert result["blocker_count"] == 1
        assert result["gated_count"] == 2
        gated_ids = [g["id"] for g in result["blockers"]["blocker"]["gated"]]
        assert "g1" in gated_ids
        assert "g2" in gated_ids

    def test_superseded_excluded(self, db_path):
        api.add_node("premise", "Supporting premise", db_path=db_path)
        api.add_node("blocker", "Defect", db_path=db_path)
        api.add_node("old", "Old conclusion", sl="premise", unless="blocker", db_path=db_path)
        api.add_node("new", "New conclusion", sl="premise", db_path=db_path)
        api.supersede("old", "new", db_path=db_path)
        result = api.list_gated(db_path=db_path)
        assert result["gated_count"] == 0

    def test_blocker_text_included(self, db_path):
        api.add_node("premise", "Supporting premise", db_path=db_path)
        api.add_node("bug-123", "File X has a null check missing", db_path=db_path)
        api.add_node("gated", "X is safe", sl="premise", unless="bug-123", db_path=db_path)
        result = api.list_gated(db_path=db_path)
        assert result["blockers"]["bug-123"]["text"] == "File X has a null check missing"


class TestListNegative:

    def test_empty_db(self, db_path):
        with patch("reasonsforge.llm.invoke_model") as mock_llm:
            result = api.list_negative(db_path=db_path)
            assert result == {"negative": [], "count": 0, "candidates": 0, "total": 0}
            mock_llm.assert_not_called()

    def test_no_keyword_matches(self, db_path):
        api.add_node("a", "The sky is blue", db_path=db_path)
        api.add_node("b", "Water flows downhill", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model") as mock_llm:
            result = api.list_negative(db_path=db_path)
            assert result["count"] == 0
            assert result["candidates"] == 0
            assert result["total"] == 2
            mock_llm.assert_not_called()

    def test_classifies_negatives(self, db_path):
        api.add_node("a", "The auth module has a bug in token refresh", db_path=db_path)
        api.add_node("b", "Error handling logs all failures", db_path=db_path)
        api.add_node("c", "The sky is blue", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model", return_value='["a"]'):
            result = api.list_negative(db_path=db_path)
            assert result["count"] == 1
            assert result["candidates"] == 2
            assert result["total"] == 3
            assert result["negative"][0]["id"] == "a"

    def test_llm_filters_all(self, db_path):
        api.add_node("a", "Error handling is comprehensive", db_path=db_path)
        api.add_node("b", "Failure modes are well documented", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model", return_value='[]'):
            result = api.list_negative(db_path=db_path)
            assert result["count"] == 0
            assert result["candidates"] == 2
            assert result["total"] == 2

    def test_multiline_json_response(self, db_path):
        api.add_node("a", "There is a critical bug here", db_path=db_path)
        api.add_node("b", "This has a missing check", db_path=db_path)
        multiline = '[\n  "a",\n  "b"\n]'
        with patch("reasonsforge.llm.invoke_model", return_value=multiline):
            result = api.list_negative(db_path=db_path)
            assert result["count"] == 2

    def test_malformed_llm_response(self, db_path):
        api.add_node("a", "There is a critical bug here", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model", return_value="Sorry, I cannot do that."):
            result = api.list_negative(db_path=db_path)
            assert result["count"] == 0

    def test_llm_returns_unknown_ids(self, db_path):
        api.add_node("a", "There is a critical bug here", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model", return_value='["a", "nonexistent", "also-fake"]'):
            result = api.list_negative(db_path=db_path)
            assert result["count"] == 1
            assert result["negative"][0]["id"] == "a"

    def test_prose_with_brackets_before_json(self, db_path):
        api.add_node("a", "There is a critical bug here", db_path=db_path)
        response = 'Based on [the analysis], here are the negative beliefs: ["a"]'
        with patch("reasonsforge.llm.invoke_model", return_value=response):
            result = api.list_negative(db_path=db_path)
            assert result["count"] == 1
            assert result["negative"][0]["id"] == "a"

    def test_claude_not_found_propagates(self, db_path):
        api.add_node("a", "There is a critical bug here", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model", side_effect=FileNotFoundError("'claude' CLI not found in PATH")):
            with pytest.raises(FileNotFoundError):
                api.list_negative(db_path=db_path)

    def test_visible_to(self, db_path):
        api.add_node("a", "Auth has a critical bug", access_tags=["internal"], db_path=db_path)
        api.add_node("b", "API has a missing validation", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model", return_value='["b"]') as mock_llm:
            result = api.list_negative(visible_to=["public"], db_path=db_path)
            assert result["count"] == 1
            assert result["total"] == 1
            assert result["negative"][0]["id"] == "b"
            prompt = mock_llm.call_args[0][0]
            assert "critical bug" not in prompt

    def test_single_batch_calls_llm_once(self, db_path):
        for i in range(5):
            api.add_node(f"bug-{i}", f"There is a bug in module {i}", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model", return_value='["bug-0"]') as mock_llm:
            result = api.list_negative(db_path=db_path)
            assert mock_llm.call_count == 1
            assert result["count"] == 1

    def test_batching_large_set(self, db_path):
        for i in range(120):
            api.add_node(f"bug-{i:03d}", f"There is a bug in module {i}", db_path=db_path)

        call_count = [0]

        def mock_invoke(prompt, model="claude"):
            call_count[0] += 1
            if call_count[0] == 1:
                return '["bug-010", "bug-020"]'
            elif call_count[0] == 2:
                return '["bug-060"]'
            else:
                return '[]'

        with patch("reasonsforge.llm.invoke_model", side_effect=mock_invoke):
            result = api.list_negative(db_path=db_path)
        assert call_count[0] == 3
        assert result["count"] == 3
        assert result["candidates"] == 120
        found_ids = {n["id"] for n in result["negative"]}
        assert found_ids == {"bug-010", "bug-020", "bug-060"}

    def test_expanded_terms_match(self, db_path):
        api.add_node("a", "The migration is stalled due to schema conflicts", db_path=db_path)
        api.add_node("b", "There is a regression in the auth flow", db_path=db_path)
        api.add_node("c", "The API docs are undocumented for v2", db_path=db_path)
        api.add_node("d", "Everything works fine", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model", return_value='["a", "b", "c"]'):
            result = api.list_negative(db_path=db_path)
            assert result["candidates"] == 3
            assert result["total"] == 4

    def test_issue_false_positive_excluded(self, db_path):
        api.add_node("jira-ref", "The child issue was closed last sprint", db_path=db_path)
        api.add_node("real-neg", "There is a known issue in the auth module", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model", return_value='["real-neg"]') as mock_llm:
            result = api.list_negative(db_path=db_path)
            assert result["candidates"] == 1
            assert result["negative"][0]["id"] == "real-neg"

    def test_skip_llm(self, db_path):
        api.add_node("a", "There is a critical bug here", db_path=db_path)
        api.add_node("b", "Everything is fine", db_path=db_path)
        with patch("reasonsforge.llm.invoke_model") as mock_llm:
            result = api.list_negative(skip_llm=True, db_path=db_path)
            mock_llm.assert_not_called()
            assert result["count"] == 1
            assert result["candidates"] == 1
            assert result["negative"][0]["id"] == "a"


class TestUpdateNode:

    @pytest.fixture
    def db_path(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("a", "Original text", db_path=db)
        api.add_node("b", "Premise B", db_path=db)
        api.add_node("derived-ab", "AB combined", sl="a,b",
                      label="combined", db_path=db)
        return db

    def test_rejects_text_mutation(self, db_path):
        with pytest.raises(ValueError, match="immutable"):
            api.update_node("a", text="Updated text", db_path=db_path)

    def test_updates_source(self, db_path):
        result = api.update_node("a", source="new/source.md", db_path=db_path)
        assert "source" in result["updated_fields"]
        node = api.show_node("a", db_path=db_path)
        assert node["source"] == "new/source.md"

    def test_nonexistent_text_raises_keyerror(self, db_path):
        with pytest.raises(KeyError):
            api.update_node("nonexistent", text="x", db_path=db_path)

    def test_nonexistent_source_raises_keyerror(self, db_path):
        with pytest.raises(KeyError):
            api.update_node("nonexistent", source="x.py", db_path=db_path)

    def test_supersede_with_text(self, db_path):
        result = api.supersede_with_text("a", "Updated text", db_path=db_path)
        assert result["old_id"] == "a"
        new_id = result["new_id"]
        old = api.show_node("a", db_path=db_path)
        new = api.show_node(new_id, db_path=db_path)
        assert old["truth_value"] == "OUT"
        assert new["text"] == "Updated text"
        assert new["truth_value"] == "IN"

    def test_supersede_with_text_custom_id(self, db_path):
        result = api.supersede_with_text("a", "New text", new_id="a-fixed",
                                          db_path=db_path)
        assert result["new_id"] == "a-fixed"
        node = api.show_node("a-fixed", db_path=db_path)
        assert node["text"] == "New text"


class TestSetMetadata:

    def test_sets_key(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("a", "A belief", db_path=db)
        result = api.set_metadata("a", "source_file", "src/foo.py", db_path=db)
        assert result == {"node_id": "a", "key": "source_file"}
        node = api.show_node("a", db_path=db)
        assert node["metadata"]["source_file"] == "src/foo.py"

    def test_overwrites_existing_key(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("a", "A belief", db_path=db)
        api.set_metadata("a", "k", "v1", db_path=db)
        api.set_metadata("a", "k", "v2", db_path=db)
        node = api.show_node("a", db_path=db)
        assert node["metadata"]["k"] == "v2"

    def test_nonexistent_raises(self, tmp_path):
        db = str(tmp_path / "test.db")
        with pytest.raises(KeyError):
            api.set_metadata("nope", "k", "v", db_path=db)


class TestListClusters:

    @pytest.fixture
    def db_with_beliefs(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)
        for i in range(10):
            api.add_node(f"in-{i}", f"Active belief {i}", db_path=db)
        for i in range(5):
            api.add_node(f"out-{i}", f"Retracted belief {i}", db_path=db)
            api.retract_node(f"out-{i}", db_path=db)
        return db

    def test_filters_by_status(self, db_with_beliefs):
        mock_result = {"clusters": [{"id": 0, "beliefs": []}], "n_clusters": 1, "embedding_model": "test"}
        with patch("reasonsforge.cluster.list_clusters", return_value=mock_result) as mock_lc:
            api.list_clusters(status="IN", db_path=db_with_beliefs)
            beliefs_arg = mock_lc.call_args[0][0]
            assert all(k.startswith("in-") for k in beliefs_arg)
            assert len(beliefs_arg) == 10

    def test_filters_out_status(self, db_with_beliefs):
        mock_result = {"clusters": [{"id": 0, "beliefs": []}], "n_clusters": 1, "embedding_model": "test"}
        with patch("reasonsforge.cluster.list_clusters", return_value=mock_result) as mock_lc:
            api.list_clusters(status="OUT", db_path=db_with_beliefs)
            beliefs_arg = mock_lc.call_args[0][0]
            assert all(k.startswith("out-") for k in beliefs_arg)
            assert len(beliefs_arg) == 5

    def test_empty_network(self, tmp_path):
        db = str(tmp_path / "empty.db")
        api.init_db(db_path=db)
        result = api.list_clusters(db_path=db)
        assert result["clusters"] == []
        assert result["n_clusters"] == 0

    def test_passes_seed(self, db_with_beliefs):
        mock_result = {"clusters": [], "n_clusters": 0, "embedding_model": "test"}
        with patch("reasonsforge.cluster.list_clusters", return_value=mock_result) as mock_lc:
            api.list_clusters(seed=42, db_path=db_with_beliefs)
            assert mock_lc.call_args[1]["seed"] == 42

    def test_passes_n_clusters(self, db_with_beliefs):
        mock_result = {"clusters": [], "n_clusters": 0, "embedding_model": "test"}
        with patch("reasonsforge.cluster.list_clusters", return_value=mock_result) as mock_lc:
            api.list_clusters(n_clusters=3, db_path=db_with_beliefs)
            assert mock_lc.call_args[1]["n_clusters"] == 3


try:
    from reasonsforge.cluster import HAS_CLUSTER_DEPS
except ImportError:
    HAS_CLUSTER_DEPS = False

skip_no_cluster = pytest.mark.skipif(
    not HAS_CLUSTER_DEPS,
    reason="sentence-transformers and scikit-learn not installed"
)


@skip_no_cluster
class TestDeduplicateSemantic:

    @pytest.fixture
    def db_with_similar(self, tmp_path):
        db = str(tmp_path / "sim.db")
        api.init_db(db_path=db)
        api.add_node("input-validation-at-boundaries",
                      "The system validates all inputs at system boundaries",
                      db_path=db)
        api.add_node("boundary-input-checking",
                      "Input validation occurs at system edges and boundaries",
                      db_path=db)
        api.add_node("database-query-performance",
                      "Database queries are optimized for read-heavy workloads",
                      db_path=db)
        return db

    def test_semantic_finds_similar_text(self, db_with_similar):
        result = api.deduplicate(threshold=0.5, semantic=True, db_path=db_with_similar)
        assert len(result["clusters"]) >= 1
        cluster_ids = {b["id"] for b in result["clusters"][0]["beliefs"]}
        assert "input-validation-at-boundaries" in cluster_ids
        assert "boundary-input-checking" in cluster_ids

    def test_semantic_skips_dissimilar(self, db_with_similar):
        result = api.deduplicate(threshold=0.8, semantic=True, db_path=db_with_similar)
        for cluster in result["clusters"]:
            ids = {b["id"] for b in cluster["beliefs"]}
            assert not ({"input-validation-at-boundaries", "database-query-performance"} <= ids)

    def test_semantic_auto_retracts(self, db_with_similar):
        result = api.deduplicate(threshold=0.5, semantic=True, auto=True,
                                  db_path=db_with_similar)
        retracted_set = set(result["retracted"])
        similar_pair = {"input-validation-at-boundaries", "boundary-input-checking"}
        assert len(retracted_set & similar_pair) == 1
        assert "database-query-performance" not in retracted_set

    def test_semantic_empty_network(self, tmp_path):
        db = str(tmp_path / "empty.db")
        api.init_db(db_path=db)
        result = api.deduplicate(threshold=0.5, semantic=True, db_path=db)
        assert result["clusters"] == []
        assert result["retracted"] == []

    def test_jaccard_mode_unchanged(self, db_with_similar):
        result = api.deduplicate(threshold=0.5, semantic=False, db_path=db_with_similar)
        assert result["retracted"] == []


class TestLifecycleTimestamps:

    def test_add_node_sets_created_at(self, db_path):
        api.add_node("ts-a", "Timestamped node", db_path=db_path)
        node = api.show_node("ts-a", db_path=db_path)
        assert node["created_at"] != ""
        assert node["updated_at"] != ""
        assert node["created_at"] == node["updated_at"]

    def test_update_node_sets_updated_at(self, db_path):
        api.add_node("ts-b", "Original", db_path=db_path)
        original = api.show_node("ts-b", db_path=db_path)
        api.update_node("ts-b", source="new-source.py", db_path=db_path)
        updated = api.show_node("ts-b", db_path=db_path)
        assert updated["updated_at"] >= original["updated_at"]

    def test_set_metadata_sets_updated_at(self, db_path):
        api.add_node("ts-c", "Meta node", db_path=db_path)
        original = api.show_node("ts-c", db_path=db_path)
        api.set_metadata("ts-c", "key", "value", db_path=db_path)
        updated = api.show_node("ts-c", db_path=db_path)
        assert updated["updated_at"] >= original["updated_at"]

    def test_retract_sets_retracted_at(self, db_path):
        api.add_node("ts-d", "To retract", db_path=db_path)
        api.retract_node("ts-d", db_path=db_path)
        node = api.show_node("ts-d", db_path=db_path)
        assert node["retracted_at"] != ""
        assert node["truth_value"] == "OUT"

    def test_show_node_includes_all_timestamps(self, db_path):
        api.add_node("ts-e", "Full timestamps", db_path=db_path)
        node = api.show_node("ts-e", db_path=db_path)
        for key in ("created_at", "updated_at", "reviewed_at", "verified_at", "retracted_at"):
            assert key in node

    def test_export_includes_timestamps(self, db_path):
        api.add_node("ts-f", "Exported node", db_path=db_path)
        result = api.export_network(db_path=db_path)
        node_data = result["nodes"]["ts-f"]
        assert "created_at" in node_data
        assert node_data["created_at"] != ""

    def test_import_roundtrips_timestamps(self, tmp_path, db_path):
        api.add_node("ts-g", "Roundtrip node", db_path=db_path)
        api.retract_node("ts-g", reason="testing", db_path=db_path)
        export = api.export_network(db_path=db_path)

        import json
        json_path = str(tmp_path / "export.json")
        with open(json_path, "w") as f:
            json.dump(export, f)

        db2 = str(tmp_path / "imported.db")
        api.init_db(db_path=db2)
        api.import_json(json_path, db_path=db2)
        node = api.show_node("ts-g", db_path=db2)
        assert node["created_at"] == export["nodes"]["ts-g"]["created_at"]
        assert node["retracted_at"] == export["nodes"]["ts-g"]["retracted_at"]
        assert node["updated_at"] == export["nodes"]["ts-g"]["updated_at"]

    def test_assert_clears_retracted_at(self, db_path):
        api.add_node("ts-h", "Retract then restore", db_path=db_path)
        api.retract_node("ts-h", db_path=db_path)
        retracted = api.show_node("ts-h", db_path=db_path)
        assert retracted["retracted_at"] != ""
        assert retracted["truth_value"] == "OUT"

        api.assert_node("ts-h", db_path=db_path)
        restored = api.show_node("ts-h", db_path=db_path)
        assert restored["retracted_at"] == ""
        assert restored["truth_value"] == "IN"
        assert restored["updated_at"] >= retracted["updated_at"]

    def test_cascade_does_not_set_retracted_at_on_dependents(self, db_path):
        api.add_node("ts-root", "Root premise", db_path=db_path)
        api.add_node("ts-dep", "Depends on root", sl="ts-root", db_path=db_path)
        api.retract_node("ts-root", db_path=db_path)

        root = api.show_node("ts-root", db_path=db_path)
        dep = api.show_node("ts-dep", db_path=db_path)
        assert root["retracted_at"] != ""
        assert dep["truth_value"] == "OUT"
        assert dep["retracted_at"] == ""

    def test_verified_at_preserved_through_export_import(self, tmp_path, db_path):
        from datetime import datetime, timezone
        api.add_node("ts-ver", "Verified node", db_path=db_path)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        from reasonsforge.storage import Storage
        store = Storage(db_path)
        net = store.load()
        net.nodes["ts-ver"].verified_at = now
        store.save(net)
        store.close()

        export = api.export_network(db_path=db_path)
        assert export["nodes"]["ts-ver"]["verified_at"] == now

        import json
        json_path = str(tmp_path / "verified.json")
        with open(json_path, "w") as f:
            json.dump(export, f)

        db2 = str(tmp_path / "verified_import.db")
        api.init_db(db_path=db2)
        api.import_json(json_path, db_path=db2)
        node = api.show_node("ts-ver", db_path=db2)
        assert node["verified_at"] == now

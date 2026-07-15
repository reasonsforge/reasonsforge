"""Tests for the PostgreSQL-native storage backend."""

import json

import pytest

from tests.conftest import skip_no_pg

pytestmark = [pytest.mark.pg, skip_no_pg]


class TestAddNode:

    def test_add_premise(self, pg_api):
        result = pg_api.add_node("a", "Alpha premise")
        assert result["node_id"] == "a"
        assert result["truth_value"] == "IN"
        assert result["type"] == "premise"

    def test_add_derived_in(self, pg_api):
        pg_api.add_node("a", "Alpha premise")
        result = pg_api.add_node("b", "Beta derived", sl="a")
        assert result["truth_value"] == "IN"
        assert result["type"] == "derived"

    def test_add_derived_out(self, pg_api):
        pg_api.add_node("a", "Alpha premise")
        pg_api.retract_node("a")
        result = pg_api.add_node("b", "Beta derived", sl="a")
        assert result["truth_value"] == "OUT"

    def test_add_with_outlist(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("blocker", "Blocker node")
        result = pg_api.add_node("c", "C unless blocker", sl="a", unless="blocker")
        assert result["truth_value"] == "OUT"

    def test_add_with_outlist_out(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("blocker", "Blocker node")
        pg_api.retract_node("blocker")
        result = pg_api.add_node("c", "C unless blocker", sl="a", unless="blocker")
        assert result["truth_value"] == "IN"

    def test_add_duplicate_raises(self, pg_api):
        pg_api.add_node("a", "Alpha")
        with pytest.raises(Exception):
            pg_api.add_node("a", "Alpha again")

    def test_add_with_access_tags(self, pg_api):
        pg_api.add_node("a", "Alpha", access_tags=["billing", "aws"])
        result = pg_api.show_node("a")
        assert result["metadata"]["access_tags"] == ["aws", "billing"]

    def test_access_tag_inheritance(self, pg_api):
        pg_api.add_node("a", "Alpha", access_tags=["billing"])
        pg_api.add_node("b", "Beta", access_tags=["aws"])
        pg_api.add_node("c", "Gamma derived", sl="a,b")
        result = pg_api.show_node("c")
        assert set(result["metadata"]["access_tags"]) == {"aws", "billing"}


class TestRefIntegrity:

    def test_add_node_phantom_antecedent(self, pg_api):
        with pytest.raises(KeyError, match="ghost"):
            pg_api.add_node("b", "Derived B", sl="ghost")

    def test_add_node_phantom_outlist(self, pg_api):
        pg_api.add_node("a", "Alpha")
        with pytest.raises(KeyError, match="ghost"):
            pg_api.add_node("b", "Derived B", sl="a", unless="ghost")

    def test_add_justification_phantom_antecedent(self, pg_api):
        pg_api.add_node("a", "Alpha")
        with pytest.raises(KeyError, match="ghost"):
            pg_api.add_justification("a", sl="ghost")

    def test_add_justification_phantom_outlist(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        with pytest.raises(KeyError, match="ghost"):
            pg_api.add_justification("b", sl="a", unless="ghost")

    def test_add_node_multiple_missing(self, pg_api):
        with pytest.raises(KeyError, match="x.*y|y.*x"):
            pg_api.add_node("b", "Derived B", sl="x,y")

    def test_add_node_valid_refs(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        result = pg_api.add_node("c", "Derived C", sl="a,b")
        assert result["truth_value"] == "IN"

    def test_add_node_premise_no_validation(self, pg_api):
        result = pg_api.add_node("a", "Alpha premise")
        assert result["truth_value"] == "IN"


class TestRetractNode:

    def test_retract_premise(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.retract_node("a")
        assert "a" in result["changed"]
        assert "a" in result["went_out"]
        status = pg_api.show_node("a")
        assert status["truth_value"] == "OUT"

    def test_retract_already_out(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.retract_node("a")
        result = pg_api.retract_node("a")
        assert result["changed"] == []

    def test_retract_cascade(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.add_node("c", "Gamma", sl="b")
        result = pg_api.retract_node("a")
        assert "a" in result["went_out"]
        assert "b" in result["went_out"]
        assert "c" in result["went_out"]

    def test_retract_with_reason(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.retract_node("a", reason="obsolete")
        result = pg_api.show_node("a")
        assert result["metadata"].get("retract_reason") == "obsolete"

    def test_retract_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.retract_node("nonexistent")


class TestAssertNode:

    def test_assert_restores(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.retract_node("a")
        result = pg_api.assert_node("a")
        assert "a" in result["changed"]
        assert "a" in result["went_in"]
        status = pg_api.show_node("a")
        assert status["truth_value"] == "IN"

    def test_assert_already_in(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.assert_node("a")
        assert result["changed"] == []

    def test_assert_restores_cascade(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.add_node("c", "Gamma", sl="b")
        pg_api.retract_node("a")
        result = pg_api.assert_node("a")
        assert "a" in result["went_in"]
        assert "b" in result["went_in"]
        assert "c" in result["went_in"]


class TestPropagation:

    def test_diamond_dependency(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.add_node("c", "Gamma", sl="a")
        pg_api.add_node("d", "Delta", sl="b,c")
        result = pg_api.retract_node("a")
        assert set(result["went_out"]) == {"a", "b", "c", "d"}

    def test_diamond_restoration(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.add_node("c", "Gamma", sl="a")
        pg_api.add_node("d", "Delta", sl="b,c")
        pg_api.retract_node("a")
        result = pg_api.assert_node("a")
        assert "b" in result["went_in"]
        assert "c" in result["went_in"]
        assert "d" in result["went_in"]

    def test_outlist_blocks(self, pg_api):
        pg_api.add_node("x", "X premise")
        pg_api.add_node("y", "Y blocker")
        pg_api.add_node("z", "Z unless Y", sl="x", unless="y")
        status = pg_api.show_node("z")
        assert status["truth_value"] == "OUT"

    def test_outlist_unblocks(self, pg_api):
        pg_api.add_node("x", "X premise")
        pg_api.add_node("y", "Y blocker")
        pg_api.add_node("z", "Z unless Y", sl="x", unless="y")
        pg_api.retract_node("y")
        status = pg_api.show_node("z")
        assert status["truth_value"] == "IN"

    def test_outlist_reblocks(self, pg_api):
        pg_api.add_node("x", "X premise")
        pg_api.add_node("y", "Y blocker")
        pg_api.add_node("z", "Z unless Y", sl="x", unless="y")
        pg_api.retract_node("y")
        pg_api.assert_node("y")
        status = pg_api.show_node("z")
        assert status["truth_value"] == "OUT"

    def test_multiple_justifications(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.add_node("c", "Gamma", sl="a")
        pg_api.add_justification("c", sl="b")
        pg_api.retract_node("a")
        status = pg_api.show_node("c")
        assert status["truth_value"] == "IN"

    def test_retracted_pin_skipped(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.retract_node("b")
        pg_api.retract_node("a")
        pg_api.assert_node("a")
        status = pg_api.show_node("b")
        assert status["truth_value"] == "OUT"


class TestGetStatus:

    def test_empty(self, pg_api):
        result = pg_api.get_status()
        assert result["nodes"] == []
        assert result["total"] == 0

    def test_with_nodes(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.retract_node("b")
        result = pg_api.get_status()
        assert result["total"] == 2
        assert result["in_count"] == 1

    def test_visible_to_filter(self, pg_api):
        pg_api.add_node("a", "Alpha", access_tags=["secret"])
        pg_api.add_node("b", "Beta")
        result = pg_api.get_status(visible_to=["public"])
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "b"


class TestShowNode:

    def test_show_premise(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.show_node("a")
        assert result["id"] == "a"
        assert result["truth_value"] == "IN"
        assert result["justifications"] == []

    def test_show_derived(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a", label="test")
        result = pg_api.show_node("b")
        assert len(result["justifications"]) == 1
        assert result["justifications"][0]["antecedents"] == ["a"]
        assert result["justifications"][0]["label"] == "test"

    def test_show_dependents(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        result = pg_api.show_node("a")
        assert "b" in result["dependents"]

    def test_show_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.show_node("nonexistent")

    def test_show_access_denied(self, pg_api):
        pg_api.add_node("a", "Alpha", access_tags=["secret"])
        with pytest.raises(PermissionError):
            pg_api.show_node("a", visible_to=["public"])


class TestSearch:

    def test_search_by_text(self, pg_api):
        pg_api.add_node("a", "Propagation uses breadth-first search")
        pg_api.add_node("b", "Retraction cascades through dependents")
        result = pg_api.search("propagation")
        assert "propagation" in result.lower() or "a" in result

    def test_search_by_id(self, pg_api):
        pg_api.add_node("prop-bfs", "Propagation uses BFS")
        result = pg_api.search("prop-bfs")
        assert "prop-bfs" in result

    def test_search_no_results(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.search("zzzznonexistent")
        assert "No results" in result

    def test_search_compact_format(self, pg_api):
        pg_api.add_node("a", "Alpha belief")
        result = pg_api.search("alpha", format="compact")
        assert "[IN] a" in result

    def test_search_json_format(self, pg_api):
        pg_api.add_node("a", "Alpha belief")
        result = pg_api.search("alpha", format="json")
        data = json.loads(result)
        assert any(d["id"] == "a" for d in data)

    def test_search_with_neighbors(self, pg_api):
        pg_api.add_node("a", "Alpha premise")
        pg_api.add_node("b", "Beta depends on alpha", sl="a")
        result = pg_api.search("beta", format="compact")
        assert "b" in result
        assert "a" in result

    def test_search_dict_format(self, pg_api):
        pg_api.add_node("a", "Alpha belief")
        pg_api.add_node("b", "Beta derived", sl="a")
        result = pg_api.search("alpha", format="dict")
        assert isinstance(result, dict)
        assert result["count"] == 1
        assert result["results"][0]["id"] == "a"
        assert result["results"][0]["truth_value"] == "IN"
        assert len(result["neighbors"]) >= 1

    def test_search_dict_no_results(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.search("zzzznonexistent", format="dict")
        assert result == {"results": [], "count": 0}


class TestListNodes:

    def test_list_all(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        result = pg_api.list_nodes()
        assert result["count"] == 2

    def test_list_by_status(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.retract_node("b")
        result = pg_api.list_nodes(status="IN")
        assert result["count"] == 1
        assert result["nodes"][0]["id"] == "a"

    def test_list_premises_only(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        result = pg_api.list_nodes(premises_only=True)
        assert result["count"] == 1
        assert result["nodes"][0]["id"] == "a"

    def test_list_has_dependents(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.add_node("c", "Gamma")
        result = pg_api.list_nodes(has_dependents=True)
        assert result["count"] == 1
        assert result["nodes"][0]["id"] == "a"

    def test_list_by_namespace(self, pg_api):
        pg_api.add_node("ns1:a", "Alpha")
        pg_api.add_node("ns2:b", "Beta")
        result = pg_api.list_nodes(namespace="ns1")
        assert result["count"] == 1
        assert result["nodes"][0]["id"] == "ns1:a"


class TestListGated:

    def test_no_gates(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.list_gated()
        assert result["blockers"] == {}
        assert result["gated_count"] == 0

    def test_active_gate(self, pg_api):
        pg_api.add_node("premise", "Supporting premise")
        pg_api.add_node("blocker", "Defect premise")
        pg_api.add_node("gated", "Conclusion unless blocker", sl="premise", unless="blocker")
        result = pg_api.list_gated()
        assert result["blocker_count"] == 1
        assert result["gated_count"] == 1
        assert "blocker" in result["blockers"]
        assert result["blockers"]["blocker"]["gated"][0]["id"] == "gated"

    def test_satisfied_gate(self, pg_api):
        pg_api.add_node("premise", "Supporting premise")
        pg_api.add_node("blocker", "Defect premise")
        pg_api.add_node("gated", "Conclusion unless blocker", sl="premise", unless="blocker")
        pg_api.retract_node("blocker")
        result = pg_api.list_gated()
        assert result["blockers"] == {}

    def test_multiple_gated_per_blocker(self, pg_api):
        pg_api.add_node("premise", "Supporting premise")
        pg_api.add_node("blocker", "Defect")
        pg_api.add_node("g1", "Gated 1", sl="premise", unless="blocker")
        pg_api.add_node("g2", "Gated 2", sl="premise", unless="blocker")
        result = pg_api.list_gated()
        assert result["blocker_count"] == 1
        assert result["gated_count"] == 2

    def test_blocker_text_included(self, pg_api):
        pg_api.add_node("premise", "Supporting premise")
        pg_api.add_node("bug-123", "Null check missing")
        pg_api.add_node("gated", "X is safe", sl="premise", unless="bug-123")
        result = pg_api.list_gated()
        assert result["blockers"]["bug-123"]["text"] == "Null check missing"


class TestGetLog:

    def test_log_after_add(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.get_log()
        assert len(result["entries"]) >= 1
        assert result["entries"][-1]["action"] == "add"

    def test_log_last(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.retract_node("a")
        result = pg_api.get_log(last=1)
        assert len(result["entries"]) == 1


class TestAddJustification:

    def test_add_justification_changes_truth(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.retract_node("b")
        result = pg_api.add_justification("b", sl="a")
        assert result["old_truth_value"] == "OUT"
        assert result["new_truth_value"] == "IN"

    def test_add_justification_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.add_justification("nonexistent", sl="a")


class TestNogood:

    def test_add_nogood_active(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        result = pg_api.add_nogood(["a", "b"])
        assert result["backtracked_to"] is not None
        # One of a or b should be retracted
        a = pg_api.show_node("a")
        b = pg_api.show_node("b")
        assert a["truth_value"] == "OUT" or b["truth_value"] == "OUT"

    def test_add_nogood_inactive(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.retract_node("b")
        result = pg_api.add_nogood(["a", "b"])
        assert result["backtracked_to"] is None
        assert result["changed"] == []

    def test_add_nogood_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.add_nogood(["nonexistent"])


class TestFindCulprits:

    def test_find_culprits(self, pg_api):
        pg_api.add_node("premise-a", "Alpha", source="code.py")
        pg_api.add_node("premise-b", "Beta")
        pg_api.add_node("derived", "Gamma", sl="premise-a,premise-b")
        result = pg_api.find_culprits(["premise-a", "premise-b"])
        assert len(result["culprits"]) >= 1
        # premise-b should be less entrenched (no source)
        assert result["culprits"][0]["premise"] == "premise-b"


class TestExplainNode:

    def test_explain_premise(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.explain_node("a")
        assert result["steps"][0]["reason"] == "premise"

    def test_explain_derived_in(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a", label="test")
        result = pg_api.explain_node("b")
        assert result["steps"][0]["truth_value"] == "IN"
        assert "SL" in result["steps"][0]["reason"]
        assert result["steps"][0]["antecedents"] == ["a"]

    def test_explain_derived_out(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.retract_node("a")
        result = pg_api.explain_node("b")
        assert result["steps"][0]["truth_value"] == "OUT"
        assert "a" in result["steps"][0]["failed_antecedents"]


class TestTraceAssumptions:

    def test_trace_premise(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.trace_assumptions("a")
        assert result["premises"] == ["a"]

    def test_trace_chain(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.add_node("c", "Gamma", sl="b")
        result = pg_api.trace_assumptions("c")
        assert "a" in result["premises"]

    def test_trace_diamond(self, pg_api):
        pg_api.add_node("p1", "Premise 1")
        pg_api.add_node("p2", "Premise 2")
        pg_api.add_node("d1", "Derived 1", sl="p1")
        pg_api.add_node("d2", "Derived 2", sl="p2")
        pg_api.add_node("top", "Top", sl="d1,d2")
        result = pg_api.trace_assumptions("top")
        assert set(result["premises"]) == {"p1", "p2"}


class TestMultiTenancy:

    def test_projects_isolated(self, pg_api):
        import os
        import uuid
        from reasonsforge.pg import PgApi

        pg_api.add_node("shared-id", "Project 1 data")

        project2 = str(uuid.uuid4())
        api2 = PgApi(os.environ["DATABASE_URL"], project2)
        api2.init_db()

        try:
            result = api2.get_status()
            assert result["total"] == 0

            api2.add_node("shared-id", "Project 2 data")
            p1 = pg_api.show_node("shared-id")
            p2 = api2.show_node("shared-id")
            assert p1["text"] == "Project 1 data"
            assert p2["text"] == "Project 2 data"
        finally:
            with api2.conn.cursor() as cur:
                for table in ("rms_propagation_log", "rms_justifications",
                              "rms_nogoods", "rms_network_meta", "rms_nodes"):
                    cur.execute(f"DELETE FROM {table} WHERE project_id = %s", (project2,))
            api2.conn.commit()
            api2.close()


class TestWhatIf:

    def test_what_if_retract_cascade(self, pg_api):
        pg_api.add_node("a", "Premise A")
        pg_api.add_node("b", "Derived B", sl="a")
        pg_api.add_node("c", "Derived C", sl="b")
        result = pg_api.what_if_retract("a")
        assert result["already_out"] is False
        assert result["total_affected"] == 2
        ids = [r["id"] for r in result["retracted"]]
        assert "b" in ids
        assert "c" in ids
        assert result["retracted"][0]["depth"] == 1  # b
        assert result["retracted"][1]["depth"] == 2  # c

    def test_what_if_retract_already_out(self, pg_api):
        pg_api.add_node("a", "Premise A")
        pg_api.retract_node("a")
        result = pg_api.what_if_retract("a")
        assert result["already_out"] is True
        assert result["total_affected"] == 0

    def test_what_if_retract_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.what_if_retract("missing")

    def test_what_if_retract_no_mutation(self, pg_api):
        pg_api.add_node("a", "Premise A")
        pg_api.add_node("b", "Derived B", sl="a")
        pg_api.what_if_retract("a")
        status = pg_api.get_status()
        assert status["in_count"] == 2

    def test_what_if_assert_restores(self, pg_api):
        pg_api.add_node("a", "Premise A")
        pg_api.add_node("b", "Derived B", sl="a")
        pg_api.retract_node("a")
        result = pg_api.what_if_assert("a")
        assert result["already_in"] is False
        assert result["total_affected"] == 1
        assert result["restored"][0]["id"] == "b"
        assert result["restored"][0]["depth"] == 1

    def test_what_if_assert_already_in(self, pg_api):
        pg_api.add_node("a", "Premise A")
        result = pg_api.what_if_assert("a")
        assert result["already_in"] is True
        assert result["total_affected"] == 0

    def test_what_if_assert_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.what_if_assert("missing")

    def test_what_if_assert_no_mutation(self, pg_api):
        pg_api.add_node("a", "Premise A")
        pg_api.add_node("b", "Derived B", sl="a")
        pg_api.retract_node("a")
        pg_api.what_if_assert("a")
        status = pg_api.get_status()
        assert status["in_count"] == 0

    def test_what_if_retract_dependents_field(self, pg_api):
        pg_api.add_node("a", "Premise A")
        pg_api.add_node("b", "Derived B", sl="a")
        pg_api.add_node("c", "Derived C", sl="b")
        result = pg_api.what_if_retract("a")
        b_info = next(r for r in result["retracted"] if r["id"] == "b")
        assert b_info["dependents"] == 1  # c depends on b

    def test_what_if_retract_with_outlist_restoration(self, pg_api):
        pg_api.add_node("premise", "Supporting premise")
        pg_api.add_node("blocker", "Blocker node")
        pg_api.add_node("gated", "Gated belief", sl="premise", unless="blocker")
        # gated is OUT because blocker is IN
        status = pg_api.show_node("gated")
        assert status["truth_value"] == "OUT"
        result = pg_api.what_if_retract("blocker")
        assert len(result["restored"]) == 1
        assert result["restored"][0]["id"] == "gated"
        assert result["restored"][0]["depth"] == 1
        # Verify no mutation
        status = pg_api.show_node("gated")
        assert status["truth_value"] == "OUT"


class TestChallenge:

    def test_challenge_premise(self, pg_api):
        pg_api.add_node("a", "Alpha premise")
        result = pg_api.challenge("a", "Alpha is wrong")
        assert result["challenge_id"] == "challenge-a"
        assert result["target_id"] == "a"
        assert "a" in result["changed"]
        status = pg_api.show_node("a")
        assert status["truth_value"] == "OUT"

    def test_challenge_derived(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        result = pg_api.challenge("b", "Beta is wrong")
        assert "b" in result["changed"]
        status = pg_api.show_node("b")
        assert status["truth_value"] == "OUT"

    def test_challenge_already_out(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.retract_node("a")
        result = pg_api.challenge("a", "Alpha is wrong")
        assert result["challenge_id"] == "challenge-a"
        # Target was already OUT, no change
        assert "a" not in result["changed"]

    def test_challenge_custom_id(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.challenge("a", "Alpha is wrong", challenge_id="my-challenge")
        assert result["challenge_id"] == "my-challenge"
        status = pg_api.show_node("my-challenge")
        assert status["truth_value"] == "IN"

    def test_challenge_auto_id_collision(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("challenge-a", "Existing node")
        result = pg_api.challenge("a", "Alpha is wrong")
        assert result["challenge_id"] == "challenge-a-2"

    def test_challenge_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.challenge("nonexistent", "reason")

    def test_challenge_source(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.challenge("a", "Alpha is wrong")
        challenge = pg_api.show_node("challenge-a")
        assert challenge["source"] == "challenge"

    def test_challenge_metadata(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.challenge("a", "Alpha is wrong")
        # Challenge node has challenge_target metadata
        challenge = pg_api.show_node("challenge-a")
        assert challenge["metadata"]["challenge_target"] == "a"
        # Target has challenges list in metadata
        target = pg_api.show_node("a")
        assert "challenge-a" in target["metadata"]["challenges"]

    def test_challenge_cascade(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.add_node("c", "Gamma", sl="b")
        result = pg_api.challenge("a", "Alpha is wrong")
        assert "a" in result["changed"]
        assert "b" in result["changed"]
        assert "c" in result["changed"]


class TestDefend:

    def test_defend_restores_target(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.challenge("a", "Alpha is wrong")
        assert pg_api.show_node("a")["truth_value"] == "OUT"
        result = pg_api.defend("a", "challenge-a", "Alpha is right")
        assert result["defense_id"] == "defense-challenge-a"
        assert pg_api.show_node("challenge-a")["truth_value"] == "OUT"
        assert pg_api.show_node("a")["truth_value"] == "IN"

    def test_defend_cascade(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.challenge("a", "Alpha is wrong")
        assert pg_api.show_node("b")["truth_value"] == "OUT"
        pg_api.defend("a", "challenge-a", "Alpha is right")
        assert pg_api.show_node("a")["truth_value"] == "IN"
        assert pg_api.show_node("b")["truth_value"] == "IN"

    def test_defend_metadata(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.challenge("a", "Alpha is wrong")
        pg_api.defend("a", "challenge-a", "Alpha is right")
        defense = pg_api.show_node("defense-challenge-a")
        assert defense["metadata"]["defense_target"] == "challenge-a"
        assert defense["metadata"]["defends"] == "a"

    def test_defend_not_found(self, pg_api):
        pg_api.add_node("a", "Alpha")
        with pytest.raises(KeyError):
            pg_api.defend("a", "nonexistent", "reason")
        with pytest.raises(KeyError):
            pg_api.defend("nonexistent", "a", "reason")

    def test_defend_custom_id(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.challenge("a", "Alpha is wrong")
        result = pg_api.defend("a", "challenge-a", "Alpha is right", defense_id="my-defense")
        assert result["defense_id"] == "my-defense"
        assert pg_api.show_node("my-defense")["truth_value"] == "IN"

    def test_defend_duplicate_id_raises(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("existing", "Existing node")
        pg_api.challenge("a", "Alpha is wrong")
        with pytest.raises(ValueError, match="Defense node"):
            pg_api.defend("a", "challenge-a", "reason", defense_id="existing")

    def test_defend_multiple_challenges(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.challenge("a", "First challenge")
        pg_api.challenge("a", "Second challenge")
        assert pg_api.show_node("a")["truth_value"] == "OUT"
        # Defend against only the first challenge
        pg_api.defend("a", "challenge-a", "First defense")
        # Target should stay OUT because second challenge remains
        assert pg_api.show_node("challenge-a")["truth_value"] == "OUT"
        assert pg_api.show_node("challenge-a-2")["truth_value"] == "IN"
        assert pg_api.show_node("a")["truth_value"] == "OUT"
        # Defend against the second challenge too
        pg_api.defend("a", "challenge-a-2", "Second defense")
        assert pg_api.show_node("a")["truth_value"] == "IN"


class TestCompact:

    def test_compact_empty(self, pg_api):
        result = pg_api.compact()
        assert "0 nodes tracked" in result
        assert "Belief State Summary" in result
        assert "Token count:" in result

    def test_compact_in_nodes(self, pg_api):
        pg_api.add_node("a", "Alpha premise")
        pg_api.add_node("b", "Beta premise")
        pg_api.add_node("c", "Gamma derived", sl="a")
        result = pg_api.compact()
        assert "3 nodes tracked" in result
        assert "## IN (active)" in result
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_compact_out_nodes(self, pg_api):
        pg_api.add_node("a", "Alpha premise")
        pg_api.retract_node("a", reason="obsolete")
        result = pg_api.compact()
        assert "## OUT (retracted)" in result
        assert "obsolete" in result

    def test_compact_budget(self, pg_api):
        for i in range(20):
            pg_api.add_node(f"node-{i:02d}", f"This is belief number {i} with some text")
        result_small = pg_api.compact(budget=50)
        result_large = pg_api.compact(budget=5000)
        assert len(result_small) < len(result_large)
        assert "omitted" in result_small or "Token count:" in result_small

    def test_compact_visible_to(self, pg_api):
        pg_api.add_node("public", "Public belief")
        pg_api.add_node("secret", "Secret belief", access_tags=["admin"])
        result = pg_api.compact(visible_to=["user"])
        assert "public" in result
        assert "secret" not in result

    def test_compact_nogoods(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.add_nogood(["a", "b"])
        result = pg_api.compact(budget=5000)
        assert "## Nogoods" in result
        assert "nogood-001" in result

    def test_compact_nogoods_filtered_by_visible_to(self, pg_api):
        pg_api.add_node("public", "Public belief")
        pg_api.add_node("secret", "Secret belief", access_tags=["admin"])
        pg_api.add_nogood(["public", "secret"])
        # Nogood references a secret node — should be hidden from non-admin
        result = pg_api.compact(budget=5000, visible_to=["user"])
        assert "Nogoods" not in result
        assert "nogood-001" not in result
        # Admin can see the nogood
        result_admin = pg_api.compact(budget=5000, visible_to=["admin"])
        assert "## Nogoods" in result_admin
        assert "nogood-001" in result_admin

    def test_compact_dependent_count_sorting(self, pg_api):
        pg_api.add_node("root", "Root node")
        pg_api.add_node("d1", "Dep 1", sl="root")
        pg_api.add_node("d2", "Dep 2", sl="root")
        pg_api.add_node("d3", "Dep 3", sl="root")
        pg_api.add_node("leaf", "Leaf node")
        result = pg_api.compact(budget=5000)
        # root has 3 dependents, should appear before leaf (0 dependents)
        root_pos = result.index("root")
        leaf_pos = result.index("leaf")
        assert root_pos < leaf_pos

    def test_compact_summary_nodes(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        # Create a summary node that covers a and b
        pg_api.add_node("summary", "Summary of a and b", sl="a,b")
        # Manually set summarizes metadata
        with pg_api.conn.cursor() as cur:
            cur.execute(
                "UPDATE rms_nodes SET metadata = %s WHERE id = %s AND project_id = %s",
                (json.dumps({"summarizes": ["a", "b"]}), "summary", pg_api.project_id),
            )
        pg_api.conn.commit()
        result = pg_api.compact(budget=5000)
        assert "[summary]" in result
        assert "hidden by summaries" in result


class TestExportNetwork:

    def test_export_empty(self, pg_api):
        result = pg_api.export_network()
        assert result["nodes"] == {}
        assert result["nogoods"] == []

    def test_export_premises(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        result = pg_api.export_network()
        assert "a" in result["nodes"]
        assert "b" in result["nodes"]
        assert result["nodes"]["a"]["text"] == "Alpha"
        assert result["nodes"]["a"]["truth_value"] == "IN"
        assert result["nodes"]["a"]["justifications"] == []

    def test_export_with_justifications(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        result = pg_api.export_network()
        justs = result["nodes"]["b"]["justifications"]
        assert len(justs) == 1
        assert justs[0]["type"] == "SL"
        assert justs[0]["antecedents"] == ["a"]

    def test_export_with_outlist(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("blocker", "Blocker")
        pg_api.retract_node("blocker")
        pg_api.add_node("c", "Gated", sl="a", unless="blocker")
        result = pg_api.export_network()
        justs = result["nodes"]["c"]["justifications"]
        assert justs[0]["outlist"] == ["blocker"]

    def test_export_with_nogoods(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.add_nogood(["a", "b"])
        result = pg_api.export_network()
        assert len(result["nogoods"]) == 1
        assert set(result["nogoods"][0]["nodes"]) == {"a", "b"}

    def test_export_source_fields(self, pg_api):
        pg_api.add_node("a", "Alpha", source="test.py", source_url="https://example.com")
        result = pg_api.export_network()
        assert result["nodes"]["a"]["source"] == "test.py"
        assert result["nodes"]["a"]["source_url"] == "https://example.com"

    def test_export_filters_private_metadata(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.retract_node("a", reason="test")
        result = pg_api.export_network()
        assert "_retracted" not in result["nodes"]["a"]["metadata"]

    def test_export_visible_to(self, pg_api):
        pg_api.add_node("a", "Alpha", access_tags=["billing"])
        pg_api.add_node("b", "Beta", access_tags=["ops"])
        result = pg_api.export_network(visible_to=["billing"])
        assert "a" in result["nodes"]
        assert "b" not in result["nodes"]


class TestRemoveJustification:

    def test_remove_one_of_two(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.add_node("c", "Gamma", sl="a")
        pg_api.add_justification("c", sl="b")
        result = pg_api.remove_justification("c", 0)
        assert result["remaining"] == 1
        assert result["removed"]["antecedents"] == ["a"]

    def test_remove_last_raises(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("c", "Gamma", sl="a")
        with pytest.raises(ValueError, match="only one justification"):
            pg_api.remove_justification("c", 0)

    def test_remove_premise_raises(self, pg_api):
        pg_api.add_node("a", "Alpha")
        with pytest.raises(ValueError, match="premise"):
            pg_api.remove_justification("a", 0)

    def test_remove_invalid_index(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.add_node("c", "Gamma", sl="a")
        pg_api.add_justification("c", sl="b")
        with pytest.raises(IndexError):
            pg_api.remove_justification("c", 5)

    def test_remove_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.remove_justification("nonexistent", 0)

    def test_remove_causes_truth_change(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.retract_node("b")
        pg_api.add_node("c", "Gamma", sl="a")
        pg_api.add_justification("c", sl="b")
        # c is IN via first justification (a is IN)
        result = pg_api.remove_justification("c", 0)
        # Now only justification is sl=b, but b is OUT → c goes OUT
        assert result["old_truth_value"] == "IN"
        assert result["new_truth_value"] == "OUT"

    def test_remove_propagates(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.add_node("c", "Gamma", sl="a")
        pg_api.add_justification("c", sl="b")
        pg_api.add_node("d", "Delta", sl="c")
        pg_api.retract_node("b")
        # Remove first justification (sl=a), leaving only sl=b (b is OUT)
        result = pg_api.remove_justification("c", 0)
        assert result["new_truth_value"] == "OUT"
        assert "d" in result["changed"]


class TestUpdateNode:

    def test_update_text_rejected(self, pg_api):
        pg_api.add_node("a", "Alpha")
        with pytest.raises(ValueError, match="immutable"):
            pg_api.update_node("a", text="Alpha updated")

    def test_update_source(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.update_node("a", source="test.py", source_url="https://example.com")
        assert "source" in result["updated_fields"]
        assert "source_url" in result["updated_fields"]
        node = pg_api.show_node("a")
        assert node["source"] == "test.py"
        assert node["source_url"] == "https://example.com"

    def test_update_nothing(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.update_node("a")
        assert result["updated_fields"] == []

    def test_update_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.update_node("nonexistent", source="test.py")


class TestConvertToPremise:

    def test_convert_derived_to_premise(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        result = pg_api.convert_to_premise("b")
        assert result["old_justifications"] == 1
        assert result["truth_value"] == "IN"
        node = pg_api.show_node("b")
        assert node["justifications"] == []

    def test_convert_out_to_in(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.retract_node("a")
        assert pg_api.show_node("b")["truth_value"] == "OUT"
        result = pg_api.convert_to_premise("b")
        assert result["truth_value"] == "IN"
        assert "b" in result["changed"]

    def test_convert_propagates(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.add_node("c", "Gamma", sl="b")
        pg_api.retract_node("a")
        assert pg_api.show_node("c")["truth_value"] == "OUT"
        result = pg_api.convert_to_premise("b")
        assert "c" in result["changed"]
        assert pg_api.show_node("c")["truth_value"] == "IN"

    def test_convert_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.convert_to_premise("nonexistent")

    def test_convert_premise_noop(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.convert_to_premise("a")
        assert result["old_justifications"] == 0
        assert result["truth_value"] == "IN"


class TestGetBeliefSet:

    def test_empty(self, pg_api):
        result = pg_api.get_belief_set()
        assert result == []

    def test_all_in(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        assert set(pg_api.get_belief_set()) == {"a", "b"}

    def test_mixed(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta")
        pg_api.retract_node("b")
        result = pg_api.get_belief_set()
        assert result == ["a"]


class TestPropagate:

    def test_no_changes(self, pg_api):
        pg_api.add_node("a", "Alpha")
        result = pg_api.propagate()
        assert result["changed"] == []

    def test_fixes_stale_truth(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        with pg_api.conn.cursor() as cur:
            cur.execute(
                "UPDATE rms_nodes SET truth_value = 'OUT' "
                "WHERE id = 'b' AND project_id = %s",
                (pg_api.project_id,),
            )
        pg_api.conn.commit()
        result = pg_api.propagate()
        assert "b" in result["changed"]
        assert pg_api.show_node("b")["truth_value"] == "IN"

    def test_cascade(self, pg_api):
        pg_api.add_node("a", "Alpha")
        pg_api.add_node("b", "Beta", sl="a")
        pg_api.add_node("c", "Gamma", sl="b")
        with pg_api.conn.cursor() as cur:
            cur.execute(
                "UPDATE rms_nodes SET truth_value = 'OUT' "
                "WHERE id IN ('b', 'c') AND project_id = %s",
                (pg_api.project_id,),
            )
        pg_api.conn.commit()
        result = pg_api.propagate()
        assert set(result["changed"]) >= {"b", "c"}

    def test_empty_network(self, pg_api):
        result = pg_api.propagate()
        assert result["changed"] == []


class TestSupersede:

    def test_makes_old_out(self, pg_api):
        pg_api.add_node("old", "Old belief")
        pg_api.add_node("new", "New belief")
        result = pg_api.supersede("old", "new")
        assert result["old_id"] == "old"
        assert result["new_id"] == "new"
        assert "old" in result["changed"]
        assert pg_api.show_node("old")["truth_value"] == "OUT"

    def test_metadata(self, pg_api):
        pg_api.add_node("old", "Old belief")
        pg_api.add_node("new", "New belief")
        pg_api.supersede("old", "new")
        old = pg_api.show_node("old")
        new = pg_api.show_node("new")
        assert old["metadata"]["superseded_by"] == "new"
        assert "old" in new["metadata"]["supersedes"]

    def test_reversible(self, pg_api):
        pg_api.add_node("old", "Old")
        pg_api.add_node("new", "New")
        pg_api.supersede("old", "new")
        assert pg_api.show_node("old")["truth_value"] == "OUT"
        pg_api.retract_node("new")
        assert pg_api.show_node("old")["truth_value"] == "IN"

    def test_cascade(self, pg_api):
        pg_api.add_node("old", "Old")
        pg_api.add_node("dep", "Depends on old", sl="old")
        pg_api.add_node("new", "New")
        result = pg_api.supersede("old", "new")
        assert "dep" in result["changed"]
        assert pg_api.show_node("dep")["truth_value"] == "OUT"

    def test_not_found(self, pg_api):
        pg_api.add_node("a", "Alpha")
        with pytest.raises(KeyError):
            pg_api.supersede("a", "missing")
        with pytest.raises(KeyError):
            pg_api.supersede("missing", "a")


class TestSummarize:

    def test_create_summary(self, pg_api):
        pg_api.add_node("a", "Premise A")
        pg_api.add_node("b", "Premise B")
        result = pg_api.summarize("s", "Summary of A and B", over=["a", "b"])
        assert result["summary_id"] == "s"
        assert result["truth_value"] == "IN"
        assert result["over"] == ["a", "b"]

    def test_metadata(self, pg_api):
        pg_api.add_node("a", "A")
        pg_api.add_node("b", "B")
        pg_api.summarize("s", "Summary", over=["a", "b"])
        summary = pg_api.show_node("s")
        assert summary["metadata"]["summarizes"] == ["a", "b"]
        a = pg_api.show_node("a")
        assert "s" in a["metadata"]["summarized_by"]

    def test_out_when_antecedent_out(self, pg_api):
        pg_api.add_node("a", "A")
        pg_api.add_node("b", "B")
        pg_api.summarize("s", "Summary", over=["a", "b"])
        pg_api.retract_node("a")
        assert pg_api.show_node("s")["truth_value"] == "OUT"

    def test_duplicate_raises(self, pg_api):
        pg_api.add_node("a", "A")
        pg_api.summarize("s", "Summary", over=["a"])
        with pytest.raises(ValueError):
            pg_api.summarize("s", "Dup", over=["a"])

    def test_missing_node_raises(self, pg_api):
        pg_api.add_node("a", "A")
        with pytest.raises(KeyError):
            pg_api.summarize("s", "Summary", over=["a", "missing"])


class TestTraceAccessTags:

    def test_premise_returns_own_tags(self, pg_api):
        pg_api.add_node("a", "A", access_tags=["finance"])
        result = pg_api.trace_access_tags("a")
        assert result["access_tags"] == ["finance"]

    def test_no_tags_returns_empty(self, pg_api):
        pg_api.add_node("a", "A")
        result = pg_api.trace_access_tags("a")
        assert result["access_tags"] == []

    def test_chain_collects_all(self, pg_api):
        pg_api.add_node("a", "A", access_tags=["finance"])
        pg_api.add_node("b", "B", sl="a", access_tags=["hr"])
        pg_api.add_node("c", "C", sl="b")
        result = pg_api.trace_access_tags("c")
        assert result["access_tags"] == ["finance", "hr"]

    def test_not_found(self, pg_api):
        with pytest.raises(KeyError):
            pg_api.trace_access_tags("missing")

    def test_visible_to_denied(self, pg_api):
        pg_api.add_node("a", "A", access_tags=["finance"])
        with pytest.raises(PermissionError):
            pg_api.trace_access_tags("a", visible_to=["hr"])


class TestEnsureNamespace:

    def test_creates_active_node(self, pg_api):
        result = pg_api.ensure_namespace("agent1")
        assert result["namespace"] == "agent1"
        assert result["active_node"] == "agent1:active"
        assert result["created"] is True
        node = pg_api.show_node("agent1:active")
        assert node["truth_value"] == "IN"
        meta = node["metadata"]
        assert meta["agent"] == "agent1"
        assert meta["role"] == "agent_premise"

    def test_idempotent(self, pg_api):
        pg_api.ensure_namespace("agent1")
        result = pg_api.ensure_namespace("agent1")
        assert result["created"] is False
        assert result["active_node"] == "agent1:active"

    def test_multiple_namespaces(self, pg_api):
        pg_api.ensure_namespace("alice")
        pg_api.ensure_namespace("bob")
        n1 = pg_api.show_node("alice:active")
        n2 = pg_api.show_node("bob:active")
        assert n1["truth_value"] == "IN"
        assert n2["truth_value"] == "IN"


class TestListNamespaces:

    def test_empty(self, pg_api):
        result = pg_api.list_namespaces()
        assert result["namespaces"] == []

    def test_with_counts(self, pg_api):
        pg_api.add_node("a", "A", namespace="ns1")
        pg_api.add_node("b", "B", namespace="ns1")
        pg_api.add_node("c", "C", namespace="ns2")
        result = pg_api.list_namespaces()
        ns_map = {ns["namespace"]: ns for ns in result["namespaces"]}
        assert "ns1" in ns_map
        assert "ns2" in ns_map
        assert ns_map["ns1"]["total_beliefs"] == 2
        assert ns_map["ns1"]["in_beliefs"] == 2
        assert ns_map["ns1"]["active"] is True
        assert ns_map["ns2"]["total_beliefs"] == 1

    def test_inactive_namespace(self, pg_api):
        pg_api.add_node("a", "A", namespace="ns1")
        pg_api.retract_node("ns1:active")
        result = pg_api.list_namespaces()
        ns = result["namespaces"][0]
        assert ns["active"] is False
        assert ns["in_beliefs"] == 0


class TestNamespaceAddNode:

    def test_prefixes_node_id(self, pg_api):
        result = pg_api.add_node("belief1", "A belief", namespace="agent1")
        assert result["node_id"] == "agent1:belief1"

    def test_auto_creates_premise(self, pg_api):
        pg_api.add_node("belief1", "A belief", namespace="agent1")
        node = pg_api.show_node("agent1:active")
        assert node["truth_value"] == "IN"
        assert node["metadata"]["role"] == "agent_premise"

    def test_cascade_on_retract(self, pg_api):
        pg_api.add_node("belief1", "A belief", namespace="agent1")
        assert pg_api.show_node("agent1:belief1")["truth_value"] == "IN"
        pg_api.retract_node("agent1:active")
        assert pg_api.show_node("agent1:belief1")["truth_value"] == "OUT"

    def test_no_double_prefix(self, pg_api):
        result = pg_api.add_node("agent1:belief1", "Already prefixed", namespace="agent1")
        assert result["node_id"] == "agent1:belief1"

    def test_resolves_antecedent_ids(self, pg_api):
        pg_api.add_node("premise", "P", namespace="agent1")
        result = pg_api.add_node("derived", "D", sl="premise", namespace="agent1")
        assert result["node_id"] == "agent1:derived"
        assert result["truth_value"] == "IN"
        info = pg_api.explain_node("agent1:derived")
        antecedents = info["steps"][0].get("antecedents", [])
        assert "agent1:premise" in antecedents
        assert "agent1:active" in antecedents


class TestNamespaceAddJustification:

    def test_prefixes_ids(self, pg_api):
        pg_api.add_node("a", "A", namespace="ns1")
        pg_api.add_node("b", "B", namespace="ns1")
        result = pg_api.add_justification("b", sl="a", namespace="ns1")
        assert result["node_id"] == "ns1:b"

    def test_works_with_namespaced_nodes(self, pg_api):
        pg_api.add_node("a", "A", namespace="ns1")
        pg_api.add_node("b", "B", namespace="ns1")
        pg_api.retract_node("ns1:b")
        assert pg_api.show_node("ns1:b")["truth_value"] == "OUT"
        pg_api.assert_node("ns1:b")
        result = pg_api.add_justification("b", sl="a", namespace="ns1")
        assert result["new_truth_value"] == "IN"

"""Tests for access control via data source provenance tags (issue #38)."""

import pytest

from reasonsforge import Justification
from reasonsforge.network import Network
from reasonsforge import api


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


class TestInheritance:

    def test_derived_inherits_single_parent_tags(self):
        net = Network()
        net.add_node("a", "Premise A", metadata={"access_tags": ["finance"]})
        j = Justification(type="SL", antecedents=["a"], outlist=[], label="")
        net.add_node("b", "Derived B", justifications=[j])

        assert net.nodes["b"].metadata.get("access_tags") == ["finance"]

    def test_derived_inherits_union_of_parents(self):
        net = Network()
        net.add_node("a", "A", metadata={"access_tags": ["finance"]})
        net.add_node("b", "B", metadata={"access_tags": ["hr"]})
        j = Justification(type="SL", antecedents=["a", "b"], outlist=[], label="")
        net.add_node("c", "Derived C", justifications=[j])

        assert net.nodes["c"].metadata.get("access_tags") == ["finance", "hr"]

    def test_derived_merges_explicit_and_inherited_tags(self):
        net = Network()
        net.add_node("a", "A", metadata={"access_tags": ["finance"]})
        j = Justification(type="SL", antecedents=["a"], outlist=[], label="")
        net.add_node("b", "B", justifications=[j], metadata={"access_tags": ["ops"]})

        assert net.nodes["b"].metadata.get("access_tags") == ["finance", "ops"]

    def test_premise_does_not_inherit(self):
        net = Network()
        net.add_node("a", "Premise")

        assert "access_tags" not in net.nodes["a"].metadata

    def test_chain_inheritance(self):
        net = Network()
        net.add_node("a", "A", metadata={"access_tags": ["finance"]})
        j1 = Justification(type="SL", antecedents=["a"], outlist=[], label="")
        net.add_node("b", "B", justifications=[j1])
        j2 = Justification(type="SL", antecedents=["b"], outlist=[], label="")
        net.add_node("c", "C", justifications=[j2])

        assert net.nodes["b"].metadata.get("access_tags") == ["finance"]
        assert net.nodes["c"].metadata.get("access_tags") == ["finance"]

    def test_diamond_inheritance(self):
        net = Network()
        net.add_node("a", "A", metadata={"access_tags": ["finance"]})
        net.add_node("b", "B", metadata={"access_tags": ["hr"]})
        j1 = Justification(type="SL", antecedents=["a"], outlist=[], label="")
        net.add_node("m1", "M1", justifications=[j1])
        j2 = Justification(type="SL", antecedents=["b"], outlist=[], label="")
        net.add_node("m2", "M2", justifications=[j2])
        j3 = Justification(type="SL", antecedents=["m1", "m2"], outlist=[], label="")
        net.add_node("c", "C", justifications=[j3])

        assert net.nodes["c"].metadata.get("access_tags") == ["finance", "hr"]

    def test_no_tags_no_inheritance(self):
        net = Network()
        net.add_node("a", "A")
        j = Justification(type="SL", antecedents=["a"], outlist=[], label="")
        net.add_node("b", "B", justifications=[j])

        assert "access_tags" not in net.nodes["b"].metadata

    def test_add_justification_updates_tags(self):
        net = Network()
        net.add_node("a", "A", metadata={"access_tags": ["finance"]})
        net.add_node("b", "B", metadata={"access_tags": ["hr"]})
        j1 = Justification(type="SL", antecedents=["a"], outlist=[], label="")
        net.add_node("c", "C", justifications=[j1])

        assert net.nodes["c"].metadata.get("access_tags") == ["finance"]

        j2 = Justification(type="SL", antecedents=["b"], outlist=[], label="")
        net.add_justification("c", j2)

        assert net.nodes["c"].metadata.get("access_tags") == ["finance", "hr"]

    def test_add_justification_propagates_tags_to_dependents(self):
        net = Network()
        net.add_node("a", "A")
        j1 = Justification(type="SL", antecedents=["a"], outlist=[], label="")
        net.add_node("b", "B", justifications=[j1])
        j2 = Justification(type="SL", antecedents=["b"], outlist=[], label="")
        net.add_node("c", "C", justifications=[j2])

        assert "access_tags" not in net.nodes["b"].metadata
        assert "access_tags" not in net.nodes["c"].metadata

        # Now give "a" access tags via a new justification from a tagged node
        net.add_node("secret", "Secret", metadata={"access_tags": ["finance"]})
        j3 = Justification(type="SL", antecedents=["secret"], outlist=[], label="")
        net.add_justification("a", j3)

        assert net.nodes["a"].metadata.get("access_tags") == ["finance"]
        assert net.nodes["b"].metadata.get("access_tags") == ["finance"]
        assert net.nodes["c"].metadata.get("access_tags") == ["finance"]

    def test_tags_persist_through_storage(self, db_path):
        api.add_node("a", "Premise A", access_tags=["finance"], db_path=db_path)
        api.add_node("b", "Derived B", sl="a", db_path=db_path)

        node_a = api.show_node("a", db_path=db_path)
        node_b = api.show_node("b", db_path=db_path)

        assert node_a["metadata"]["access_tags"] == ["finance"]
        assert node_b["metadata"]["access_tags"] == ["finance"]


class TestVisibleTo:

    def test_list_nodes_filters_by_access(self, db_path):
        api.add_node("pub", "Public", db_path=db_path)
        api.add_node("fin", "Finance data", access_tags=["finance"], db_path=db_path)
        api.add_node("hr", "HR data", access_tags=["hr"], db_path=db_path)

        result = api.list_nodes(visible_to=["finance"], db_path=db_path)
        ids = {n["id"] for n in result["nodes"]}
        assert "pub" in ids
        assert "fin" in ids
        assert "hr" not in ids

    def test_list_nodes_visible_to_none_shows_all(self, db_path):
        api.add_node("pub", "Public", db_path=db_path)
        api.add_node("fin", "Finance", access_tags=["finance"], db_path=db_path)

        result = api.list_nodes(db_path=db_path)
        assert result["count"] == 2

    def test_list_nodes_untagged_always_visible(self, db_path):
        api.add_node("a", "No tags", db_path=db_path)

        result = api.list_nodes(visible_to=["anything"], db_path=db_path)
        assert result["count"] == 1

    def test_list_nodes_subset_match(self, db_path):
        api.add_node("a", "Finance", access_tags=["finance"], db_path=db_path)

        result = api.list_nodes(visible_to=["finance", "hr"], db_path=db_path)
        assert result["count"] == 1

    def test_list_nodes_multi_tag_requires_all(self, db_path):
        api.add_node("a", "Multi-source", access_tags=["finance", "hr"], db_path=db_path)

        only_finance = api.list_nodes(visible_to=["finance"], db_path=db_path)
        assert only_finance["count"] == 0

        both = api.list_nodes(visible_to=["finance", "hr"], db_path=db_path)
        assert both["count"] == 1

    def test_show_node_raises_on_forbidden(self, db_path):
        api.add_node("fin", "Finance", access_tags=["finance"], db_path=db_path)

        with pytest.raises(PermissionError):
            api.show_node("fin", visible_to=["hr"], db_path=db_path)

    def test_show_node_allowed(self, db_path):
        api.add_node("fin", "Finance", access_tags=["finance"], db_path=db_path)

        node = api.show_node("fin", visible_to=["finance"], db_path=db_path)
        assert node["id"] == "fin"

    def test_show_node_no_filter(self, db_path):
        api.add_node("fin", "Finance", access_tags=["finance"], db_path=db_path)

        node = api.show_node("fin", db_path=db_path)
        assert node["id"] == "fin"

    def test_lookup_respects_visible_to(self, db_path):
        api.add_node("fin-data", "Finance data point", access_tags=["finance"], db_path=db_path)
        api.add_node("pub-data", "Public data point", db_path=db_path)

        result = api.lookup("data", visible_to=["public"], db_path=db_path)
        assert "pub-data" in result
        assert "fin-data" not in result

    def test_search_respects_visible_to(self, db_path):
        api.add_node("secret-item", "Secret finance item", access_tags=["finance"], db_path=db_path)
        api.add_node("public-item", "Public item", db_path=db_path)

        result = api.search("item", visible_to=["public"], db_path=db_path)
        assert "public-item" in result
        assert "secret-item" not in result

    def test_search_neighbors_filtered(self, db_path):
        api.add_node("secret", "Secret finance data", access_tags=["finance"], db_path=db_path)
        api.add_node("public", "Public derived fact", sl="secret", db_path=db_path)

        result = api.search("public", visible_to=["public"], db_path=db_path)
        assert "secret" not in result

    def test_status_respects_visible_to(self, db_path):
        api.add_node("pub", "Public", db_path=db_path)
        api.add_node("fin", "Finance", access_tags=["finance"], db_path=db_path)

        result = api.get_status(visible_to=["public"], db_path=db_path)
        ids = {n["id"] for n in result["nodes"]}
        assert "pub" in ids
        assert "fin" not in ids
        assert result["total"] == 1

    def test_explain_raises_on_forbidden(self, db_path):
        api.add_node("fin", "Finance", access_tags=["finance"], db_path=db_path)

        with pytest.raises(PermissionError):
            api.explain_node("fin", visible_to=["hr"], db_path=db_path)

    def test_explain_inherits_restriction_from_antecedent(self, db_path):
        """Tag inheritance prevents explain from leaking restricted antecedents.

        A derived node inherits its parent's tags, so a caller without
        access to the parent also can't explain the derived node.
        """
        api.add_node("secret", "Secret finance data", access_tags=["finance"], db_path=db_path)
        api.add_node("derived", "Derived from secret", sl="secret", db_path=db_path)

        # derived inherits ["finance"] — caller without finance can't explain it
        with pytest.raises(PermissionError):
            api.explain_node("derived", visible_to=["public"], db_path=db_path)

        # caller with finance can explain and sees the antecedent (allowed)
        result = api.explain_node("derived", visible_to=["finance"], db_path=db_path)
        step_ids = [s["node"] for s in result["steps"]]
        assert "derived" in step_ids
        assert "secret" in step_ids

    def test_trace_raises_on_forbidden(self, db_path):
        api.add_node("fin", "Finance", access_tags=["finance"], db_path=db_path)
        api.add_node("derived", "Derived", sl="fin", db_path=db_path)

        with pytest.raises(PermissionError):
            api.trace_assumptions("derived", visible_to=["public"], db_path=db_path)

    def test_trace_filters_premises(self, db_path):
        api.add_node("pub", "Public premise", db_path=db_path)
        api.add_node("fin", "Finance premise", access_tags=["finance"], db_path=db_path)
        api.add_node("derived", "Derived", sl="pub,fin", db_path=db_path)

        result = api.trace_assumptions("derived", visible_to=["finance", "public"], db_path=db_path)
        assert "pub" in result["premises"]
        assert "fin" in result["premises"]

        # derived inherits ["finance"] from fin, so visible_to=["public"] can't see it
        with pytest.raises(PermissionError):
            api.trace_assumptions("derived", visible_to=["public"], db_path=db_path)

    def test_export_network_respects_visible_to(self, db_path):
        api.add_node("pub", "Public", db_path=db_path)
        api.add_node("fin", "Finance", access_tags=["finance"], db_path=db_path)

        data = api.export_network(visible_to=["public"], db_path=db_path)
        assert "pub" in data["nodes"]
        assert "fin" not in data["nodes"]

    def test_export_network_filters_nogoods(self, db_path):
        api.add_node("pub", "Public", db_path=db_path)
        api.add_node("fin", "Finance", access_tags=["finance"], db_path=db_path)
        api.add_nogood(["pub", "fin"], db_path=db_path)

        data = api.export_network(visible_to=["public"], db_path=db_path)
        assert len(data["nogoods"]) == 0

        data_all = api.export_network(visible_to=["finance", "public"], db_path=db_path)
        assert len(data_all["nogoods"]) == 1

    def test_export_markdown_respects_visible_to(self, db_path):
        api.add_node("pub", "Public", db_path=db_path)
        api.add_node("fin", "Finance", access_tags=["finance"], db_path=db_path)

        md = api.export_markdown(visible_to=["public"], db_path=db_path)
        assert "pub" in md
        assert "fin" not in md

    def test_compact_respects_visible_to(self, db_path):
        api.add_node("pub", "Public belief", db_path=db_path)
        api.add_node("fin", "Finance belief", access_tags=["finance"], db_path=db_path)

        result = api.compact(visible_to=["public"], db_path=db_path)
        assert "pub" in result
        assert "fin" not in result


class TestTraceAccessTags:

    def test_premise_returns_own_tags(self):
        net = Network()
        net.add_node("a", "A", metadata={"access_tags": ["finance"]})

        assert net.trace_access_tags("a") == ["finance"]

    def test_no_tags_returns_empty(self):
        net = Network()
        net.add_node("a", "A")

        assert net.trace_access_tags("a") == []

    def test_chain_collects_all_tags(self):
        net = Network()
        net.add_node("a", "A", metadata={"access_tags": ["finance"]})
        j1 = Justification(type="SL", antecedents=["a"], outlist=[], label="")
        net.add_node("b", "B", justifications=[j1], metadata={"access_tags": ["hr"]})
        j2 = Justification(type="SL", antecedents=["b"], outlist=[], label="")
        net.add_node("c", "C", justifications=[j2])

        assert net.trace_access_tags("c") == ["finance", "hr"]

    def test_diamond_collects_union(self):
        net = Network()
        net.add_node("a", "A", metadata={"access_tags": ["finance"]})
        net.add_node("b", "B", metadata={"access_tags": ["hr"]})
        j = Justification(type="SL", antecedents=["a", "b"], outlist=[], label="")
        net.add_node("c", "C", justifications=[j])

        assert net.trace_access_tags("c") == ["finance", "hr"]

    def test_missing_node_raises(self):
        net = Network()
        with pytest.raises(KeyError):
            net.trace_access_tags("nonexistent")

    def test_api_trace_access_tags(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("a", "Finance premise", access_tags=["finance"], db_path=db)
        api.add_node("b", "Derived", sl="a", db_path=db)

        result = api.trace_access_tags("b", db_path=db)
        assert result["node_id"] == "b"
        assert result["access_tags"] == ["finance"]

    def test_api_trace_access_tags_raises_on_forbidden(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("a", "Finance premise", access_tags=["finance"], db_path=db)

        with pytest.raises(PermissionError):
            api.trace_access_tags("a", visible_to=["hr"], db_path=db)

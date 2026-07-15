"""Tests for export_card — HuggingFace EEM card generation."""

import yaml

from reasonsforge import Justification
from reasonsforge.metadata import SCHEMA_VERSION
from reasonsforge.network import Network
from reasonsforge.export_card import export_card


def _parse_frontmatter(text):
    """Extract YAML frontmatter from card text."""
    parts = text.split("---", 2)
    assert len(parts) >= 3, "Missing YAML frontmatter delimiters"
    return yaml.safe_load(parts[1])


def _make_network(nodes=None, project_name="test-project", nogoods=None):
    net = Network()
    net.meta = {"project_name": project_name}
    if nodes:
        for nid, data in nodes.items():
            from reasonsforge import Node
            node = Node(nid, data.get("text", f"Text for {nid}"))
            node.truth_value = data.get("truth_value", "IN")
            for jdata in data.get("justifications", []):
                j = Justification(
                    type=jdata.get("type", "SL"),
                    antecedents=jdata.get("antecedents", []),
                    outlist=jdata.get("outlist", []),
                )
                node.justifications.append(j)
            net.nodes[nid] = node
    if nogoods:
        from reasonsforge import Nogood
        for ng in nogoods:
            net.nogoods.append(Nogood(**ng))
    return net


class TestFrontmatter:

    def test_schema_version(self):
        net = _make_network()
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["schema_version"] == SCHEMA_VERSION

    def test_type_is_eem(self):
        net = _make_network()
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["type"] == "eem"

    def test_project_name(self):
        net = _make_network(project_name="my-expert")
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["project_name"] == "my-expert"

    def test_domain_tags(self):
        net = _make_network()
        card = export_card(net, domain=["kubernetes", "devops"])
        fm = _parse_frontmatter(card)
        assert fm["domain"] == ["kubernetes", "devops"]

    def test_no_domain_defaults_to_empty(self):
        net = _make_network()
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["domain"] == []

    def test_license(self):
        net = _make_network()
        card = export_card(net, license="apache-2.0")
        fm = _parse_frontmatter(card)
        assert fm["license"] == "apache-2.0"

    def test_default_license(self):
        net = _make_network()
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["license"] == "mit"

    def test_base_network(self):
        net = _make_network()
        card = export_card(net, base_network="user/parent-eem")
        fm = _parse_frontmatter(card)
        assert fm["base_network"] == "user/parent-eem"

    def test_no_base_network(self):
        net = _make_network()
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["base_network"] is None

    def test_source_repos(self):
        net = _make_network()
        card = export_card(net, source_repos=["user/repo1", "user/repo2"])
        fm = _parse_frontmatter(card)
        assert fm["source_repos"] == ["user/repo1", "user/repo2"]

    def test_generator(self):
        net = _make_network()
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert "reasonsforge/" in fm["generator"]


class TestBeliefCounts:

    def test_all_in(self):
        net = _make_network(nodes={
            "a": {"text": "A"},
            "b": {"text": "B"},
        })
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["beliefs_total"] == 2
        assert fm["beliefs_in"] == 2
        assert fm["beliefs_out"] == 0

    def test_mixed_in_out(self):
        net = _make_network(nodes={
            "a": {"text": "A", "truth_value": "IN"},
            "b": {"text": "B", "truth_value": "OUT"},
            "c": {"text": "C", "truth_value": "IN"},
        })
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["beliefs_total"] == 3
        assert fm["beliefs_in"] == 2
        assert fm["beliefs_out"] == 1

    def test_premises_and_derived(self):
        net = _make_network(nodes={
            "premise-a": {"text": "A"},
            "premise-b": {"text": "B"},
            "derived-c": {
                "text": "C",
                "justifications": [{"antecedents": ["premise-a", "premise-b"]}],
            },
        })
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["premises"] == 2
        assert fm["derived"] == 1

    def test_nogoods_count(self):
        net = _make_network(
            nodes={"a": {"text": "A"}, "b": {"text": "B"}},
            nogoods=[{"id": "ng1", "nodes": ["a", "b"], "discovered": "", "resolution": ""}],
        )
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["nogoods"] == 1


class TestEmptyNetwork:

    def test_empty_produces_valid_card(self):
        net = _make_network()
        card = export_card(net)
        fm = _parse_frontmatter(card)
        assert fm["beliefs_total"] == 0
        assert fm["beliefs_in"] == 0
        assert fm["premises"] == 0
        assert fm["derived"] == 0
        assert fm["nogoods"] == 0

    def test_empty_has_title(self):
        net = _make_network(project_name="my-project")
        card = export_card(net)
        assert "# My Project" in card


class TestMaxDepth:

    def test_premises_only_depth_zero(self):
        net = _make_network(nodes={"a": {"text": "A"}, "b": {"text": "B"}})
        card = export_card(net)
        assert "| Max derivation depth | 0 |" in card

    def test_single_level_derivation(self):
        net = _make_network(nodes={
            "a": {"text": "A"},
            "d": {"text": "D", "justifications": [{"antecedents": ["a"]}]},
        })
        card = export_card(net)
        assert "| Max derivation depth | 1 |" in card

    def test_multi_level_derivation(self):
        net = _make_network(nodes={
            "a": {"text": "A"},
            "b": {"text": "B", "justifications": [{"antecedents": ["a"]}]},
            "c": {"text": "C", "justifications": [{"antecedents": ["b"]}]},
        })
        card = export_card(net)
        assert "| Max derivation depth | 2 |" in card


class TestMarkdownBody:

    def test_stats_table(self):
        net = _make_network(nodes={"x": {"text": "X"}})
        card = export_card(net)
        assert "| Total beliefs | 1 |" in card
        assert "| Premises (observations) | 1 |" in card

    def test_retraction_rate(self):
        net = _make_network(nodes={
            "a": {"text": "A", "truth_value": "IN"},
            "b": {"text": "B", "truth_value": "OUT"},
        })
        card = export_card(net)
        assert "| Retraction rate | 50% |" in card

    def test_how_to_use_section(self):
        net = _make_network()
        card = export_card(net)
        assert "## How to Use" in card
        assert "reasons import-json network.json" in card

    def test_quality_section(self):
        net = _make_network(nodes={"a": {"text": "A"}})
        card = export_card(net)
        assert "## Quality" in card
        assert "1 beliefs IN, 0 OUT" in card

    def test_license_section(self):
        net = _make_network()
        card = export_card(net, license="apache-2.0")
        assert "## License" in card
        assert "apache-2.0" in card

    def test_title_from_project_name(self):
        net = _make_network(project_name="k8s-ops-expert")
        card = export_card(net)
        assert "# K8S Ops Expert" in card

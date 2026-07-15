"""Tests for importing beliefs.md into RMS."""

from pathlib import Path

import pytest

from reasonsforge.network import Network
from reasonsforge.import_beliefs import parse_beliefs, parse_nogoods, import_into_network


SAMPLE_BELIEFS = """\
# Belief Registry

## Repos
- physics: ~/git/physics

## Claims

### premise-a [IN] OBSERVATION
First premise with no dependencies
- Source: repo/file.md
- Source hash: abc123
- Date: 2026-03-17

### premise-b [IN] OBSERVATION
Second independent premise
- Source: repo/other.md
- Source hash: def456
- Date: 2026-03-17

### derived-c [IN] DERIVED
Derived from both premises
- Source: repo/entry.md
- Source hash: ghi789
- Date: 2026-03-17
- Depends on: premise-a, premise-b

### stale-d [STALE] DERIVED
This was superseded by new evidence
- Source: repo/old.md
- Date: 2026-03-15
- Stale reason: New data invalidated this
- Superseded by: derived-c
- Depends on: premise-a

### chain-e [IN] DERIVED
Depends on derived-c (depth 2)
- Source: repo/chain.md
- Date: 2026-03-17
- Depends on: derived-c
"""

SAMPLE_NOGOODS = """\
# Nogoods

### nogood-001: A contradicts B
- Discovered: 2026-03-15
- Discovered by: test
- Resolution: Choose one
- Affects: premise-a, premise-b

### nogood-002: Stale vs new
- Discovered: 2026-03-17
- Discovered by: experiment
- Resolution: New evidence wins
- Affects: stale-d, derived-c
"""


class TestParseBeliefs:

    def test_parses_all_claims(self):
        claims = parse_beliefs(SAMPLE_BELIEFS)
        assert len(claims) == 5

    def test_parses_id_and_status(self):
        claims = parse_beliefs(SAMPLE_BELIEFS)
        by_id = {c["id"]: c for c in claims}
        assert by_id["premise-a"]["status"] == "IN"
        assert by_id["stale-d"]["status"] == "STALE"

    def test_parses_type(self):
        claims = parse_beliefs(SAMPLE_BELIEFS)
        by_id = {c["id"]: c for c in claims}
        assert by_id["premise-a"]["type"] == "OBSERVATION"
        assert by_id["derived-c"]["type"] == "DERIVED"

    def test_parses_text(self):
        claims = parse_beliefs(SAMPLE_BELIEFS)
        by_id = {c["id"]: c for c in claims}
        assert by_id["premise-a"]["text"] == "First premise with no dependencies"

    def test_parses_source(self):
        claims = parse_beliefs(SAMPLE_BELIEFS)
        by_id = {c["id"]: c for c in claims}
        assert by_id["premise-a"]["source"] == "repo/file.md"
        assert by_id["premise-a"]["source_hash"] == "abc123"

    def test_parses_depends_on(self):
        claims = parse_beliefs(SAMPLE_BELIEFS)
        by_id = {c["id"]: c for c in claims}
        assert by_id["derived-c"]["depends_on"] == ["premise-a", "premise-b"]
        assert by_id["premise-a"]["depends_on"] == []

    def test_parses_unless(self):
        text = """\
### gated [IN] DERIVED
A gated belief
- Depends on: fact-a
- Unless: blocker-x, blocker-y
"""
        claims = parse_beliefs(text)
        assert claims[0]["unless"] == ["blocker-x", "blocker-y"]

    def test_parses_unless_empty_default(self):
        claims = parse_beliefs(SAMPLE_BELIEFS)
        by_id = {c["id"]: c for c in claims}
        assert by_id["premise-a"]["unless"] == []

    def test_parses_stale_metadata(self):
        claims = parse_beliefs(SAMPLE_BELIEFS)
        by_id = {c["id"]: c for c in claims}
        assert by_id["stale-d"]["stale_reason"] == "New data invalidated this"
        assert by_id["stale-d"]["superseded_by"] == "derived-c"


class TestParseNogoods:

    def test_parses_all_nogoods(self):
        nogoods = parse_nogoods(SAMPLE_NOGOODS)
        assert len(nogoods) == 2

    def test_parses_id_and_label(self):
        nogoods = parse_nogoods(SAMPLE_NOGOODS)
        assert nogoods[0]["id"] == "nogood-001"
        assert nogoods[0]["label"] == "A contradicts B"

    def test_parses_affects(self):
        nogoods = parse_nogoods(SAMPLE_NOGOODS)
        assert nogoods[0]["affects"] == ["premise-a", "premise-b"]

    def test_parses_resolution(self):
        nogoods = parse_nogoods(SAMPLE_NOGOODS)
        assert nogoods[0]["resolution"] == "Choose one"


class TestImportIntoNetwork:

    def test_imports_premises(self):
        net = Network()
        result = import_into_network(net, SAMPLE_BELIEFS)
        assert "premise-a" in net.nodes
        assert "premise-b" in net.nodes
        assert net.nodes["premise-a"].truth_value == "IN"
        assert net.nodes["premise-a"].justifications == []  # premise

    def test_imports_derived_with_justification(self):
        net = Network()
        import_into_network(net, SAMPLE_BELIEFS)
        node = net.nodes["derived-c"]
        assert node.truth_value == "IN"
        assert len(node.justifications) == 1
        assert node.justifications[0].type == "SL"
        assert set(node.justifications[0].antecedents) == {"premise-a", "premise-b"}

    def test_stale_claims_retracted(self):
        net = Network()
        import_into_network(net, SAMPLE_BELIEFS)
        assert net.nodes["stale-d"].truth_value == "OUT"

    def test_stale_metadata_preserved(self):
        net = Network()
        import_into_network(net, SAMPLE_BELIEFS)
        assert net.nodes["stale-d"].metadata["stale_reason"] == "New data invalidated this"
        assert net.nodes["stale-d"].metadata["superseded_by"] == "derived-c"

    def test_dependency_chain(self):
        """chain-e depends on derived-c which depends on premise-a + premise-b."""
        net = Network()
        import_into_network(net, SAMPLE_BELIEFS)
        assert net.nodes["chain-e"].truth_value == "IN"
        assert net.nodes["chain-e"].justifications[0].antecedents == ["derived-c"]

    def test_retraction_cascades_from_stale(self):
        """stale-d is retracted but derived-c has independent justification."""
        net = Network()
        import_into_network(net, SAMPLE_BELIEFS)
        # derived-c depends on premise-a and premise-b (both IN), not on stale-d
        assert net.nodes["derived-c"].truth_value == "IN"

    def test_import_counts(self):
        net = Network()
        result = import_into_network(net, SAMPLE_BELIEFS)
        assert result["claims_imported"] == 5
        assert result["claims_retracted"] == 1  # stale-d
        assert result["claims_skipped"] == 0

    def test_skip_duplicates(self):
        net = Network()
        import_into_network(net, SAMPLE_BELIEFS)
        result = import_into_network(net, SAMPLE_BELIEFS)
        assert result["claims_imported"] == 0
        assert result["claims_skipped"] == 5

    def test_source_preserved(self):
        net = Network()
        import_into_network(net, SAMPLE_BELIEFS)
        assert net.nodes["premise-a"].source == "repo/file.md"
        assert net.nodes["premise-a"].source_hash == "abc123"
        assert net.nodes["premise-a"].date == "2026-03-17"

    def test_dependents_registered(self):
        """Reverse index: premise-a should list derived-c as dependent."""
        net = Network()
        import_into_network(net, SAMPLE_BELIEFS)
        assert "derived-c" in net.nodes["premise-a"].dependents

    def test_cascading_works_after_import(self):
        """Retract premise-a → derived-c and chain-e go OUT."""
        net = Network()
        import_into_network(net, SAMPLE_BELIEFS)
        changed = net.retract("premise-a")
        assert net.nodes["derived-c"].truth_value == "OUT"
        assert net.nodes["chain-e"].truth_value == "OUT"

    def test_import_nogoods(self):
        net = Network()
        result = import_into_network(net, SAMPLE_BELIEFS, SAMPLE_NOGOODS)
        assert result["nogoods_imported"] == 2
        assert len(net.nogoods) == 2
        assert net.nogoods[0].id == "nogood-001"
        assert net.nogoods[0].nodes == ["premise-a", "premise-b"]

    def test_nogoods_without_valid_nodes_skipped(self):
        """Nogoods referencing non-existent nodes are skipped."""
        bad_nogoods = """\
### nogood-001: Missing refs
- Discovered: 2026-03-17
- Resolution: n/a
- Affects: missing-x, missing-y
"""
        net = Network()
        import_into_network(net, SAMPLE_BELIEFS)
        result = import_into_network(net, "", bad_nogoods)
        assert result["nogoods_imported"] == 0


class TestImportRealRegistry:
    """Test with the actual beliefs-pi beliefs.md if available."""

    @pytest.fixture
    def real_beliefs(self):
        path = Path.home() / "git" / "beliefs-pi" / "beliefs.md"
        if not path.exists():
            pytest.skip("beliefs-pi not found")
        return path.read_text()

    @pytest.fixture
    def real_nogoods(self):
        path = Path.home() / "git" / "beliefs-pi" / "nogoods.md"
        if not path.exists():
            return None
        return path.read_text()

    def test_import_real_registry(self, real_beliefs, real_nogoods):
        net = Network()
        result = import_into_network(net, real_beliefs, real_nogoods)
        assert result["claims_imported"] >= 1
        assert len(net.nodes) >= result["claims_imported"]

        # Most claims should be IN
        in_count = sum(1 for n in net.nodes.values() if n.truth_value == "IN")
        out_count = sum(1 for n in net.nodes.values() if n.truth_value == "OUT")
        assert in_count > out_count

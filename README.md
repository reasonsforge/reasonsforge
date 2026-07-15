# Reasons Forge

A belief forge built on Doyle's (1979) Truth Maintenance System. Ingest domain sources, extract beliefs, derive new knowledge, and maintain consistency — all backed by automatic retraction cascades and dependency-directed backtracking.

## What It Does

Reasons Forge analyzes domain-specific sources (codebases, issue trackers, academic papers, documents) and builds a dependency network of beliefs. The TMS engine tracks which beliefs are IN (believed) or OUT (retracted), automatically propagating changes when evidence shifts.

**Core TMS engine:**
- Nodes with SL/CP justifications and non-monotonic reasoning (outlist)
- Retraction cascades and automatic restoration
- Nogoods with dependency-directed backtracking
- Dialectical argumentation (challenge/defend)
- Merkle integrity verification

**LLM-powered analysis:**
- `derive` — generate new beliefs from existing ones
- `review-beliefs` — validate derived beliefs
- `contradictions` — detect contradictions between IN beliefs
- `verify` — re-examine beliefs against source documents
- `repair` — fix invalid beliefs (search-and-link, soften, abandon)
- `deduplicate` — find and merge duplicate beliefs
- `ask` — natural language questions over the belief network

## Install

```bash
pip install reasonsforge
```

Or with uv:

```bash
uv tool install reasonsforge
```

## Quick Start

```bash
# Initialize database
reasonsforge init

# Add premises
reasonsforge add source-uses-langgraph "Source code uses LangGraph" --source "src/graph.py"
reasonsforge add graph-has-cycles "Graph contains cycles"

# Add derived nodes with SL justifications
reasonsforge add topology-is-static "Graph topology is static" --sl source-uses-langgraph
reasonsforge add no-runtime-modification "No runtime graph modification" --sl topology-is-static

# See what's believed
reasonsforge status
#   [+] graph-has-cycles: Graph contains cycles  (premise)
#   [+] no-runtime-modification: No runtime graph modification  (1 justification)
#   [+] source-uses-langgraph: Source code uses LangGraph  (premise)
#   [+] topology-is-static: Graph topology is static  (1 justification)
#
# 4/4 IN

# Retract a premise — cascade propagates
reasonsforge retract source-uses-langgraph
# Retracted: source-uses-langgraph, topology-is-static, no-runtime-modification

# Restore — dependents come back automatically
reasonsforge assert source-uses-langgraph
# Asserted: source-uses-langgraph, topology-is-static, no-runtime-modification

# Record a contradiction
reasonsforge add graph-is-dynamic "Graph is dynamically modified"
reasonsforge nogood topology-is-static graph-is-dynamic
# Recorded nogood-001: topology-is-static, graph-is-dynamic
# Retracted: graph-is-dynamic

# Explain why a node is IN or OUT
reasonsforge explain no-runtime-modification

# Challenge a belief — target goes OUT
reasonsforge challenge velocity-constraint "Not derived — postulated"

# Defend against a challenge — target restored
reasonsforge defend velocity-constraint challenge-velocity-constraint \
  "Follows from variational principle"

# Non-monotonic reasoning: believe X unless Y
reasonsforge add default-approx "Newtonian approximation holds" --unless strong-field

# LLM-powered derivation
reasonsforge derive --sample -m claude

# Review derived beliefs
reasonsforge review-beliefs -m claude

# Export
reasonsforge export -o network.json
reasonsforge export-markdown -o beliefs.md
```

## Commands

### Core TMS

| Command | Description |
|---------|-------------|
| `init` | Create reasons.db |
| `add ID "text"` | Add a premise |
| `add ID "text" --sl a,b` | Add with SL justification |
| `add ID "text" --unless y` | Add with outlist (non-monotonic) |
| `retract ID` | Mark OUT + cascade |
| `assert ID` | Mark IN + restore dependents |
| `status` | Show all nodes with truth values |
| `show ID` | Node details, justifications, dependents |
| `explain ID` | Trace why a node is IN or OUT |
| `trace ID` | Find all premises a conclusion rests on |
| `propagate` | Recompute all truth values |
| `log` | Propagation audit trail |

### Dialectical

| Command | Description |
|---------|-------------|
| `challenge ID "reason"` | Challenge a node — target goes OUT |
| `defend TARGET CHALLENGE "reason"` | Defend — target restored |
| `nogood A B ...` | Record contradiction, backtrack to responsible premise |

### Search & Query

| Command | Description |
|---------|-------------|
| `search QUERY` | Full-text search |
| `list` | Filter by `--status`, `--premises`, `--has-dependents`, `--challenged` |
| `ask QUESTION` | Natural language question over the network |
| `compact` | Token-budgeted summary (`--budget N`) |

### LLM Analysis

| Command | Description |
|---------|-------------|
| `derive` | Generate new beliefs from existing ones |
| `review-beliefs` | Validate derived beliefs |
| `contradictions` | Detect contradictions |
| `verify` | Re-examine against source documents |
| `repair` | Fix invalid beliefs |
| `deduplicate` | Find and merge duplicates |

### Import & Export

| Command | Description |
|---------|-------------|
| `import-beliefs FILE` | Import beliefs.md registry |
| `import-json FILE` | Import from JSON |
| `export` | Export as JSON |
| `export-markdown` | Export as beliefs.md |
| `export-card` | Export as HuggingFace model card |
| `check-stale` | Check for source file changes |
| `hash-sources` | Backfill source hashes |

## Tests

```bash
uv run --extra test pytest tests/ -v
```

1613 tests covering propagation, retraction cascades, restoration, multiple justifications, diamond dependencies, nogoods, dependency-directed backtracking, non-monotonic justifications, dialectical argumentation, Merkle integrity, SQLite round-trips, import/export, and more.

## References

Doyle, J. (1979). A Truth Maintenance System. *Artificial Intelligence*, 12(3), 231–272.

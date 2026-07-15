"""Ask natural language questions against a belief network.

Uses FTS5 search to find relevant beliefs, then optionally synthesizes
an answer via an LLM with a tool loop that allows the model to
request additional belief searches.
"""

import json
import re
import sqlite3
import subprocess
import sys

from . import api
from .llm import invoke_model


_CITE_RULE = "- Cite belief IDs in [brackets] when referencing specific beliefs."
_CITE_RULE_NATURAL = "- Do not cite belief IDs. Answer in plain natural language."

ASK_PROMPT = """\
You are answering a question using a belief network (a Truth Maintenance System).
Each belief has an ID, text, truth value (IN = held true, OUT = retracted), and
may have justifications tracing why it is believed.

{tools_section}

Rules:
- If the belief matches below are sufficient to answer the question, write your
  answer directly. Do NOT call a tool.
- If you need more information, respond with ONLY a single JSON line
  (no other text). The system will run the tool and give you the results.
{cite_rule}
- ONLY answer based on the beliefs and data provided. Do NOT use your training
  data or general knowledge to fill gaps.
- If the beliefs are insufficient to answer, respond EXACTLY with:
  "I don't have enough beliefs in the network to answer this question."
  Do NOT attempt a partial or speculative answer.
{mcp_instructions}
## Question

{question}

## Belief matches

{beliefs_context}
{tool_history}"""


_DEFAULT_TOOLS_SECTION = """\
You have one tool available:

{"tool": "search_beliefs", "query": "search terms"}"""


FINAL_ASK_PROMPT = """\
You are answering a question using a belief network (a Truth Maintenance System).
Each belief has an ID, text, truth value (IN = held true, OUT = retracted), and
may have justifications tracing why it is believed.

Rules:
{cite_rule}
- ONLY answer based on the beliefs provided. Do NOT use your training data or
  general knowledge to fill gaps.
- If the beliefs are insufficient to answer, respond EXACTLY with:
  "I don't have enough beliefs in the network to answer this question."
  Do NOT attempt a partial or speculative answer.
- Write your answer now.

## Question

{question}

## Belief matches

{beliefs_context}
{tool_history}"""


SIMPLE_ASK_PROMPT = """\
You are answering a question using a belief network (a Truth Maintenance System).
Each belief has an ID, text, truth value (IN = held true, OUT = retracted), and
may have justifications tracing why it is believed.

Rules:
{cite_rule}
- ONLY answer based on the beliefs provided. Do NOT use your training data or
  general knowledge to fill gaps.
- If the beliefs are insufficient to answer, respond EXACTLY with:
  "I don't have enough beliefs in the network to answer this question."
  Do NOT attempt a partial or speculative answer.

## Question

{question}

## Belief matches

{beliefs_context}"""


def build_simple_prompt(question, beliefs_context, natural=False):
    """Build prompt for simple single-pass synthesis — no tool definitions."""
    return SIMPLE_ASK_PROMPT.format(
        question=question,
        beliefs_context=beliefs_context,
        cite_rule=_CITE_RULE_NATURAL if natural else _CITE_RULE,
    )


def extract_tool_call(text):
    """Extract a tool call from LLM response text.

    Scans each line for valid JSON with a "tool" key.
    Returns the parsed dict or None.
    """
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "tool" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def build_ask_prompt(question, beliefs_context, tool_history=None, natural=False,
                     tools_section=None, mcp_instructions=""):
    """Build the full prompt for LLM synthesis."""
    history_section = ""
    if tool_history:
        parts = []
        for entry in tool_history:
            parts.append(
                f"### Tool call: {entry['tool_label']}\n\n"
                f"{entry['result']}"
            )
        history_section = "\n\n## Additional search results\n\n" + "\n\n---\n\n".join(parts)

    if mcp_instructions:
        mcp_instructions = f"\n## Data Source Instructions\n\n{mcp_instructions}\n"

    return ASK_PROMPT.format(
        question=question,
        beliefs_context=beliefs_context,
        tool_history=history_section,
        cite_rule=_CITE_RULE_NATURAL if natural else _CITE_RULE,
        tools_section=tools_section or _DEFAULT_TOOLS_SECTION,
        mcp_instructions=mcp_instructions,
    )


def build_final_prompt(question, beliefs_context, tool_history=None, natural=False):
    """Build prompt for final synthesis — no tool definition."""
    history_section = ""
    if tool_history:
        parts = []
        for entry in tool_history:
            parts.append(
                f"### Tool call: {entry['tool_label']}\n\n"
                f"{entry['result']}"
            )
        history_section = "\n\n## Additional search results\n\n" + "\n\n---\n\n".join(parts)

    return FINAL_ASK_PROMPT.format(
        question=question,
        beliefs_context=beliefs_context,
        tool_history=history_section,
        cite_rule=_CITE_RULE_NATURAL if natural else _CITE_RULE,
    )


def _invoke_claude(prompt, timeout=300):
    """Call the default LLM (claude). Backward-compat wrapper."""
    return invoke_model(prompt, model="claude", timeout=timeout)


def _strip_belief_metadata(beliefs_context):
    """Strip IDs, status markers, and justification metadata from belief context.

    Converts structured belief format to plain natural language paragraphs.
    """
    if not beliefs_context:
        return beliefs_context
    lines = beliefs_context.split("\n")
    out = []
    for line in lines:
        if line.startswith("### "):
            continue
        if line.startswith("**Status:**"):
            continue
        if line.startswith("**Source:**"):
            continue
        if line.startswith("**Depends on:**"):
            continue
        if line.startswith("**Justification:**"):
            continue
        if line.startswith("**Supported by:**"):
            continue
        if line.startswith("**Supports:**"):
            continue
        if line.startswith("**Depended on by:**"):
            continue
        if line.startswith("**Related nodes:**"):
            continue
        if re.match(r'^- \*\*\S+\*\* \((?:IN|OUT)\):', line):
            continue
        if line.strip() == "---":
            continue
        out.append(line)
    result = "\n".join(out).strip()
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result


def search_source_chunks(query, sources_db, top_k=10):
    """Search FTS5 index and return structured results.

    Returns list of dicts with keys: filename, section, text, cluster.
    """
    from .api import _STOP_WORDS

    raw_words = re.findall(r'\w+', query)
    words = [w for w in raw_words if w.lower() not in _STOP_WORDS and len(w) > 1]
    if not words:
        words = [w for w in raw_words if len(w) > 1]
    if not words:
        return []
    fts_query = " OR ".join(f'"{w}"' for w in words)

    conn = sqlite3.connect(sources_db)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT c.text, c.cluster, c.filename, c.section
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY chunks_fts.rank
            LIMIT ?
        """, (fts_query, top_k))
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _search_source_chunks(question, sources_db, top_k=10):
    """Search FTS5 index of source document chunks. Returns formatted text."""
    try:
        rows = search_source_chunks(question, sources_db, top_k)
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return ""
    if not rows:
        return ""
    parts = []
    for i, row in enumerate(rows, 1):
        header = f"[{i}] {row['filename']}"
        if row["section"]:
            header += f" > {row['section']}"
        parts.append(f"### {header}\n\n{row['text']}")
    return "\n\n---\n\n".join(parts)


FTS_RAG_PROMPT = """\
You are answering questions using retrieved document excerpts.

Below are the most relevant excerpts from source documents, retrieved via
full-text search. Use them to answer the question. Cite your sources by referencing
the document filename in [brackets].

If the excerpts don't contain enough information to answer the question, say so honestly.
Do not fabricate information that isn't in the provided excerpts.

## Retrieved Documents

{context}

## Question

{question}

## Instructions

- Answer the question based on the retrieved documents above
- Cite sources using [filename] notation
- If information is insufficient, say what you can and note the gaps
- Be specific and concise
"""

MERGE_PROMPT = """\
You are merging two answers to the same question. Each answer was produced
independently using a different retrieval method:

- Answer A used a structured belief network with dependency chains
- Answer B used full-text search over source documents

Produce a single merged answer that:
- Combines information from both answers
- When both answers cover the same point, use the more specific/detailed version
- Preserve all citations (belief IDs in [brackets] from Answer A, [filenames] from Answer B)
- Do not add information that neither answer contains
- If the answers contradict each other, note the disagreement

## Question

{question}

## Answer A (Belief Network)

{answer_tms}

## Answer B (Source Documents)

{answer_fts}
"""


def _fts_rag_answer(question, sources_db, model="claude", timeout=300):
    """Run FTS5 RAG over source document chunks and synthesize an answer."""
    context = _search_source_chunks(question, sources_db)
    if not context:
        return "No relevant documents found for this question."
    prompt = FTS_RAG_PROMPT.format(context=context, question=question)
    return invoke_model(prompt, model=model, timeout=timeout).strip()


def _merge_answers(question, answer_tms, answer_fts, model="claude", timeout=300):
    """Merge two answers from different retrieval paths."""
    prompt = MERGE_PROMPT.format(
        question=question, answer_tms=answer_tms, answer_fts=answer_fts,
    )
    return invoke_model(prompt, model=model, timeout=timeout).strip()


MAX_ITERATIONS = 3

NO_BELIEFS_MSG = "No matching beliefs found. Cannot answer from the belief network."


def _beliefs_or_no_match(beliefs_context):
    if not beliefs_context or beliefs_context.strip() == "No results found.":
        return NO_BELIEFS_MSG
    return beliefs_context


def _build_tools_section(mcp_servers):
    """Build the tools section for the prompt, including MCP server tools."""
    lines = ["You have the following tools available:", "",
             '{"tool": "search_beliefs", "query": "search terms"}']
    if mcp_servers:
        for bridge in mcp_servers:
            for tool in bridge.list_tools():
                schema = tool.get("input_schema", {})
                props = schema.get("properties", {})
                param_parts = []
                for pname, pinfo in props.items():
                    param_parts.append(f'"{pname}": "<{pinfo.get("description", pname)}>"')
                example = ', '.join(param_parts)
                if example:
                    lines.append(f'{{"tool": "{tool["name"]}", {example}}}')
                else:
                    lines.append(f'{{"tool": "{tool["name"]}"}}')
                if tool["description"]:
                    desc = tool["description"].split("\n")[0]
                    lines.append(f"  # {desc}")
    return "\n".join(lines)


def _build_mcp_instructions(mcp_servers):
    """Collect server instructions from all MCP servers."""
    parts = []
    for bridge in mcp_servers:
        instructions = bridge.get_instructions()
        if instructions:
            parts.append(instructions)
    return "\n\n".join(parts)


def ask(question, db_path="reasons.db", timeout=300, no_synth=False, format=None,
        model="claude", simple=False, sources_db=None, natural=False, dual=False,
        mcp_servers=None):
    """Answer a question using FTS5 belief search and optional LLM synthesis.

    Args:
        sources_db: path to FTS5 index of source document chunks (rag_fts.db).
                    When provided, appends retrieved document excerpts to the
                    belief context for fuller coverage.
        natural: strip belief IDs, status markers, and justification metadata
                 from context, presenting beliefs as plain natural language.
        dual: run TMS and FTS RAG in separate calls, then merge the two
              answers in a third call. Requires sources_db. Makes 3 LLM
              calls with simple=True (1 TMS + 1 FTS + 1 merge), or up to
              5 with simple=False (up to 3 TMS tool-loop rounds + 1 FTS
              + 1 merge). Timeout applies per call.
        mcp_servers: list of connected McpBridge instances for external tools.

    Returns the answer text.
    """
    if dual and not sources_db:
        raise ValueError("--dual requires --full-sources")

    if dual and sources_db:
        print("Dual path: running TMS...", file=sys.stderr)
        answer_tms = ask(question, db_path=db_path, timeout=timeout,
                         model=model, simple=simple, natural=natural,
                         mcp_servers=mcp_servers)
        print("Dual path: running FTS RAG...", file=sys.stderr)
        answer_fts = _fts_rag_answer(question, sources_db, model=model,
                                     timeout=timeout)
        tms_empty = (answer_tms.strip() == NO_BELIEFS_MSG
                     or "don't have enough beliefs" in answer_tms.lower())
        fts_empty = (answer_fts.strip().startswith("No relevant documents"))
        if tms_empty and not fts_empty:
            print("Dual path: TMS empty, using FTS answer directly.", file=sys.stderr)
            return answer_fts
        if fts_empty and not tms_empty:
            print("Dual path: FTS empty, using TMS answer directly.", file=sys.stderr)
            return answer_tms
        if tms_empty and fts_empty:
            return NO_BELIEFS_MSG
        print("Dual path: merging answers...", file=sys.stderr)
        return _merge_answers(question, answer_tms, answer_fts, model=model,
                              timeout=timeout)

    if no_synth:
        fmt = format or "compact"
        return api.search(question, db_path=db_path, format=fmt)

    if simple:
        beliefs_context = api.search(question, db_path=db_path, format="markdown",
                                     depth=2)
        if not beliefs_context or beliefs_context.strip() == "No results found.":
            beliefs_context = ""

        if natural and beliefs_context:
            beliefs_context = _strip_belief_metadata(beliefs_context)

        if sources_db:
            sources_context = _search_source_chunks(question, sources_db)
            if sources_context:
                beliefs_context = beliefs_context + "\n\n## Source Documents\n\n" + sources_context

        if not beliefs_context.strip():
            return NO_BELIEFS_MSG

        prompt = build_simple_prompt(question, beliefs_context, natural=natural)
        print("Synthesizing (simple)...", file=sys.stderr)
        try:
            response = invoke_model(prompt, model=model, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"LLM timed out after {timeout}s", file=sys.stderr)
            return _beliefs_or_no_match(beliefs_context)
        except Exception as e:
            print(f"LLM error: {e}", file=sys.stderr)
            return _beliefs_or_no_match(beliefs_context)
        return response.strip()

    beliefs_context = api.search(question, db_path=db_path, format="markdown")

    if natural and beliefs_context:
        beliefs_context = _strip_belief_metadata(beliefs_context)

    sources_suffix = ""
    if sources_db:
        sources_context = _search_source_chunks(question, sources_db)
        if sources_context:
            sources_suffix = "\n\n## Source Documents\n\n" + sources_context

    def _full_context():
        if sources_suffix:
            return (beliefs_context or "") + sources_suffix
        return beliefs_context

    tools_section = None
    mcp_instructions = ""
    mcp_tool_map = {}
    max_iters = MAX_ITERATIONS

    if mcp_servers:
        tools_section = _build_tools_section(mcp_servers)
        mcp_instructions = _build_mcp_instructions(mcp_servers)
        for bridge in mcp_servers:
            for tool in bridge.list_tools():
                mcp_tool_map[tool["name"]] = bridge
        max_iters = max(MAX_ITERATIONS, 5)

    tool_history = []

    for iteration in range(max_iters):
        ctx = _full_context()
        if iteration == max_iters - 1:
            prompt = build_final_prompt(question, ctx, tool_history, natural=natural)
        else:
            prompt = build_ask_prompt(question, ctx, tool_history, natural=natural,
                                      tools_section=tools_section,
                                      mcp_instructions=mcp_instructions)

        print(f"Synthesizing (round {iteration + 1}/{max_iters})...",
              file=sys.stderr)

        try:
            response = invoke_model(prompt, model=model, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"LLM timed out after {timeout}s", file=sys.stderr)
            return _beliefs_or_no_match(ctx)
        except Exception as e:
            print(f"LLM error: {e}", file=sys.stderr)
            return _beliefs_or_no_match(ctx)

        tool_call = extract_tool_call(response)

        if tool_call is None:
            return response.strip()

        tool_name = tool_call.get("tool")

        if tool_name == "search_beliefs":
            query = tool_call.get("query", "")
            print(f"  Searching: {query}", file=sys.stderr)
            result = api.search(query, db_path=db_path, format="markdown")
            history_result = _strip_belief_metadata(result) if natural and result else result
            tool_history.append({
                "tool_label": f'search_beliefs("{query}")',
                "result": history_result,
            })
            if result and result.strip() != "No results found.":
                beliefs_context = result
                if natural:
                    beliefs_context = _strip_belief_metadata(beliefs_context)
        elif tool_name in mcp_tool_map:
            bridge = mcp_tool_map[tool_name]
            args = {k: v for k, v in tool_call.items() if k != "tool"}
            print(f"  MCP tool: {tool_name}({json.dumps(args)[:80]})", file=sys.stderr)
            try:
                result = bridge.call_tool(tool_name, args)
            except Exception as e:
                result = f"Error calling {tool_name}: {e}"
            tool_history.append({
                "tool_label": f"{tool_name}(...)",
                "result": result,
            })
        else:
            return response.strip()

        if iteration == max_iters - 1:
            print(f"Synthesizing (final)...", file=sys.stderr)
            ctx = _full_context()
            prompt = build_final_prompt(question, ctx, tool_history, natural=natural)
            try:
                response = invoke_model(prompt, model=model, timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"LLM timed out after {timeout}s", file=sys.stderr)
                return _beliefs_or_no_match(ctx)
            except Exception as e:
                print(f"LLM error: {e}", file=sys.stderr)
                return _beliefs_or_no_match(ctx)
            if extract_tool_call(response):
                return _beliefs_or_no_match(ctx)
            return response.strip()

    return _beliefs_or_no_match(beliefs_context)

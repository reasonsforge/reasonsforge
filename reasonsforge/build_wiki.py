"""Build a static wiki from the belief network.

Exports beliefs as interlinked markdown pages grouped by topic
(word-frequency) or semantic cluster. Optionally uses an LLM to
synthesize each topic page into a coherent narrative.
"""

import os
import re
import sys


_TOPIC_STOP_WORDS = {
    "the", "is", "in", "to", "of", "and", "or", "not", "as", "by",
    "via", "can", "with", "from", "than", "that", "this", "be", "has",
    "have", "it", "its", "no", "do", "if", "so", "up", "out", "all",
    "but", "get", "set", "only", "per", "use", "may", "one", "two",
    "new", "any", "each", "must", "when", "how", "also", "into",
    "over", "more", "both", "same", "own", "used", "using", "based",
    "does", "then", "for",
}


def _assign_topics(node_ids, topics):
    """Assign each node to its best-matching topic based on ID segments.

    Returns {topic_label: [node_id, ...], ...} with "Other" for unmatched.
    """
    topic_set = {t["topic"] for t in topics}
    groups = {t["topic"]: [] for t in topics}
    groups["Other"] = []

    for nid in node_ids:
        words = [w for w in re.split(r'[-._:]', nid) if w and len(w) > 2]
        matched = False
        for word in words:
            if word in topic_set:
                groups[word].append(nid)
                matched = True
                break
        if not matched:
            groups["Other"].append(nid)

    return {k: v for k, v in groups.items() if v}


def _page_name(label):
    """Sanitize a topic/cluster label to a valid markdown filename."""
    safe = re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')
    return safe or "other"


def _format_node(node_id, node_detail, node_to_page, all_details=None):
    """Render one node as markdown with cross-reference links."""
    lines = []
    lines.append(f"### {node_id}")
    lines.append(f"**Status:** {node_detail['truth_value']}")
    lines.append("")

    # Render duplicate-of or superseded-by relationships
    metadata = node_detail.get("metadata") or {}
    if "duplicate_of" in metadata:
        canonical_id = metadata["duplicate_of"]
        page = node_to_page.get(canonical_id)
        if page:
            link = f"[{canonical_id}]({page}#{canonical_id})"
        else:
            link = canonical_id
        lines.append(f"**Duplicate of:** {link}")
        lines.append("")

    if "superseded_by" in metadata:
        new_id = metadata["superseded_by"]
        page = node_to_page.get(new_id)
        if page:
            link = f"[{new_id}]({page}#{new_id})"
        else:
            link = new_id
        lines.append(f"**Superseded by:** {link}")
        lines.append("")

    # Render defeaters from outlist
    if all_details:
        defeaters = []
        for j in node_detail.get("justifications", []):
            for o in j.get("outlist", []):
                o_detail = all_details.get(o)
                if o_detail:
                    o_meta = o_detail.get("metadata") or {}
                    if o_meta.get("defeats_node") == node_id:
                        defeaters.append((o, o_meta, o_detail.get("text", "")))
        if defeaters:
            for d_id, d_meta, d_text in defeaters:
                d_type = d_meta.get("defeater_type", "defeater")
                r_type = d_meta.get("defeat_reason_type", "")
                label = f"{d_type}, {r_type}" if r_type else d_type
                page = node_to_page.get(d_id)
                if page:
                    link = f"[{d_id}]({page}#{d_id})"
                else:
                    link = d_id
                lines.append(f"**Defeated by:** {link} ({label})")
            lines.append("")

    lines.append(node_detail["text"])
    lines.append("")

    justifications = node_detail.get("justifications", [])
    si = node_detail.get("supporting_justification")
    if si is not None and 0 <= si < len(justifications):
        designated = justifications[si]
        antecedents = set(designated.get("antecedents", []))
        other_antecedents = set()
        for idx, j in enumerate(justifications):
            if idx != si:
                for a in j.get("antecedents", []):
                    if a not in antecedents:
                        other_antecedents.add(a)
    else:
        antecedents = set()
        other_antecedents = set()
        for j in justifications:
            for a in j.get("antecedents", []):
                antecedents.add(a)

    if antecedents:
        links = []
        for a in sorted(antecedents):
            page = node_to_page.get(a)
            if page:
                links.append(f"[{a}]({page}#{a})")
            else:
                links.append(a)
        label = "**Depends on (active):**" if other_antecedents else "**Depends on:**"
        lines.append(f"{label} {', '.join(links)}")
    if other_antecedents:
        links = []
        for a in sorted(other_antecedents):
            page = node_to_page.get(a)
            if page:
                links.append(f"[{a}]({page}#{a})")
            else:
                links.append(a)
        lines.append(f"**Depends on (other):** {', '.join(links)}")

    dependents = node_detail.get("dependents", [])
    if dependents:
        links = []
        for d in sorted(dependents):
            page = node_to_page.get(d)
            if page:
                links.append(f"[{d}]({page}#{d})")
            else:
                links.append(d)
        lines.append(f"**Supports:** {', '.join(links)}")

    lines.append("")
    return "\n".join(lines)


WIKI_PAGE_PROMPT = """\
You are writing a wiki page about "{topic}" for a knowledge base built from \
a belief network (Truth Maintenance System). The page should be a coherent, \
readable narrative that synthesizes the beliefs below into an informative article.

## Guidelines

- Write in clear, encyclopedic prose — not a list of beliefs
- Use markdown headers (##, ###) to organize sections
- Start with a brief overview paragraph
- Group related beliefs into thematic sections
- Mention the status (IN = currently held, OUT = retracted) only when relevant
- Include belief IDs in parentheses after key claims so readers can trace sources, \
  e.g. "The system uses SL justifications (sl-justification-mechanism)."
- Note important dependency relationships between beliefs
- If some beliefs contradict or qualify others, explain the nuance
- Do NOT include a title — the page already has one
- Keep the page concise but comprehensive

## Beliefs

{beliefs}

Write the wiki page content now.
"""


def _format_beliefs_for_prompt(node_ids, node_details):
    """Format beliefs into a structured text block for the LLM prompt."""
    lines = []
    for nid in sorted(node_ids):
        detail = node_details.get(nid)
        if not detail:
            continue
        lines.append(f"### {nid}")
        lines.append(f"Status: {detail['truth_value']}")
        lines.append(f"Text: {detail['text']}")

        justifications = detail.get("justifications", [])
        si = detail.get("supporting_justification")
        if si is not None and 0 <= si < len(justifications):
            antecedents = set(justifications[si].get("antecedents", []))
        else:
            antecedents = set()
            for j in justifications:
                for a in j.get("antecedents", []):
                    antecedents.add(a)
        if antecedents:
            lines.append(f"Depends on: {', '.join(sorted(antecedents))}")

        dependents = detail.get("dependents", [])
        if dependents:
            lines.append(f"Supports: {', '.join(sorted(dependents))}")
        lines.append("")
    return "\n".join(lines)


def generate_wiki_page(topic, node_ids, node_details, model, timeout):
    """Generate a wiki page for a topic group using an LLM."""
    from .llm import invoke_model

    beliefs_text = _format_beliefs_for_prompt(node_ids, node_details)
    prompt = WIKI_PAGE_PROMPT.format(topic=topic, beliefs=beliefs_text)
    return invoke_model(prompt, model=model, timeout=timeout)


def _linkify(content, current_page, node_to_page, all_ids):
    """Replace cross-page belief IDs with markdown links."""
    for nid in sorted(all_ids, key=len, reverse=True):
        target = node_to_page.get(nid)
        if not target or target == current_page:
            continue
        if nid not in content:
            continue
        if "[" + nid + "](" in content:
            continue
        link = "[" + nid + "](" + target + "#" + nid + ")"
        pattern = r'(?<![a-z0-9\-])' + re.escape(nid) + r'(?![a-z0-9\-])'
        content = re.sub(pattern, link, content)
    return content


_RESERVED_SLUGS = {"index"}


def build_wiki(node_details, groups, output_dir, model="", timeout=300,
               parallel=0):
    """Write index.md and per-group pages to output_dir.

    Args:
        node_details: {node_id: show_node dict}
        groups: {group_label: [node_id, ...]}
        output_dir: directory to write markdown files into
        model: LLM model for page generation (empty = no LLM)
        timeout: LLM timeout in seconds
        parallel: number of concurrent LLM workers (0 = sequential)
    """
    os.makedirs(output_dir, exist_ok=True)

    used_slugs: dict[str, str] = {}
    label_to_file: dict[str, str] = {}
    for label in groups:
        slug = _page_name(label)
        if slug in _RESERVED_SLUGS:
            slug = f"{slug}-topic"
        while slug in used_slugs:
            slug = f"{slug}-2"
        used_slugs[slug] = label
        label_to_file[label] = slug + ".md"

    node_to_page = {}
    for label, nids in groups.items():
        page_file = label_to_file[label]
        for nid in nids:
            node_to_page[nid] = page_file

    index_lines = ["# Belief Wiki", ""]
    index_lines.append("| Topic | Beliefs |")
    index_lines.append("|-------|---------|")
    for label in sorted(groups, key=lambda l: (-len(groups[l]), l)):
        page_file = label_to_file[label]
        count = len(groups[label])
        index_lines.append(f"| [{label}]({page_file}) | {count} |")
    index_lines.append("")

    total = sum(len(nids) for nids in groups.values())
    index_lines.append(f"*{total} beliefs across {len(groups)} pages*")
    index_lines.append("")

    with open(os.path.join(output_dir, "index.md"), "w") as f:
        f.write("\n".join(index_lines))

    total_groups = len(groups)
    generated_content: dict[str, str] = {}

    if model and parallel > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _gen(label, nids):
            return label, generate_wiki_page(label, nids, node_details,
                                             model, timeout)

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(_gen, label, nids): label
                for label, nids in groups.items()
            }
            done = 0
            for future in as_completed(futures):
                label = futures[future]
                done += 1
                try:
                    _, content = future.result()
                    generated_content[label] = content
                    print(f"  Generated {label} ({done}/{total_groups})",
                          file=sys.stderr)
                except Exception as e:
                    print(f"  WARN: {label} failed: {e} ({done}/{total_groups})",
                          file=sys.stderr)
    elif model:
        for i, (label, nids) in enumerate(groups.items(), 1):
            print(f"  Generating {label} ({i}/{total_groups})...",
                  file=sys.stderr)
            try:
                generated_content[label] = generate_wiki_page(
                    label, nids, node_details, model, timeout)
            except Exception as e:
                print(f"  WARN: {label} failed: {e}", file=sys.stderr)

    for label, nids in groups.items():
        page_file = label_to_file[label]
        page_lines = [f"# {label}", ""]
        page_lines.append(f"[Back to index](index.md)")
        page_lines.append("")
        if label in generated_content:
            page_lines.append(generated_content[label])
            page_lines.append("")
        else:
            for nid in sorted(nids):
                detail = node_details.get(nid)
                if detail:
                    page_lines.append(_format_node(nid, detail, node_to_page, all_details=node_details))
        page_text = "\n".join(page_lines)
        page_text = _linkify(page_text, page_file, node_to_page,
                             node_details.keys())
        with open(os.path.join(output_dir, page_file), "w") as f:
            f.write(page_text)

    return {"output_dir": output_dir, "pages": len(groups), "total_nodes": total}

"""CLI for the Reason Maintenance System.

Thin wrappers around reasons.api — each command calls an api function
and formats the result dict for terminal output.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from importlib.metadata import version as _pkg_version

from . import api


def cmd_init(args):
    try:
        result = api.init_db(
            force=args.force,
            project_name=getattr(args, "project_name", "") or "",
            **_backend_kwargs(args),
        )
        if "db_path" in result:
            print(f"Initialized RMS database: {result['db_path']}")
        else:
            print(f"Initialized PostgreSQL project: {result['project_id']}")
    except FileExistsError as e:
        print(f"{e}", file=sys.stderr)
        print("Use --force to reinitialize.", file=sys.stderr)
        sys.exit(1)


def _warn_multi_premise(premise_count, any_mode):
    """Print a tip when an SL has 3+ premises and --any was not used."""
    if premise_count >= 3 and not any_mode:
        print(f"  Tip: This SL requires ALL {premise_count} premises to be IN. If any single")
        print(f"  premise is sufficient, use --any to create separate justifications.")


def cmd_add(args):
    access_tags = None
    if getattr(args, "access_tags", None):
        access_tags = [t.strip() for t in args.access_tags.split(",") if t.strip()]
    try:
        result = api.add_node(
            node_id=args.node_id,
            text=args.text,
            sl=args.sl or "",
            cp=args.cp or "",
            unless=args.unless or "",
            label=args.label or "",
            source=args.source or "",
            source_url=args.source_url or "",
            namespace=getattr(args, "namespace", None),
            any_mode=getattr(args, "any", False),
            access_tags=access_tags,
            example=getattr(args, "example", None),
            source_type=getattr(args, "source_type", None) or "",
            accepted_pr=getattr(args, "accepted_pr", None) or "",
            **_backend_kwargs(args),
        )
        print(f"Added {result['node_id']} [{result['truth_value']}] ({result['type']})")
        _warn_multi_premise(result.get("premise_count", 0), getattr(args, "any", False))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_add_justification(args):
    try:
        result = api.add_justification(
            node_id=args.node_id,
            sl=args.sl or "",
            cp=args.cp or "",
            unless=args.unless or "",
            label=args.label or "",
            namespace=getattr(args, "namespace", None),
            any_mode=getattr(args, "any", False),
            **_backend_kwargs(args),
        )
        print(f"Added justification to {result['node_id']}")
        print(f"  Truth value: {result['old_truth_value']} → {result['new_truth_value']}")
        if result["changed"]:
            print(f"  Cascade: {', '.join(result['changed'])}")
        _warn_multi_premise(result.get("premise_count", 0), getattr(args, "any", False))
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_remove_justification(args):
    try:
        result = api.remove_justification(
            node_id=args.node_id,
            index=args.index,
            **_backend_kwargs(args),
        )
        removed = result["removed"]
        ants = ", ".join(removed["antecedents"])
        label = f" [{removed['label']}]" if removed["label"] else ""
        print(f"Removed justification {args.index} from {result['node_id']}")
        print(f"  Was: {removed['type']}({ants}){label}")
        print(f"  Truth value: {result['old_truth_value']} → {result['new_truth_value']}")
        print(f"  Remaining justifications: {result['remaining']}")
        if result["changed"]:
            print(f"  Cascade: {', '.join(result['changed'])}")
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _print_cascade(result):
    """Print cascade results, splitting went_out from went_in."""
    went_out = result.get("went_out", [])
    went_in = result.get("went_in", [])
    if went_out:
        print(f"  Went OUT ({len(went_out)}):")
        for nid in went_out:
            print(f"    [-] {nid}")
    if went_in:
        print(f"  Went IN ({len(went_in)}):")
        for nid in went_in:
            print(f"    [+] {nid}")


def _print_restoration_hints(hints):
    """Print hints when multi-premise SL nodes go OUT with surviving premises."""
    for hint in hints:
        surviving = ", ".join(hint["surviving_premises"])
        print(f"  Note: {hint['node_id']} went OUT because its justification required ALL of")
        print(f"    {hint['all_premises']}")
        print(f"    Surviving premises still IN: {surviving}")
        print(f"    If any single premise is sufficient, restore with:")
        print(f"      reasons add-justification {hint['node_id']} --sl {surviving} --any")


def cmd_retract(args):
    try:
        result = api.retract_node(args.node_id, reason=args.reason or "", **_backend_kwargs(args))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not result["changed"]:
        print(f"{args.node_id} is already OUT")
    else:
        print(f"Retracted {args.node_id}")
        _print_cascade(result)
        if result.get("restoration_hints"):
            _print_restoration_hints(result["restoration_hints"])


def cmd_mark_duplicate(args):
    if getattr(args, "pg", None) or os.environ.get("REASONSFORGE_PG_CONNINFO"):
        print("Error: mark-duplicate is not supported with --pg (no PgApi implementation)", file=sys.stderr)
        sys.exit(1)
    try:
        result = api.mark_duplicate(
            args.source_id,
            args.canonical_id,
            db_path=args.db,
        )
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Marked {result['source_id']} as duplicate of {result['canonical_id']}")
    print(f"  Status: Retracted with metadata duplicate_of={result['canonical_id']}")
    if result["changed"]:
        went_out = [nid for nid in result["changed"] if nid != args.source_id]
        if went_out:
            print(f"  Cascade: {len(went_out)} dependent belief(s) went OUT")


def cmd_mark_superseded(args):
    if getattr(args, "pg", None) or os.environ.get("REASONSFORGE_PG_CONNINFO"):
        print("Error: mark-superseded is not supported with --pg (no PgApi implementation)", file=sys.stderr)
        sys.exit(1)
    try:
        result = api.mark_superseded(
            args.old_id,
            args.new_id,
            db_path=args.db,
        )
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Marked {result['old_id']} as superseded by {result['new_id']}")
    print(f"  Status: Retracted with metadata superseded_by={result['new_id']}")
    if result["changed"]:
        went_out = [nid for nid in result["changed"] if nid != args.old_id]
        if went_out:
            print(f"  Cascade: {len(went_out)} dependent belief(s) went OUT")


def cmd_defeat_justification(args):
    if getattr(args, "pg", None) or os.environ.get("REASONSFORGE_PG_CONNINFO"):
        print("Error: defeat-justification is not supported with --pg (no PgApi implementation)", file=sys.stderr)
        sys.exit(1)
    try:
        result = api.defeat_justification(
            args.node_id,
            args.justification_index,
            args.reason,
            defeater_type=args.type or "invalid-inference",
            defeater_id=getattr(args, "defeater_id", None),
            defeat_reason_type=getattr(args, "reason_type", None) or "",
            db_path=args.db,
        )
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    dtype = result['defeater_type']
    rtype = result.get('defeat_reason_type', '')
    label = f"{dtype}, {rtype}" if rtype else dtype
    print(f"Defeated justification {result['justification_index']} of {result['node_id']}")
    print(f"  Defeater: {result['defeater_id']} ({label})")
    if result["changed"]:
        went_out = [nid for nid in result["changed"] if nid != result["node_id"]]
        if result["node_id"] in result["changed"]:
            print(f"  {result['node_id']} went OUT")
        if went_out:
            print(f"  Cascade: {len(went_out)} dependent belief(s) affected")


def cmd_defeat_with_scope(args):
    if getattr(args, "pg", None) or os.environ.get("REASONSFORGE_PG_CONNINFO"):
        print("Error: defeat-with-scope is not supported with --pg", file=sys.stderr)
        sys.exit(1)

    import json as json_mod
    scope_path = Path(args.scope_file)
    if not scope_path.exists():
        print(f"Error: file not found: {args.scope_file}", file=sys.stderr)
        sys.exit(1)

    data = json_mod.loads(scope_path.read_text())
    scope_findings = data.get("scope_findings", [])
    missing_property = data.get("missing_property", "")

    if not scope_findings:
        print("Error: scope_findings is empty in the provided file", file=sys.stderr)
        sys.exit(1)
    if not missing_property:
        print("Error: missing_property is empty in the provided file", file=sys.stderr)
        sys.exit(1)

    reason_type = getattr(args, "reason_type", None) or data.get("defeat_reason_type", "")

    try:
        result = api.defeat_with_scope(
            args.node_id,
            args.justification_index,
            scope_findings,
            missing_property,
            defeater_type=args.type or "invalid-inference",
            defeater_id=getattr(args, "defeater_id", None),
            defeat_reason_type=reason_type,
            db_path=args.db,
        )
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Defeated justification {result['justification_index']} of {result['node_id']}")
    dtype = result['defeater_type']
    rtype = result.get('defeat_reason_type', '')
    label = f"{dtype}, {rtype}" if rtype else dtype
    print(f"  Defeater: {result['defeater_id']} ({label})")
    print(f"  Scope beliefs: {len(result['scope_belief_ids'])}")
    for sid in result["scope_belief_ids"]:
        print(f"    - {sid}")
    if result["changed"]:
        went_out = [nid for nid in result["changed"] if nid != result["node_id"]]
        if result["node_id"] in result["changed"]:
            print(f"  {result['node_id']} went OUT")
        if went_out:
            print(f"  Cascade: {len(went_out)} dependent belief(s) affected")


def cmd_migrate_defeaters(args):
    if getattr(args, "pg", None) or os.environ.get("REASONSFORGE_PG_CONNINFO"):
        print("Error: migrate-defeaters is not supported with --pg (no PgApi implementation)", file=sys.stderr)
        sys.exit(1)

    node_ids = getattr(args, "node_ids", None) or None
    dry_run = not args.apply

    result = api.migrate_retract_to_defeaters(
        node_ids=node_ids, dry_run=dry_run, db_path=args.db,
    )

    if dry_run:
        print("Dry run — no changes applied")
        print()

    if result["migrated"]:
        print(f"{'Would migrate' if dry_run else 'Migrated'}: {len(result['migrated'])}")
        for m in result["migrated"]:
            defeater = m.get("defeater_id", f"migrated-retraction-{m['id']}-j{m['justification_index']}")
            print(f"  {m['id']} j{m['justification_index']} -> {defeater}")
            print(f"    Reason: {m['retract_reason']}")

    if result["skipped"]:
        print(f"Skipped: {len(result['skipped'])}")
        for s in result["skipped"]:
            print(f"  {s['id']}: {s['reason']}")

    if result["errors"]:
        print(f"Errors: {len(result['errors'])}")
        for e in result["errors"]:
            print(f"  {e['id']}: {e['reason']}")

    if not result["migrated"] and not result["skipped"] and not result["errors"]:
        print("No candidates found")


def cmd_classify_defeaters(args):
    if getattr(args, "pg", None) or os.environ.get("REASONSFORGE_PG_CONNINFO"):
        print("Error: classify-defeaters is not supported with --pg", file=sys.stderr)
        sys.exit(1)

    dry_run = not args.apply
    result = api.classify_defeat_reason_types(
        defeater_type_filter=getattr(args, "type", None),
        model=args.model,
        timeout=args.timeout,
        dry_run=dry_run,
        db_path=args.db,
    )

    if dry_run:
        print("Dry run — no changes applied")
        print()

    if result["classified"]:
        print(f"{'Would classify' if dry_run else 'Classified'}: {len(result['classified'])}")
        for c in result["classified"]:
            print(f"  {c['id']}: {c['defeat_reason_type']}")

    if result["skipped"]:
        print(f"Skipped: {len(result['skipped'])}")
        for s in result["skipped"]:
            print(f"  {s['id']}: {s['reason']}")

    if result["errors"]:
        print(f"Errors: {len(result['errors'])}")
        for e in result["errors"]:
            print(f"  {e['id']}: {e['reason']}")

    if not result["classified"] and not result["skipped"] and not result["errors"]:
        print("No unclassified defeaters found")


def cmd_assert(args):
    try:
        result = api.assert_node(args.node_id, **_backend_kwargs(args))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not result["changed"]:
        print(f"{args.node_id} is already IN")
    else:
        print(f"Asserted {args.node_id}")
        _print_cascade(result)


def _print_what_if_results(result, action, node_id):
    """Shared output formatting for what-if retract and assert."""
    if not result["retracted"] and not result["restored"]:
        verb = "Retracting" if action == "retract" else "Asserting"
        print(f"{verb} {node_id} would affect no other nodes.")
        return

    verb = "retracted" if action == "retract" else "asserted"
    print(f"What if '{node_id}' were {verb}?\n")

    if result["retracted"]:
        print("  Would go OUT:")
        current_depth = 0
        for item in result["retracted"]:
            if item["depth"] != current_depth:
                current_depth = item["depth"]
                print(f"  --- depth {current_depth} ---")
            deps = f"  ({item['dependents']} dependents)" if item["dependents"] else ""
            text = item["text"][:80]
            print(f"  [-] {item['id']}: {text}{deps}")

    if result["restored"]:
        if result["retracted"]:
            print()
        print("  Would go IN:")
        current_depth = 0
        for item in result["restored"]:
            if item["depth"] != current_depth:
                current_depth = item["depth"]
                print(f"  --- depth {current_depth} ---")
            deps = f"  ({item['dependents']} dependents)" if item["dependents"] else ""
            text = item["text"][:80]
            print(f"  [+] {item['id']}: {text}{deps}")

    parts = []
    if result["retracted"]:
        parts.append(f"{len(result['retracted'])} would go OUT")
    if result["restored"]:
        parts.append(f"{len(result['restored'])} would go IN")
    print(f"\nTotal: {', '.join(parts)} (database NOT modified)")


def cmd_what_if(args):
    action = args.action
    try:
        if action == "retract":
            result = api.what_if_retract(args.node_id, **_backend_kwargs(args))
            if result.get("already_out"):
                print(f"{args.node_id} is already OUT — nothing to simulate.")
                return
        else:
            result = api.what_if_assert(args.node_id, **_backend_kwargs(args))
            if result.get("already_in"):
                print(f"{args.node_id} is already IN — nothing to simulate.")
                return
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    _print_what_if_results(result, action, args.node_id)


def cmd_status(args):
    result = api.get_status(visible_to=_parse_visible_to(args), **_backend_kwargs(args))

    if not result["nodes"]:
        print("No nodes in the network.")
        return

    for node in result["nodes"]:
        marker = "+" if node["truth_value"] == "IN" else "-"
        jcount = node["justification_count"]
        jinfo = f"  ({jcount} justification{'s' if jcount != 1 else ''})" if jcount else "  (premise)"
        print(f"  [{marker}] {node['id']}: {node['text']}{jinfo}")

    print(f"\n{result['in_count']}/{result['total']} IN")


def cmd_show(args):
    visible_to = _parse_visible_to(args)
    try:
        node = api.show_node(args.node_id, visible_to=visible_to, **_backend_kwargs(args))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Access denied: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"ID:     {node['id']}")
    print(f"Text:   {node['text']}")
    print(f"Status: {node['truth_value']}")
    if node["source"]:
        print(f"Source: {node['source']}")
    if node.get("source_url"):
        print(f"URL:    {node['source_url']}")
    if node["source_hash"]:
        print(f"Hash:   {node['source_hash']}")
    if node["metadata"].get("source_type"):
        print(f"Source type: {node['metadata']['source_type']}")
    if node["metadata"].get("accepted_pr"):
        print(f"Accepted PR: {node['metadata']['accepted_pr']}")
    if node["metadata"].get("pinned_sha"):
        sha = node["metadata"]["pinned_sha"][:12]
        lines = node["metadata"].get("pinned_lines", "")
        line_info = f" (lines {lines})" if lines else ""
        print(f"Pinned: {sha}{line_info}")

    if node["justifications"]:
        print(f"\nJustifications ({len(node['justifications'])}):")
        for j in node["justifications"]:
            ants = ", ".join(j["antecedents"])
            label = f" [{j['label']}]" if j["label"] else ""
            print(f"  {j['type']}({ants}){label}")
    else:
        print("\nPremise (no justifications)")

    if node["metadata"].get("retract_reason"):
        print(f"\nRetract reason: {node['metadata']['retract_reason']}")

    if node["metadata"].get("example"):
        indented = node["metadata"]["example"].replace("\n", "\n  ")
        print(f"\nExample:\n  {indented}")

    timestamps = []
    for ts_key in ("created_at", "updated_at", "reviewed_at", "verified_at", "retracted_at"):
        ts_val = node.get(ts_key, "")
        if ts_val:
            label = ts_key.replace("_", " ").title().replace(" At", "")
            timestamps.append((label, ts_val))
    if timestamps:
        print()
        for label, val in timestamps:
            print(f"{label}: {val}")

    if node["dependents"]:
        print(f"\nDependents: {', '.join(node['dependents'])}")


def cmd_explain(args):
    try:
        result = api.explain_node(args.node_id, visible_to=_parse_visible_to(args), **_backend_kwargs(args))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Access denied: {e}", file=sys.stderr)
        sys.exit(1)

    for step in result["steps"]:
        nid = step["node"]
        tv = step["truth_value"]
        reason = step["reason"]
        marker = "+" if tv == "IN" else "-"
        line = f"  [{marker}] {nid}: {reason}"
        if "antecedents" in step:
            line += f" — antecedents: {', '.join(step['antecedents'])}"
        if "outlist" in step:
            line += f" — unless: {', '.join(step['outlist'])}"
        if "failed_antecedents" in step:
            line += f" — failed: {', '.join(step['failed_antecedents'])}"
        if "violated_outlist" in step:
            line += f" — violated unless: {', '.join(step['violated_outlist'])}"
        if step.get("label"):
            line += f" [{step['label']}]"
        print(line)


def cmd_convert_to_premise(args):
    try:
        result = api.convert_to_premise(args.node_id, **_backend_kwargs(args))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Converted {result['node_id']} to premise (stripped {result['old_justifications']} justification(s))")
    if result["changed"]:
        print(f"Changed: {', '.join(result['changed'])}")


def cmd_summarize(args):
    over = [n.strip() for n in args.over.split(",")]
    try:
        result = api.summarize(
            args.summary_id, args.text, over,
            source=args.source or "",
            **_backend_kwargs(args),
        )
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Created summary {result['summary_id']} [{result['truth_value']}] over {len(result['over'])} nodes")


def cmd_supersede(args):
    new_id = args.new_id
    text = getattr(args, "text", None)
    custom_id = getattr(args, "id", None)

    if text and new_id:
        print("Error: cannot specify both new_id and --text", file=sys.stderr)
        sys.exit(1)
    if not text and not new_id:
        print("Error: either new_id or --text is required", file=sys.stderr)
        sys.exit(1)

    try:
        if text:
            result = api.supersede_with_text(
                args.old_id, text, new_id=custom_id, db_path=args.db,
            )
        else:
            result = api.supersede(args.old_id, new_id, **_backend_kwargs(args))
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Superseded {result['old_id']} by {result['new_id']}")
    if result["changed"]:
        print(f"Changed: {', '.join(result['changed'])}")


def cmd_update(args):
    if not any([args.source, args.source_url, args.example]):
        print("Error: at least one of --source, --source-url, or --example required",
              file=sys.stderr)
        sys.exit(1)
    try:
        result = api.update_node(
            args.node_id,
            source=args.source,
            source_url=args.source_url,
            example=args.example,
            **_backend_kwargs(args),
        )
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    fields = ", ".join(result["updated_fields"])
    print(f"Updated {result['node_id']} ({fields})")


def cmd_set_metadata(args):
    try:
        result = api.set_metadata(
            args.node_id, args.key, args.value,
            **_backend_kwargs(args),
        )
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Set {result['key']} on {result['node_id']}")


def cmd_get_metadata(args):
    try:
        node = api.show_node(args.node_id, **_backend_kwargs(args))
    except (KeyError, PermissionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    metadata = node.get("metadata") or {}
    if args.key:
        if args.key not in metadata:
            print(f"No metadata key '{args.key}' on {args.node_id}", file=sys.stderr)
            sys.exit(1)
        print(metadata[args.key])
    else:
        if not metadata:
            print(f"No metadata on {args.node_id}")
            return
        for k, v in sorted(metadata.items()):
            print(f"{k}: {v}")


def cmd_challenge(args):
    try:
        result = api.challenge(
            args.target_id, args.reason,
            challenge_id=args.id,
            **_backend_kwargs(args),
        )
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Challenged {result['target_id']} with {result['challenge_id']}")
    if result["changed"]:
        print(f"Changed: {', '.join(result['changed'])}")


def cmd_defend(args):
    try:
        result = api.defend(
            args.target_id, args.challenge_id, args.reason,
            defense_id=args.id,
            **_backend_kwargs(args),
        )
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Defended {result['target_id']} against {result['challenge_id']} with {result['defense_id']}")
    if result["changed"]:
        print(f"Changed: {', '.join(result['changed'])}")


def cmd_nogood(args):
    try:
        result = api.add_nogood(args.node_ids, **_backend_kwargs(args))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Recorded {result['nogood_id']}: {', '.join(result['nodes'])}")
    if result["backtracked_to"]:
        print(f"Backtracked to premise: {result['backtracked_to']}")
    if result["changed"]:
        print(f"Retracted: {', '.join(result['changed'])}")


def cmd_trace_access_tags(args):
    try:
        result = api.trace_access_tags(args.node_id, visible_to=_parse_visible_to(args),
                                        **_backend_kwargs(args))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Access denied: {e}", file=sys.stderr)
        sys.exit(1)

    if not result["access_tags"]:
        print(f"{args.node_id} has no access tags in its dependency chain (unrestricted).")
        return

    print(f"{args.node_id} depends on data tagged: {', '.join(result['access_tags'])}")


def cmd_trace(args):
    try:
        result = api.trace_assumptions(args.node_id, visible_to=_parse_visible_to(args), **_backend_kwargs(args))
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Access denied: {e}", file=sys.stderr)
        sys.exit(1)

    if not result["premises"]:
        print(f"{args.node_id} is a premise (no dependencies).")
        return

    print(f"{args.node_id} rests on {len(result['premises'])} premise(s):")
    for pid in result["premises"]:
        node = api.show_node(pid, **_backend_kwargs(args))
        marker = "+" if node["truth_value"] == "IN" else "-"
        deps = f"  ({len(node['dependents'])} dependents)" if node["dependents"] else ""
        print(f"  [{marker}] {pid}: {node['text'][:80]}{deps}")


def cmd_propagate(args):
    result = api.propagate(**_backend_kwargs(args))
    changed = result["changed"]
    if changed:
        print(f"Updated: {', '.join(changed)}")
    else:
        print("All truth values are current.")


def cmd_log(args):
    result = api.get_log(last=args.last, **_backend_kwargs(args))

    if not result["entries"]:
        print("No propagation events.")
        return

    for entry in result["entries"]:
        print(f"  {entry['timestamp']}  {entry['action']:10s}  {entry['target']:20s}  {entry['value']}")


def cmd_add_repo(args):
    result = api.add_repo(args.name, args.path, **_backend_kwargs(args))
    print(f"Added repo {result['name']}: {result['path']}")


def cmd_repos(args):
    result = api.list_repos(**_backend_kwargs(args))
    if not result["repos"]:
        print("No repos registered.")
        return
    for name, path in sorted(result["repos"].items()):
        print(f"  {name}: {path}")
    print(f"\n{len(result['repos'])} repo(s)")


def cmd_import_agent(args):
    try:
        result = api.import_agent(
            agent_name=args.agent_name,
            beliefs_file=args.beliefs_file,
            nogoods_file=args.nogoods_file,
            only_in=args.only_in,
            **_backend_kwargs(args),
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Agent '{result['agent']}' imported:")
    if result['created_premise']:
        print(f"  Created premise: {result['active_node']}")
    else:
        print(f"  Premise exists:  {result['active_node']}")
    print(f"  Imported:  {result['claims_imported']} beliefs (as {result['prefix']}*)")
    if result['claims_skipped']:
        print(f"  Skipped:   {result['claims_skipped']} (already in network)")
    if result['claims_retracted']:
        print(f"  Retracted: {result['claims_retracted']} (STALE/OUT in source)")
    if result.get('claims_propagated'):
        print(f"  Propagated: {result['claims_propagated']} (truth values recomputed)")
    if result['nogoods_imported']:
        print(f"  Nogoods:   {result['nogoods_imported']}")
    print(f"\n  To revoke all: reasons retract {result['active_node']}")


def cmd_sync_agent(args):
    try:
        result = api.sync_agent(
            agent_name=args.agent_name,
            beliefs_file=args.beliefs_file,
            nogoods_file=args.nogoods_file,
            only_in=args.only_in,
            **_backend_kwargs(args),
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Agent '{result['agent']}' synced:")
    if result['created_premise']:
        print(f"  Created premise: {result['active_node']}")
    if result['beliefs_added']:
        print(f"  Added:     {result['beliefs_added']} new beliefs")
    if result['beliefs_updated']:
        print(f"  Updated:   {result['beliefs_updated']} beliefs")
    if result['beliefs_removed']:
        print(f"  Removed:   {result['beliefs_removed']} beliefs (retracted)")
    if result['beliefs_retracted']:
        print(f"  Retracted: {result['beliefs_retracted']} (OUT/STALE in source)")
    if result['beliefs_unchanged']:
        print(f"  Unchanged: {result['beliefs_unchanged']}")
    if result.get('beliefs_propagated'):
        print(f"  Propagated: {result['beliefs_propagated']} (truth values recomputed)")
    if result['nogoods_imported']:
        print(f"  Nogoods:   {result['nogoods_imported']}")


def cmd_import_beliefs(args):
    try:
        result = api.import_beliefs(
            beliefs_file=args.beliefs_file,
            nogoods_file=args.nogoods_file,
            **_backend_kwargs(args),
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Imported {result['claims_imported']} claims ({result['claims_retracted']} retracted)")
    if result['claims_skipped']:
        print(f"Skipped {result['claims_skipped']} (already in network)")
    if result['nogoods_imported']:
        print(f"Imported {result['nogoods_imported']} nogoods")


def cmd_import_json(args):
    try:
        result = api.import_json(args.json_file, **_backend_kwargs(args))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Imported {result['nodes_imported']} nodes")
    if result['nogoods_imported']:
        print(f"Imported {result['nogoods_imported']} nogoods")


def cmd_import_hf(args):
    try:
        result = api.import_hf(
            repo_id=args.repo_id,
            init=args.init,
            token=args.token,
            **_backend_kwargs(args),
        )
    except (RuntimeError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Imported {result['nodes_imported']} nodes from {result['repo_id']}")
    if result['nogoods_imported']:
        print(f"Imported {result['nogoods_imported']} nogoods")


def cmd_pull(args):
    cmd_import_hf(args)


def cmd_publish(args):
    try:
        result = api.publish_hf(
            repo_id=args.repo_id,
            token=args.token,
            private=getattr(args, "private", False),
            visible_to=_parse_visible_to(args),
            domain=getattr(args, "domain", None),
            license=getattr(args, "license", "mit"),
            base_network=getattr(args, "base_network", None),
            source_repos=getattr(args, "source_repos", None),
            **_backend_kwargs(args),
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Published to {result['url']}")
    for f in result["files_uploaded"]:
        print(f"  uploaded {f}")


def cmd_import_api(args):
    try:
        result = api.import_api(
            url=args.url,
            agent_id=args.agent_id,
            api_key=args.api_key,
            init=args.init,
            **_backend_kwargs(args),
        )
    except (RuntimeError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Imported {result['nodes_imported']} nodes")
    if result['nogoods_imported']:
        print(f"Imported {result['nogoods_imported']} nogoods")


def cmd_export_api(args):
    try:
        result = api.export_api(
            url=args.url,
            agent_id=args.agent_id,
            api_key=args.api_key,
            **_backend_kwargs(args),
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Exported {result['nodes_exported']} nodes")
    if result['errors']:
        print(f"Errors: {result['errors']}")


def cmd_export(args):
    data = api.export_network(visible_to=_parse_visible_to(args), **_backend_kwargs(args))
    output = json.dumps(data, indent=2)
    if args.output == "-":
        print(output)
    else:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}")


def cmd_export_markdown(args):
    md = api.export_markdown(visible_to=_parse_visible_to(args), **_backend_kwargs(args))
    if args.output == "-":
        print(md)
    else:
        Path(args.output).write_text(md)
        print(f"Written to {args.output}")


def cmd_export_card(args):
    md = api.export_card(
        visible_to=_parse_visible_to(args),
        domain=args.domain,
        license=args.license,
        base_network=args.base_network,
        source_repos=args.source_repos,
        **_backend_kwargs(args),
    )
    if args.output == "-":
        print(md)
    else:
        Path(args.output).write_text(md)
        print(f"Written to {args.output}")


def cmd_hash_sources(args):
    result = api.hash_sources(force=args.force, **_backend_kwargs(args))

    if not result["hashed"]:
        print("No nodes to hash (all sources already have hashes, or source files not found).")
        if not args.force:
            print("Use --force to re-hash nodes that already have hashes.")
        return

    for item in result["hashed"]:
        action = "backfilled" if item["was_empty"] else "re-hashed"
        print(f"  {action}  {item['node_id']}  {item['hash']}  ({item['source']})")

    backfilled = sum(1 for h in result["hashed"] if h["was_empty"])
    rehashed = result["count"] - backfilled
    parts = []
    if backfilled:
        parts.append(f"{backfilled} backfilled")
    if rehashed:
        parts.append(f"{rehashed} re-hashed")
    print(f"\n{', '.join(parts)}")


def cmd_check_stale(args):
    result = api.check_stale(upgrade_hashes=args.upgrade_hashes,
                             git_aware=getattr(args, "git", False),
                             **_backend_kwargs(args))

    if result.get("upgraded"):
        print(f"Upgraded {result['upgraded']} truncated hash(es) to full length.")
    if result.get("sha_bumped"):
        print(f"Auto-bumped {result['sha_bumped']} pinned SHA(s) (content unchanged).")

    if not result["stale"]:
        print(f"All {result['checked']} nodes with sources are fresh.")
        return

    truncated = [i for i in result["stale"] if i.get("reason") == "truncated_hash"]
    stale = [i for i in result["stale"] if i.get("reason") != "truncated_hash"]

    for item in stale:
        if item.get("reason") == "source_deleted":
            print(f"  DELETED  {item['node_id']}")
            print(f"           source: {item['source']}")
        else:
            print(f"  STALE  {item['node_id']}")
            print(f"         source: {item['source']}")
            print(f"         hash: {item['old_hash']} -> {item['new_hash']}")
        print()

    if truncated:
        print(f"WARNING: {len(truncated)} node(s) have truncated hashes.")
        print("  Run 'reasons check-stale --upgrade-hashes' to upgrade them.\n")

    fresh = result["checked"] - len(stale)
    print(f"{fresh} fresh, {len(stale)} stale (of {result['checked']} checked)")
    if stale:
        sys.exit(1)


def cmd_check_integrity(args):
    _require_sqlite(args, "check-integrity")
    result = api.check_integrity(db_path=args.db)

    if result["text_mutations"]:
        print(f"Text mutations: {len(result['text_mutations'])}")
        for f in result["text_mutations"]:
            print(f"  {f['node_id']}: text changed since creation")
        print()

    if result["chain_mutations"]:
        print(f"Chain mutations: {len(result['chain_mutations'])}")
        for f in result["chain_mutations"]:
            print(f"  {f['node_id']} j{f['justification_index']}: antecedent text changed")
        print()

    if result["missing_hashes"]:
        print(f"Missing hashes: {result['missing_hashes']} (run 'reasons backfill-hashes' to compute)")
        print()

    total = len(result["text_mutations"]) + len(result["chain_mutations"])
    if total:
        print(f"{total} integrity issue(s) found")
        sys.exit(1)
    else:
        print("All Merkle hashes verified — no mutations detected")


def cmd_backfill_hashes(args):
    _require_sqlite(args, "backfill-hashes")
    result = api.backfill_hashes(db_path=args.db)
    print(f"Nodes updated: {result['nodes_updated']}")
    print(f"Justifications updated: {result['justifications_updated']}")


def cmd_pin_sources(args):
    _require_sqlite(args, "pin-sources")
    result = api.pin_sources(
        force=args.force,
        pin_urls=getattr(args, "pin_urls", False),
        db_path=args.db,
    )

    if not result["pinned"]:
        print("No nodes to pin (all sources already pinned, or source files not in git).")
        if not args.force:
            print("Use --force to re-pin nodes that already have a pinned_sha.")
        return

    for item in result["pinned"]:
        action = "pinned" if item["was_empty"] else "re-pinned"
        print(f"  {action}  {item['node_id']}  {item['pinned_sha'][:12]}  ({item['source']})")

    new = sum(1 for p in result["pinned"] if p["was_empty"])
    re = result["count"] - new
    parts = []
    if new:
        parts.append(f"{new} pinned")
    if re:
        parts.append(f"{re} re-pinned")
    print(f"\n{', '.join(parts)}")


def cmd_pin_update(args):
    _require_sqlite(args, "pin-update")
    result = api.pin_update(node_ids=args.node_ids, db_path=args.db)

    for item in result["updated"]:
        if "error" in item:
            print(f"  ERROR  {item['node_id']}: {item['error']}")
        else:
            old = item['old_sha'][:12] if item['old_sha'] else "(none)"
            print(f"  UPDATED  {item['node_id']}  {old} -> {item['new_sha'][:12]}")

    if result["errors"]:
        print(f"\n{result['count']} updated, {result['errors']} errors")
    else:
        print(f"\n{result['count']} updated")


def cmd_pin_lines(args):
    _require_sqlite(args, "pin-lines")
    try:
        result = api.pin_lines(
            node_id=args.node_id,
            line_start=args.start,
            line_end=args.end,
            db_path=args.db,
        )
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    lines = result["pinned_lines"]
    sha = result["pinned_sha"][:12] if result.get("pinned_sha") else "(none)"
    auto = " (auto-pinned SHA)" if result.get("auto_pinned") else ""
    print(f"  PINNED  {result['node_id']}  lines {lines}  sha {sha}{auto}")


def cmd_compact(args):
    summary = api.compact(
        budget=args.budget,
        truncate=not args.no_truncate,
        visible_to=_parse_visible_to(args),
        **_backend_kwargs(args),
    )
    print(summary)


def _parse_visible_to(args):
    val = getattr(args, "visible_to", None)
    if val is not None:
        return [t.strip() for t in val.split(",") if t.strip()]
    return None


def _backend_kwargs(args):
    pg = getattr(args, "pg", None) or os.environ.get("REASONSFORGE_PG_CONNINFO")
    pid = getattr(args, "project_id", None) or os.environ.get("REASONSFORGE_PROJECT_ID")
    if pg:
        if not pid:
            print("Error: --project-id is required with --pg", file=sys.stderr)
            sys.exit(1)
        return {"pg_conninfo": pg, "project_id": pid}
    return {"db_path": args.db}


def _require_sqlite(args, command_name):
    pg = getattr(args, "pg", None) or os.environ.get("REASONSFORGE_PG_CONNINFO")
    if pg:
        print(f"Error: {command_name} is not supported with --pg", file=sys.stderr)
        sys.exit(1)


def cmd_search(args):
    fmt = getattr(args, "format", "markdown")
    result = api.search(args.query, visible_to=_parse_visible_to(args), format=fmt, **_backend_kwargs(args))
    print(result)


def cmd_lookup(args):
    result = api.lookup(args.query, visible_to=_parse_visible_to(args), **_backend_kwargs(args))
    print(result)


def cmd_search_sources(args):
    from .ask import search_source_chunks
    results = search_source_chunks(args.query, args.db, top_k=args.top_k)
    if not results:
        print("No matching chunks found.")
        return
    fmt = getattr(args, "format", "text")
    if fmt == "json":
        print(json.dumps(results, indent=2))
        return
    for i, row in enumerate(results, 1):
        header = f"[{i}] {row['filename']}"
        if row.get("section"):
            header += f" > {row['section']}"
        print(f"### {header}\n")
        print(row["text"])
        if i < len(results):
            print("\n---\n")


def cmd_ask(args):
    _require_sqlite(args, "ask")
    from .ask import ask

    mcp_servers = []
    try:
        if args.mcp:
            from .mcp_client import McpBridge
            for cmd in args.mcp:
                bridge = McpBridge(cmd)
                bridge.connect()
                mcp_servers.append(bridge)

        result = ask(
            question=args.question,
            db_path=args.db,
            timeout=args.timeout,
            no_synth=args.no_synth,
            format=getattr(args, "format", None),
            model=args.model or "claude",
            simple=args.simple,
            sources_db=args.full_sources,
            natural=args.natural,
            dual=args.dual,
            mcp_servers=mcp_servers or None,
        )
        print(result)
    finally:
        for bridge in mcp_servers:
            bridge.close()


def cmd_cluster_list(args):
    _require_sqlite(args, "cluster-list")
    result = api.list_clusters(
        status=args.status,
        n_clusters=args.n_clusters,
        seed=args.seed,
        embedding_model=args.embedding_model,
        visible_to=_parse_visible_to(args),
        db_path=args.db,
    )

    if not result["clusters"]:
        print("No beliefs to cluster.")
        return

    fmt = args.format

    if fmt == "json":
        print(json.dumps(result, indent=2))
        return

    if fmt == "markdown":
        for i, cluster in enumerate(result["clusters"], 1):
            size = len(cluster["beliefs"])
            print(f"\n## Cluster {i} ({size} belief{'s' if size != 1 else ''})\n")
            for b in cluster["beliefs"]:
                print(f"- **{b['id']}**: {b['text']}")
        return

    total = 0
    for i, cluster in enumerate(result["clusters"], 1):
        size = len(cluster["beliefs"])
        total += size
        print(f"\nCluster {i} ({size} belief{'s' if size != 1 else ''}):")
        for b in cluster["beliefs"]:
            marker = "+" if args.status == "IN" else "-"
            print(f"  [{marker}] {b['id']}")
            text = b['text'][:100] + "..." if len(b['text']) > 100 else b['text']
            print(f"      {text}")

    print(f"\n{result['n_clusters']} cluster(s), {total} beliefs")
    print(f"Model: {result['embedding_model']}")


def cmd_deduplicate(args):
    _require_sqlite(args, "deduplicate")
    if args.accept:
        accept_path = Path(args.accept)
        if not accept_path.exists():
            print(f"File not found: {accept_path}", file=sys.stderr)
            sys.exit(1)
        plan = api.parse_dedup_plan(accept_path.read_text())
        if not plan:
            print("No clusters found in plan file.")
            return
        result = api.apply_dedup_plan(plan, db_path=args.db)
        for err in result["errors"]:
            print(f"  ERROR: {err}", file=sys.stderr)
        if result["retracted"]:
            print(f"Retracted {len(result['retracted'])} duplicates "
                  f"(from {result['applied']} cluster(s)).")
            for nid in result["retracted"]:
                print(f"  RETRACTED {nid}")
        else:
            print("No duplicates to retract.")
        return

    result = api.deduplicate(
        threshold=args.threshold,
        auto=args.auto,
        semantic=args.semantic,
        embedding_model=args.embedding_model,
        db_path=args.db,
    )

    if not result["clusters"]:
        print("No duplicate clusters found.")
        return

    for i, cluster in enumerate(result["clusters"], 1):
        print(f"\nCluster {i} ({cluster['size']} beliefs):")
        for b in cluster["beliefs"]:
            deps = f"  [{b['dependents']} dependents]" if b["dependents"] else ""
            kept = "  <- kept" if cluster.get("kept") == b["id"] else ""
            retracted = "  RETRACTED" if b["id"] in result["retracted"] else ""
            print(f"  {b['id']}{deps}{kept}{retracted}")
            print(f"    {b['text'][:100]}")

    print(f"\n{len(result['clusters'])} cluster(s), "
          f"{sum(c['size'] for c in result['clusters'])} beliefs involved")
    if result["retracted"]:
        print(f"Retracted {len(result['retracted'])} duplicates.")
    elif not args.auto:
        output = args.output
        api.write_dedup_plan(result["clusters"], output)
        print(f"\nWrote {output} — review, then run:")
        print(f"  reasons deduplicate --accept {output}")


def _derive_one_round(args, round_num=None, report_state=None,
                      cluster_cache=None, prompt_template=None):
    """Run a single derive round. Returns number of beliefs added (0 = saturated).

    Used by cmd_derive for both single-round and --exhaust mode.
    If report_state is provided, appends round results and writes the report.
    """
    import subprocess

    from .derive import (
        build_prompt,
        parse_proposals,
        validate_proposals,
        apply_proposals,
        write_proposals_file,
    )

    prefix = f"[round {round_num}] " if round_num is not None else ""

    # Load network (fresh each round)
    try:
        result = api.export_network(db_path=args.db)
    except Exception as e:
        print(f"{prefix}Error loading network: {e}", file=sys.stderr)
        return -1

    nodes = result.get("nodes", {})
    if not nodes:
        print(f"{prefix}No nodes in the network.", file=sys.stderr)
        return -1

    prompt, stats = build_prompt(
        nodes, domain=args.domain, topic=args.topic,
        budget=args.budget, sample=args.sample, seed=args.seed,
        min_depth=args.min_depth, max_depth_filter=args.max_depth,
        premises_only=args.premises, has_dependents=args.has_dependents,
        cluster=args.cluster, intra_cluster=args.intra_cluster,
        round_num=round_num or 0, cluster_cache=cluster_cache,
        embedding_model=args.embedding_model, n_clusters=args.n_clusters,
        prompt_template=prompt_template,
    )

    print(f"{prefix}Network: {stats['total_in']} IN beliefs, "
          f"{stats['total_derived']} derived, max depth {stats['max_depth']}",
          file=sys.stderr)
    if stats.get("topic"):
        print(f"{prefix}Topic filter: {stats['topic']}", file=sys.stderr)
    if stats.get("min_depth") is not None or stats.get("max_depth_filter") is not None:
        lo = stats.get("min_depth", 0)
        hi = stats.get("max_depth_filter", "∞")
        print(f"{prefix}Depth filter: {lo}–{hi}", file=sys.stderr)
    if stats.get("cluster"):
        cluster_mode = "intra" if stats.get("intra_cluster") else "inter"
        print(f"{prefix}Clustering ({cluster_mode}): {stats['n_clusters']} clusters, "
              f"model={stats['embedding_model']}", file=sys.stderr)
        if stats.get("focus_cluster") is not None:
            print(f"{prefix}Focus: cluster {stats['focus_cluster']}", file=sys.stderr)
    elif stats.get("sample"):
        print(f"{prefix}Sampling: {stats['budget']} beliefs (random)", file=sys.stderr)
    elif stats.get("budget", 300) != 300:
        print(f"{prefix}Budget: {stats['budget']} beliefs", file=sys.stderr)
    if stats["agents"]:
        print(f"{prefix}Agents: {', '.join(stats['agent_names'])}", file=sys.stderr)

    if args.dry_run:
        print(f"\n=== Prompt ({len(prompt)} chars) ===\n")
        print(prompt[:3000])
        if len(prompt) > 3000:
            print(f"\n... ({len(prompt) - 3000} more chars)")
        return 0

    # Model invocation via CLI
    model = args.model or "claude"

    print(f"{prefix}Deriving with {model}...", file=sys.stderr)

    from .llm import invoke_model
    try:
        response = invoke_model(prompt, model=model, timeout=args.timeout)
    except FileNotFoundError as e:
        print(f"{prefix}Error: {e}", file=sys.stderr)
        return -1
    except subprocess.TimeoutExpired:
        print(f"{prefix}Model timed out after {args.timeout}s", file=sys.stderr)
        return -1
    except Exception as e:
        print(f"{prefix}Error: {e}", file=sys.stderr)
        return -1

    # Parse and validate proposals
    proposals = parse_proposals(response)

    if not proposals:
        print(f"{prefix}No new proposals — network saturated.", file=sys.stderr)
        if report_state is not None:
            report_state["rounds"].append({
                "round": round_num or 1,
                "network_stats": stats,
                "proposals_found": 0, "valid": 0,
                "skipped": [], "applied": [], "added": 0,
            })
            _write_derive_report(report_state, "partial")
        return 0

    valid, skipped = validate_proposals(proposals, nodes)

    for p, reason in skipped:
        print(f"  SKIP {p['id']}: {reason}", file=sys.stderr)

    print(f"\n{prefix}{len(valid)} valid proposals "
          f"({len(skipped)} skipped)", file=sys.stderr)

    round_result = {
        "round": round_num or 1,
        "network_stats": stats,
        "proposals_found": len(proposals),
        "valid": len(valid),
        "skipped": [{"id": p["id"], "reason": reason} for p, reason in skipped],
        "applied": [],
        "added": 0,
    }

    if not valid:
        if report_state is not None:
            report_state["rounds"].append(round_result)
            _write_derive_report(report_state, "partial")
        return 0

    if args.auto or args.exhaust:
        results = apply_proposals(valid, db_path=args.db)
        added = 0
        for p, result in results:
            if isinstance(result, dict):
                print(f"  Added {p['id']} [{result['truth_value']}]")
                round_result["applied"].append({
                    "id": p["id"], "truth_value": result["truth_value"],
                })
                added += 1
            else:
                print(f"  FAIL {p['id']}: {result}", file=sys.stderr)
        round_result["added"] = added
        if added:
            print(f"\n{prefix}Added {added} derived beliefs.", file=sys.stderr)
        if report_state is not None:
            report_state["rounds"].append(round_result)
            _write_derive_report(report_state, "partial")
        return added
    else:
        output_path = Path(args.output)
        write_proposals_file(valid, output_path)
        print(f"\n{prefix}Wrote {output_path} ({len(valid)} proposals)")
        round_result["added"] = 0
        round_result["proposed"] = len(valid)
        if report_state is not None:
            report_state["rounds"].append(round_result)
            _write_derive_report(report_state, "partial")
        return len(valid)


def _write_derive_report(report_state, status):
    """Write derive JSON report to disk."""
    import json
    if report_state.get("report_path") is None:
        return
    total = sum(r["added"] for r in report_state["rounds"])
    report = {
        "timestamp": report_state["ts"],
        "status": status,
        "model": report_state["model"],
        "timeout": report_state["timeout"],
        "exhaust": report_state["exhaust"],
        "filters": report_state["filters"],
        "rounds": report_state["rounds"],
        "total_added": total,
    }
    report_state["report_path"].write_text(json.dumps(report, indent=2))


def cmd_derive(args):
    _require_sqlite(args, "derive")
    from datetime import datetime
    from .llm import reset_cost_tracker, format_cost_summary
    reset_cost_tracker()

    prompt_template = None
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"Error: prompt file not found: {prompt_path}",
                  file=sys.stderr)
            sys.exit(1)
        prompt_template = prompt_path.read_text()

    if (args.cluster or args.intra_cluster) and args.sample:
        print("Error: --cluster/--intra-cluster and --sample are mutually exclusive.",
              file=sys.stderr)
        sys.exit(1)

    if args.cluster and args.intra_cluster:
        print("Error: --cluster and --intra-cluster are mutually exclusive.",
              file=sys.stderr)
        sys.exit(1)

    cluster_cache = None
    if args.cluster or args.intra_cluster:
        try:
            from .cluster import ClusterCache
            print("Loading embedding model...", file=sys.stderr)
            cluster_cache = ClusterCache(
                model_name=args.embedding_model or "all-MiniLM-L6-v2"
            )
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    report_state = None
    if not args.no_report:
        ts = datetime.now().isoformat(timespec="seconds")
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"derive-{ts.replace(':', '')}.json"
        model = args.model or "claude"
        report_state = {
            "report_path": report_path,
            "ts": ts,
            "model": model,
            "timeout": args.timeout,
            "exhaust": args.exhaust,
            "filters": {
                "domain": args.domain,
                "topic": args.topic,
                "min_depth": args.min_depth,
                "max_depth": args.max_depth,
                "premises": args.premises,
                "has_dependents": args.has_dependents,
                "budget": args.budget,
                "sample": args.sample,
                "cluster": args.cluster,
                "intra_cluster": args.intra_cluster,
                "embedding_model": args.embedding_model,
                "n_clusters": args.n_clusters,
                "prompt_file": args.prompt_file,
            },
            "rounds": [],
        }

    if args.exhaust:
        max_rounds = args.max_rounds
        total_added = 0
        for round_num in range(1, max_rounds + 1):
            print(f"\n{'=' * 40}", file=sys.stderr)
            print(f"Round {round_num}/{max_rounds}", file=sys.stderr)
            print(f"{'=' * 40}", file=sys.stderr)
            added = _derive_one_round(args, round_num=round_num,
                                      report_state=report_state,
                                      cluster_cache=cluster_cache,
                                      prompt_template=prompt_template)
            if added < 0:
                print(f"\nExhaust stopped: error in round {round_num}.",
                      file=sys.stderr)
                sys.exit(1)
            if added == 0:
                print(f"\nExhaust complete: saturated after {round_num} rounds. "
                      f"Total added: {total_added}.", file=sys.stderr)
                break
            total_added += added
        else:
            print(f"\nExhaust complete: hit max rounds ({max_rounds}). "
                  f"Total added: {total_added}.", file=sys.stderr)
    else:
        added = _derive_one_round(args, report_state=report_state,
                                  cluster_cache=cluster_cache,
                                  prompt_template=prompt_template)
        if added < 0:
            sys.exit(1)

    if report_state is not None:
        _write_derive_report(report_state, "complete")
        print(f"  Report: {report_state['report_path']}")

    cost = format_cost_summary()
    if cost:
        print(f"  {cost}", file=sys.stderr)


def cmd_accept(args):
    _require_sqlite(args, "accept")
    from .derive import parse_proposals, validate_proposals, apply_proposals

    proposals_path = Path(args.file)
    if not proposals_path.exists():
        print(f"File not found: {proposals_path}", file=sys.stderr)
        sys.exit(1)

    text = proposals_path.read_text()
    proposals = parse_proposals(text)

    if not proposals:
        print("No proposals found in file.")
        return

    # Load network for validation
    result = api.export_network(db_path=args.db)
    nodes = result.get("nodes", {})

    valid, skipped = validate_proposals(proposals, nodes)

    for p, reason in skipped:
        print(f"  SKIP {p['id']}: {reason}", file=sys.stderr)

    if not valid:
        print("No valid proposals to accept.")
        return

    results = apply_proposals(valid, db_path=args.db)
    added = 0
    for p, result in results:
        if isinstance(result, dict):
            print(f"  Added {p['id']} [{result['truth_value']}]")
            added += 1
        else:
            print(f"  FAIL {p['id']}: {result}", file=sys.stderr)

    print(f"\nAccepted {added} of {len(proposals)} proposals "
          f"({len(skipped)} skipped).", file=sys.stderr)


def cmd_list(args):
    result = api.list_nodes(
        status=args.status,
        premises_only=args.premises,
        has_dependents=args.has_dependents,
        challenged=args.challenged,
        namespace=getattr(args, "namespace", None),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        visible_to=_parse_visible_to(args),
        not_reviewed_since=args.not_reviewed_since,
        never_reviewed=args.never_reviewed,
        by_impact=args.by_impact,
        label=args.label,
        **_backend_kwargs(args),
    )

    if args.never_reviewed and args.not_reviewed_since is not None:
        print("Warning: --never-reviewed makes --not-reviewed-since a no-op",
              file=sys.stderr)

    if not result["nodes"]:
        print("No matching nodes.")
        return

    show_review = args.never_reviewed or args.not_reviewed_since is not None
    for node in result["nodes"]:
        marker = "+" if node["truth_value"] == "IN" else "-"
        jinfo = f"  ({node['justification_count']} justification{'s' if node['justification_count'] != 1 else ''})" if node["justification_count"] else "  (premise)"
        deps = f"  [{node['dependent_count']} dependents]" if node["dependent_count"] else ""
        stype = f"  <{node['source_type']}>" if node.get("source_type") else ""
        review_info = ""
        if show_review:
            if node.get("last_reviewed"):
                review_info = f"  (reviewed: {node['last_reviewed']}, {node.get('review_result', '?')})"
            elif node.get("justification_count", 0) > 0:
                review_info = "  (never reviewed)"
        print(f"  [{marker}] {node['id']}{jinfo}{deps}{stype}{review_info}")

    print(f"\n{result['count']} node{'s' if result['count'] != 1 else ''}")


def cmd_list_gated(args):
    result = api.list_gated(
        visible_to=_parse_visible_to(args),
        **_backend_kwargs(args),
    )

    if not result["blockers"]:
        print("No active gates found. All gated beliefs are satisfied.")
        return

    for blocker_id, info in sorted(result["blockers"].items()):
        print(f"  [{blocker_id}] {info['text']}")
        for gated in info["gated"]:
            print(f"    ⊢ {gated['id']}: {gated['text']}")
        print()

    print(f"{result['blocker_count']} blocker(s) gating {result['gated_count']} belief(s)")


def cmd_report_gated(args):
    model = getattr(args, "model", None) or ""
    if model:
        from .llm import reset_cost_tracker
        reset_cost_tracker()

    result = api.report_gated(
        visible_to=_parse_visible_to(args),
        model=model,
        timeout=args.timeout,
        **_backend_kwargs(args),
    )

    report = result["report"]
    output_path = getattr(args, "output", None)
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report written to {output_path}")
    else:
        print(report)

    print(
        f"  {result['blocker_count']} blocker(s), "
        f"{result['gated_count']} gated belief(s), "
        f"{result['retracted_count']} retracted premise(s)",
        file=sys.stderr,
    )

    if model:
        from .llm import format_cost_summary
        cost = format_cost_summary()
        if cost:
            print(f"  {cost}", file=sys.stderr)


def cmd_report(args):
    model = getattr(args, "model", None) or ""
    if model:
        from .llm import reset_cost_tracker
        reset_cost_tracker()

    result = api.report_belief(
        args.node_id,
        sources_db=getattr(args, "sources_db", None),
        model=model,
        timeout=args.timeout,
        **_backend_kwargs(args),
    )

    report = result["report"]
    output_path = getattr(args, "output", None)
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report written to {output_path}")
    else:
        print(report)

    print(
        f"  {result['premise_count']} root premise(s)",
        file=sys.stderr,
    )

    if model:
        from .llm import format_cost_summary
        cost = format_cost_summary()
        if cost:
            print(f"  {cost}", file=sys.stderr)


def cmd_verify(args):
    from .llm import reset_cost_tracker, format_cost_summary
    reset_cost_tracker()

    result = api.verify_belief(
        args.node_id,
        trace=getattr(args, "trace", False),
        retract=getattr(args, "retract", False),
        dry_run=getattr(args, "dry_run", False),
        model=getattr(args, "model", "claude") or "claude",
        timeout=args.timeout,
        **_backend_kwargs(args),
    )

    beliefs = result["beliefs_checked"]

    if result["dry_run"]:
        print(f"--dry-run: {len(beliefs)} belief(s) would be verified:")
        for b in beliefs:
            has_src = "yes" if b.get("source") else "NO"
            print(f"  {b['id']} [source: {has_src}]")
        return

    markers = {
        "CONFIRMED": "+", "STALE": "-",
        "PARTIAL": "~", "INCONCLUSIVE": "?",
    }
    verdicts = result["results"]

    confirmed, stale, partial, inconclusive = [], [], [], []
    for b in beliefs:
        bid = b["id"]
        v = verdicts.get(bid, {})
        verdict = v.get("verdict", "INCONCLUSIVE")
        reason = v.get("reason", "")
        quote = v.get("quote")
        marker = markers.get(verdict, "?")

        print(f"\n  [{marker}] {verdict}: {bid}")
        if reason:
            print(f"      {reason}")
        if quote:
            print(f'      Quote: "{quote}"')

        if verdict == "CONFIRMED":
            confirmed.append(bid)
        elif verdict == "STALE":
            stale.append(bid)
        elif verdict == "PARTIAL":
            partial.append(bid)
        else:
            inconclusive.append(bid)

    if result["is_derived"] and getattr(args, "trace", False):
        total = len(beliefs)
        print(f"\n  Antecedent summary: {len(confirmed)}/{total} confirmed, "
              f"{len(stale)} stale, {len(partial)} partial, "
              f"{len(inconclusive)} inconclusive")

    if result.get("retract_failed"):
        print(f"\n  WARNING: failed to retract: {', '.join(result['retract_failed'])}", file=sys.stderr)
    if result.get("stamp_failed"):
        print(f"\n  WARNING: failed to stamp verified_at: {', '.join(result['stamp_failed'])}", file=sys.stderr)

    print(f"\nResults: {len(confirmed)} confirmed, {len(stale)} stale, "
          f"{len(partial)} partial, {len(inconclusive)} inconclusive")

    cost = format_cost_summary()
    if cost:
        print(f"  {cost}", file=sys.stderr)


def cmd_list_negative(args):
    result = api.list_negative(
        visible_to=_parse_visible_to(args),
        model=getattr(args, "model", None) or "claude",
        skip_llm=getattr(args, "no_llm", False),
        **_backend_kwargs(args),
    )

    if not result["negative"]:
        print("No negative beliefs found.")
        return

    for item in result["negative"]:
        print(f"  [-] {item['id']}: {item['text']}")

    print(f"\n{result['count']} negative belief(s) "
          f"({result['candidates']} candidates from {result['total']} IN nodes)")


def cmd_topics(args):
    result = api.topics(
        limit=args.limit,
        **_backend_kwargs(args),
    )
    if getattr(args, "json", False):
        import json
        print(json.dumps(result, indent=2))
        return
    if not result["topics"]:
        print("No topics found.")
        return
    for item in result["topics"]:
        print(f"  {item['topic']} ({item['count']})")
    print(f"\n{len(result['topics'])} topics from {result['total_nodes']} nodes")


def cmd_build_wiki(args):
    _require_sqlite(args, "build-wiki")
    model = getattr(args, "model", None) or ""
    if model:
        from .llm import reset_cost_tracker
        reset_cost_tracker()
    result = api.build_wiki(
        output_dir=args.output,
        status=args.status or None,
        max_topics=args.max_topics,
        cluster=args.cluster,
        n_clusters=args.n_clusters,
        seed=args.seed,
        embedding_model=args.embedding_model,
        visible_to=_parse_visible_to(args),
        model=model,
        timeout=args.timeout,
        parallel=getattr(args, "parallel", 0) or 0,
        db_path=args.db,
    )
    print(f"Wiki written to {result['output_dir']}/")
    print(f"  {result['total_nodes']} beliefs across {result['pages']} pages")
    if model:
        from .llm import format_cost_summary
        cost = format_cost_summary()
        if cost:
            print(f"  {cost}", file=sys.stderr)


def cmd_review_beliefs(args):
    _require_sqlite(args, "review-beliefs")
    import json
    from datetime import datetime
    from .llm import reset_cost_tracker, format_cost_summary
    reset_cost_tracker()

    model = getattr(args, "model", None) or "claude"
    ts = datetime.now().isoformat(timespec="seconds")
    write_report = not args.no_report

    report_path = None
    if write_report:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"review-beliefs-{ts.replace(':', '')}.json"

    def _build_report(results, status):
        invalid = sum(1 for r in results if not r.get("valid", True))
        insufficient = sum(1 for r in results if not r.get("sufficient", True))
        unnecessary = sum(1 for r in results if not r.get("necessary", True))
        return {
            "timestamp": ts,
            "status": status,
            "model": model,
            "timeout": args.timeout,
            "dry_run": args.dry_run,
            "filters": {
                "belief_ids": args.ids or None,
                "min_depth": args.min_depth,
                "depends_on": args.depends_on,
                "namespace": args.namespace,
                "sample": args.sample,
                "visible_to": _parse_visible_to(args),
            },
            "reviewed": len(results),
            "total_derived": None,
            "summary": {
                "invalid": invalid,
                "insufficient": insufficient,
                "unnecessary": unnecessary,
            },
            "results": results,
        }

    def _write_report(results, status):
        if report_path is not None:
            report_path.write_text(json.dumps(_build_report(results, status), indent=2))

    on_batch = (lambda results: _write_report(results, "partial")) if write_report else None

    result = api.review_beliefs(
        belief_ids=args.ids or None,
        model=model,
        timeout=args.timeout,
        min_depth=args.min_depth,
        depends_on=args.depends_on,
        namespace=args.namespace,
        sample=args.sample,
        visible_to=_parse_visible_to(args),
        dry_run=args.dry_run,
        on_batch=on_batch,
        include_out=args.include_out,
        db_path=args.db,
    )

    reviews = result["results"]
    if not reviews:
        print("No derived beliefs to review.")
        return

    invalid = [r for r in reviews if not r.get("valid", True)]
    insufficient = [r for r in reviews if not r.get("sufficient", True)]
    unnecessary = [r for r in reviews if not r.get("necessary", True)]

    for r in reviews:
        flags = []
        if not r.get("valid", True):
            flags.append("INVALID")
        if not r.get("sufficient", True):
            flags.append("INSUFFICIENT")
        if not r.get("necessary", True):
            unneeded = r.get("unnecessary_antecedents", [])
            flag = "UNNECESSARY"
            if unneeded:
                flag += f"({', '.join(unneeded)})"
            flags.append(flag)

        if flags:
            print(f"  [{' | '.join(flags)}] {r['id']}")
            if r.get("comment"):
                print(f"    {r['comment']}")

    print(f"\nReviewed {result['reviewed']} of {result['total_derived']} derived beliefs")
    print(f"  Invalid: {result['invalid']}  Insufficient: {result['insufficient']}"
          f"  Unnecessary: {result['unnecessary']}")

    if report_path is not None:
        report = _build_report(reviews, "complete")
        report["total_derived"] = result["total_derived"]
        report_path.write_text(json.dumps(report, indent=2))
        print(f"  Report: {report_path}")

    if args.output:
        with open(args.output, "w") as f:
            f.write("# Belief Review Findings\n\n")
            for r in reviews:
                v = "PASS" if r.get("valid", True) else "FAIL"
                s = "PASS" if r.get("sufficient", True) else "FAIL"
                n = "PASS" if r.get("necessary", True) else "FAIL"
                f.write(f"### {r['id']}\n")
                f.write(f"- Valid: {v}\n- Sufficient: {s}\n- Necessary: {n}\n")
                if r.get("unnecessary_antecedents"):
                    f.write(f"- Unnecessary antecedents: {', '.join(r['unnecessary_antecedents'])}\n")
                if r.get("comment"):
                    f.write(f"- Comment: {r['comment']}\n")
                f.write("\n")
        print(f"\nWrote findings to {args.output}")

    if args.auto_retract and not args.dry_run and invalid:
        print(f"\nRetracting {len(invalid)} invalid belief(s)...")
        for r in invalid:
            reason = f"review-beliefs: {r.get('comment', 'invalid')}"
            try:
                node_data = api.export_network(db_path=args.db)["nodes"].get(r["id"], {})
                if node_data.get("truth_value") == "OUT":
                    api.set_metadata(r["id"], "retract_reason", reason, db_path=args.db)
                    print(f"  UPDATED reason for {r['id']}")
                else:
                    api.retract_node(r["id"], reason=reason, db_path=args.db)
                    print(f"  RETRACTED {r['id']}")
            except Exception as e:
                print(f"  ERROR retracting {r['id']}: {e}", file=sys.stderr)

    if args.auto_defeat and not args.dry_run and invalid:
        print(f"\nDefeating {len(invalid)} invalid belief(s) with scope beliefs...")
        for r in invalid:
            scope_findings = r.get("scope_findings", [])
            missing_property = r.get("missing_property", r.get("comment", "invalid"))
            reason_type = r.get("defeat_reason_type", "")
            try:
                if scope_findings:
                    result = api.defeat_with_scope(
                        r["id"], 0, scope_findings, missing_property,
                        defeater_type="invalid-inference",
                        defeat_reason_type=reason_type, db_path=args.db)
                    rt = f" [{reason_type}]" if reason_type else ""
                    print(f"  DEFEATED {r['id']} with {len(result['scope_belief_ids'])} scope belief(s){rt}")
                else:
                    api.defeat_justification(
                        r["id"], 0, r.get("comment", "invalid"),
                        defeater_type="invalid-inference",
                        defeat_reason_type=reason_type, db_path=args.db)
                    rt = f" [{reason_type}]" if reason_type else ""
                    print(f"  DEFEATED {r['id']} (bare defeater, no scope findings){rt}")
            except Exception as e:
                print(f"  ERROR defeating {r['id']}: {e}", file=sys.stderr)

    cost = format_cost_summary()
    if cost:
        print(f"  {cost}", file=sys.stderr)


def cmd_review_justifications(args):
    _require_sqlite(args, "review-justifications")
    from .llm import reset_cost_tracker, format_cost_summary
    reset_cost_tracker()

    model = getattr(args, "model", None) or "claude"

    result = api.review_justifications(
        belief_ids=[args.node] if args.node else None,
        model=model,
        timeout=args.timeout,
        min_antecedents=args.min_antecedents,
        parallel=args.parallel,
        db_path=args.db,
    )

    reviews = result["results"]
    if not reviews:
        print("No multi-antecedent SL justifications to review.")
        return

    for r in reviews:
        cls = r.get("classification", "ALL")
        if cls in ("ANY", "MIXED"):
            print(f"  [{cls}] {r['id']}")
            if r.get("comment"):
                print(f"    {r['comment']}")
            if cls == "ANY":
                indep = r.get("independent_antecedents", [])
                if indep:
                    sl = ",".join(indep)
                    print(f"    Fix: reasons add-justification {r['id']} --sl {sl} --any")
            elif cls == "MIXED":
                req = r.get("required_antecedents", [])
                indep = r.get("independent_antecedents", [])
                if req:
                    print(f"    Required together: {', '.join(req)}")
                if indep:
                    print(f"    Independent: {', '.join(indep)}")

    print(f"\nReviewed {result['reviewed']} justifications")
    print(f"  Keep ALL: {result['keep_all']}  Convert ANY: {result['convert_any']}"
          f"  Mixed: {result['convert_mixed']}")

    if args.output:
        with open(args.output, "w") as f:
            f.write("# Justification Review\n\n")
            for r in reviews:
                cls = r.get("classification", "ALL")
                f.write(f"### {r['id']}\n")
                f.write(f"- Classification: {cls}\n")
                if r.get("required_antecedents"):
                    f.write(f"- Required: {', '.join(r['required_antecedents'])}\n")
                if r.get("independent_antecedents"):
                    f.write(f"- Independent: {', '.join(r['independent_antecedents'])}\n")
                if r.get("comment"):
                    f.write(f"- Comment: {r['comment']}\n")
                f.write("\n")
        print(f"\nWrote findings to {args.output}")

    cost = format_cost_summary()
    if cost:
        print(f"  {cost}", file=sys.stderr)


def cmd_review_premises(args):
    _require_sqlite(args, "review-premises")
    import json
    from datetime import datetime
    from .llm import reset_cost_tracker, format_cost_summary
    reset_cost_tracker()

    model = getattr(args, "model", None) or "claude"
    ts = datetime.now().isoformat(timespec="seconds")
    write_report = not args.no_report

    report_path = None
    if write_report:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"review-premises-{ts.replace(':', '')}.json"

    def _build_report(results, status, total_premises=None, skipped=0):
        inaccurate = sum(1 for r in results if not r.get("accurate", True))
        overgeneralized = sum(1 for r in results
                             if r.get("accurate", True) and not r.get("well_scoped", True))
        return {
            "timestamp": ts,
            "status": status,
            "model": model,
            "timeout": args.timeout,
            "dry_run": args.dry_run,
            "filters": {
                "belief_ids": args.ids or None,
                "sample": args.sample,
                "visible_to": _parse_visible_to(args),
            },
            "reviewed": len(results),
            "total_premises": total_premises,
            "skipped_no_source": skipped,
            "summary": {
                "inaccurate": inaccurate,
                "overgeneralized": overgeneralized,
            },
            "results": results,
        }

    def _write_report(results, status):
        if report_path is not None:
            report_path.write_text(json.dumps(_build_report(results, status), indent=2))

    on_batch = (lambda results: _write_report(results, "partial")) if write_report else None

    result = api.review_premises(
        belief_ids=args.ids or None,
        model=model,
        timeout=args.timeout,
        sample=args.sample,
        visible_to=_parse_visible_to(args),
        dry_run=args.dry_run,
        on_batch=on_batch,
        parallel=getattr(args, "parallel", 0),
        db_path=args.db,
    )

    reviews = result["results"]
    if not reviews:
        print("No premises to review (no premises with resolvable sources found).")
        cost = format_cost_summary()
        if cost:
            print(f"  {cost}", file=sys.stderr)
        return

    inaccurate = [r for r in reviews if not r.get("accurate", True)]
    overgeneralized = [r for r in reviews
                       if r.get("accurate", True) and not r.get("well_scoped", True)]

    for r in reviews:
        flags = []
        if not r.get("accurate", True):
            err = r.get("error_type", "inaccurate")
            flags.append(f"INACCURATE({err})")
        if not r.get("well_scoped", True):
            flags.append("OVERGENERALIZED")

        if flags:
            print(f"  [{' | '.join(flags)}] {r['id']}")
            if r.get("comment"):
                print(f"    {r['comment']}")

    print(f"\nReviewed {result['reviewed']} of {result['total_premises']} premises"
          f" ({result['skipped_no_source']} skipped, no source)")
    print(f"  Inaccurate: {result['inaccurate']}  Overgeneralized: {result['overgeneralized']}")

    if report_path is not None:
        report = _build_report(reviews, "complete",
                               total_premises=result["total_premises"],
                               skipped=result["skipped_no_source"])
        report_path.write_text(json.dumps(report, indent=2))
        print(f"  Report: {report_path}")

    if args.auto_retract and not args.dry_run and inaccurate:
        print(f"\nRetracting {len(inaccurate)} inaccurate premise(s)...")
        for r in inaccurate:
            try:
                api.retract_node(r["id"],
                                 reason=f"review-premises: {r.get('error_type', 'inaccurate')}: {r.get('comment', '')}",
                                 db_path=args.db)
                print(f"  RETRACTED {r['id']}")
            except Exception as e:
                print(f"  ERROR retracting {r['id']}: {e}", file=sys.stderr)

    cost = format_cost_summary()
    if cost:
        print(f"  {cost}", file=sys.stderr)


def cmd_repair_premises(args):
    _require_sqlite(args, "repair-premises")
    import json
    from datetime import datetime
    from .llm import reset_cost_tracker, format_cost_summary
    reset_cost_tracker()

    model = getattr(args, "model", None) or "claude"
    ts = datetime.now().isoformat(timespec="seconds")
    write_report = not args.no_report

    report_path = None
    if write_report:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"repair-premises-{ts.replace(':', '')}.json"

    review_file = getattr(args, "review_file", None)
    belief_ids = args.ids or None

    if not review_file and not belief_ids:
        print("Error: provide premise IDs or --review-file", file=sys.stderr)
        return

    def _write_report(results, status):
        if report_path is not None:
            report = {
                "timestamp": ts,
                "status": status,
                "model": model,
                "dry_run": args.dry_run,
                "results": results,
            }
            report_path.write_text(json.dumps(report, indent=2))

    on_result = (lambda results: _write_report(results, "partial")) if write_report else None

    result = api.repair_premises(
        review_file=review_file,
        belief_ids=belief_ids,
        model=model,
        timeout=args.timeout,
        dry_run=args.dry_run,
        parallel=getattr(args, "parallel", 0),
        on_result=on_result,
        db_path=args.db,
    )

    repairs = result["results"]
    if not repairs:
        print("No inaccurate premises to repair.")
        cost = format_cost_summary()
        if cost:
            print(f"  {cost}", file=sys.stderr)
        return

    for r in repairs:
        action = r.get("action", "error")
        if action == "rewrite":
            label = "REWRITE"
        elif action == "retract":
            label = "RETRACT"
        else:
            label = "ERROR"
        print(f"  [{label}] {r['id']}")
        if r.get("rationale"):
            print(f"    {r['rationale']}")
        if action == "rewrite" and r.get("corrected_text"):
            text = r["corrected_text"][:100]
            if len(r["corrected_text"]) > 100:
                text += "..."
            print(f"    -> {text}")

    print(f"\nRepaired {len(repairs)} of {result['total_inaccurate']} inaccurate premises")
    print(f"  Rewritten: {result['rewritten']}  Retracted: {result['retracted']}"
          f"  Failed: {result['failed']}")

    if report_path is not None:
        _write_report(repairs, "complete")
        print(f"  Report: {report_path}")

    cost = format_cost_summary()
    if cost:
        print(f"  {cost}", file=sys.stderr)


def cmd_propose_update(args):
    _require_sqlite(args, "propose-update")
    import json as _json
    from datetime import datetime
    from .llm import reset_cost_tracker, format_cost_summary
    reset_cost_tracker()

    from .propose_update import format_proposals_file

    model = getattr(args, "model", None) or "claude"
    ts = datetime.now().isoformat(timespec="seconds")
    write_report = not args.no_report
    output_format = getattr(args, "format", "markdown") or "markdown"

    report_path = None
    if write_report:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"propose-update-{ts.replace(':', '')}.json"

    def on_batch(results):
        if report_path is not None:
            report_path.write_text(_json.dumps({
                "timestamp": ts,
                "status": "partial",
                "model": model,
                "proposals": results,
            }, indent=2))

    result = api.propose_update(
        belief_ids=args.ids or None,
        model=model,
        timeout=args.timeout,
        stale_only=args.stale_only,
        namespace=args.namespace,
        sample=args.sample,
        on_batch=on_batch,
        db_path=args.db,
    )

    proposals = result["proposals"]
    cascades = result.get("cascades", {})

    if not proposals:
        print("No updates proposed.")
        return

    for p in proposals:
        action = p["action"].upper()
        basis = p.get("basis", "")
        fm = p.get("failure_mode", "")
        print(f"  [{action}] {p['id']}  ({fm}, {basis})")
        if p.get("comment"):
            print(f"    {p['comment']}")

    print(f"\nReviewed {result['reviewed']} beliefs, {len(proposals)} update(s) proposed")

    if output_format == "markdown":
        net_result = api.export_network(db_path=args.db)
        nodes = net_result.get("nodes", {})
        output_file = args.output or "proposed-updates.md"
        content = format_proposals_file(proposals, nodes=nodes, cascades=cascades)
        Path(output_file).write_text(content)
        print(f"\nWrote proposals to {output_file}")
    else:
        output_file = args.output or "proposed-updates.json"
        Path(output_file).write_text(_json.dumps({
            "timestamp": ts,
            "model": model,
            "reviewed": result["reviewed"],
            "proposals": proposals,
            "cascades": {k: v for k, v in cascades.items()},
        }, indent=2))
        print(f"\nWrote proposals to {output_file}")

    if write_report and report_path is not None:
        report_path.write_text(_json.dumps({
            "timestamp": ts,
            "status": "complete",
            "model": model,
            "reviewed": result["reviewed"],
            "proposals": proposals,
        }, indent=2))
        print(f"  Report: {report_path}")

    cost = format_cost_summary()
    if cost:
        print(f"  {cost}", file=sys.stderr)


def cmd_repair_smuggled(args):
    _require_sqlite(args, "repair-smuggled")

    model = getattr(args, "model", None) or "claude"
    review_file = getattr(args, "review_file", None)
    belief_ids = args.ids if args.ids else None

    if not review_file and not belief_ids:
        print("Error: provide either --review-file or belief IDs", file=sys.stderr)
        sys.exit(1)

    result = api.repair_smuggled(
        review_file=review_file,
        belief_ids=belief_ids,
        model=model,
        timeout=args.timeout,
        dry_run=args.dry_run,
        db_path=args.db,
    )

    repairs = result["repairs"]
    for r in repairs:
        status = r["status"].upper()
        print(f"  [{status}] {r['id']}")
        if r.get("smuggled_claim"):
            print(f"    Smuggled: {r['smuggled_claim']}")
        if r.get("matched_premises"):
            print(f"    Linked: {', '.join(r['matched_premises'])}")
        if r.get("rationale"):
            print(f"    Rationale: {r['rationale']}")
        if r.get("error"):
            print(f"    Error: {r['error']}")

    print(f"\nTotal invalid: {result['total_invalid']}")
    print(f"  Repaired: {result['repaired']}  No candidates: {result['no_candidates']}"
          f"  No match: {result['no_match']}  Extract failed: {result['extraction_failed']}"
          f"  Errors: {result['errors']}")

    if args.dry_run:
        print("\n  (dry run -- no changes applied)")


def cmd_repair(args):
    _require_sqlite(args, "repair")
    from .llm import reset_cost_tracker, format_cost_summary
    reset_cost_tracker()

    model = getattr(args, "model", None) or "claude"
    review_file = getattr(args, "review_file", None)
    belief_ids = args.ids if args.ids else None

    if not review_file and not belief_ids:
        print("Error: provide either --review-file or belief IDs", file=sys.stderr)
        sys.exit(1)

    result = api.repair(
        review_file=review_file,
        belief_ids=belief_ids,
        model=model,
        timeout=args.timeout,
        dry_run=args.dry_run,
        db_path=args.db,
    )

    for r in result["results"]:
        status = r["status"].upper()
        pattern = r.get("pattern") or "?"
        print(f"  [{status}] {r['id']} ({pattern})")
        if r.get("rationale"):
            print(f"    {r['rationale']}")
        if r.get("smuggled_claim"):
            print(f"    Smuggled: {r['smuggled_claim']}")
        if r.get("matched_premises"):
            print(f"    Linked: {', '.join(r['matched_premises'])}")
        if r.get("softened_text"):
            print(f"    Softened: {r['softened_text']}")
        if r.get("error"):
            print(f"    Error: {r['error']}")

    print(f"\nTotal invalid: {result['total_invalid']}")
    print(f"  Linked: {result['linked']}  Softened: {result['softened']}"
          f"  Abandoned: {result['abandoned']}  Research: {result['needs_research']}"
          f"  Failed: {result['failed']}  Errors: {result['errors']}")

    if args.dry_run:
        print("\n  (dry run -- no changes applied)")

    cost = format_cost_summary()
    if cost:
        print(f"  {cost}", file=sys.stderr)


def cmd_contradictions(args):
    _require_sqlite(args, "detect-contradictions")

    if args.accept:
        accept_path = Path(args.accept)
        if not accept_path.exists():
            print(f"File not found: {accept_path}", file=sys.stderr)
            sys.exit(1)
        plan = api.parse_contradiction_plan(accept_path.read_text())
        if not plan:
            print("No APPLY entries found in plan file.")
            return
        result = api.apply_contradiction_plan(plan, db_path=args.db)
        for err in result["errors"]:
            print(f"  ERROR: {err}", file=sys.stderr)
        if result["nogoods"]:
            print(f"Applied {result['applied']} nogood(s):")
            for n in result["nogoods"]:
                changed = n.get("changed", [])
                print(f"  {n['id']}: nogood={n.get('nogood_id', '?')}, "
                      f"changed {len(changed)} node(s)")
        else:
            print("No nogoods to apply.")
        return

    from .llm import reset_cost_tracker, format_cost_summary
    reset_cost_tracker()

    model = getattr(args, "model", None) or "claude"
    output = args.output
    result = api.detect_contradictions(
        belief_ids=args.ids or None,
        model=model,
        timeout=args.timeout,
        sample=args.sample,
        auto_apply=args.auto_apply,
        semantic=args.semantic,
        embedding_model=args.embedding_model,
        output_path=output if not args.auto_apply else None,
        db_path=args.db,
    )

    contradictions = result["contradictions"]
    if not contradictions:
        print(f"No contradictions detected among {result['checked']} IN beliefs.")
        return

    for c in contradictions:
        severity = c.get("severity", "")
        sev_str = f" ({severity})" if severity else ""
        print(f"  [NOGOOD] {c['id']}{sev_str}")
        print(f"    Claims: {', '.join(c['claims'])}")
        if c.get("analysis"):
            print(f"    Analysis: {c['analysis']}")

    print(f"\nChecked {result['checked']} of {result['total_in']} IN beliefs")
    print(f"  Found: {result['found']}  Applied: {result['applied']}")

    if args.auto_apply and result.get("applied_details"):
        print(f"\nApplied {result['applied']} nogood(s):")
        for d in result["applied_details"]:
            changed = d.get("changed", [])
            print(f"  {d.get('id', '?')}: nogood={d.get('nogood_id', '?')}, "
                  f"changed {len(changed)} node(s)")
    elif not args.auto_apply:
        print(f"\nWrote {output} — review, then run:")
        print(f"  reasons contradictions --accept {output}")

    cost = format_cost_summary()
    if cost:
        print(f"  {cost}", file=sys.stderr)


def cmd_namespaces(args):
    result = api.list_namespaces(**_backend_kwargs(args))
    if not result["namespaces"]:
        print("No namespaces found. Use --namespace/-n with 'add' or 'import-agent' to create one.")
        return
    for ns in result["namespaces"]:
        status = "ACTIVE" if ns["active"] else "INACTIVE"
        print(f"  {ns['namespace']:30s} {status:8s} {ns['in_beliefs']:3d} IN / {ns['total_beliefs']} total")
    print(f"\n{len(result['namespaces'])} namespace(s)")


def main():
    parser = argparse.ArgumentParser(
        prog="reasonsforge",
        description="Reasons — automatic belief retraction and dependency-directed backtracking",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_pkg_version('reasonsforge')}")
    parser.add_argument("--db", default=api.DEFAULT_DB, help="Path to database (default: reasons.db)")
    parser.add_argument("--pg", default=None, metavar="CONNINFO",
                        help="PostgreSQL connection string (or set REASONSFORGE_PG_CONNINFO)")
    parser.add_argument("--project-id", default=None,
                        help="Project ID for PostgreSQL (or set REASONSFORGE_PROJECT_ID)")
    sub = parser.add_subparsers(dest="command")

    # Register forge subcommands
    from .forge.cli import register_forge_commands, register_forge_type_commands
    _forge_commands = register_forge_commands(sub)
    _forge_type_commands = register_forge_type_commands(sub)

    # init
    p = sub.add_parser("init", help="Initialize a new RMS database")
    p.add_argument("--force", action="store_true", help="Overwrite existing database")
    p.add_argument("--project-name", default="", help="Name for this belief network (defaults to DB filename stem)")

    # add
    p = sub.add_parser("add", help="Add a node")
    p.add_argument("node_id", help="Node identifier")
    p.add_argument("text", help="Node text")
    p.add_argument("--sl", metavar="A,B", help="SL justification: comma-separated antecedent IDs")
    p.add_argument("--cp", metavar="A,B", help="CP justification: comma-separated antecedent IDs")
    p.add_argument("--unless", metavar="X,Y", help="Outlist: comma-separated node IDs that must be OUT")
    p.add_argument("--any", action="store_true", help="Expand SL into one justification per premise (OR instead of AND)")
    p.add_argument("--label", help="Justification label")
    p.add_argument("--source", help="Provenance (repo:path)")
    p.add_argument("--source-url", help="URL for the source document")
    p.add_argument("-n", "--namespace", help="Namespace prefix (auto-creates ns:active premise)")
    p.add_argument("--access-tags", metavar="TAG,TAG", help="Data source provenance tags (comma-separated)")
    p.add_argument("--example", default=None, help="Code example demonstrating the belief")
    p.add_argument("--source-type", choices=["code", "document", "self-description", "derived"],
                   help="Epistemic source type (code, document, self-description, derived)")
    p.add_argument("--accepted-pr", default=None,
                   help="URL of the PR that accepted this belief")

    # add-justification
    p = sub.add_parser("add-justification", help="Add a justification to an existing node")
    p.add_argument("node_id", help="Node to add justification to")
    p.add_argument("--sl", metavar="A,B", help="SL justification: comma-separated antecedent IDs")
    p.add_argument("--cp", metavar="A,B", help="CP justification: comma-separated antecedent IDs")
    p.add_argument("--unless", metavar="X,Y", help="Outlist: comma-separated node IDs that must be OUT")
    p.add_argument("--any", action="store_true", help="Expand SL into one justification per premise (OR instead of AND)")
    p.add_argument("--label", help="Justification label")
    p.add_argument("-n", "--namespace", help="Namespace prefix")

    # remove-justification
    p = sub.add_parser("remove-justification", help="Remove a justification by index")
    p.add_argument("node_id", help="Node to remove justification from")
    p.add_argument("index", type=int, help="0-based justification index (see 'show' output)")

    # retract
    p = sub.add_parser("retract", help="Retract a node (mark OUT + cascade)")
    p.add_argument("node_id", help="Node to retract")
    p.add_argument("--reason", help="Why this node is being retracted")

    # assert
    p = sub.add_parser("assert", help="Assert a node (mark IN + cascade)")
    p.add_argument("node_id", help="Node to assert")

    # mark-duplicate
    p = sub.add_parser("mark-duplicate", help="Mark a node as duplicate of a canonical version")
    p.add_argument("source_id", help="Duplicate node to retract")
    p.add_argument("--of", dest="canonical_id", required=True, help="Canonical node ID")

    # mark-superseded
    p = sub.add_parser("mark-superseded", help="Retract a node as superseded with metadata (hard retract; see 'supersede' for outlist-based)")
    p.add_argument("old_id", help="Obsolete node to retract")
    p.add_argument("--by", dest="new_id", required=True, help="Replacement node ID")

    # defeat-justification
    p = sub.add_parser("defeat-justification", help="Defeat a justification by adding a defeater to its outlist")
    p.add_argument("node_id", help="Node whose justification to defeat")
    p.add_argument("justification_index", type=int, help="Justification index (0-based)")
    p.add_argument("reason", help="Why this justification is invalid")
    p.add_argument("--type", choices=["invalid-inference", "over-generalizes", "duplicate-of", "superseded-by"],
                   help="Type of defeater (default: invalid-inference)")
    p.add_argument("--reason-type", dest="reason_type",
                   choices=["unsupported-conjunct", "over-generalizes", "false-causal-claim",
                            "internal-contradiction", "circular-reasoning", "missing-bridge",
                            "scope-mismatch"],
                   help="Logical failure mode classification")
    p.add_argument("--defeater-id", help="Custom defeater belief ID")

    # defeat-with-scope
    p = sub.add_parser("defeat-with-scope", help="Defeat a justification with scope beliefs as antecedents")
    p.add_argument("node_id", help="Node whose justification to defeat")
    p.add_argument("justification_index", type=int, help="Justification index (0-based)")
    p.add_argument("--scope-file", required=True,
                   help="JSON file with scope_findings and missing_property")
    p.add_argument("--type", choices=["invalid-inference", "over-generalizes", "duplicate-of", "superseded-by"],
                   help="Type of defeater (default: invalid-inference)")
    p.add_argument("--reason-type", dest="reason_type",
                   choices=["unsupported-conjunct", "over-generalizes", "false-causal-claim",
                            "internal-contradiction", "circular-reasoning", "missing-bridge",
                            "scope-mismatch"],
                   help="Logical failure mode classification")
    p.add_argument("--defeater-id", help="Custom defeater belief ID")

    # migrate-defeaters
    p = sub.add_parser("migrate-defeaters", help="Convert string-based retract_reason to graph-native defeaters")
    p.add_argument("node_ids", nargs="*", help="Specific nodes to migrate (default: all candidates)")
    p.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")

    # classify-defeaters
    p = sub.add_parser("classify-defeaters", help="Classify unclassified defeaters by logical failure mode via LLM")
    p.add_argument("--model", "-m", required=True, help="LLM model for classification")
    p.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    p.add_argument("--type", help="Only classify defeaters with this defeater_type")
    p.add_argument("--timeout", type=int, default=300, help="LLM timeout in seconds (default: 300)")

    # what-if
    p = sub.add_parser("what-if", help="Simulate retracting or asserting a node (read-only)")
    p.add_argument("action", choices=["retract", "assert"], help="Action to simulate")
    p.add_argument("node_id", help="Node to simulate")

    # status
    p = sub.add_parser("status", help="Show all nodes with truth values")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only show nodes whose access_tags are a subset of these tags")

    # show
    p = sub.add_parser("show", help="Show node details")
    p.add_argument("node_id", help="Node to show")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only show if access_tags are a subset of these tags")

    # explain
    p = sub.add_parser("explain", help="Explain why a node is IN or OUT")
    p.add_argument("node_id", help="Node to explain")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only show if access_tags are a subset of these tags")

    # convert-to-premise
    p = sub.add_parser("convert-to-premise", help="Strip justifications, make a node a premise")
    p.add_argument("node_id", help="Node to convert")

    # summarize
    p = sub.add_parser("summarize", help="Create a summary node over a group of nodes")
    p.add_argument("summary_id", help="Summary node ID")
    p.add_argument("text", help="High-level summary text")
    p.add_argument("--over", required=True, metavar="A,B,C", help="Comma-separated node IDs to summarize")
    p.add_argument("--source", help="Provenance (repo:path)")

    # supersede
    p = sub.add_parser("supersede", help="Reversible supersession via outlist (old comes back if new is retracted)")
    p.add_argument("old_id", help="Belief being superseded")
    p.add_argument("new_id", nargs="?", default=None, help="Belief that supersedes it (omit when using --text)")
    p.add_argument("--text", default=None, help="Create a successor node with this text and supersede")
    p.add_argument("--id", default=None, help="Custom ID for the successor node (used with --text)")

    # update
    p = sub.add_parser("update", help="Update a belief's metadata (source, example)")
    p.add_argument("node_id", help="Belief to update")
    p.add_argument("--source", default=None, help="Update source path")
    p.add_argument("--source-url", default=None, help="Update source URL")
    p.add_argument("--example", default=None, help="Code example demonstrating the belief")

    # set-metadata
    p = sub.add_parser("set-metadata", help="Set a metadata key on a belief")
    p.add_argument("node_id", help="Belief to update")
    p.add_argument("key", help="Metadata key")
    p.add_argument("value", help="Metadata value")

    # get-metadata
    p = sub.add_parser("get-metadata", help="Show metadata for a belief")
    p.add_argument("node_id", help="Belief to inspect")
    p.add_argument("key", nargs="?", default=None, help="Specific key to show (default: all)")

    # challenge
    p = sub.add_parser("challenge", help="Challenge a node — target goes OUT")
    p.add_argument("target_id", help="Node to challenge")
    p.add_argument("reason", help="Why the node is being challenged")
    p.add_argument("--id", help="Custom challenge node ID (default: challenge-TARGET)")

    # defend
    p = sub.add_parser("defend", help="Defend a node against a challenge")
    p.add_argument("target_id", help="Node being defended")
    p.add_argument("challenge_id", help="Challenge to defend against")
    p.add_argument("reason", help="Defense argument")
    p.add_argument("--id", help="Custom defense node ID")

    # nogood
    p = sub.add_parser("nogood", help="Record a contradiction")
    p.add_argument("node_ids", nargs="+", help="Node IDs that cannot all be IN")

    # trace
    p = sub.add_parser("trace-access-tags", help="Trace all access tags in a node's dependency chain")
    p.add_argument("node_id", help="Node to trace")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only allow if access_tags are a subset of these tags")

    p = sub.add_parser("trace", help="Trace backward to find premises a node rests on")
    p.add_argument("node_id", help="Node to trace")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only show premises whose access_tags are a subset of these tags")

    # propagate
    sub.add_parser("propagate", help="Recompute all truth values")

    # log
    p = sub.add_parser("log", help="Show propagation history")
    p.add_argument("--last", type=int, help="Show only last N entries")

    # add-repo
    p = sub.add_parser("add-repo", help="Register a repo name and path")
    p.add_argument("name", help="Repo name (used in source paths)")
    p.add_argument("path", help="Filesystem path to the repo")

    # repos
    sub.add_parser("repos", help="List registered repos")

    # derive
    p = sub.add_parser("derive", help="Derive deeper reasoning chains from existing beliefs")
    p.add_argument("-o", "--output", default="proposed-derivations.md",
                   help="Output file for proposals (default: proposed-derivations.md)")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude). Prefixes: ollama:<model>, api:<model>, vertex:<model>")
    p.add_argument("--auto", action="store_true",
                   help="Automatically add proposals (no review step)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show prompt without invoking the model")
    p.add_argument("--domain", default=None,
                   help="Domain description for context (auto-detected from agents)")
    p.add_argument("--topic", default=None,
                   help="Keyword filter — only include beliefs matching these keywords")
    p.add_argument("--budget", type=int, default=300,
                   help="Maximum number of beliefs in prompt (default: 300)")
    p.add_argument("--sample", action="store_true",
                   help="Randomly sample beliefs instead of alphabetical truncation")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible sampling")
    p.add_argument("--timeout", type=int, default=600,
                   help="Model timeout in seconds (default: 600)")
    p.add_argument("--premises", action="store_true",
                   help="Only include premises (no justifications)")
    p.add_argument("--has-dependents", action="store_true",
                   help="Only include nodes that others depend on")
    p.add_argument("--min-depth", type=int, default=None,
                   help="Only include beliefs at this depth or deeper (0=premises)")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Only include beliefs at this depth or shallower")
    p.add_argument("--exhaust", action="store_true",
                   help="Repeat derive until no new proposals (implies --auto)")
    p.add_argument("--max-rounds", type=int, default=10,
                   help="Maximum rounds for --exhaust (default: 10)")
    p.add_argument("--report-dir", default="reviews/",
                   help="Directory for JSON reports (default: reviews/)")
    p.add_argument("--no-report", action="store_true",
                   help="Suppress JSON report generation")
    p.add_argument("--cluster", action="store_true",
                   help="Use semantic clustering to sample across domains")
    p.add_argument("--intra-cluster", action="store_true",
                   help="Focus on one cluster per round (rotate in --exhaust mode)")
    p.add_argument("--embedding-model", default=None,
                   help="Sentence-transformers model for --cluster/--intra-cluster "
                        "(default: all-MiniLM-L6-v2)")
    p.add_argument("--n-clusters", type=int, default=None,
                   help="Override automatic cluster count for --cluster/--intra-cluster")
    p.add_argument("--prompt-file", default=None,
                   help="Custom prompt template file (overrides built-in DERIVE_PROMPT)")

    # accept
    p = sub.add_parser("accept", help="Accept proposals from a derive proposals file")
    p.add_argument("file", nargs="?", default="proposed-derivations.md",
                   help="Proposals file (default: proposed-derivations.md)")

    # import-agent
    p = sub.add_parser("import-agent", help="Import another agent's beliefs with namespacing")
    p.add_argument("agent_name", help="Agent name (used as namespace prefix)")
    p.add_argument("beliefs_file", help="Path to the agent's beliefs.md or network.json")
    p.add_argument("--nogoods", dest="nogoods_file", help="Path to nogoods.md (auto-detected if next to beliefs.md)")
    p.add_argument("--only-in", action="store_true", help="Only import beliefs with status IN")

    # sync-agent
    p = sub.add_parser("sync-agent", help="Sync another agent's beliefs (remote wins)")
    p.add_argument("agent_name", help="Agent name (must match previous import)")
    p.add_argument("beliefs_file", help="Path to the agent's beliefs.md or network.json")
    p.add_argument("--nogoods", dest="nogoods_file", help="Path to nogoods.md (auto-detected if next to beliefs.md)")
    p.add_argument("--only-in", action="store_true", help="Only sync beliefs with status IN")

    # import-beliefs
    p = sub.add_parser("import-beliefs", help="Import a beliefs.md registry")
    p.add_argument("beliefs_file", help="Path to beliefs.md")
    p.add_argument("--nogoods", dest="nogoods_file", help="Path to nogoods.md (auto-detected if next to beliefs.md)")

    # import-json
    p = sub.add_parser("import-json", help="Import network from JSON (produced by export)")
    p.add_argument("json_file", help="Path to JSON file")

    # import-hf
    p = sub.add_parser("import-hf", help="Import network from HuggingFace repo")
    p.add_argument("repo_id", help="HuggingFace repo (bare name, user/repo, or URL)")
    p.add_argument("--init", action="store_true",
                   help="Initialize reasons.db if it doesn't exist")
    p.add_argument("--token", help="HuggingFace token (default: from HF_TOKEN or ~/.cache/huggingface/token)")

    # pull (alias for import-hf with default org)
    p = sub.add_parser("pull", help="Pull EEM from HuggingFace (default org: EEM-Hub)")
    p.add_argument("repo_id", help="EEM name (bare name defaults to EEM-Hub/name)")
    p.add_argument("--init", action="store_true",
                   help="Initialize reasons.db if it doesn't exist")
    p.add_argument("--token", help="HuggingFace token (default: from HF_TOKEN or ~/.cache/huggingface/token)")

    # publish
    p = sub.add_parser("publish", help="Publish EEM to HuggingFace (default org: EEM-Hub)")
    p.add_argument("repo_id", help="EEM name (bare name defaults to EEM-Hub/name)")
    p.add_argument("--token", help="HuggingFace token (default: from HF_TOKEN or ~/.cache/huggingface/token)")
    p.add_argument("--private", action="store_true", help="Create a private repo")
    p.add_argument("--domain", nargs="*", help="Domain tags (e.g. kubernetes devops)")
    p.add_argument("--license", default="mit", help="License identifier (default: mit)")
    p.add_argument("--base-network", help="Parent EEM this was derived from")
    p.add_argument("--source-repos", nargs="*", help="Source repository identifiers")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only publish nodes whose access_tags are a subset of these tags")

    # import-api
    p = sub.add_parser("import-api", help="Import beliefs from agentic-mind-service")
    p.add_argument("--url", help="Service URL (default: MIND_SERVICE_URL env)")
    p.add_argument("--agent-id", help="Agent UUID (default: MIND_AGENT_ID env)")
    p.add_argument("--api-key", help="API key (default: MIND_API_KEY env)")
    p.add_argument("--init", action="store_true",
                   help="Initialize reasons.db if it doesn't exist")

    # export-api
    p = sub.add_parser("export-api", help="Export beliefs to agentic-mind-service")
    p.add_argument("--url", help="Service URL (default: MIND_SERVICE_URL env)")
    p.add_argument("--agent-id", help="Agent UUID (default: MIND_AGENT_ID env)")
    p.add_argument("--api-key", help="API key (default: MIND_API_KEY env)")

    # export
    p = sub.add_parser("export", help="Export network as JSON")
    p.add_argument("-o", "--output", default="network.json", nargs="?", const="network.json",
                   help="Output file (default: network.json). Use --output=- for stdout")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only export nodes whose access_tags are a subset of these tags")

    # export-markdown
    p = sub.add_parser("export-markdown", help="Export network as beliefs.md-compatible markdown")
    p.add_argument("-o", "--output", default="beliefs.md", nargs="?", const="beliefs.md",
                   help="Output file (default: beliefs.md). Use --output=- for stdout")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only export nodes whose access_tags are a subset of these tags")

    # export-card
    p = sub.add_parser("export-card", help="Export network as HuggingFace EEM card (README.md)")
    p.add_argument("-o", "--output", default="README.md", nargs="?", const="README.md",
                   help="Output file (default: README.md). Use --output=- for stdout")
    p.add_argument("--domain", nargs="*", help="Domain tags (e.g. kubernetes devops)")
    p.add_argument("--license", default="mit", help="License identifier (default: mit)")
    p.add_argument("--base-network", help="Parent EEM this was derived from")
    p.add_argument("--source-repos", nargs="*", help="Source repository identifiers")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only export nodes whose access_tags are a subset of these tags")

    # hash-sources
    p = sub.add_parser("hash-sources", help="Backfill source hashes for nodes without them")
    p.add_argument("--force", action="store_true", help="Re-hash all nodes, even those with existing hashes")

    # check-stale
    p = sub.add_parser("check-stale", help="Check IN nodes for source file staleness")
    p.add_argument("--upgrade-hashes", action="store_true",
                   help="Upgrade truncated hashes to full length in place")
    p.add_argument("--git", action="store_true",
                   help="Use git commit SHA for faster staleness detection")

    # check-integrity
    p = sub.add_parser("check-integrity", help="Verify Merkle hashes for text mutation detection")

    # backfill-hashes
    p = sub.add_parser("backfill-hashes", help="Compute Merkle hashes for nodes/justifications missing them")

    # pin-sources
    p = sub.add_parser("pin-sources", help="Pin source links to git commit SHA")
    p.add_argument("--force", action="store_true",
                   help="Re-pin even nodes that already have a pinned_sha")
    p.add_argument("--pin-urls", action="store_true",
                   help="Rewrite source_url from branch URLs to SHA-pinned URLs")

    # pin-update
    p = sub.add_parser("pin-update", help="Bump pinned_sha to current HEAD for beliefs")
    p.add_argument("node_ids", nargs="+", help="Belief IDs to update")

    # pin-lines
    p = sub.add_parser("pin-lines", help="Pin a belief to specific source file lines")
    p.add_argument("node_id", help="Belief ID to pin")
    p.add_argument("start", type=int, help="Start line number (1-based)")
    p.add_argument("end", type=int, help="End line number (inclusive)")

    # compact
    p = sub.add_parser("compact", help="Token-budgeted belief state summary")
    p.add_argument("--budget", type=int, default=500, help="Token budget (default: 500)")
    p.add_argument("--no-truncate", action="store_true", help="Show full node text")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only include nodes whose access_tags are a subset of these tags")

    # search
    p = sub.add_parser("search", help="Search nodes using full-text search with neighbor expansion")
    p.add_argument("query", help="Search terms (FTS5 all-terms matching)")
    p.add_argument("--format", choices=["markdown", "json", "minimal"], default="markdown",
                   help="Output format (default: markdown)")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only show nodes whose access_tags are a subset of these tags")

    # lookup
    p = sub.add_parser("lookup", help="Simple keyword search over beliefs (no neighbor expansion)")
    p.add_argument("query", help="Search terms (all must match, case-insensitive)")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only show nodes whose access_tags are a subset of these tags")

    # ask
    p = sub.add_parser("ask", help="Ask a question about beliefs (FTS5 search + LLM synthesis)")
    p.add_argument("question", help="Natural language question")
    p.add_argument("--no-synth", action="store_true",
                   help="Show belief matches only, no LLM synthesis")
    p.add_argument("--format", choices=["compact", "markdown", "json", "minimal"],
                   default=None,
                   help="Output format for --no-synth (default: compact)")
    p.add_argument("--timeout", type=int, default=600,
                   help="LLM timeout in seconds (default: 600)")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude). Prefixes: ollama:<model>, api:<model>, vertex:<model>")
    p.add_argument("--simple", action="store_true",
                   help="Single-pass synthesis with pre-retrieved beliefs (better for smaller models)")
    p.add_argument("--full-sources", default=None, metavar="FTS_DB",
                   help="Also search source document chunks from FTS5 index (e.g. rag_fts.db)")
    p.add_argument("--natural", action="store_true",
                   help="Strip belief IDs, status, and justification metadata from context")
    p.add_argument("--dual", action="store_true",
                   help="Run TMS and FTS RAG separately, then merge (requires --full-sources)")
    p.add_argument("--mcp", action="append", default=None,
                   help="MCP server command as data source (repeatable, e.g. --mcp snowflake-mcp)")

    # search-sources
    p = sub.add_parser("search-sources", help="Search source document chunks from an FTS5 index (no LLM)")
    p.add_argument("query", help="Search query")
    p.add_argument("--db", required=True, metavar="FTS_DB",
                   help="Path to FTS5 chunks database (e.g. rag_fts.db)")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of results to return (default: 10)")
    p.add_argument("--format", choices=["text", "json"], default="text",
                   help="Output format (default: text)")

    # deduplicate
    p = sub.add_parser("deduplicate", help="Find and optionally retract duplicate IN beliefs")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Similarity threshold (Jaccard for ID mode, cosine for --semantic; default: 0.5)")
    p.add_argument("--semantic", action="store_true",
                   help="Use embedding similarity instead of ID token similarity")
    p.add_argument("--embedding-model", default=None,
                   help="Sentence-transformers model (default: all-MiniLM-L6-v2)")
    p.add_argument("--auto", action="store_true",
                   help="Automatically retract duplicates (keeps one per cluster)")
    p.add_argument("-o", "--output", default="proposed-dedup.md",
                   help="Output file for dedup plan (default: proposed-dedup.md)")
    p.add_argument("--accept", metavar="FILE",
                   help="Apply a reviewed dedup plan file")

    # cluster-list
    p = sub.add_parser("cluster-list", help="List semantic similarity clusters")
    p.add_argument("--status", choices=["IN", "OUT"], default="IN",
                   help="Filter by truth value (default: IN)")
    p.add_argument("--n-clusters", type=int, default=None,
                   help="Override automatic cluster count")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible clustering")
    p.add_argument("--embedding-model", default=None,
                   help="Sentence-transformers model (default: all-MiniLM-L6-v2)")
    p.add_argument("--visible-to", metavar="TAG,TAG",
                   help="Only show nodes whose access_tags are a subset of these tags")
    p.add_argument("--format", choices=["text", "json", "markdown"], default="text",
                   help="Output format (default: text)")

    # namespaces
    p = sub.add_parser("list-gated", help="List OUT nodes blocked by IN outlist nodes")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only show nodes whose access_tags are a subset of these tags")

    # report-gated
    p = sub.add_parser("report-gated", help="Generate a problems/open-issues report from gated beliefs")
    p.add_argument("-o", "--output", default=None, help="Write report to file (default: stdout)")
    p.add_argument("-m", "--model", default=None,
                   help="LLM model for narrative synthesis (default: structured report)")
    p.add_argument("--timeout", type=int, default=300, help="LLM timeout in seconds (default: 300)")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only include nodes whose access_tags are a subset of these tags")

    # report
    p = sub.add_parser("report", help="Trace a belief to its root premises with source evidence")
    p.add_argument("node_id", help="Belief ID to report on")
    p.add_argument("--sources-db", default=None, metavar="FTS_DB",
                   help="FTS5 chunks database for source evidence (e.g. rag_fts.db)")
    p.add_argument("-o", "--output", default=None, help="Write report to file (default: stdout)")
    p.add_argument("-m", "--model", default=None,
                   help="LLM model for narrative synthesis (default: structured report)")
    p.add_argument("--timeout", type=int, default=300, help="LLM timeout in seconds (default: 300)")

    # verify
    p = sub.add_parser("verify", help="Verify beliefs against their source documents")
    p.add_argument("node_id", help="Belief ID to verify")
    p.add_argument("--trace", action="store_true",
                   help="Trace derived beliefs to leaf premises and verify each")
    p.add_argument("--retract", action="store_true",
                   help="Retract beliefs with STALE verdicts")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be verified without calling LLM")
    p.add_argument("-m", "--model", default="claude",
                   help="LLM model to use (default: claude)")
    p.add_argument("--timeout", type=int, default=120, help="LLM timeout in seconds (default: 120)")

    p = sub.add_parser("list-negative", help="Find IN beliefs describing problems/defects/risks (LLM-classified)")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only show nodes whose access_tags are a subset of these tags")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude). Prefixes: ollama:<model>, api:<model>, vertex:<model>")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip LLM classification; return keyword-filtered candidates directly")

    sub.add_parser("namespaces", help="List all agent namespaces in the database")

    # topics
    p = sub.add_parser("topics", help="Extract topics from node IDs by word frequency")
    p.add_argument("--limit", type=int, default=20, help="Max topics to show (default: 20)")
    p.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    # build-wiki
    p = sub.add_parser("build-wiki", help="Export beliefs as interlinked markdown wiki pages")
    p.add_argument("-o", "--output", default="wiki", help="Output directory (default: wiki)")
    p.add_argument("-m", "--model", default=None,
                   help="LLM model for page generation (e.g. claude, gemini). Without this, pages are structured dumps")
    p.add_argument("--timeout", type=int, default=300, help="LLM timeout in seconds (default: 300)")
    p.add_argument("--parallel", type=int, default=0, help="Number of concurrent LLM workers (default: 0 = sequential)")
    p.add_argument("--status", choices=["IN", "OUT"], default=None, help="Filter by truth value")
    p.add_argument("--max-topics", type=int, default=20, help="Max topics for word-frequency grouping (default: 20)")
    p.add_argument("--cluster", action="store_true", help="Use semantic clustering instead of topic grouping")
    p.add_argument("--n-clusters", type=int, default=None, help="Override automatic cluster count")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible clustering")
    p.add_argument("--embedding-model", default=None, help="Sentence-transformers model for clustering")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only export nodes whose access_tags are a subset of these tags")

    # review-beliefs
    p = sub.add_parser("review-beliefs", help="Review derived beliefs for validity, sufficiency, and necessity")
    p.add_argument("ids", nargs="*", help="Specific belief IDs to review (default: all derived)")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude). Prefixes: ollama:<model>, api:<model>, vertex:<model>")
    p.add_argument("--timeout", type=int, default=600,
                   help="LLM timeout in seconds (default: 600)")
    p.add_argument("--min-depth", type=int, default=None,
                   help="Only review beliefs at this depth or deeper")
    p.add_argument("--depends-on", default=None,
                   help="Only review beliefs depending on this node")
    p.add_argument("-n", "--namespace", default=None,
                   help="Filter by agent namespace (use empty string '' for local beliefs only)")
    p.add_argument("--sample", type=int, default=None,
                   help="Randomly sample N beliefs to review")
    p.add_argument("--dry-run", action="store_true",
                   help="Report findings without taking action")
    p.add_argument("--auto-retract", action="store_true",
                   help="Retract beliefs found invalid")
    p.add_argument("--auto-defeat", action="store_true",
                   help="Defeat invalid beliefs with justified scope defeaters")
    p.add_argument("-o", "--output", default=None,
                   help="Write findings to markdown file")
    p.add_argument("--visible-to", metavar="TAG,TAG",
                   help="Only review nodes whose access_tags are a subset of these tags")
    p.add_argument("--report-dir", default="reviews",
                   help="Directory for JSON reports (default: reviews/)")
    p.add_argument("--no-report", action="store_true",
                   help="Skip JSON report generation")
    p.add_argument("--include-out", action="store_true",
                   help="Include OUT beliefs (e.g. to re-review previously retracted beliefs)")

    # review-justifications
    p = sub.add_parser("review-justifications",
                       help="Review SL justifications for ALL vs ANY misclassification")
    p.add_argument("--node", default=None, help="Review a specific belief")
    p.add_argument("--min-antecedents", type=int, default=2,
                   help="Only review justifications with at least N antecedents (default: 2)")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude)")
    p.add_argument("--timeout", type=int, default=600,
                   help="LLM timeout in seconds (default: 600)")
    p.add_argument("--parallel", type=int, default=0,
                   help="Number of concurrent LLM workers (default: 0 = sequential)")
    p.add_argument("-o", "--output", default=None,
                   help="Write findings to markdown file")

    # review-premises
    p = sub.add_parser("review-premises", help="Review premises against source material for factual accuracy")
    p.add_argument("ids", nargs="*", help="Specific premise IDs to review (default: all IN premises)")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude). Prefixes: ollama:<model>, api:<model>, vertex:<model>")
    p.add_argument("--timeout", type=int, default=600,
                   help="LLM timeout in seconds (default: 600)")
    p.add_argument("--sample", type=int, default=None,
                   help="Randomly sample N premises to review")
    p.add_argument("--dry-run", action="store_true",
                   help="Report findings without taking action")
    p.add_argument("--auto-retract", action="store_true",
                   help="Retract premises found inaccurate")
    p.add_argument("--visible-to", metavar="TAG,TAG",
                   help="Only review nodes whose access_tags are a subset of these tags")
    p.add_argument("--report-dir", default="reviews",
                   help="Directory for JSON reports (default: reviews/)")
    p.add_argument("--no-report", action="store_true",
                   help="Skip JSON report generation")
    p.add_argument("--parallel", type=int, default=0, metavar="N",
                   help="Number of concurrent workers (0 = sequential)")

    # repair-premises
    p = sub.add_parser("repair-premises", help="Repair inaccurate premises by rewriting from source or retracting")
    p.add_argument("ids", nargs="*", help="Premise IDs to repair (requires --review-file or re-reviews these IDs)")
    p.add_argument("--review-file", default=None,
                   help="Path to review-premises JSON report")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude). Prefixes: ollama:<model>, api:<model>, vertex:<model>")
    p.add_argument("--timeout", type=int, default=600,
                   help="LLM timeout in seconds (default: 600)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report findings without applying changes")
    p.add_argument("--parallel", type=int, default=0, metavar="N",
                   help="Number of concurrent workers (0 = sequential)")
    p.add_argument("--report-dir", default="reviews",
                   help="Directory for JSON reports (default: reviews/)")
    p.add_argument("--no-report", action="store_true",
                   help="Skip JSON report generation")

    # propose-update
    p = sub.add_parser("propose-update",
        help="Propose structured updates or retractions for beliefs (LLM-driven)")
    p.add_argument("ids", nargs="*", help="Specific belief IDs to evaluate (default: all IN)")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude). Prefixes: ollama:<model>, api:<model>, vertex:<model>")
    p.add_argument("--timeout", type=int, default=600,
                   help="LLM timeout in seconds (default: 600)")
    p.add_argument("--stale-only", action="store_true",
                   help="Only evaluate beliefs flagged as stale")
    p.add_argument("-n", "--namespace", default=None,
                   help="Filter by agent namespace (use empty string '' for local beliefs only)")
    p.add_argument("--sample", type=int, default=None,
                   help="Randomly sample N beliefs to evaluate")
    p.add_argument("-o", "--output", default=None,
                   help="Output file path (default: proposed-updates.md or .json)")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown",
                   help="Output format (default: markdown)")
    p.add_argument("--report-dir", default="reviews",
                   help="Directory for JSON reports (default: reviews/)")
    p.add_argument("--no-report", action="store_true",
                   help="Skip JSON report generation")

    # repair-smuggled
    p = sub.add_parser("repair-smuggled",
        help="Repair smuggled premises by finding and linking existing premises")
    p.add_argument("ids", nargs="*", help="Belief IDs to review and repair")
    p.add_argument("--review-file", default=None,
                   help="Path to review-beliefs JSON report")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude)")
    p.add_argument("--timeout", type=int, default=600,
                   help="LLM timeout in seconds (default: 600)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report findings without applying repairs")

    # repair (formerly research)
    p = sub.add_parser("repair",
        help="Repair flagged beliefs: search-and-link, soften, or abandon")
    p.add_argument("ids", nargs="*", help="Belief IDs to repair")
    p.add_argument("--review-file", default=None,
                   help="Path to review-beliefs JSON report")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude)")
    p.add_argument("--timeout", type=int, default=600,
                   help="LLM timeout in seconds (default: 600)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report findings without applying changes")

    # research (backward-compatible alias for repair)
    p = sub.add_parser("research", help="Alias for 'repair' (deprecated)")
    p.add_argument("ids", nargs="*", help="Belief IDs to repair")
    p.add_argument("--review-file", default=None,
                   help="Path to review-beliefs JSON report")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude)")
    p.add_argument("--timeout", type=int, default=600,
                   help="LLM timeout in seconds (default: 600)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report findings without applying changes")

    # contradictions
    p = sub.add_parser("contradictions", help="Detect contradictions between IN beliefs")
    p.add_argument("ids", nargs="*", help="Specific belief IDs to check (default: all IN)")
    p.add_argument("-m", "--model", default=None,
                   help="Model to use (default: claude). Prefixes: ollama:<model>, api:<model>, vertex:<model>")
    p.add_argument("--timeout", type=int, default=600,
                   help="LLM timeout in seconds (default: 600)")
    p.add_argument("--sample", type=int, default=None,
                   help="Randomly sample N beliefs to check")
    p.add_argument("--auto-apply", action="store_true",
                   help="Auto-apply detected nogoods via dependency-directed backtracking")
    p.add_argument("--semantic", action="store_true",
                   help="Group beliefs by semantic similarity before LLM analysis")
    p.add_argument("--embedding-model", default=None,
                   help="Sentence-transformers model (default: all-MiniLM-L6-v2)")
    p.add_argument("-o", "--output", default="proposed-contradictions.md",
                   help="Output file for contradiction plan (default: proposed-contradictions.md)")
    p.add_argument("--accept", metavar="FILE",
                   help="Apply a reviewed contradiction plan file")

    # list
    p = sub.add_parser("list", help="List nodes with filters")
    p.add_argument("--status", choices=["IN", "OUT"], help="Filter by truth value")
    p.add_argument("--premises", action="store_true", help="Only show premises (no justifications)")
    p.add_argument("--has-dependents", action="store_true", help="Only show nodes that others depend on")
    p.add_argument("--challenged", action="store_true", help="Only show nodes with active challenges")
    p.add_argument("-n", "--namespace", help="Filter to nodes in this namespace")
    p.add_argument("--min-depth", type=int, default=None,
                   help="Only show beliefs at this depth or deeper (0=premises)")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Only show beliefs at this depth or shallower")
    p.add_argument("--visible-to", metavar="TAG,TAG", help="Only show nodes whose access_tags are a subset of these tags")
    p.add_argument("--not-reviewed-since", type=int, default=None, metavar="DAYS",
                   help="Derived beliefs not reviewed in the last N days (or never)")
    p.add_argument("--never-reviewed", action="store_true",
                   help="Derived beliefs that have never been reviewed")
    p.add_argument("--by-impact", action="store_true",
                   help="Sort output by dependent count (descending)")
    p.add_argument("--label", help="Filter to nodes with a justification matching this label")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "init": cmd_init,
        "add": cmd_add,
        "add-justification": cmd_add_justification,
        "remove-justification": cmd_remove_justification,
        "retract": cmd_retract,
        "assert": cmd_assert,
        "mark-duplicate": cmd_mark_duplicate,
        "mark-superseded": cmd_mark_superseded,
        "defeat-justification": cmd_defeat_justification,
        "defeat-with-scope": cmd_defeat_with_scope,
        "migrate-defeaters": cmd_migrate_defeaters,
        "classify-defeaters": cmd_classify_defeaters,
        "what-if": cmd_what_if,
        "status": cmd_status,
        "show": cmd_show,
        "explain": cmd_explain,
        "nogood": cmd_nogood,
        "propagate": cmd_propagate,
        "log": cmd_log,
        "add-repo": cmd_add_repo,
        "repos": cmd_repos,
        "derive": cmd_derive,
        "accept": cmd_accept,
        "import-agent": cmd_import_agent,
        "sync-agent": cmd_sync_agent,
        "import-beliefs": cmd_import_beliefs,
        "import-json": cmd_import_json,
        "import-hf": cmd_import_hf,
        "pull": cmd_pull,
        "publish": cmd_publish,
        "import-api": cmd_import_api,
        "export-api": cmd_export_api,
        "export": cmd_export,
        "export-markdown": cmd_export_markdown,
        "export-card": cmd_export_card,
        "hash-sources": cmd_hash_sources,
        "check-stale": cmd_check_stale,
        "check-integrity": cmd_check_integrity,
        "backfill-hashes": cmd_backfill_hashes,
        "pin-sources": cmd_pin_sources,
        "pin-update": cmd_pin_update,
        "pin-lines": cmd_pin_lines,
        "compact": cmd_compact,
        "convert-to-premise": cmd_convert_to_premise,
        "summarize": cmd_summarize,
        "supersede": cmd_supersede,
        "update": cmd_update,
        "get-metadata": cmd_get_metadata,
        "set-metadata": cmd_set_metadata,
        "challenge": cmd_challenge,
        "defend": cmd_defend,
        "trace": cmd_trace,
        "trace-access-tags": cmd_trace_access_tags,
        "search": cmd_search,
        "lookup": cmd_lookup,
        "ask": cmd_ask,
        "search-sources": cmd_search_sources,
        "deduplicate": cmd_deduplicate,
        "cluster-list": cmd_cluster_list,
        "list": cmd_list,
        "list-gated": cmd_list_gated,
        "report-gated": cmd_report_gated,
        "report": cmd_report,
        "verify": cmd_verify,
        "list-negative": cmd_list_negative,
        "topics": cmd_topics,
        "build-wiki": cmd_build_wiki,
        "review-beliefs": cmd_review_beliefs,
        "review-justifications": cmd_review_justifications,
        "review-premises": cmd_review_premises,
        "repair-premises": cmd_repair_premises,
        "propose-update": cmd_propose_update,
        "repair-smuggled": cmd_repair_smuggled,
        "repair": cmd_repair,
        "research": cmd_repair,
        "contradictions": cmd_contradictions,
        "namespaces": cmd_namespaces,
        "forge": lambda args: _dispatch_forge(args, _forge_commands),
        **_forge_type_commands,
    }
    commands[args.command](args)


def _dispatch_forge(args, forge_commands):
    if not args.forge_command:
        print("Usage: reasonsforge forge <command>", file=sys.stderr)
        print("Commands: " + ", ".join(sorted(forge_commands.keys())),
              file=sys.stderr)
        sys.exit(1)
    forge_commands[args.forge_command](args)

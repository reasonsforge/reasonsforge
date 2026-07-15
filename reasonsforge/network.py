"""Dependency network with automatic truth value propagation.

Implements Doyle's (1979) TMS algorithm:
- Nodes have justifications (SL or CP)
- Truth values propagate automatically through the dependency graph
- Retraction cascades: when a node goes OUT, dependents are recomputed
- Restoration: when a node comes back IN, dependents are recomputed
- Retracted nodes stay in the network (enables restoration without rederivation)
"""

from collections import deque
from datetime import datetime, timezone

from . import Node, Justification, Nogood


class Network:
    """The dependency network — core data structure of the RMS."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.nogoods: list[Nogood] = []
        self._next_nogood_id: int = 1
        self.repos: dict[str, str] = {}  # name → path mapping
        self.log: list[dict] = []  # propagation audit trail
        self.meta: dict[str, str] = {}  # schema_version, project_name, etc.

    def _rebuild_dependents(self) -> None:
        """Rebuild the dependents reverse index from justifications.

        This is the single canonical implementation. All other rebuild
        call sites should delegate here.
        """
        for node in self.nodes.values():
            node.dependents = set()
        for node in self.nodes.values():
            for j in node.justifications:
                for ant_id in j.antecedents:
                    if ant_id in self.nodes:
                        self.nodes[ant_id].dependents.add(node.id)
                for out_id in j.outlist:
                    if out_id in self.nodes:
                        self.nodes[out_id].dependents.add(node.id)

    def verify_dependents(self) -> list[str]:
        """Compare live dependents index against what justifications imply.

        Returns a list of error strings. Empty list means consistent.
        """
        expected: dict[str, set[str]] = {nid: set() for nid in self.nodes}
        for node in self.nodes.values():
            for j in node.justifications:
                for ant_id in j.antecedents:
                    if ant_id in self.nodes:
                        expected[ant_id].add(node.id)
                for out_id in j.outlist:
                    if out_id in self.nodes:
                        expected[out_id].add(node.id)

        errors = []
        for nid in self.nodes:
            live = self.nodes[nid].dependents
            exp = expected.get(nid, set())
            extras = live - exp
            missing = exp - live
            if extras:
                errors.append(f"{nid}: extra dependents {extras}")
            if missing:
                errors.append(f"{nid}: missing dependents {missing}")
        return errors

    def add_node(
        self,
        id: str,
        text: str,
        justifications: list[Justification] | None = None,
        source: str = "",
        source_url: str = "",
        source_hash: str = "",
        date: str = "",
        metadata: dict | None = None,
        created_at: str = "",
        updated_at: str = "",
        reviewed_at: str = "",
        verified_at: str = "",
        retracted_at: str = "",
    ) -> Node:
        """Add a node to the network and propagate.

        If no justifications are provided, the node is a premise (IN by default).
        If justifications are provided, truth value is computed from them.
        """
        if id in self.nodes:
            raise ValueError(f"Node '{id}' already exists")

        if not created_at:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            created_at = now
            updated_at = now

        node = Node(
            id=id,
            text=text,
            justifications=justifications or [],
            source=source,
            source_url=source_url,
            source_hash=source_hash,
            date=date,
            metadata=metadata or {},
            created_at=created_at,
            updated_at=updated_at,
            reviewed_at=reviewed_at,
            verified_at=verified_at,
            retracted_at=retracted_at,
        )

        # Register as dependent of antecedents (inlist) and outlist nodes.
        # Both can affect this node's truth value when they change.
        for j in node.justifications:
            for ant_id in j.antecedents:
                if ant_id in self.nodes:
                    self.nodes[ant_id].dependents.add(id)
            for out_id in j.outlist:
                if out_id in self.nodes:
                    self.nodes[out_id].dependents.add(id)

        self.nodes[id] = node

        # Compute Merkle hashes
        if not node.text_hash:
            from .merkle import compute_text_hash, compute_content_hash
            node.text_hash = compute_text_hash(node.text)
            for j in node.justifications:
                if not j.content_hash:
                    j.content_hash = compute_content_hash(node, j, self)

        self._inherit_access_tags(node)

        # Compute initial truth value
        if node.justifications:
            node.truth_value = self._compute_truth(node)
        else:
            # Premise — IN by default
            node.truth_value = "IN"

        self._log("add", id, node.truth_value)
        return node

    def retract(self, node_id: str, reason: str = "") -> list[str]:
        """Mark a node OUT and propagate the retraction cascade.

        Args:
            node_id: Node to retract
            reason: Why this node is being retracted (stored in metadata)

        Returns list of all node IDs whose truth value changed.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        node = self.nodes[node_id]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if node.truth_value == "OUT":
            node.metadata["_retracted"] = True
            node.retracted_at = now
            node.updated_at = now
            return []

        node.truth_value = "OUT"
        node.metadata["_retracted"] = True
        node.retracted_at = now
        node.updated_at = now
        if reason:
            node.metadata["retract_reason"] = reason
        changed = [node_id]
        self._log("retract", node_id, reason or "OUT")

        # Propagate to dependents
        changed.extend(self._propagate(node_id))
        return changed

    def assert_node(self, node_id: str) -> list[str]:
        """Mark a node IN and propagate restoration.

        Returns list of all node IDs whose truth value changed.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        node = self.nodes[node_id]
        if node.truth_value == "IN":
            return []

        node.truth_value = "IN"
        node.metadata.pop("_retracted", None)
        node.retracted_at = ""
        node.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        changed = [node_id]
        self._log("assert", node_id, "IN")

        # Propagate to dependents
        changed.extend(self._propagate(node_id))
        return changed

    def _inherit_access_tags(self, node: "Node") -> None:
        """Merge access_tags from all justification antecedents into this node."""
        if not node.justifications:
            return
        inherited = set(node.metadata.get("access_tags", []))
        for j in node.justifications:
            for ant_id in j.antecedents:
                if ant_id in self.nodes:
                    inherited.update(
                        self.nodes[ant_id].metadata.get("access_tags", [])
                    )
        if inherited:
            node.metadata["access_tags"] = sorted(inherited)

    def _propagate_access_tags(self, changed_id: str) -> None:
        """Push access_tags forward to dependents that derive from this node."""
        queue = deque([changed_id])
        visited = {changed_id}
        while queue:
            current_id = queue.popleft()
            current = self.nodes[current_id]
            for dep_id in current.dependents:
                if dep_id in visited or dep_id not in self.nodes:
                    continue
                dep = self.nodes[dep_id]
                old_tags = dep.metadata.get("access_tags", [])
                self._inherit_access_tags(dep)
                new_tags = dep.metadata.get("access_tags", [])
                if old_tags != new_tags:
                    visited.add(dep_id)
                    queue.append(dep_id)

    def trace_access_tags(self, node_id: str) -> list[str]:
        """Return the union of all access_tags in the dependency subgraph.

        Traces backward through all justification chains, collecting
        access_tags from every node visited (not just premises).
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        all_tags: set[str] = set()
        visited: set[str] = set()

        def _walk(nid: str) -> None:
            if nid in visited or nid not in self.nodes:
                return
            visited.add(nid)
            node = self.nodes[nid]
            all_tags.update(node.metadata.get("access_tags", []))
            for j in node.justifications:
                for ant_id in j.antecedents:
                    _walk(ant_id)

        _walk(node_id)
        return sorted(all_tags)

    def trace_assumptions(self, node_id: str) -> list[str]:
        """Trace backward through justification chains to find all premises.

        Returns the set of premise IDs (nodes with no justifications) that
        support this node. These are the assumptions the conclusion rests on.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        premises = []
        visited = set()

        def _walk(nid: str) -> None:
            if nid in visited or nid not in self.nodes:
                return
            visited.add(nid)
            node = self.nodes[nid]
            if not node.justifications:
                # This is a premise
                if nid not in premises:
                    premises.append(nid)
                return
            # Walk through all justifications' antecedents
            for j in node.justifications:
                for ant_id in j.antecedents:
                    _walk(ant_id)

        _walk(node_id)
        return premises

    def _entrenchment(self, node_id: str) -> int:
        """Compute entrenchment score for a node.

        Higher score = more entrenched = harder to retract.
        Premises (evidence) are more entrenched than derived nodes (speculation).
        Nodes with more dependents are more entrenched (more things break).
        Source-backed nodes are more entrenched than unsourced.

        The score is used by find_culprits to prefer retracting less-entrenched
        nodes (speculative assumptions) over more-entrenched ones (evidence).
        """
        if node_id not in self.nodes:
            return 0
        node = self.nodes[node_id]
        score = 0

        # Premises (no justifications) are evidence — protect them
        if not node.justifications:
            score += 100

        # Source-backed nodes are more trustworthy
        if node.source:
            score += 50
        if node.source_hash:
            score += 25

        # More dependents = more entrenched (more things break if retracted)
        score += len(node.dependents) * 10

        # Metadata-based type scoring
        btype = node.metadata.get("beliefs_type", "").upper()
        type_scores = {
            "AXIOM": 90, "WARNING": 90,
            "OBSERVATION": 80,
            "DERIVED": 40,
            "PREDICTED": 30,
            "NOTE": 10,
        }
        score += type_scores.get(btype, 20)

        return score

    def find_culprits(self, nogood_node_ids: list[str]) -> list[dict]:
        """Find premises that could be retracted to resolve a contradiction.

        For each nogood node, traces back to its premises. Then identifies
        which premises, if retracted, would cause at least one nogood node
        to go OUT (resolving the contradiction).

        Sorted by entrenchment (least entrenched first). This ensures
        evidence/observations are protected and speculative assumptions
        are retracted first.

        Returns:
            [{"premise": str, "would_resolve": list[str], "dependent_count": int, "entrenchment": int}]
        """
        # Collect assumptions for each nogood node
        assumptions_by_node: dict[str, list[str]] = {}
        all_premises: set[str] = set()
        for nid in nogood_node_ids:
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            if node.truth_value != "IN":
                continue
            assumptions = self.trace_assumptions(nid)
            assumptions_by_node[nid] = assumptions
            all_premises.update(assumptions)

        # For each premise, check which nogood nodes depend on it
        candidates = []
        for premise_id in all_premises:
            would_resolve = []
            for nid, assumptions in assumptions_by_node.items():
                if premise_id in assumptions:
                    would_resolve.append(nid)
            if would_resolve:
                entrenchment = self._entrenchment(premise_id)
                candidates.append({
                    "premise": premise_id,
                    "would_resolve": would_resolve,
                    "dependent_count": len(self.nodes[premise_id].dependents),
                    "entrenchment": entrenchment,
                })

        # Sort: least entrenched first (retract speculative assumptions first)
        candidates.sort(key=lambda c: c["entrenchment"])
        return candidates

    def add_nogood(self, node_ids: list[str]) -> list[str]:
        """Record a contradiction and use dependency-directed backtracking to resolve.

        Traces backward through justification chains to find the premises
        (assumptions) responsible for the contradiction, then retracts the
        premise with the fewest dependents (minimal disruption).

        Returns list of all node IDs whose truth value changed.
        """
        # Verify all nodes exist
        for nid in node_ids:
            if nid not in self.nodes:
                raise KeyError(f"Node '{nid}' not found")

        nogood_id = f"nogood-{self._next_nogood_id:03d}"
        self._next_nogood_id += 1
        nogood = Nogood(
            id=nogood_id,
            nodes=list(node_ids),
            discovered=datetime.now().isoformat(timespec="seconds"),
        )
        self.nogoods.append(nogood)
        self._log("nogood", nogood_id, str(node_ids))

        # Check if contradiction is active (all nodes IN)
        all_in = all(self.nodes[nid].truth_value == "IN" for nid in node_ids)
        if not all_in:
            return []

        # Dependency-directed backtracking: find responsible premises
        culprits = self.find_culprits(node_ids)

        if culprits:
            # Retract the premise with fewest dependents
            victim_id = culprits[0]["premise"]
            self._log("backtrack", victim_id, f"culprit for {nogood_id}")
        else:
            # Fallback: retract the nogood node with fewest dependents
            candidates = [(nid, len(self.nodes[nid].dependents)) for nid in node_ids]
            candidates.sort(key=lambda x: x[1])
            victim_id = candidates[0][0]

        return self.retract(victim_id)

    def supersede(self, old_id: str, new_id: str) -> dict:
        """Mark old_id as superseded by new_id using the outlist mechanism.

        Adds new_id to old_id's outlist. When new_id is IN, old_id
        automatically goes OUT. If new_id is later retracted, old_id
        comes back IN — the supersession is reversible.

        Records supersession in both nodes' metadata for display.

        Returns: {"old_id": str, "new_id": str, "changed": list[str]}
        """
        if old_id not in self.nodes:
            raise KeyError(f"Node '{old_id}' not found")
        if new_id not in self.nodes:
            raise KeyError(f"Node '{new_id}' not found")

        old_node = self.nodes[old_id]

        # Add new_id to old_node's outlist
        if old_node.justifications:
            for j in old_node.justifications:
                if new_id not in j.outlist:
                    j.outlist.append(new_id)
        else:
            # Old node is a premise — convert to justified with outlist
            old_node.justifications.append(
                Justification(type="SL", antecedents=[], outlist=[new_id])
            )

        # Register new_id as affecting old_id
        self.nodes[new_id].dependents.add(old_id)

        # Record in metadata
        old_node.metadata["superseded_by"] = new_id
        supersedes = self.nodes[new_id].metadata.get("supersedes", [])
        if old_id not in supersedes:
            supersedes.append(old_id)
        self.nodes[new_id].metadata["supersedes"] = supersedes

        # Recompute and propagate
        old_value = old_node.truth_value
        new_value = self._compute_truth(old_node)
        changed = []

        if old_value != new_value:
            old_node.truth_value = new_value
            changed.append(old_id)
            self._log("supersede", old_id, f"superseded by {new_id}")
            changed.extend(self._propagate(old_id))
        else:
            self._log("supersede", old_id, f"superseded by {new_id} (unchanged)")

        return {"old_id": old_id, "new_id": new_id, "changed": changed}

    def add_justification(self, node_id: str, justification: "Justification") -> dict:
        """Add a justification to an existing node and propagate."""
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        node = self.nodes[node_id]
        old_value = node.truth_value

        for ant_id in justification.antecedents:
            if ant_id in self.nodes:
                self.nodes[ant_id].dependents.add(node_id)
        for out_id in justification.outlist:
            if out_id in self.nodes:
                self.nodes[out_id].dependents.add(node_id)

        node.justifications.append(justification)

        if not justification.content_hash:
            from .merkle import compute_content_hash
            if not node.text_hash:
                from .merkle import compute_text_hash
                node.text_hash = compute_text_hash(node.text)
            justification.content_hash = compute_content_hash(node, justification, self)

        self._inherit_access_tags(node)
        self._propagate_access_tags(node_id)

        new_value = self._compute_truth(node)
        changed = []

        if old_value != new_value:
            node.truth_value = new_value
            changed.append(node_id)
            changed.extend(self._propagate(node_id))

        self._log("add-justification", node_id, new_value)
        return {
            "node_id": node_id,
            "old_truth_value": old_value,
            "new_truth_value": new_value,
            "changed": changed,
        }

    def challenge(self, target_id: str, reason: str, challenge_id: str | None = None) -> dict:
        """Challenge a node — create a challenge node and add it to the target's outlist.

        The challenge node is a premise (IN by default), so the target
        immediately goes OUT (unless it has another justification that
        doesn't include this challenge in its outlist).

        If the target is a premise (no justifications), it is converted
        to a justified node with an SL justification that has the
        challenge in its outlist.

        Returns: {"challenge_id": str, "target_id": str, "changed": list[str]}
        """
        if target_id not in self.nodes:
            raise KeyError(f"Node '{target_id}' not found")

        target = self.nodes[target_id]

        # Generate challenge ID
        if challenge_id is None:
            challenge_id = f"challenge-{target_id}"
            # Handle multiple challenges to the same target
            suffix = 1
            while challenge_id in self.nodes:
                suffix += 1
                challenge_id = f"challenge-{target_id}-{suffix}"

        if challenge_id in self.nodes:
            raise ValueError(f"Challenge node '{challenge_id}' already exists")

        # Create the challenge node (premise — IN by default)
        challenge_node = Node(
            id=challenge_id,
            text=reason,
            metadata={"challenge_target": target_id},
        )
        self.nodes[challenge_id] = challenge_node
        self._log("add", challenge_id, "IN")

        # Add challenge to target's outlist
        if target.justifications:
            # Add challenge to outlist of all existing justifications
            for j in target.justifications:
                j.outlist.append(challenge_id)
        else:
            # Target is a premise — convert to justified node
            # It was IN because it was a premise; now it's IN because
            # of an SL justification with the challenge in the outlist
            target.justifications.append(
                Justification(type="SL", antecedents=[], outlist=[challenge_id])
            )

        # Register challenge node as affecting target
        challenge_node.dependents.add(target_id)

        # Track challenge on target metadata
        challenges = target.metadata.get("challenges", [])
        challenges.append(challenge_id)
        target.metadata["challenges"] = challenges

        # Recompute target truth value and propagate
        old_value = target.truth_value
        new_value = self._compute_truth(target)
        changed = []

        if old_value != new_value:
            target.truth_value = new_value
            changed.append(target_id)
            self._log("challenge", target_id, new_value)
            changed.extend(self._propagate(target_id))
        else:
            self._log("challenge", target_id, f"unchanged ({old_value})")

        return {"challenge_id": challenge_id, "target_id": target_id, "changed": changed}

    def defend(
        self,
        target_id: str,
        challenge_id: str,
        reason: str,
        defense_id: str | None = None,
    ) -> dict:
        """Defend a node against a challenge — create a defense that neutralises the challenge.

        The defense node has the challenge in its outlist: "the defense
        holds unless the challenge is sustained." Since the defense is
        a premise (IN by default), the challenge gets the defense in
        its outlist, which makes the challenge go OUT, which restores
        the target.

        Returns: {"defense_id": str, "challenge_id": str, "target_id": str, "changed": list[str]}
        """
        if target_id not in self.nodes:
            raise KeyError(f"Node '{target_id}' not found")
        if challenge_id not in self.nodes:
            raise KeyError(f"Challenge '{challenge_id}' not found")

        if defense_id is None:
            defense_id = f"defense-{challenge_id}"
            suffix = 1
            while defense_id in self.nodes:
                suffix += 1
                defense_id = f"defense-{challenge_id}-{suffix}"

        if defense_id in self.nodes:
            raise ValueError(f"Defense node '{defense_id}' already exists")

        # The defense challenges the challenge — same mechanism
        result = self.challenge(challenge_id, reason, challenge_id=defense_id)

        # Update metadata
        self.nodes[defense_id].metadata["defense_target"] = challenge_id
        self.nodes[defense_id].metadata["defends"] = target_id

        return {
            "defense_id": defense_id,
            "challenge_id": challenge_id,
            "target_id": target_id,
            "changed": result["changed"],
        }

    def convert_to_premise(self, node_id: str) -> dict:
        """Strip all justifications from a node, making it a premise.

        Use after import when a 'Depends on:' relationship was contextual
        (derived in the context of investigating X) rather than logical
        (true only if X is true). The node becomes IN by default.

        Returns: {"node_id": str, "old_justifications": int, "truth_value": str, "changed": list[str]}
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        node = self.nodes[node_id]
        old_count = len(node.justifications)

        # Remove this node from the dependents set of its antecedents/outlist
        for j in node.justifications:
            for ant_id in j.antecedents:
                if ant_id in self.nodes:
                    self.nodes[ant_id].dependents.discard(node_id)
            for out_id in j.outlist:
                if out_id in self.nodes:
                    self.nodes[out_id].dependents.discard(node_id)

        node.justifications = []
        node.supporting_justification = None

        # A premise is IN by default
        changed = []
        if node.truth_value != "IN":
            node.truth_value = "IN"
            changed.append(node_id)
            self._log("convert-to-premise", node_id, "IN")
            changed.extend(self._propagate(node_id))
        else:
            self._log("convert-to-premise", node_id, "IN (unchanged)")

        return {
            "node_id": node_id,
            "old_justifications": old_count,
            "truth_value": node.truth_value,
            "changed": changed,
        }

    def remove_justification(self, node_id: str, index: int) -> dict:
        """Remove a single justification by index and propagate."""
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        node = self.nodes[node_id]

        if not node.justifications:
            raise ValueError(f"Node '{node_id}' is a premise (no justifications)")

        if index < 0 or index >= len(node.justifications):
            raise IndexError(
                f"Justification index {index} out of range "
                f"(node has {len(node.justifications)})"
            )

        if len(node.justifications) == 1:
            raise ValueError(
                f"Node '{node_id}' has only one justification; "
                f"use 'convert-to-premise' or 'retract' instead"
            )

        old_value = node.truth_value
        removed = node.justifications.pop(index)

        # Clean up dependents for antecedents/outlist that no longer appear
        # in any remaining justification
        remaining_refs = set()
        for j in node.justifications:
            remaining_refs.update(j.antecedents)
            remaining_refs.update(j.outlist)

        for ant_id in removed.antecedents:
            if ant_id not in remaining_refs and ant_id in self.nodes:
                self.nodes[ant_id].dependents.discard(node_id)
        for out_id in removed.outlist:
            if out_id not in remaining_refs and out_id in self.nodes:
                self.nodes[out_id].dependents.discard(node_id)

        new_value = self._compute_truth(node)
        changed = []

        if old_value != new_value:
            node.truth_value = new_value
            changed.append(node_id)
            changed.extend(self._propagate(node_id))

        self._log("remove-justification", node_id, new_value)

        removed_dict = {
            "type": removed.type,
            "antecedents": removed.antecedents,
            "outlist": removed.outlist,
            "label": removed.label,
        }
        return {
            "node_id": node_id,
            "old_truth_value": old_value,
            "new_truth_value": new_value,
            "removed": removed_dict,
            "remaining": len(node.justifications),
            "changed": changed,
        }

    def summarize(
        self,
        summary_id: str,
        text: str,
        over: list[str],
        source: str = "",
    ) -> dict:
        """Create a summary node that abstracts over a group of nodes.

        The summary is IN when ALL summarized nodes are IN (SL justification).
        In compact output, the summary replaces the individual nodes it covers,
        saving token budget while preserving the high-level picture.

        Returns: {"summary_id": str, "over": list[str], "truth_value": str}
        """
        for nid in over:
            if nid not in self.nodes:
                raise KeyError(f"Node '{nid}' not found")

        if summary_id in self.nodes:
            raise ValueError(f"Node '{summary_id}' already exists")

        node = self.add_node(
            id=summary_id,
            text=text,
            justifications=[
                Justification(
                    type="SL",
                    antecedents=list(over),
                    label="summarizes",
                ),
            ],
            source=source,
            metadata={"summarizes": list(over)},
        )

        # Mark the summarized nodes as covered
        for nid in over:
            covered = self.nodes[nid].metadata.get("summarized_by", [])
            if summary_id not in covered:
                covered.append(summary_id)
            self.nodes[nid].metadata["summarized_by"] = covered

        return {
            "summary_id": summary_id,
            "over": list(over),
            "truth_value": node.truth_value,
        }

    def explain(self, node_id: str, _visited: set | None = None,
                _path: set | None = None) -> list[dict]:
        """Trace why a node is IN or OUT.

        Returns a list of explanation steps tracing back through justifications.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found")

        if _visited is None:
            _visited = set()
        if _path is None:
            _path = set()

        if node_id in _path:
            return [{"node": node_id, "truth_value": self.nodes[node_id].truth_value,
                     "reason": "circular dependency"}]
        if node_id in _visited:
            return []
        _visited.add(node_id)
        _path = _path | {node_id}

        node = self.nodes[node_id]
        steps = []

        if not node.justifications:
            steps.append({
                "node": node_id,
                "truth_value": node.truth_value,
                "reason": "premise" if node.truth_value == "IN" else "retracted premise",
            })
            return steps

        if node.truth_value == "IN":
            # Use designated supporting justification when available
            j = None
            si = node.supporting_justification
            if si is not None and 0 <= si < len(node.justifications):
                candidate = node.justifications[si]
                if self._justification_valid(candidate):
                    j = candidate
            if j is None:
                for candidate in reversed(node.justifications):
                    if self._justification_valid(candidate):
                        j = candidate
                        break
            if j is not None:
                step = {
                    "node": node_id,
                    "truth_value": "IN",
                    "reason": f"{j.type} justification valid",
                    "antecedents": list(j.antecedents),
                    "label": j.label,
                }
                if j.outlist:
                    step["outlist"] = list(j.outlist)
                steps.append(step)
                for ant_id in j.antecedents:
                    steps.extend(self.explain(ant_id, _visited, _path))
        else:
            # All justifications invalid — explain why
            for j in node.justifications:
                failed = [
                    a for a in j.antecedents
                    if a in self.nodes and self.nodes[a].truth_value == "OUT"
                ]
                violated_outlist = [
                    o for o in j.outlist
                    if o in self.nodes and self.nodes[o].truth_value == "IN"
                ]
                step = {
                    "node": node_id,
                    "truth_value": "OUT",
                    "reason": f"{j.type} justification invalid",
                    "failed_antecedents": failed,
                    "label": j.label,
                }
                if violated_outlist:
                    step["violated_outlist"] = violated_outlist
                steps.append(step)

        return steps

    def get_belief_set(self) -> list[str]:
        """Return all node IDs currently IN."""
        return [nid for nid, node in self.nodes.items() if node.truth_value == "IN"]

    def recompute_all(self) -> list[str]:
        """Recompute truth values for all derived nodes from the justification graph.

        Iterates to a fixpoint so cascading changes propagate regardless of
        node insertion order.  Returns list of node IDs whose truth values changed.
        """
        all_changed: set[str] = set()
        max_iterations = len(self.nodes) + 1
        for _ in range(max_iterations):
            changed_this_pass = []
            for nid, node in self.nodes.items():
                if node.justifications and not node.metadata.get("_retracted"):
                    old = node.truth_value
                    new = self._compute_truth(node)
                    if old != new:
                        node.truth_value = new
                        changed_this_pass.append(nid)
                        self._log("recompute", nid, new)
            if not changed_this_pass:
                break
            all_changed.update(changed_this_pass)
        return list(all_changed)

    def _propagate(self, changed_id: str) -> list[str]:
        """BFS propagation of truth value changes through dependents."""
        changed = []
        queue = deque([changed_id])
        visited = {changed_id}

        while queue:
            current_id = queue.popleft()
            current = self.nodes[current_id]

            for dep_id in current.dependents:
                if dep_id in visited:
                    continue

                if dep_id not in self.nodes:
                    self._log("warn", dep_id, f"dangling dependent of {current_id}")
                    continue

                dep = self.nodes[dep_id]
                if dep.metadata.get("_retracted"):
                    continue
                old_value = dep.truth_value
                new_value = self._compute_truth(dep)

                if old_value != new_value:
                    dep.truth_value = new_value
                    changed.append(dep_id)
                    self._log("propagate", dep_id, new_value)
                    visited.add(dep_id)
                    queue.append(dep_id)

        return changed

    def _compute_truth(self, node: Node) -> str:
        """Compute truth value from justifications.

        A node is IN if ANY justification is valid.  Prefers the newest
        (last-added) valid justification as the designated support.
        """
        if not node.justifications:
            return node.truth_value  # premise — keep current value

        for i in range(len(node.justifications) - 1, -1, -1):
            if self._justification_valid(node.justifications[i]):
                node.supporting_justification = i
                return "IN"
        node.supporting_justification = None
        return "OUT"

    def _justification_valid(self, j: Justification) -> bool:
        """Check if a justification is currently valid.

        SL: all antecedents (inlist) must be IN AND all outlist must be OUT.
        This enables non-monotonic reasoning: "believe X unless Y."
        """
        if j.type in ("SL", "CP"):
            inlist_ok = all(
                a in self.nodes and self.nodes[a].truth_value == "IN"
                for a in j.antecedents
            )
            outlist_ok = all(
                o not in self.nodes or self.nodes[o].truth_value == "OUT"
                for o in j.outlist
            )
            return inlist_ok and outlist_ok
        return False

    def _log(self, action: str, target: str, value: str) -> None:
        """Record a propagation event."""
        self.log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "target": target,
            "value": value,
        })

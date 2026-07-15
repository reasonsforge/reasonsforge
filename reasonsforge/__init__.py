"""Reason Maintenance System — data model based on Doyle (1979)."""

from dataclasses import dataclass, field


@dataclass
class Justification:
    """A reason for believing a node.

    Two types:
    - SL (Support List): node is IN iff ALL antecedents (inlist) are IN
      AND ALL outlist nodes are OUT
    - CP (Conditional Proof): node is IN iff assumptions are consistent

    The outlist enables non-monotonic reasoning: "believe X unless Y is
    believed." If Y comes IN, the justification becomes invalid and X
    may go OUT (if no other justification supports it).
    """
    type: str  # "SL" or "CP"
    antecedents: list[str] = field(default_factory=list)  # inlist: must be IN
    outlist: list[str] = field(default_factory=list)  # must be OUT
    label: str = ""
    content_hash: str = ""


@dataclass
class Node:
    """A node in the dependency network.

    A node is IN if ANY of its justifications is valid.
    A premise has no justifications and is IN by default.
    """
    id: str
    text: str
    truth_value: str = "IN"  # IN or OUT
    justifications: list[Justification] = field(default_factory=list)
    supporting_justification: int | None = None  # index into justifications list
    dependents: set[str] = field(default_factory=set)  # reverse index
    source: str = ""
    source_url: str = ""
    source_hash: str = ""
    text_hash: str = ""
    date: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    reviewed_at: str = ""
    verified_at: str = ""
    retracted_at: str = ""


@dataclass
class Nogood:
    """A recorded contradiction — these node IDs cannot all be IN simultaneously."""
    id: str
    nodes: list[str] = field(default_factory=list)
    discovered: str = ""
    resolution: str = ""

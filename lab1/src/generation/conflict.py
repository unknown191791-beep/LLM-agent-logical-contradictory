"""Conflict injection for transitivity breaks."""

from src.core.types import Fact, RelationType, ConflictPosition


class ConflictInjector:
    """
    Injects transitivity-break conflicts into chain structures.

    For a chain A→B→C→...→Z and question "Is A∈Z?":
      - Transitivity says A∈Z (derived).
      - We add negation fact(s) "A∉Z" breaking transitivity.
      - Each negation is a vote against A∈Z.

    Conflict position semantics (for transitivity_break on chain):
      - late:   negate the full chain (A∉Z where Z is the last entity)
      - middle: negate a middle hop (e.g., A∉C for chain A→B→C→D)
      - early:  negate an early hop (e.g., A∉B)

    For MVP, all conflicts are at the "late" position (A∉last_entity).
    """

    def inject_transitivity_break(
        self,
        chain_facts: list[Fact],
        chain_entities: list[str],
        conflict_count: int,
        position: ConflictPosition = ConflictPosition.LATE,
        start_id: int = 0,
    ) -> list[Fact]:
        """
        Create negation facts that break transitivity.

        Args:
            chain_facts: The original chain facts.
            chain_entities: Ordered entities in the chain.
            conflict_count: Number of negation facts to create.
            position: Where in the chain to place the conflict.
            start_id: Starting number for fact IDs.

        Returns:
            List of negation Facts.

        Raises:
            ValueError: If conflict_count is negative or position is invalid.
        """
        if conflict_count < 0:
            raise ValueError(f"conflict_count must be ≥ 0, got {conflict_count}")
        if conflict_count == 0:
            return []

        first = chain_entities[0]
        chain_len = len(chain_entities) - 1  # number of edges

        # Determine which entity pair to negate based on position
        if position == ConflictPosition.LATE:
            # Negate the full chain: A∉Z
            negated_subject = first
            negated_object = chain_entities[-1]
        elif position == ConflictPosition.MIDDLE:
            # Negate middle hop: A∉Y where Y is the middle entity
            mid_idx = max(1, chain_len // 2)
            negated_subject = first
            negated_object = chain_entities[mid_idx]
        elif position == ConflictPosition.EARLY:
            # Negate first hop: A∉B
            negated_subject = first
            negated_object = chain_entities[1]
        else:
            raise ValueError(f"Unknown conflict position: {position}")

        negation_facts = []
        for i in range(conflict_count):
            fid = f"n{start_id + i}"
            negation_facts.append(Fact(
                id=fid,
                subject=negated_subject,
                predicate=RelationType.NOT_MEMBER_OF,
                obj=negated_object,
                is_noise=False,
            ))

        return negation_facts

    def compute_mis(
        self,
        chain_length: int,
        conflict_count: int,
        position: ConflictPosition = ConflictPosition.LATE,
    ) -> int:
        """
        Compute Minimum Inconsistent Subset (MIS) size.

        For transitivity_break at 'late' position:
          Need all chain facts + 1 negation = chain_length + 1 facts.

        Args:
            chain_length: Number of edges in the chain.
            conflict_count: Number of negations.
            position: Where the conflict is.

        Returns:
            MIS size.
        """
        # The minimal inconsistent set = all chain facts + 1 negation
        # (You need all chain links to derive transitivity, plus one negation
        #  to create the contradiction.)
        return chain_length + 1

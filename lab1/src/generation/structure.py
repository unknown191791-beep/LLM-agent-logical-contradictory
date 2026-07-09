"""Chain structure builder and ground truth computation via majority voting."""

from dataclasses import dataclass
from typing import Optional

from src.core.types import Fact, Question, RelationType


# Pool of entity names for abstract scheme
_CHAIN_ENTITIES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NOISE_ENTITIES = "abcdefghijklmnopqrstuvwxyz"


def _entity_name(index: int) -> str:
    """Convert an integer index to a letter-based entity name.

    0->A, 1->B, ..., 25->Z, 26->AA, 27->AB, ..., 51->AZ, 52->BA, ...
    Works for arbitrarily large indices.
    """
    if index < 26:
        return _CHAIN_ENTITIES[index]
    # Convert to base-26-like representation
    result = []
    n = index
    while n >= 0:
        result.append(_CHAIN_ENTITIES[n % 26])
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(result))


class ChainBuilder:
    """
    Builds linear membership chains for transitive reasoning.

    A chain of length L has L edges connecting L+1 entities:
      chain_length=2: A → B → C  (2 hops, 3 entities)
      chain_length=3: A → B → C → D  (3 hops, 4 entities)

    The question always asks: Is the first entity a member of the last entity?
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._entity_offset = 0

    def build_chain(self, chain_length: int, entity_offset: int = 0) -> tuple[list[Fact], list[str]]:
        """
        Build a linear chain of membership relations.

        Args:
            chain_length: Number of hops (≥2).
            entity_offset: Starting index into the entity name pool.

        Returns:
            (chain_facts, entities) where:
              - chain_facts: Facts representing the chain edges
              - entities: Ordered list of entity names
        """
        if chain_length < 2:
            raise ValueError(f"chain_length must be ≥ 2, got {chain_length}")

        entities = []
        for i in range(chain_length + 1):
            idx = entity_offset + i
            name = _entity_name(idx)
            entities.append(name)

        facts = []
        for i in range(chain_length):
            fid = f"c{i}"
            facts.append(Fact(
                id=fid,
                subject=entities[i],
                predicate=RelationType.MEMBER_OF,
                obj=entities[i + 1],
                is_noise=False,
            ))

        return facts, entities

    def make_question(
        self, entities: list[str],
        predicate: RelationType = RelationType.MEMBER_OF,
    ) -> Question:
        """Create a question about transitivity from first to last entity."""
        return Question(
            subject=entities[0],
            predicate=predicate,
            obj=entities[-1],
        )


@dataclass
class GroundTruth:
    """Ground truth for a sample under majority voting."""
    answer: Optional[bool]       # None if unresolvable (tie)
    vote_ratio: float            # supporting / total relevant
    supporting_count: int
    opposing_count: int
    is_resolvable: bool
    rationale: str


class GroundTruthComputer:
    """
    Computes ground truth using majority voting (Plan A).

    Algorithm:
      1. Collect all facts relevant to the question's subject-object path.
      2. For each fact, determine whether it supports or opposes the conclusion.
      3. GT = majority verdict.
      4. If tie → sample is UNRESOLVABLE.
    """

    def compute(
        self,
        chain_facts: list[Fact],
        chain_entities: list[str],
        negation_facts: list[Fact],
        noise_facts: list[Fact],
        question: Question,
    ) -> GroundTruth:
        """
        Compute ground truth for a sample.

        Args:
            chain_facts: The transitive chain facts (all support transitivity).
            chain_entities: Ordered entity list.
            negation_facts: Conflict facts that oppose transitivity.
            noise_facts: Irrelevant facts.
            question: The query.

        Returns:
            GroundTruth with answer, confidence, and rationale.
        """
        # Count votes
        # Each chain fact supports the transitive conclusion
        supporting = len(chain_facts)

        # Each negation fact opposes the transitive conclusion
        opposing = len(negation_facts)

        # Noise facts are irrelevant → don't count
        total_relevant = supporting + opposing

        if total_relevant == 0:
            raise ValueError("No relevant facts to compute ground truth")

        vote_ratio = supporting / total_relevant

        if supporting > opposing:
            answer = True
            rationale = (
                f"Majority voting: {supporting} supporting facts "
                f"(chain transitivity) vs {opposing} opposing facts "
                f"(direct negations). GT = True ({supporting}:{opposing})."
            )
        elif opposing > supporting:
            answer = False
            rationale = (
                f"Majority voting: {supporting} supporting facts "
                f"(chain transitivity) vs {opposing} opposing facts "
                f"(direct negations). GT = False ({supporting}:{opposing})."
            )
        else:
            # Tie — unresolvable
            answer = None  # Mark as unresolvable
            rationale = (
                f"Tie: {supporting} supporting vs {opposing} opposing facts. "
                f"GT is unresolvable under majority voting."
            )

        return GroundTruth(
            answer=answer,
            vote_ratio=vote_ratio,
            supporting_count=supporting,
            opposing_count=opposing,
            is_resolvable=(supporting != opposing),
            rationale=rationale,
        )

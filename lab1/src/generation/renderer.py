"""Natural language rendering for abstract logical facts."""

import random as _random

from src.core.types import Fact, Question, RelationType, Sample


class NaturalLanguageRenderer:
    """
    Renders symbolic facts and questions as English natural language.

    Abstract scheme mapping:
      MEMBER_OF:     "X is a member of Y."
      NOT_MEMBER_OF: "X is not a member of Y."

    The renderer also supports shuffling facts to counter ordering bias.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = _random.Random(seed)

    def render_fact(self, fact: Fact) -> str:
        """Render a single fact as natural language text."""
        if fact.natural_text:
            return fact.natural_text

        if fact.predicate == RelationType.MEMBER_OF:
            return f"{fact.subject} is a member of {fact.obj}."
        elif fact.predicate == RelationType.NOT_MEMBER_OF:
            return f"{fact.subject} is not a member of {fact.obj}."
        else:
            raise ValueError(f"Unknown relation type: {fact.predicate}")

    def render_question(self, question: Question) -> str:
        """Render a question as natural language."""
        if question.natural_text:
            return question.natural_text

        if question.predicate == RelationType.MEMBER_OF:
            return f"Is {question.subject} a member of {question.obj}?"
        else:
            return f"Is {question.subject} not a member of {question.obj}?"

    def render_facts_and_question(
        self,
        facts: list[Fact],
        question: Question,
        shuffle: bool = True,
        shuffle_seed: int = 0,
    ) -> tuple[str, str]:
        """
        Render facts and question, optionally shuffling facts.

        Args:
            facts: List of all facts (chain + negation + noise).
            question: The question to ask.
            shuffle: Whether to randomize fact order.
            shuffle_seed: Seed for the shuffle.

        Returns:
            (facts_text, question_text) ready for prompt construction.
        """
        # Render all facts
        for fact in facts:
            if not fact.natural_text:
                fact.natural_text = self.render_fact(fact)

        if not question.natural_text:
            question.natural_text = self.render_question(question)

        # Shuffle if requested
        ordered_facts = list(facts)
        if shuffle:
            rng = _random.Random(shuffle_seed)
            rng.shuffle(ordered_facts)

        # Build facts block
        facts_lines = ["Facts:"]
        for f in ordered_facts:
            facts_lines.append(f"- {f.natural_text}")

        facts_text = "\n".join(facts_lines)
        question_text = f"Question: {question.natural_text}"

        return facts_text, question_text

    def render_sample_prompt(
        self,
        sample: Sample,
        shuffle_seed: int = 0,
    ) -> str:
        """
        Render a complete sample as a prompt-ready string.

        Args:
            sample: The sample to render.
            shuffle_seed: Seed for fact order randomization.

        Returns:
            Full prompt text (facts + question).
        """
        facts_text, question_text = self.render_facts_and_question(
            facts=sample.facts,
            question=sample.question,
            shuffle=True,
            shuffle_seed=shuffle_seed,
        )
        return facts_text + "\n\n" + question_text

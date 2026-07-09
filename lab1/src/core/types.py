"""Core data types for the experiment framework."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────────

class ConflictType(str, Enum):
    """Type of logical conflict injected into a sample."""
    NONE = "none"
    DIRECT_NEGATION = "direct_negation"       # A∈C vs A∉C
    TRANSITIVITY_BREAK = "transitivity_break"  # A∈B, B∈C, but A∉C
    CYCLE = "cycle"                            # A∈B, B∈C, C∈A
    # Future: EXCEPTION, AMBIGUITY, ...


class RelationType(str, Enum):
    """Kind of relationship between two entities."""
    MEMBER_OF = "member_of"                # A is a member of B
    NOT_MEMBER_OF = "not_member_of"        # A is NOT a member of B
    # Future: SUBSET_OF, EQUAL_TO, DISJOINT_FROM, ...


class ConflictPosition(str, Enum):
    """Where in the reasoning chain the conflict appears."""
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


class NoiseType(str, Enum):
    """Type of noise facts added to a sample."""
    IRRELEVANT = "irrelevant"    # Facts unrelated to the question chain
    MISLEADING = "misleading"    # Facts that appear relevant but aren't
    # Future: REDUNDANT, CONTRADICTORY_TO_NOISE, ...


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Fact:
    """A single logical fact (atomic statement)."""
    id: str                          # Unique identifier, e.g. "f0"
    subject: str                     # Entity name, e.g. "A"
    predicate: RelationType          # Relationship kind
    obj: str                         # Entity name, e.g. "B"
    is_noise: bool = False           # Whether this fact is noise
    natural_text: str = ""           # Rendered natural language text


@dataclass
class Question:
    """A query posed to the agent."""
    subject: str                     # e.g. "A"
    predicate: RelationType          # e.g. MEMBER_OF
    obj: str                         # e.g. "C"
    natural_text: str = ""           # Rendered natural language text


@dataclass
class SampleMetadata:
    """All structural and conflict parameters for a sample."""
    # Structural
    chain_length: int                # Number of hops in main chain (≥2)
    total_facts: int                 # Total facts in the sample

    # Conflict
    conflict_type: ConflictType
    conflict_count: int              # Number of independent conflicts
    conflict_positions: list[ConflictPosition]  # Where each conflict sits
    mis_size: int                    # Minimum Inconsistent Subset size

    # Noise
    noise_count: int
    noise_type: NoiseType

    # Generation
    seed: int                        # Random seed used for this sample
    entity_count: int                # Number of distinct entities
    supporting_fact_count: int       # Facts supporting the correct answer
    opposing_fact_count: int         # Facts opposing the correct answer
    gt_vote_ratio: float             # supporting / total relevant facts

    def to_dict(self) -> dict:
        return {
            "chain_length": self.chain_length,
            "total_facts": self.total_facts,
            "conflict_type": self.conflict_type.value,
            "conflict_count": self.conflict_count,
            "conflict_positions": [p.value for p in self.conflict_positions],
            "mis_size": self.mis_size,
            "noise_count": self.noise_count,
            "noise_type": self.noise_type.value,
            "seed": self.seed,
            "entity_count": self.entity_count,
            "supporting_fact_count": self.supporting_fact_count,
            "opposing_fact_count": self.opposing_fact_count,
            "gt_vote_ratio": self.gt_vote_ratio,
        }


@dataclass
class Sample:
    """Complete experimental sample (facts + question + ground truth)."""
    id: str                          # Unique sample ID
    facts: list[Fact]
    question: Question
    ground_truth: bool               # Majority-vote answer (Plan A)
    ground_truth_rationale: str      # How GT was derived
    metadata: SampleMetadata

    def render(self, shuffle_seed: Optional[int] = None) -> str:
        """
        Render facts and question as natural language.

        Args:
            shuffle_seed: If provided, shuffle facts with this seed
                          before rendering (to counter ordering bias).
        """
        import random as _random
        facts = list(self.facts)
        if shuffle_seed is not None:
            rng = _random.Random(shuffle_seed)
            rng.shuffle(facts)

        lines = ["Facts:"]
        for f in facts:
            lines.append(f"- {f.natural_text}")
        lines.append("")
        lines.append(f"Question: {self.question.natural_text}")
        return "\n".join(lines)


# ── Agent Result ─────────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    """A single step in the agent's reasoning trace."""
    step_index: int
    step_type: str                   # "thought", "action", "observation",
                                     # "plan_step", "solve_step", "conclusion"
    content: str
    tokens_spent: int = 0


@dataclass
class AgentResult:
    """Complete result of a single agent run."""
    sample_id: str
    workflow_name: str               # "react", "plan_solve", ...
    run_index: int                   # Which repeat (0, 1, 2, ...)
    seed: int                        # Shuffle seed used for this run

    # ── Core output ──
    final_answer: Optional[bool]     # True/False; None if unparseable
    extracted_answer_text: str       # Raw "YES"/"NO"/"UNCERTAIN" matched
    raw_response: str                # Full LLM response text
    answered_uncertain: bool         # Agent explicitly said UNCERTAIN

    # ── Conflict detection ──
    conflict_detected_explicit: Optional[bool]  # From CONFLICT_DETECTED field
    conflict_detected_implicit: bool            # From keyword matching
    conflict_keywords_found: list[str]          # Which keywords matched

    # ── Performance ──
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    reasoning_steps: int             # Number of steps extracted

    # ── Trace ──
    reasoning_trace: list[ReasoningStep] = field(default_factory=list)

    # ── Meta ──
    model_name: str = ""
    prompt_version: str = ""
    timestamp: str = ""
    error: Optional[str] = None

    @property
    def is_correct(self) -> bool:
        """Whether this result can be compared and is correct.
        Returns False when answer is None (unparseable)."""
        # This is set externally after comparison with ground_truth
        return getattr(self, '_is_correct', False)

    @is_correct.setter
    def is_correct(self, value: bool) -> None:
        self._is_correct = value


# ── Experiment Config ────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Full experiment configuration loaded from YAML."""
    # Sampling
    chain_lengths: list[int] = field(default_factory=lambda: [2, 3, 4])
    conflict_counts: list[int] = field(default_factory=lambda: [0, 1, 2])
    noise_counts: list[int] = field(default_factory=lambda: [0, 3, 6])
    samples_per_condition: int = 10
    fixed_conflict_type: ConflictType = ConflictType.TRANSITIVITY_BREAK
    fixed_conflict_position: ConflictPosition = ConflictPosition.LATE
    fixed_noise_type: NoiseType = NoiseType.IRRELEVANT
    seed: int = 42

    # Execution
    workflows: list[str] = field(default_factory=lambda: ["react", "plan_solve"])
    repeats_per_sample: int = 3
    adaptive_repeats: bool = True
    max_total_repeats: int = 5
    rate_limit_delay: float = 0.5

    # Output
    data_dir: str = "data"
    results_dir: str = "data/results"
    figures_dir: str = "outputs/figures"
    reports_dir: str = "outputs/reports"
    save_raw_responses: bool = True
    save_reasoning_traces: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from a YAML file."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        samp = data.get("sampling", {})
        exe = data.get("execution", {})
        out = data.get("output", {})
        fixed = samp.get("fixed", {})

        return cls(
            chain_lengths=samp.get("chain_lengths", [2, 3, 4]),
            conflict_counts=samp.get("conflict_counts", [0, 1, 2]),
            noise_counts=samp.get("noise_counts", [0, 3, 6]),
            samples_per_condition=samp.get("samples_per_condition", 10),
            fixed_conflict_type=ConflictType(fixed.get("conflict_type", "transitivity_break")),
            fixed_conflict_position=ConflictPosition(fixed.get("conflict_position", "late")),
            fixed_noise_type=NoiseType(fixed.get("noise_type", "irrelevant")),
            seed=samp.get("seed", 42),
            workflows=exe.get("workflows", ["react", "plan_solve"]),
            repeats_per_sample=exe.get("repeats_per_sample", 3),
            adaptive_repeats=exe.get("adaptive_repeats", True),
            max_total_repeats=exe.get("max_total_repeats", 5),
            rate_limit_delay=exe.get("rate_limit_delay", 0.5),
            data_dir=out.get("data_dir", "data"),
            results_dir=out.get("results_dir", "data/results"),
            figures_dir=out.get("figures_dir", "outputs/figures"),
            reports_dir=out.get("reports_dir", "outputs/reports"),
            save_raw_responses=out.get("save_raw_responses", True),
            save_reasoning_traces=out.get("save_reasoning_traces", True),
        )

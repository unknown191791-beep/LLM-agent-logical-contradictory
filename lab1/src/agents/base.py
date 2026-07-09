"""
AgentRunner abstract base class and AnswerExtractor.

AgentRunner defines the unified interface for all agent workflows.
AnswerExtractor provides robust boolean extraction from LLM output.
"""

import re
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from src.core.types import Sample, AgentResult, ReasoningStep
from src.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ── Conflict Detection Keywords ─────────────────────────────────────────────

_CONFLICT_KEYWORDS = [
    "conflict", "contradiction", "contradicts", "contradictory",
    "inconsistent", "inconsistency",
    "does not match", "doesn't match", "does not align",
    "on the other hand", "however, this",
    "but then", "this contradicts", "cannot both be true",
    "mutually exclusive", "paradox",
]

_HESITATION_KEYWORDS = [
    "possibly", "might be", "maybe", "could be",
    "uncertain", "unclear", "ambiguous",
    "it depends", "not sure",
]


# ── Answer Extractor ────────────────────────────────────────────────────────

class AnswerExtractor:
    """
    Multi-layer fallback extraction of boolean answers and conflict detection
    from free-form LLM responses.
    """

    # Layer 1: Strong format matching
    _ANSWER_PATTERN = re.compile(
        r'ANSWER\s*:\s*(YES|NO|UNCERTAIN)',
        re.IGNORECASE,
    )
    _CONFLICT_PATTERN = re.compile(
        r'CONFLICT_DETECTED\s*:\s*(YES|NO)',
        re.IGNORECASE,
    )

    # Layer 2: "answer is yes/no" pattern, flexible prefix
    _ANSWER_IS = re.compile(
        r'(?:the\s+)?answer\s+is\s+(yes|no)\b',
        re.IGNORECASE,
    )

    # Layer 3: Answer-like patterns anywhere
    _ANY_YES_NO = re.compile(
        r'\b(yes|no)\b',
        re.IGNORECASE,
    )

    # Member-of inference pattern
    _MEMBER_PATTERN = re.compile(
        r'(\w+)\s+is\s+(not\s+)?(?:a\s+)?member\s+of\s+(\w+)',
        re.IGNORECASE,
    )

    def extract_answer(self, raw: str) -> tuple[Optional[bool], str, str]:
        """
        Extract boolean answer from raw LLM response.

        Returns:
            (answer, extracted_text, method)
              - answer: True/False/None (None = couldn't extract)
              - extracted_text: The raw match string ("YES", "NO", "UNCERTAIN")
              - method: Which layer was used
        """
        # Layer 1: ANSWER: YES/NO/UNCERTAIN
        match = self._ANSWER_PATTERN.search(raw)
        if match:
            text = match.group(1).upper()
            if text == "YES":
                return True, text, "layer1_format"
            elif text == "NO":
                return False, text, "layer1_format"
            elif text == "UNCERTAIN":
                return None, text, "layer1_uncertain"

        # Layer 2: "answer is yes/no" — take last occurrence
        answer_is_matches = list(self._ANSWER_IS.finditer(raw))
        if answer_is_matches:
            m = answer_is_matches[-1]  # Last occurrence
            text = m.group(1).upper()
            if text == "YES":
                return True, text, "layer2_answer_is"
            elif text == "NO":
                return False, text, "layer2_answer_is"

        # Layer 3: "X is (not) a member of Y" pattern near the end
        # Look for the question entity in the last portion of the response
        last_portion = raw[-500:] if len(raw) > 500 else raw
        member_matches = self._MEMBER_PATTERN.findall(last_portion)
        if member_matches:
            # Take the last such statement
            subj, negation, obj = member_matches[-1]
            if negation.strip():
                return False, f"{subj} is not a member of {obj}", "layer3_member_inference"
            else:
                return True, f"{subj} is a member of {obj}", "layer3_member_inference"

        # Layer 4: Count YES/NO occurrences (fallback)
        yeses = len(re.findall(r'\byes\b', last_portion, re.IGNORECASE))
        nos = len(re.findall(r'\bno\b', last_portion, re.IGNORECASE))
        if yeses > nos:
            return True, "YES (counted)", "layer4_count"
        elif nos > yeses:
            return False, "NO (counted)", "layer4_count"

        # All layers failed
        return None, "", "failed"

    def extract_conflict_explicit(self, raw: str) -> Optional[bool]:
        """Extract CONFLICT_DETECTED: YES/NO field."""
        match = self._CONFLICT_PATTERN.search(raw)
        if match:
            text = match.group(1).upper()
            return text == "YES"
        return None

    def extract_conflict_implicit(self, raw: str) -> tuple[bool, list[str]]:
        """
        Check for implicit conflict awareness via keywords.

        Returns:
            (detected, list_of_matched_keywords)
        """
        raw_lower = raw.lower()
        matched = [kw for kw in _CONFLICT_KEYWORDS if kw in raw_lower]
        return len(matched) > 0, matched

    def extract_hesitation(self, raw: str) -> list[str]:
        """Check for hesitation/uncertainty markers."""
        raw_lower = raw.lower()
        return [kw for kw in _HESITATION_KEYWORDS if kw in raw_lower]

    def extract_reasoning_steps(self, raw: str) -> list[ReasoningStep]:
        """
        Extract reasoning steps from a response.
        Handles both ReAct format (Thought/Action/Observation)
        and Plan-Solve format (PLAN/SOLVE sections).
        """
        steps = []
        idx = 0

        # Try ReAct format first
        react_blocks = re.split(
            r'((?:Thought|Action|Observation)\s*:)',
            raw,
            flags=re.IGNORECASE,
        )
        if len(react_blocks) > 1:
            # Re-group: blocks alternate between separator and content
            i = 1
            while i < len(react_blocks) - 1:
                step_type = react_blocks[i].strip().rstrip(":").lower()
                content = react_blocks[i + 1].strip()
                if step_type in ("thought", "action", "observation"):
                    steps.append(ReasoningStep(
                        step_index=idx,
                        step_type=step_type,
                        content=content[:500],  # Truncate long content
                    ))
                    idx += 1
                i += 2
            if steps:
                return steps

        # Try Plan-Solve format
        plan_match = re.search(
            r'###\s*PLAN\s*\n(.*?)(?:###\s*SOLVE|$)',
            raw, re.IGNORECASE | re.DOTALL,
        )
        if plan_match:
            plan_text = plan_match.group(1).strip()
            steps.append(ReasoningStep(
                step_index=idx, step_type="plan", content=plan_text[:500],
            ))
            idx += 1

        solve_match = re.search(
            r'###\s*SOLVE\s*\n(.*?)$',
            raw, re.IGNORECASE | re.DOTALL,
        )
        if solve_match:
            solve_text = solve_match.group(1).strip()
            # Split solve section into individual steps
            solve_steps = re.split(r'\n(?=\d+\.)', solve_text)
            for ss in solve_steps:
                if ss.strip():
                    steps.append(ReasoningStep(
                        step_index=idx,
                        step_type="solve_step",
                        content=ss.strip()[:500],
                    ))
                    idx += 1

        return steps


# ── AgentRunner Abstract Base ────────────────────────────────────────────────

class AgentRunner(ABC):
    """
    Unified interface for all agent workflows.

    Subclasses implement:
      - build_prompt(sample, shuffle_seed) → full prompt string
      - parse_answer(raw_response) → Optional[bool]
      - name property → str

    The base class handles:
      - LLM calling, timing, token tracking
      - Answer extraction (via AnswerExtractor)
      - Conflict detection (explicit + implicit)
      - Result assembly as AgentResult
    """

    def __init__(
        self,
        llm_client: LLMClient,
        task_prompt: str = "",
        workflow_prompt: str = "",
    ):
        self.llm = llm_client
        self.task_prompt = task_prompt        # Shared Layer 1+3
        self.workflow_prompt = workflow_prompt  # Workflow-specific Layer 2
        self.extractor = AnswerExtractor()

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, sample: Sample, shuffle_seed: int = 0) -> AgentResult:
        """
        Execute the agent on a sample and return a structured result.

        Args:
            sample: The sample to reason about.
            shuffle_seed: Fact order randomization seed for this run.

        Returns:
            AgentResult with all metrics and traces.
        """
        # Build prompt
        prompt = self.build_prompt(sample, shuffle_seed)

        # Call LLM
        t0 = time.time()
        try:
            llm_result = self.llm.send(prompt=prompt)
        except Exception as e:
            logger.error(f"LLM call failed for sample {sample.id}: {e}")
            return AgentResult(
                sample_id=sample.id,
                workflow_name=self.name,
                run_index=0,  # overwritten by caller
                seed=shuffle_seed,
                final_answer=None,
                extracted_answer_text="",
                raw_response="",
                answered_uncertain=False,
                conflict_detected_explicit=None,
                conflict_detected_implicit=False,
                conflict_keywords_found=[],
                total_tokens=0,
                prompt_tokens=0,
                completion_tokens=0,
                latency_seconds=time.time() - t0,
                reasoning_steps=0,
                model_name=self.llm.model,
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=str(e),
            )

        latency = time.time() - t0

        # Extract answer
        answer, answer_text, method = self.extractor.extract_answer(llm_result.content)

        # Conflict detection
        conflict_explicit = self.extractor.extract_conflict_explicit(llm_result.content)
        conflict_implicit, conflict_kw = self.extractor.extract_conflict_implicit(llm_result.content)

        # Reasoning trace
        trace = self.extractor.extract_reasoning_steps(llm_result.content)

        # Assemble result
        result = AgentResult(
            sample_id=sample.id,
            workflow_name=self.name,
            run_index=0,  # overwritten by caller
            seed=shuffle_seed,
            final_answer=answer,
            extracted_answer_text=answer_text,
            raw_response=llm_result.content if not llm_result.content.startswith("mock:") else llm_result.content,
            answered_uncertain=(answer_text.upper() == "UNCERTAIN" if answer_text else False),
            conflict_detected_explicit=conflict_explicit,
            conflict_detected_implicit=conflict_implicit,
            conflict_keywords_found=conflict_kw,
            total_tokens=llm_result.total_tokens,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            latency_seconds=latency,
            reasoning_steps=len(trace),
            reasoning_trace=trace,
            model_name=self.llm.model,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return result

    # ── Subclass Interface ───────────────────────────────────────────────

    @abstractmethod
    def build_prompt(self, sample: Sample, shuffle_seed: int = 0) -> str:
        """Construct the full prompt for this workflow."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique workflow name for result labeling."""
        ...

    # ── Helpers for Subclasses ───────────────────────────────────────────

    def _render_sample(self, sample: Sample, shuffle_seed: int = 0) -> str:
        """Render sample facts and question as text."""
        from src.generation.renderer import NaturalLanguageRenderer
        renderer = NaturalLanguageRenderer(seed=shuffle_seed)
        return renderer.render_sample_prompt(sample, shuffle_seed=shuffle_seed)

"""ReAct (Reasoning + Acting) agent workflow.

Simulated ReAct: The LLM is prompted to produce Thought/Action/Observation
cycles in a single response without actual tool calling.

This is a known limitation — see the experimental caveats.
"""

from src.core.types import Sample
from src.agents.base import AgentRunner
from src.agents.llm_client import LLMClient


class ReActAgent(AgentRunner):
    """
    ReAct workflow: interleaved reasoning and action steps.

    Prompt structure:
      1. Shared task definition (Layer 1+3)
      2. ReAct-specific workflow instructions (Layer 2)
      3. Sample facts + question
    """

    def __init__(self, llm_client: LLMClient,
                 task_prompt: str = "",
                 workflow_prompt: str = ""):
        super().__init__(
            llm_client=llm_client,
            task_prompt=task_prompt,
            workflow_prompt=workflow_prompt,
        )

    @property
    def name(self) -> str:
        return "react"

    def build_prompt(self, sample: Sample, shuffle_seed: int = 0) -> str:
        """Build the full ReAct prompt."""
        sample_text = self._render_sample(sample, shuffle_seed)

        parts = []

        if self.task_prompt:
            parts.append(self.task_prompt)

        if self.workflow_prompt:
            parts.append(self.workflow_prompt)
        else:
            # Fallback ReAct instructions if no template provided
            parts.append(self._default_workflow_prompt())

        parts.append("---")
        parts.append(sample_text)
        parts.append("")
        parts.append("Begin your reasoning:")

        return "\n\n".join(parts)

    def _default_workflow_prompt(self) -> str:
        """Fallback ReAct instructions."""
        return """## Reasoning Method: Step-by-Step with Actions (ReAct)

You must reason using the ReAct method. For each step, use this EXACT format:

Thought: <your reasoning about what to do next>
Action: <one of: check_fact, deduce, verify>
Observation: <the result of your action>

Available actions:
- check_fact: Review a specific fact from the list
- deduce(A, relation, B): Try to derive whether A is related to B using logical rules
  (e.g., transitivity: if A∈B and B∈C, then A∈C)
- verify(statement): Check if a statement is consistent with known facts

Rules:
- Transitivity: If X is a member of Y, and Y is a member of Z, then X is a member of Z.
- A direct fact overrides a derived conclusion.
- Facts may be incomplete or inconsistent. You must detect and report any contradictions.

Continue until you reach a conclusion. Do not exceed 10 reasoning steps."""

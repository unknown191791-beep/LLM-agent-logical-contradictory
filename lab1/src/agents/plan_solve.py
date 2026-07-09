"""Plan-and-Solve agent workflow.

Two-phase reasoning:
  1. PLAN: Analyze the problem and devise a solution strategy.
  2. SOLVE: Execute the plan step by step.
"""

from src.core.types import Sample
from src.agents.base import AgentRunner
from src.agents.llm_client import LLMClient


class PlanSolveAgent(AgentRunner):
    """
    Plan-and-Solve workflow: plan first, then execute.

    Prompt structure:
      1. Shared task definition (Layer 1+3)
      2. Plan-Solve-specific workflow instructions (Layer 2)
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
        return "plan_solve"

    def build_prompt(self, sample: Sample, shuffle_seed: int = 0) -> str:
        """Build the full Plan-and-Solve prompt."""
        sample_text = self._render_sample(sample, shuffle_seed)

        parts = []

        if self.task_prompt:
            parts.append(self.task_prompt)

        if self.workflow_prompt:
            parts.append(self.workflow_prompt)
        else:
            parts.append(self._default_workflow_prompt())

        parts.append("---")
        parts.append(sample_text)
        parts.append("")
        parts.append("Begin your reasoning:")

        return "\n\n".join(parts)

    def _default_workflow_prompt(self) -> str:
        """Fallback Plan-and-Solve instructions."""
        return """## Reasoning Method: Plan-then-Solve

You must reason using the Plan-and-Solve method. Your response MUST have these two sections:

### PLAN
First, analyze the facts and create a step-by-step plan to answer the question. Your plan should:
1. Identify all relevant facts
2. Check if any facts conflict with each other
3. Determine which reasoning steps are needed
4. Note any assumptions you will make

### SOLVE
Execute your plan step by step. For each step:
1. State what you are doing
2. Show your work
3. Note the intermediate result

Rules:
- Transitivity: If X is a member of Y, and Y is a member of Z, then X is a member of Z.
- A direct fact overrides a derived conclusion.
- Facts may be incomplete or inconsistent. You must detect and report any contradictions."""

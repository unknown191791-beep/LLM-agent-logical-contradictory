# Agent Workflow Reasoning under Logical Conflict

Experimental framework for comparing ReAct vs Plan-and-Solve workflows on logical conflict reasoning tasks.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (skip for mock mode)
export ANTHROPIC_API_KEY=your-key-here

# Run unit tests
python -m pytest tests/test_generation.py -v

# Smoke test (no API calls required)
python tests/test_smoke.py

# Generate samples only
python run_experiment.py --generate-only

# Full experiment (mock mode, no API)
python run_experiment.py --mock

# Full experiment (real API calls)
python run_experiment.py

# Use a different model
python run_experiment.py --model-profile sonnet
```

## Project Structure

```
lab1/
├── config/
│   ├── experiment.yaml       # Experiment parameters (factor design)
│   ├── model.yaml            # LLM model configuration
│   └── prompts/              # Prompt templates (3-layer design)
├── src/
│   ├── core/                 # Data types and exceptions
│   ├── generation/           # Sample generation pipeline
│   ├── agents/               # Agent workflows (ReAct, Plan-and-Solve)
│   ├── experiment/           # Runner, checkpointing
│   ├── metrics/              # Accuracy, conflict detection, cost, etc.
│   └── analysis/             # Visualization and statistics
├── tests/                    # Unit and smoke tests
├── data/                     # Generated samples and results
├── outputs/                  # Figures and reports
├── run_experiment.py         # Main entry point
└── requirements.txt
```

## Experimental Design

### Factor Design (3×3×3)

| Parameter | Levels | Description |
|-----------|--------|-------------|
| chain_length | 2, 3, 4 | Number of reasoning hops |
| conflict_count | 0, 1, 2 | Number of negation facts |
| noise_count | 0, 3, 6 | Irrelevant facts added |

### Ground Truth

Majority voting (Plan A): GT = conclusion supported by more facts.

### Metrics

1. Accuracy (with 95% CI)
2. Conflict Detection Rate (explicit + implicit)
3. Token Cost
4. Latency
5. Reasoning Steps

### Auto-Generated Charts

- Accuracy vs Chain Length
- Accuracy vs Conflict Count
- Accuracy vs Noise Count

## Adding a New Workflow

1. Create a class extending `AgentRunner` in `src/agents/`
2. Implement `build_prompt()`, `name`
3. Add to `run_experiment.py` agents dict
4. Add workflow prompt template in `config/prompts/`

See `src/agents/react.py` for an example.

## Limitations

- Simulated ReAct (prompt-only, no real tool calls)
- Abstract entity names only (A, B, C, ...)
- Single model per experiment run
- Limited to transitivity_break conflicts (MVP)

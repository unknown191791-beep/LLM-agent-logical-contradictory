"""R3 rerun: Fixed Rule 2 + free-form reasoning + full traces."""

import json, os, sys, time, random
sys.path.insert(0, '.')
from src.agents.llm_client import LLMClient

# ─── Clean prompt (no bias, no caps emphasis) ───

TASK = """You are a logical reasoning agent. You will be given FACTS and a QUESTION about who likes whom.

Reason carefully using the logical rules below. Evaluate all evidence equally.
At the end, state your judgment as: JUDGMENT: YES, JUDGMENT: NO, or JUDGMENT: UNCERTAIN.
Then: CONFLICT: YES or CONFLICT: NO."""

RULES = """Logical Rules:
1. If X likes Y, and Y likes Z, then X likes Z.
2. If X dislikes Y, and Y likes Z, then X dislikes Z.
3. If X dislikes Y, then X does not like Y."""

REACT_WF = """## Method: ReAct

Format each step as:
Thought: <reasoning>
Action: <check_fact | deduce | verify>
Observation: <result>

Use the logical rules. Detect and report contradictions."""

PS_WF = """## Method: Plan-and-Solve

### PLAN
Analyze facts. Identify conflicts. Devise strategy.

### SOLVE
Execute step by step. Use the logical rules. Detect and report contradictions."""

# ─── Load samples ───
samples = []
with open('data/samples/predicate_samples.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))
print(f'Loaded {len(samples)} samples')

# ─── LLM ───
llm = LLMClient.from_yaml('config/model.yaml')
print(f'Model: {llm.model}')

total = len(samples) * 2 * 2
count = 0; t0 = time.time()
all_lines = []

for si, s in enumerate(samples):
    # Render facts
    rng = random.Random(si)
    shuffled = list(s['facts'])
    rng.shuffle(shuffled)
    facts_lines = ['Facts:']
    for subj, rel, obj, role in shuffled:
        text = f'{subj} likes {obj}.' if rel == 'likes' else f'{subj} dislikes {obj}.'
        facts_lines.append(f'- {text}')
    q_subj, q_obj = s['question']
    sample_text = '\n'.join(facts_lines) + f'\n\nQuestion: Does {q_subj} like {q_obj}?'

    for wf_name, wf_prompt in [('REACT', REACT_WF), ('PLAN_SOLVE', PS_WF)]:
        prompt = TASK + '\n\n' + RULES + '\n\n' + wf_prompt + '\n\n---\n\n' + sample_text + '\n\nBegin your reasoning:'

        for run_idx in range(2):
            result = llm.send(prompt=prompt)
            clean = result.content.encode('ascii', errors='replace').decode('ascii')

            import re
            judgment = None
            jm = re.search(r'JUDGMENT\s*:\s*(YES|NO|UNCERTAIN)', clean, re.IGNORECASE)
            if jm:
                j = jm.group(1).upper()
                judgment = True if j == 'YES' else (False if j == 'NO' else None)
            conflict = None
            cm = re.search(r'CONFLICT\s*:\s*(YES|NO)', clean, re.IGNORECASE)
            if cm:
                conflict = cm.group(1).upper() == 'YES'

            correct = None if judgment is None or s['gt'] is None else (judgment == s['gt'])
            status = 'OK' if correct else ('XX' if judgment is not None else '??')

            all_lines.append(f'{"="*80}')
            all_lines.append(f'[{s["id"]}] {wf_name} r{run_idx} | GT={s["gt"]} judgment={judgment} {status} | CONFLICT={conflict}')
            all_lines.append(f'S={s["support_len"]} O={s["oppose_len"]} | {s["description"]}')
            all_lines.append(f'{"="*80}')
            all_lines.append(clean)
            all_lines.append('')

            count += 1
            elapsed = time.time() - t0
            rate = count/elapsed if elapsed>0 else 0
            eta = (total-count)/rate if rate>0 else 0
            print(f'[{count}/{total}] {s["id"]} {wf_name} r{run_idx} {status} '
                  f'S={s["support_len"]}v{s["oppose_len"]} [{elapsed:.0f}s ETA {eta:.0f}s]', flush=True)
            time.sleep(0.3)

out_path = 'outputs/reports/predicate_v2_traces.txt'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_lines))
print(f'\nSaved to {out_path}')
llm.print_usage_summary()

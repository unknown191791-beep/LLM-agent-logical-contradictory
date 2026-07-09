"""
Rerun Round 1 + Round 2 with:
  - NO "direct fact overrides" instruction
  - Free-form reasoning (no forced ANSWER: format)
  - Full traces saved to file
"""

import json, os, sys, time, random
sys.path.insert(0, '.')
from src.agents.llm_client import LLMClient

# ─── Free-form task prompt (no ANSWER: constraint) ───
FREE_TASK = """You are a logical reasoning agent. You will be given a set of FACTS and a QUESTION.

Your task:
1. Reason through the facts carefully, step by step.
2. If any facts conflict, identify the contradiction explicitly.
3. Based on the facts, determine which answer is most likely to be correct.
4. At the very end, state your final judgment on a new line as: JUDGMENT: YES, JUDGMENT: NO, or JUDGMENT: UNCERTAIN.
5. On the next line, state: CONFLICT: YES or CONFLICT: NO.

Do NOT assume any fact is more reliable than another. Evaluate all evidence equally."""

# ─── Workflow instructions (minimal, no bias) ───
REACT_WF = """## Method: ReAct

Use this format for each step:
Thought: <your reasoning>
Action: <check_fact | deduce | verify>
Observation: <result>

Rules:
- Transitivity: If X is a member of Y, and Y is a member of Z, then X is a member of Z.
- Facts may be incomplete or inconsistent. Detect and report any contradictions."""

PS_WF = """## Method: Plan-and-Solve

Your response must have two sections:

### PLAN
Analyze the facts. Identify conflicts. Devise a strategy.

### SOLVE
Execute your plan step by step. Show your work.

Rules:
- Transitivity: If X is a member of Y, and Y is a member of Z, then X is a member of Z.
- Facts may be incomplete or inconsistent. Detect and report any contradictions."""

# ═══════════════════════════════════════════════════════════════
# SAMPLES
# ═══════════════════════════════════════════════════════════════

# Round 1: basic conflict samples
r1_samples = [
    {
        'id': 'R1_no_conflict',
        'round': 'Round1',
        'facts': [('A','in','B'), ('B','in','C')],
        'question': ('A','C'),
        'gt': True,
        'desc': 'No conflict: A in B, B in C. Q: A in C?'
    },
    {
        'id': 'R1_conflict_2v1',
        'round': 'Round1',
        'facts': [('A','in','B'), ('B','in','C'), ('A','not_in','C')],
        'question': ('A','C'),
        'gt': True,
        'desc': 'Conflict 2v1: A in B, B in C, A not-in C. Q: A in C?'
    },
    {
        'id': 'R1_conflict_3v1',
        'round': 'Round1',
        'facts': [('A','in','B'), ('B','in','C'), ('C','in','D'), ('A','not_in','D')],
        'question': ('A','D'),
        'gt': True,
        'desc': 'Conflict 3v1: A in B, B in C, C in D, A not-in D. Q: A in D?'
    },
    {
        'id': 'R1_conflict_4v1',
        'round': 'Round1',
        'facts': [('A','in','B'), ('B','in','C'), ('C','in','D'), ('D','in','E'), ('A','not_in','E')],
        'question': ('A','E'),
        'gt': True,
        'desc': 'Conflict 4v1: 4-hop chain, A not-in E. Q: A in E?'
    },
    {
        'id': 'R1_gt_false',
        'round': 'Round1',
        'facts': [('A','in','B'), ('B','in','C'), ('A','not_in','C'), ('A','not_in','C'), ('A','not_in','C')],
        'question': ('A','C'),
        'gt': False,
        'desc': 'GT=False: 2 support vs 3 oppose. Q: A in C?'
    },
    {
        'id': 'R1_gt_false_light',
        'round': 'Round1',
        'facts': [('A','in','B'), ('B','in','C'), ('A','not_in','C'), ('A','not_in','C')],
        'question': ('A','C'),
        'gt': False,
        'desc': 'GT=False: 2 support vs 2 oppose (tie-break by extra negation). Q: A in C?'
    },
]

# Round 2: diverse conflict types (1 sample per type)
r2_samples = [
    # TYPE 1: No conflict
    {'id': 'R2_no_conflict', 'round': 'Round2',
     'facts': [('A','in','B'), ('B','in','C'), ('C','in','D')],
     'question': ('A','D'), 'gt': True,
     'desc': 'Type1: No conflict. 3-hop chain.'},

    # TYPE 2: Conclusion negation (original type)
    {'id': 'R2_conclusion_neg', 'round': 'Round2',
     'facts': [('A','in','B'), ('B','in','C'), ('C','in','D'), ('A','not_in','D')],
     'question': ('A','D'), 'gt': True,
     'desc': 'Type2: Conclusion negation. A not-in D at end of chain.'},

    # TYPE 3: Direct contradiction (both A in C and A not-in C)
    {'id': 'R2_direct_contradiction', 'round': 'Round2',
     'facts': [('A','in','B'), ('B','in','C'), ('A','in','C'), ('A','not_in','C')],
     'question': ('A','C'), 'gt': True,
     'desc': 'Type3: Direct contradiction. BOTH A in C AND A not-in C as facts.'},

    # TYPE 4: Flat contradiction
    {'id': 'R2_flat', 'round': 'Round2',
     'facts': [('A','in','C'), ('A','not_in','C')],
     'question': ('A','C'), 'gt': None,
     'desc': 'Type4: Flat contradiction. Only A in C vs A not-in C. No chain.'},

    # TYPE 5: Early negation
    {'id': 'R2_early_neg', 'round': 'Round2',
     'facts': [('A','not_in','B'), ('A','in','B'), ('B','in','C'), ('C','in','D')],
     'question': ('B','D'), 'gt': True,
     'desc': 'Type5: Early negation A not-in B. Q: B in D? (negation off-target).'},

    # TYPE 6: Middle negation (pre)
    {'id': 'R2_mid_pre', 'round': 'Round2',
     'facts': [('A','in','B'), ('B','not_in','C'), ('B','in','C'), ('C','in','D')],
     'question': ('A','B'), 'gt': True,
     'desc': 'Type6: Middle negation B not-in C. Q: A in B? (before break).'},

    # TYPE 7: Middle negation (post)
    {'id': 'R2_mid_post', 'round': 'Round2',
     'facts': [('A','in','B'), ('B','not_in','C'), ('B','in','C'), ('C','in','D')],
     'question': ('C','D'), 'gt': True,
     'desc': 'Type7: Middle negation B not-in C. Q: C in D? (after break).'},

    # TYPE 8: Negation dominant (GT=False)
    {'id': 'R2_neg_dominant', 'round': 'Round2',
     'facts': [('A','in','B'), ('B','in','C'), ('A','not_in','C'), ('A','not_in','C'),
               ('A','not_in','C'), ('A','not_in','C')],
     'question': ('A','C'), 'gt': False,
     'desc': 'Type8: Negation dominant. 2 support vs 4 oppose. GT=False.'},

    # TYPE 9: Irrelevant negation
    {'id': 'R2_irrelevant', 'round': 'Round2',
     'facts': [('A','in','B'), ('B','in','C'), ('X','not_in','Y')],
     'question': ('A','C'), 'gt': True,
     'desc': 'Type9: Irrelevant negation X not-in Y. Q: A in C? (negation off-target).'},

    # TYPE 10: Tangential negation
    {'id': 'R2_tangential', 'round': 'Round2',
     'facts': [('A','in','B'), ('B','in','C'), ('C','in','D'), ('B','not_in','D')],
     'question': ('A','D'), 'gt': True,
     'desc': 'Type10: Tangential negation B not-in D. Q: A in D? (B vs A, different subject).'},
]

# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

llm = LLMClient.from_yaml('config/model.yaml')
print(f'Model: {llm.model}')

all_samples = r1_samples + r2_samples
all_lines = []
total = len(all_samples) * 2 * 2  # 16 samples * 2 workflows * 2 repeats = 64
count = 0
t0 = time.time()

for si, s in enumerate(all_samples):
    # Build sample text
    rng = random.Random(si)
    shuffled = list(s['facts'])
    rng.shuffle(shuffled)
    facts_lines = ['Facts:']
    for subj, pred, obj in shuffled:
        text = f'{subj} is a member of {obj}.' if pred == 'in' else f'{subj} is not a member of {obj}.'
        facts_lines.append(f'- {text}')
    q_subj, q_obj = s['question']
    sample_text = '\n'.join(facts_lines) + f'\n\nQuestion: Is {q_subj} a member of {q_obj}?'

    for wf_name, wf_prompt in [('REACT', REACT_WF), ('PLAN_SOLVE', PS_WF)]:
        prompt = FREE_TASK + '\n\n' + wf_prompt + '\n\n---\n\n' + sample_text + '\n\nBegin your reasoning:'

        for run_idx in range(2):
            result = llm.send(prompt=prompt)
            clean = result.content.encode('ascii', errors='replace').decode('ascii')

            # Extract judgment
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

            correct = None
            if judgment is not None and s['gt'] is not None:
                correct = (judgment == s['gt'])
            status = 'OK' if correct else ('XX' if judgment is not None else '??')

            all_lines.append(f'{"="*80}')
            all_lines.append(f'[{s["id"]}] {wf_name} run={run_idx} | GT={s["gt"]} judgment={judgment} {status} | CONFLICT={conflict}')
            all_lines.append(f'[{s["desc"]}]')
            all_lines.append(f'[ROUND: {s["round"]}]')
            all_lines.append(f'{"="*80}')
            all_lines.append(clean)
            all_lines.append('')

            count += 1
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            eta = (total - count) / rate if rate > 0 else 0
            print(f'[{count}/{total}] {s["id"]:30s} {wf_name} r{run_idx} {status} '
                  f'[{elapsed:.0f}s ETA {eta:.0f}s]', flush=True)
            time.sleep(0.3)

# Save
out_path = 'outputs/reports/free_reasoning_traces.txt'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_lines))

print(f'\nSaved {len(all_lines)} lines to {out_path}')
llm.print_usage_summary()

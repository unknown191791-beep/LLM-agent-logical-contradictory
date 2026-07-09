"""Run predicate logic experiment on DeepSeek."""

import json, os, sys, time
sys.path.insert(0, '.')

from src.agents.llm_client import LLMClient
from src.agents.base import AnswerExtractor

# Load samples
samples = []
with open('data/samples/predicate_samples.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))
print(f'Loaded {len(samples)} predicate logic samples')

# Load prompts
def load(name):
    path = f'config/prompts/{name}.txt'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

task_p = load('base_task')
react_p = load('react_workflow')
ps_p = load('plan_solve_workflow')

# ─── Prompt Builder ───────────────────────────────────────────────

RULES_TEXT = """
Logical Rules:
1. Like transitivity: If X likes Y, and Y likes Z, then X likes Z.
2. Dislike propagation: If X dislikes Y, and Y likes Z, then X dislikes Z.
3. Mutual exclusion: If X dislikes Y, then X does NOT like Y.

Use these rules to reason from the facts."""

def build_prompt(s, workflow):
    # Render facts
    facts_lines = ['Facts:']
    for subj, rel, obj, role in s['facts']:
        if rel == 'likes':
            text = f'{subj} likes {obj}.'
        else:
            text = f'{subj} dislikes {obj}.'
        facts_lines.append(f'- {text}')

    q_subj, q_obj = s['question']
    question = f'Question: Does {q_subj} like {q_obj}?'

    # Build parts
    parts = [task_p if task_p else 'You are a logical reasoning agent.']
    parts.append(RULES_TEXT)
    parts.append('---')
    parts.append('\n'.join(facts_lines))
    parts.append(question)
    parts.append('')
    parts.append('Reason step by step. End with ANSWER: YES or ANSWER: NO or ANSWER: UNCERTAIN.')
    parts.append('Then: CONFLICT_DETECTED: YES or CONFLICT_DETECTED: NO.')

    prompt = '\n\n'.join(parts)

    if workflow == 'react':
        wf_extra = '\n\nUse the ReAct format: Thought, then Action, then Observation for each step.'
    else:
        wf_extra = '\n\nUse Plan-and-Solve format: first write a PLAN section, then a SOLVE section.'
    prompt += wf_extra

    return prompt

# ─── LLM ──────────────────────────────────────────────────────────

llm = LLMClient.from_yaml('config/model.yaml')
print(f'Model: {llm.model} | Provider: {llm.provider}')

ext = AnswerExtractor()
total = len(samples) * 2 * 2
out_path = 'data/results/experiment_predicate.jsonl'
os.makedirs('data/results', exist_ok=True)

count = 0
t0 = time.time()

with open(out_path, 'w', encoding='utf-8') as out:
    for si, s in enumerate(samples):
        for workflow in ['react', 'plan_solve']:
            for run_idx in range(2):
                prompt = build_prompt(s, workflow)
                llm_result = llm.send(prompt=prompt)

                answer, answer_text, method = ext.extract_answer(llm_result.content)
                conflict_explicit = ext.extract_conflict_explicit(llm_result.content)
                conflict_implicit, conflict_kw = ext.extract_conflict_implicit(llm_result.content)

                correct = None
                if answer is not None and s['gt'] is not None:
                    correct = (answer == s['gt'])

                rec = {
                    'sample_id': s['id'],
                    'workflow': workflow,
                    'run': run_idx,
                    'answer': answer,
                    'gt': s['gt'],
                    'correct': correct,
                    'conflict_explicit': conflict_explicit,
                    'conflict_implicit': conflict_implicit,
                    'conflict_keywords': conflict_kw,
                    'tokens': llm_result.total_tokens,
                    'latency': llm_result.latency_seconds,
                    'support_len': s['support_len'],
                    'oppose_len': s['oppose_len'],
                    'support_oppose': f"{s['supporting']}:{s['opposing']}",
                    'tags': s['tags'],
                    'description': s['description'],
                    'raw_preview': llm_result.content[:200],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                count += 1

                elapsed = time.time() - t0
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total - count) / rate if rate > 0 else 0
                status = 'OK' if correct else ('??' if answer is None else ('--' if s['gt'] is None else 'XX'))
                cd = 'CD' if conflict_explicit else '--'
                print(f'[{count}/{total}] {s["id"]} {workflow:11s} r{run_idx} '
                      f'ans={str(answer):5s} gt={s["gt"]} {status} {cd} '
                      f'slvso={s["support_len"]}v{s["oppose_len"]} '
                      f'tok={llm_result.total_tokens} [{elapsed:.0f}s ETA {eta:.0f}s]',
                      flush=True)

print(f'\nDone! {count} runs in {time.time()-t0:.0f}s')
llm.print_usage_summary()
print(f'Saved to {out_path}')

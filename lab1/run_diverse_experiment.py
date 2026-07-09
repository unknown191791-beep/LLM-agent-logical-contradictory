"""Run experiment on diverse sample set with DeepSeek."""

import json, os, sys, time
sys.path.insert(0, '.')
from src.agents.llm_client import LLMClient
from src.generation.renderer import NaturalLanguageRenderer

# Load samples
samples = []
with open('data/samples/diverse_samples.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))
print(f'Loaded {len(samples)} diverse samples')

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

# Build prompts manually (bypassing Sample object for flexibility)
def build_prompt(s, workflow):
    """Build prompt from raw sample dict."""
    # Render facts
    facts_lines = ['Facts:']
    for subj, pred, obj in s['facts']:
        if pred == 'in':
            text = f'{subj} is a member of {obj}.'
        else:
            text = f'{subj} is not a member of {obj}.'
        facts_lines.append(f'- {text}')

    q_subj, q_obj = s['question']
    question = f'Question: Is {q_subj} a member of {q_obj}?'

    sample_text = '\n'.join(facts_lines) + '\n\n' + question

    parts = []
    if task_p:
        parts.append(task_p)
    if workflow == 'react':
        parts.append(react_p if react_p else 'Use Thought/Action/Observation format.')
    else:
        parts.append(ps_p if ps_p else 'Use PLAN and SOLVE sections.')
    parts.append('---')
    parts.append(sample_text)
    parts.append('')
    parts.append('Begin your reasoning:')

    return '\n\n'.join(parts)

# LLM
llm = LLMClient.from_yaml('config/model.yaml')
print(f'Model: {llm.model} | Provider: {llm.provider}')

# Run
total = len(samples) * 2 * 2  # 38 samples * 2 workflows * 2 repeats = 152
out_path = 'data/results/experiment_diverse.jsonl'
os.makedirs('data/results', exist_ok=True)

count = 0
t0 = time.time()

with open(out_path, 'w', encoding='utf-8') as out:
    for si, s in enumerate(samples):
        for workflow in ['react', 'plan_solve']:
            for run_idx in range(2):
                prompt = build_prompt(s, workflow)
                llm_result = llm.send(prompt=prompt)

                # Extract answer
                from src.agents.base import AnswerExtractor
                ext = AnswerExtractor()
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
                    'conflict_type': s['conflict_type'],
                    'conflict_explicit': conflict_explicit,
                    'conflict_implicit': conflict_implicit,
                    'conflict_keywords': conflict_kw,
                    'tokens': llm_result.total_tokens,
                    'latency': llm_result.latency_seconds,
                    'chain_length': s['chain_length'],
                    'noise_count': s['noise_count'],
                    'support_oppose': f"{s['supporting']}:{s['opposing']}",
                    'tags': s['tags'],
                    'raw_preview': llm_result.content[:150],
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
                      f'type={s["conflict_type"][:25]:25s} tok={llm_result.total_tokens} '
                      f'[{elapsed:.0f}s ETA {eta:.0f}s]', flush=True)

print(f'\nDone! {count} runs in {time.time()-t0:.0f}s')
llm.print_usage_summary()
print(f'Saved to {out_path}')

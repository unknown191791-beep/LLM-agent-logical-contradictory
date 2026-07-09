"""Generate ~50 diverse samples with multiple conflict types."""

import json, random, os, sys

random.seed(42)
samples = []
sid = 0

# Noise entity pool
NOISE_POOL = [('P','Q'), ('X','Y'), ('U','V'), ('W','Z'), ('R','S'),
              ('K','L'), ('M','N'), ('H','I'), ('J','T'), ('O','G'),
              ('AA','BB'), ('CC','DD'), ('EE','FF'), ('GG','HH')]

def render_facts(facts):
    lines = ['Facts:']
    for f in facts:
        pred = 'is a member of' if f[1] == 'in' else 'is not a member of'
        lines.append(f'- {f[0]} {pred} {f[2]}.')
    return '\n'.join(lines)

def render_question(q):
    return f'Question: Is {q[0]} a member of {q[1]}?'

def make_sample(conflict_type, chain_len, noise_cnt, facts, q_subj, q_obj, gt, rationale, tags):
    global sid
    # Add noise
    all_f = list(facts)
    for i in range(noise_cnt):
        s, o = NOISE_POOL[(sid + i) % len(NOISE_POOL)]
        all_f.append((s, 'in', o))
    random.shuffle(all_f)

    supporting = sum(1 for f in facts if f[1] == 'in')
    opposing = sum(1 for f in facts if f[1] == 'not_in')

    return {
        'id': f's{sid:04d}',
        'facts': all_f,
        'question': (q_subj, q_obj),
        'gt': gt,
        'rationale': rationale,
        'conflict_type': conflict_type,
        'chain_length': chain_len,
        'noise_count': noise_cnt,
        'supporting': supporting,
        'opposing': opposing,
        'tags': tags,
    }

def emit(conflict_type, chain_len, noise_cnt, facts, q_subj, q_obj, gt, rationale, tags):
    global sid
    s = make_sample(conflict_type=conflict_type, chain_len=chain_len, noise_cnt=noise_cnt,
                    facts=facts, q_subj=q_subj, q_obj=q_obj, gt=gt, rationale=rationale, tags=tags)
    samples.append(s)
    sid += 1
    return s

# ═══════════════════════════════════════════════════════════════
# TYPE 1: No Conflict (pure chain) - 9 samples
# ═══════════════════════════════════════════════════════════════
for cl in [2, 3, 4]:
    for nc in [0, 2, 4]:
        ents = [chr(65+i) for i in range(cl+1)]
        facts = [(ents[i], 'in', ents[i+1]) for i in range(cl)]
        emit('no_conflict', cl, nc, facts, ents[0], ents[-1], True,
             f'Pure chain: {cl} facts all support transitivity. GT=True.',
             ['baseline'])

# ═══════════════════════════════════════════════════════════════
# TYPE 2: Conclusion Negation (original) - 9 samples
# ═══════════════════════════════════════════════════════════════
for cl in [2, 3, 4]:
    for nc in [0, 3, 5]:
        ents = [chr(65+i) for i in range(cl+1)]
        facts = [(ents[i], 'in', ents[i+1]) for i in range(cl)]
        facts.append((ents[0], 'not_in', ents[-1]))
        emit('conclusion_negation', cl, nc, facts, ents[0], ents[-1], True,
             f'{cl} chain facts support via transitivity, 1 direct negation opposes. GT=True ({cl}:1).',
             ['negation_at_end', 'original_type'])

# ═══════════════════════════════════════════════════════════════
# TYPE 3: Direct Self-Contradiction - 6 samples
# Both A in Z and A not-in Z as direct facts + full chain
# ═══════════════════════════════════════════════════════════════
for cl in [2, 3]:
    for nc in [0, 3]:
        ents = [chr(65+i) for i in range(cl+1)]
        facts = [(ents[i], 'in', ents[i+1]) for i in range(cl)]
        facts.append((ents[0], 'in', ents[-1]))       # direct YES
        facts.append((ents[0], 'not_in', ents[-1]))   # direct NO
        emit('direct_contradiction', cl, nc, facts, ents[0], ents[-1], True,
             f'Direct YES+NO conflict. {cl} transitive + 1 direct support vs 1 oppose. GT=True ({cl+1}:1).',
             ['direct_both', 'tiebreaker'])

    # Also: direct contradiction WITHOUT chain support
    ents = [chr(65+i) for i in range(cl+1)]
    facts = [(ents[0], 'in', ents[-1]), (ents[0], 'not_in', ents[-1])]
    emit('direct_contradiction_flat', cl, 0, facts, ents[0], ents[-1], None,
         '1 direct YES vs 1 direct NO. No chain support. TIE - unresolvable.',
         ['direct_both', 'flat_tie'])

# ═══════════════════════════════════════════════════════════════
# TYPE 4: Position-Varied Negation - 6 samples (cl >= 4 only)
# ═══════════════════════════════════════════════════════════════
for cl in [4, 5]:
    ents = [chr(65+i) for i in range(cl+1)]  # A..E or A..F

    # Early negation: break at first link A not-in B, Q about B to Z
    facts = [(ents[i], 'in', ents[i+1]) for i in range(cl)]
    facts.append((ents[0], 'not_in', ents[1]))
    emit('negation_early', cl, 2, facts, ents[1], ents[-1], True,
         f'Early negation (A not-in B). Q: B to Z (unbroken path, {cl-1} hops). GT=True.',
         ['position_early', 'partial_path'])

    # Middle negation: break at cl//2, Q about path that skips the break
    mid = cl // 2  # e.g., cl=4: mid=2, break at B not-in C
    facts2 = [(ents[i], 'in', ents[i+1]) for i in range(cl)]
    facts2.append((ents[mid-1], 'not_in', ents[mid]))
    # Q: A to entity just before break (this path is unbroken)
    emit('negation_middle', cl, 2, facts2, ents[0], ents[mid-1], True,
         f'Middle negation at pos {mid}. Q: A to entity before break (unbroken). GT=True.',
         ['position_middle', 'partial_path'])

    # Q: entity after break to Z (also unbroken)
    emit('negation_middle2', cl, 2, facts2, ents[mid], ents[-1], True,
         f'Middle negation at pos {mid}. Q: entity after break to Z (unbroken). GT=True.',
         ['position_middle', 'partial_path'])

# ═══════════════════════════════════════════════════════════════
# TYPE 5: GT=False (negation dominates) - 6 samples
# ═══════════════════════════════════════════════════════════════
for cl in [2, 3]:
    for extra_neg in [1, 3]:  # extra negations beyond chain_length
        ents = [chr(65+i) for i in range(cl+1)]
        facts = [(ents[i], 'in', ents[i+1]) for i in range(cl)]
        total_neg = cl + extra_neg
        for _ in range(total_neg):
            facts.append((ents[0], 'not_in', ents[-1]))
        emit('negation_dominant', cl, 0, facts, ents[0], ents[-1], False,
             f'{cl} support vs {total_neg} oppose. GT=False ({cl}:{total_neg}).',
             ['gt_false', 'negation_wins'])

# ═══════════════════════════════════════════════════════════════
# TYPE 6: Irrelevant Negation - 4 samples
# Negation that does NOT conflict with the question
# ═══════════════════════════════════════════════════════════════
for cl in [3, 4]:
    ents = [chr(65+i) for i in range(cl+1)]
    facts = [(ents[i], 'in', ents[i+1]) for i in range(cl)]
    # Add a negation about unrelated entities
    facts.append(('X', 'not_in', 'Y'))
    emit('irrelevant_negation', cl, 2, facts, ents[0], ents[-1], True,
         'Unrelated negation (X not-in Y) does not affect A to Z transitivity. GT=True.',
         ['irrelevant', 'distractor'])

    # Add a negation that SEEMS related but isn't about the question target
    facts2 = [(ents[i], 'in', ents[i+1]) for i in range(cl)]
    facts2.append((ents[1], 'not_in', ents[-1]))  # B not-in Z, but Q is A in Z?
    emit('tangential_negation', cl, 2, facts2, ents[0], ents[-1], True,
         f'Tangential negation (B not-in Z) does not directly contradict A in Z. GT=True (cl:0).',
         ['tangential', 'distractor'])

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f'Generated {len(samples)} samples')

from collections import Counter
types = Counter(s['conflict_type'] for s in samples)
gts = Counter(s['gt'] for s in samples)
print('\nBy conflict type:')
for t, c in types.most_common():
    tags = samples[[s['conflict_type'] for s in samples].index(t)]['tags']
    print(f'  {t:35s}: {c:>2}  tags={tags}')

print(f'\nGT: True={gts[True]}, False={gts[False]}, None={gts.get(None,0)}')

# Save
os.makedirs('data/samples', exist_ok=True)
path = 'data/samples/diverse_samples.jsonl'
with open(path, 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
print(f'\nSaved to {path}')

# Print preview
print('\n' + '='*70)
print('PREVIEW (first of each type)')
print('='*70)
printed = set()
for s in samples:
    t = s['conflict_type']
    if t not in printed:
        printed.add(t)
        gt_val = s['gt']
        rat = s['rationale']
        print(f'\n[{t}]  GT={gt_val}')
        print(render_facts(s['facts']))
        print(render_question(s['question']))
        print(f'  Rationale: {rat}')
        print()

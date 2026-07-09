"""
Predicate Logic Experiment: both support and opposition can use transitive chains.

Rules:
  R1: Likes(x,y) AND Likes(y,z) -> Likes(x,z)
  R2: Dislikes(x,y) AND Likes(y,z) -> Dislikes(x,z)
  R3: Dislikes(x,y) -> NOT Likes(x,y)

Both "Likes(A,Z)" and "Dislikes(A,Z)" can be derived via chains.
This decouples directness from the support/opposition dimension.
"""

import json, os, sys, time, random
sys.path.insert(0, '.')
from src.agents.llm_client import LLMClient
from src.agents.base import AnswerExtractor

random.seed(42)
samples = []
sid = 0

# ─── Sample Generator ────────────────────────────────────────────────

_ENTITIES = [chr(65+i) for i in range(26)]  # A-Z

def make_sample(support_len, oppose_len, noise_cnt, gt, description, tags):
    """
    support_len: number of Likes facts forming the support chain (A likes Z via k hops)
    oppose_len:  number of facts forming the opposition chain
                 (1 Dislikes + (n-1) Likes via n hops)
    """
    global sid

    # Allocate entities
    # Support chain: A -> B -> C -> ... -> Z  (support_len+1 entities)
    # Opposition chain: A -> X -> Y -> ... -> Z  (oppose_len+1 entities, A and Z shared)

    entities_used = 0

    # Support chain
    s_chain = []
    for i in range(support_len + 1):
        s_chain.append(_ENTITIES[entities_used])
        entities_used += 1
    A = s_chain[0]
    Z = s_chain[-1]

    # Opposition chain must start at A and end at Z
    # Uses 1 Dislikes(A, O1) + (oppose_len-1) Likes(O1, O2, ..., Z)
    # Need oppose_len entities for the opposition path (A is shared, Z is shared)

    facts = []
    supporting = 0
    opposing = 0

    # Support facts: all Likes along the support chain
    for i in range(support_len):
        facts.append((s_chain[i], 'likes', s_chain[i+1], 'support'))
        supporting += 1

    # Opposition chain
    if oppose_len > 0:
        o_chain = [A]  # starts at A
        for i in range(oppose_len):
            o_chain.append(_ENTITIES[entities_used])
            entities_used += 1
        o_chain[-1] = Z  # last must be Z

        # First link: Dislikes(A, o_chain[1])
        facts.append((A, 'dislikes', o_chain[1], 'oppose'))
        opposing += 1
        # Remaining links: Likes(o_chain[1], o_chain[2]), ..., Likes(o_chain[n-1], Z)
        for i in range(1, oppose_len):
            facts.append((o_chain[i], 'likes', o_chain[i+1], 'oppose'))
            opposing += 1

    # Noise: irrelevant Likes/Dislikes
    noise_pool = [(_ENTITIES[(entities_used+j*2)%26], _ENTITIES[(entities_used+j*2+1)%26])
                  for j in range(20)]
    noise_facts = []
    for i in range(noise_cnt):
        s, o = noise_pool[(sid*3 + i) % len(noise_pool)]
        if s in (A, Z) or o in (A, Z):
            continue  # avoid confusion
        rel = 'likes' if i % 2 == 0 else 'dislikes'
        noise_facts.append((s, rel, o, 'noise'))

    all_facts = facts + noise_facts
    random.shuffle(all_facts)

    sample = {
        'id': f'p{sid:04d}',
        'facts': all_facts,
        'question': (A, Z),
        'gt': gt,
        'support_len': support_len,
        'oppose_len': oppose_len,
        'supporting': supporting,
        'opposing': opposing,
        'noise_cnt': noise_cnt,
        'description': description,
        'tags': tags,
    }
    samples.append(sample)
    sid += 1
    return sample

# ─── Generate ~30 Diverse Samples ──────────────────────────────────

# Type 1: No conflict (pure Likes chain)
for sl in [2, 3, 4]:
    for nc in [0, 2]:
        make_sample(sl, 0, nc, True,
            f'Pure Likes chain ({sl} hops). GT=True.', ['no_conflict'])

# Type 2: Symmetric: support and opposition both via chains
# support=2, oppose=1 -> GT=True (2:1)
for sl, ol in [(2,1), (3,1), (3,2), (4,2), (4,3)]:
    if sl > ol:
        gt = True
    elif ol > sl:
        gt = False
    else:
        gt = None  # tie
    make_sample(sl, ol, 2, gt,
        f'Support={sl} hops vs Oppose={ol} hops. GT={gt} ({sl}:{ol}).',
        ['chain_vs_chain'])

# Type 3: Opposition wins
for sl, ol in [(1,2), (2,3), (1,3)]:
    make_sample(sl, ol, 1, False,
        f'Opposition chain longer ({sl}:{ol}). GT=False.', ['oppose_wins'])

# Type 4: Tie
for sl in [2, 3]:
    make_sample(sl, sl, 2, None,
        f'Equal chains ({sl}:{sl}). GT=unresolvable.', ['tie'])

# Type 5: Support dominant
for sl, ol in [(3,1), (4,1), (4,2)]:
    make_sample(sl, ol, 1, True,
        f'Support chain longer ({sl}:{ol}). GT=True.', ['support_wins'])

# ─── Summary ─────────────────────────────────────────────────────

print(f'Generated {len(samples)} predicate logic samples')
from collections import Counter
gt_dist = Counter(s['gt'] for s in samples)
print(f'GT: True={gt_dist[True]}, False={gt_dist[False]}, None={gt_dist[None]}')
by_type = Counter(s['tags'][0] for s in samples)
for t, c in by_type.most_common():
    print(f'  {t}: {c}')

os.makedirs('data/samples', exist_ok=True)
with open('data/samples/predicate_samples.jsonl', 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

# ─── Print Preview ────────────────────────────────────────────────

def render(s):
    lines = ['Facts:']
    for subj, rel, obj, role in s['facts']:
        if rel == 'likes':
            text = f'{subj} likes {obj}.'
        else:
            text = f'{subj} dislikes {obj}.'
        lines.append(f'- {text}')
    q_subj, q_obj = s['question']
    lines.append(f'')
    lines.append(f'Question: Does {q_subj} like {q_obj}?')
    lines.append(f'Rules: Liking is transitive. If someone dislikes X and Y likes X, that person dislikes Y too. If someone dislikes Y, they do not like Y.')
    return '\n'.join(lines)

print('\n' + '='*70)
print('SAMPLE PREVIEW')
print('='*70)
for s in samples:
    if s['id'] in ('p0000', 'p0006', 'p0010', 'p0016', 'p0020', 'p0025'):
        sid_val = s['id']
        desc = s['description']
        print(f'\n[{sid_val}] {desc}')
        print(render(s))
        print()

print(f'\nSaved {len(samples)} samples. Ready to run experiment.')

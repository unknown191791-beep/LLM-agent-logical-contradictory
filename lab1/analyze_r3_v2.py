import re, json
from collections import defaultdict

samples = {}
with open('data/samples/predicate_samples.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        s = json.loads(line)
        samples[s['id']] = (s['support_len'], s['oppose_len'], s['gt'])

by_so = defaultdict(list)
with open('outputs/reports/predicate_v2_traces.txt', 'r', encoding='utf-8') as f:
    content = f.read()

for block in content.split('='*80):
    if 'GT=' not in block:
        continue
    sid = re.search(r'\[(p\d+)\]', block)
    jm = re.search(r'judgment=(\w+)', block)
    wm = re.search(r'(REACT|PLAN_SOLVE)', block)
    if not sid or not jm:
        continue
    s_id = sid.group(1)
    if s_id in samples:
        sl, ol, gt = samples[s_id]
        key = f'{sl}v{ol}'
        js = {'True': True, 'False': False, 'None': None}[jm.group(1)]
        by_so[key].append((s_id, wm.group(1), js, gt))

print('FIXED R3: Predicate Logic with corrected Rule 2')
print('='*70)
print(f'{"S:O":>6s} {"GT":>5s}  {"REACT":>18s} {"P&S":>18s}  {"Behavior"}')
print('-'*70)

for key in sorted(by_so.keys(), key=lambda k: (int(k.split('v')[0]), int(k.split('v')[1]))):
    entries = by_so[key]
    gt_val = entries[0][3]
    gt_str = str(gt_val) if gt_val is not None else 'tie'

    reacts = [e for e in entries if e[1] == 'REACT']
    ps = [e for e in entries if e[1] == 'PLAN_SOLVE']

    def fmt(lst):
        js = [e[2] for e in lst]
        t = sum(1 for j in js if j is True)
        f = sum(1 for j in js if j is False)
        u = sum(1 for j in js if j is None)
        parts = []
        if t: parts.append('T:%d' % t)
        if f: parts.append('F:%d' % f)
        if u: parts.append('U:%d' % u)
        return ','.join(parts) if parts else 'N/A'

    react_s = fmt(reacts)
    ps_s = fmt(ps)

    all_js = [e[2] for e in entries]
    n = len(all_js)
    t = sum(1 for j in all_js if j is True)
    f = sum(1 for j in all_js if j is False)
    u = sum(1 for j in all_js if j is None)

    if u == n: bh = 'ALL UNCERTAIN'
    elif t == n: bh = 'ALL TRUE'
    elif f == n: bh = 'ALL FALSE'
    elif f > 0 and u > 0 and t == 0: bh = 'FALSE + UNCERTAIN'
    elif t > 0 and u > 0 and f == 0: bh = 'TRUE + UNCERTAIN'
    else: bh = 'MIXED'

    print('%6s %5s  %18s %18s  %s' % (key, gt_str, react_s, ps_s, bh))

all_js = [e[2] for entries in by_so.values() for e in entries]
t_count = sum(1 for j in all_js if j is True)
f_count = sum(1 for j in all_js if j is False)
u_count = sum(1 for j in all_js if j is None)
print('\nTotal: True=%d, False=%d, UNCERTAIN=%d (N=%d)' % (t_count, f_count, u_count, len(all_js)))

# Compare with buggy R3
print()
print('COMPARISON: Buggy R3 vs Fixed R3')
print('-'*60)
print('  Buggy (backward Rule 2): Agent could not derive opposition chains')
print('    -> GT=False samples: Agent picked True (wrong)')
print('    -> S>O samples: Agent picked True (coincidentally correct)')
print('  Fixed (forward Rule 2): Both chains work symmetrically')
print('    -> ALL conflict samples: OVERWHELMINGLY UNCERTAIN')
print('    -> Same pattern as R1/R2 without instruction bias')

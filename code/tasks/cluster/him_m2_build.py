#!/usr/bin/env python3
"""HIM 2026-11-13: rerun every L16 kappa=.27 task with the retrained model L16_k027_buf_s2 (isis 61668, seed 2).
Writes only him_tc/m2_20261113/lines_{cpu,imh}.txt. Nothing old is overwritten: outputs go to new dirs
(<dir>_m2) and carry a new stem (..._k027m2...), so pregen dirs and resume files never collide with the s0 runs.
Sources: isis task lines (isis_lines.tsv, file<TAB>line) + HIM stage lines; per output the HIM line wins, else the
newest isis task file. GPU IMH lines are rewritten to the HIM CPU-sharded form (same args as the HIM s0 lines)."""
import glob, re, sys
B = '/lustre/him/him-mam/mqian'; W = f'{B}/him_tc/m2_20261113'
OLD = 'L16_k0.27_buf_s0'; NEW = 'revision_runs/phi4_buffer/L16_k027_buf_s2'
WANT = ([f'longchain_min/E2_L16_k027_{m}' for m in ('rt', 'rtp12', 'imhG', 'rtp2', 'rtp4', 'rtp6', 'rtp8', 'rtp10', 'rtp14')]
        + [f'logz_chains/E2_L16_k027_{m}' for m in ('rt', 'rtp12', 'imh')] + ['raw_snis/E2_L16_k027']
        + [f'longchain_min/T16_L{L}_k027_{m}' for L in (4, 8, 12, 24, 32) for m in ('rt', 'rtp12', 'imhG')]
        + [f'raw_snis/T16_L{L}_k027' for L in (4, 8, 12, 24, 32)]
        + [f'longchain_min/T16_L{L}_k027_rtp12' for L in (48, 64)]
        + [f'pilot_bigL/T16_L{L}_k027_{m}' for L in (48, 64, 96) for m in ('rt', 'rtp12')])

def key(l):
    O = re.search(r'O=/tmp/mq/revision_runs/(\w+);', l); o = re.search(r'--out \$O/(\S+)\.json', l)
    return f'{O.group(1)}/{o.group(1)}' if O and o else None

cand = {}                                         # key -> (priority, line)
for f, l in (x.rstrip('\n').split('\t', 1) for x in open(f'{W}/isis_lines.tsv')):
    l = re.sub(r'^# HIM [0-9-]+ ', '', l).replace('/media/scratch/mqian/', '/tmp/mq/')
    k = key(l)
    if k in WANT and (k not in cand or f > cand[k][0]): cand[k] = (f, l)
for f in sorted(glob.glob(f'{B}/logs/stage-*/lines.txt')):
    for l in open(f):
        l = re.sub(r'/tmp/mq_\d+/', '/tmp/mq/', l.rstrip('\n'))
        if OLD in l and key(l) in WANT: cand[key(l)] = ('~him', l)   # '~' sorts after 'tasks_'
miss = [k for k in WANT if k not in cand]
assert not miss, miss

def shard(l, L, nmax, batch, ns):
    """GPU IMH line -> HIM CPU-sharded line (template of the HIM s0 imhG lines)."""
    k = key(l); d, s = k.split('/')
    head = l.split('[ -f $O/')[0]
    args = (f'--kind phi --run-dir $D --ckpt kernel_final.pt --L {L} --kappa 0.27 --method imh --imh-pregen --pregen-batch {batch} '
            f'--n-chains 4 --n-start 10000 --n-max {nmax} --device cpu --threads 1 --seed 20261109 --pregen-dir $PD')
    P = '/tmp/mq/envs/sps/bin/python'; PD = f'/tmp/mq/revision_runs/pregen/{s}'
    return (head.replace('O=', 'export CUDA_VISIBLE_DEVICES=; O=', 1) + f'[ -f $O/{s}.json ] || {{ [ -f $D/kernel_final.pt ] || exit 3; '
            f'mkdir -p {PD}; PD={PD}; for i in $(seq 0 {ns - 1}); do {P} longchain.py {args} --pregen-shard $i/{ns} '
            f'--out $PD/shard.json > $PD.sh$i.log 2>&1 & done; {P} longchain.py {args} --pregen-wait 3600 --out $O/{s}.json > '
            f'$O/{s}.log 2>&1; rc=$?; kill $(jobs -p) 2>/dev/null; wait; [ $rc -eq 0 ] && rm -rf $PD $PD.sh*.log; exit $rc; }}')

out = {'cpu': [], 'imhB': [], 'imhS': []}
for k in WANT:
    src, l = cand[k]
    if '--device cuda' in l:
        L = int(re.search(r'_L(\d+)_', k).group(1))
        l = shard(l, L, 3200000, 8192, 7) if L == 16 else shard(l, L, 800000, 2048, 15)
    # the model: both the revision_runs and the results_sync fallback path -> the new s2 model
    l = re.sub(r'/tmp/mq/revision_(runs|code/results_sync/models)/phi4_buffer/' + re.escape(OLD), '/tmp/mq/' + NEW, l)
    l = re.sub(r'(/tmp/mq/revision_runs/(longchain_min|logz_chains|raw_snis|pilot_bigL))\b', r'\1_m2', l)
    l = re.sub(r'\b(E2|T16|buf)_L(\d+)_k027(?=[_.])', r'\1_L\2_k027m2', l)
    assert OLD not in l and 'k027_' not in l.replace('k027_buf_s2', '') and '--device cuda' not in l, (k, l[:300])
    kind = 'cpu' if '--imh-pregen' not in l else 'imhB' if re.search(r'--L (24|32) ', l) else 'imhS'
    out[kind].append(l); print(kind, k, src)
for kind, ls in out.items():
    open(f'{W}/lines_{kind}.txt', 'x').write('\n'.join(ls) + '\n')
print({k: len(v) for k, v in out.items()})

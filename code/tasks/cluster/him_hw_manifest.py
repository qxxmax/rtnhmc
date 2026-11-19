#!/usr/bin/env python3
"""Hardware manifest of the HIMSTER3 runs (2026-11-14): for every output (model dir D=... or --out $O/<name>.json) of
every staged job, record job id, node, CPU threads of the line (OMP_NUM_THREADS), allocated cores, line exit code,
and the wall time from the line log timestamps if available. Read-only; prints JSON to stdout.
Usage: python3 him_hw_manifest.py > hw_manifest_him.json   (run on the HIM login node in /lustre/him/him-mam/mqian)"""
import glob
import json
import os
import re
import subprocess

B = '/lustre/him/him-mam/mqian'
jobs = {}
for f in glob.glob(f'{B}/logs/*.out') + glob.glob(f'{B}/prest_cpu/logs/*.out'):
    try:
        t = open(f, errors='replace').read()
    except OSError:
        continue
    m = re.search(r'^job (\d+) (\S+) lines (\d+) P (\d+)', t, re.M)
    if m:
        rc = {int(i): int(r) for i, r in re.findall(r' line (\d+) rc (\d+)', t)}
        jobs[m.group(1)] = dict(log=os.path.basename(f), node=m.group(2), n_lines=int(m.group(3)), P=int(m.group(4)), rc=rc)


def alloc(j):
    try:
        o = subprocess.run(['sacct', '-j', j, '-X', '-n', '-P', '-o', 'AllocCPUS,Elapsed,Start,End,Partition'],
                           capture_output=True, text=True, timeout=30).stdout.strip().split('|')
        return dict(alloc_cpus=int(o[0]), elapsed=o[1], start=o[2], end=o[3], partition=o[4])
    except Exception:
        return {}


out = []
for j, info in sorted(jobs.items()):
    p = f'{B}/logs/stage-{j}/lines.txt'
    if not os.path.exists(p):
        continue
    acc = alloc(j)
    for i, l in enumerate(open(p).read().split('\n'), 1):
        if not l.strip() or l.startswith('true'):
            continue
        thr = re.search(r'OMP_NUM_THREADS=(\d+)', l)
        train = 'train_cond_phi4' in l or ('cmcd_phi4.py' in l and '--load' not in l)   # D= is an output only here
        targets = (re.findall(r'(?<![A-Z])D=(?:\$O/|/tmp/mq_\d+/)(\S+?);', l) if train else []) + \
            re.findall(r'--out \$O/(\S+?\.json)(?:\s|$)', l)
        O = re.search(r'O=/tmp/mq_\d+/(\S+?);', l)
        for tg in targets:
            path = (O.group(1) + '/' + tg) if (O and not tg.startswith('revision_runs')) else tg
            out.append(dict(output=path, kind='training' if train and not tg.endswith('.json') else 'chain/eval', job=j, line=i, cluster='HIMSTER3', device='cpu', node=info['node'],
                            threads=int(thr.group(1)) if thr else None, lines_in_job=info['n_lines'],
                            parallel_lines=info['P'], rc=info['rc'].get(i), **acc))
print(json.dumps(out, indent=0))

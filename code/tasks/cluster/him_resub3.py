#!/usr/bin/env python3
"""Round-N resubmission with 4 threads per rt / rtp chain (2026-11-13). Same selection and state as him_resub2.py, but
cpu parts hold 7 lines x 4 cores, rt / rtp lines run with --threads 4 (thread test 92992, L32 rt: 1 / 2 / 4 / 8 threads
-> 8.2 / 4.5 / 3.3 / 4.6 min, outputs identical; resume does not check --threads), and every cpu part also carries the
retrained L16 kappa=.27 models (him_tc/m2_20261113/models.tar) for the *_m2 lines. L >= 24 IMH lines get 31 pregen shards on 32 cores.
Fixes to the him_resub.py selection: (a) a failed line counts only in the job that owns it now (the r2b failures were
rerun in r2c / r2d with the identical line text, which made those running lines look todo); (b) skip the thread tests.
Usage: him_resub3.py TAG [--submit]  (never overwrites: TAG dir must not exist)."""
T_ = '/lustre/him/him-mam/mqian/him_tc'


def patch(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


base = open(f'{T_}/him_resub.py').read().split("if not submit: sys.exit()")[0]
base = patch(base, "for suf in ('.ckpt.pt', '.log'):", "for suf in ('.ckpt.pt', '.log', '.part.npz'):")
base = patch(base, "failed.add(re.sub(r'/tmp/mq_\\d+/', '/tmp/mq/', ls_[int(i) - 1]))",
             "failed.add((m.group(1), norm(re.sub(r'/tmp/mq_\\d+/', '/tmp/mq/', ls_[int(i) - 1]))))")
# (c) r4f+ lines carry --threads 4 / 31 shards: map them back to the original text so ownership and done-state match
base = patch(base, "l = re.sub(r'/tmp/mq_\\d+/', '/tmp/mq/', l.rstrip('\\n'))",
             "l = norm(re.sub(r'/tmp/mq_\\d+/', '/tmp/mq/', l.rstrip('\\n')))")
base = patch(base, "l not in failed", "(j, l) not in failed")
base = patch(base, "'revision_runs/cc' in l", "'revision_runs/cc' in l or 'thrtest' in l or 'udb_train_' in l or 'nhmc_cpu_pilot' in l")
# (e) CPU K-scan trainings (revision_runs/kscan_cpu/<model>) are handled exactly like the l6train lines
base = patch(base, "D = re.search(r'D=/tmp/mq/(revision_runs/l6train/\\S+?);', l)",
             "D = re.search(r'D=/tmp/mq/(revision_runs/(?:l6train|kscan_cpu)/\\S+?);', l)")
base = patch(base, "    if rel.startswith('revision_runs/l6train/'):", "    if rel.startswith(('revision_runs/l6train/', 'revision_runs/kscan_cpu/')):")
# (d) cmcd_phi4.py training lines (udb_train_*) write a directory, not $O/<name>.json: resubmitted by hand (--resume)

body = open(f'{T_}/him_resub2.py').read().split("exec(src)\n", 1)[1]
body = patch(body, "    if rel.startswith('revision_runs/l6train/'):", "    if rel.startswith(('revision_runs/l6train/', 'revision_runs/kscan_cpu/')):")
body = patch(body, "SPEC = {'cpu': (28, 1), 'imhB': (2, 16),", "SPEC = {'cpu': (7, 4), 'imhB': (2, 32),")
body = patch(body, "for l in todo: groups[kind(l)].append(l)",
             "for l in todo: groups[kind(l)].append(thr4(l) if kind(l) == 'cpu' else shard31(l) if kind(l) == 'imhB' else l)")
body = patch(body, "            for rel, s in sorted(st.items()): tf.add(s, arcname=rel)",
             "            for rel, s in sorted(st.items()): tf.add(s, arcname=rel)\n"
             "            if k in ('cpu', 'imhB'):\n"
             "                with tarfile.open(f'{T}/m2_20261113/models.tar') as mt:\n"
             "                    for m in mt.getmembers(): tf.addfile(m, mt.extractfile(m) if m.isfile() else None)")


def thr4(l):
    import re
    if re.search(r'longchain\.py .*--method rtp?(:\d+)? ', l):
        l = l.replace('--threads 1 ', '--threads 4 ').replace('OMP_NUM_THREADS=1 MKL_NUM_THREADS=1',
                                                             'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4')
    return l


def shard31(l):
    """L >= 24 IMH: 31 pregen shards + the chain on 32 cores (was 15 + 1); the batch store does not depend on sharding."""
    assert l.count('$(seq 0 14)') == 1 and l.count('--pregen-shard $i/15 ') == 1, l[:200]
    return l.replace('$(seq 0 14)', '$(seq 0 30)').replace('--pregen-shard $i/15 ', '--pregen-shard $i/31 ')


def norm(l):
    return (l.replace('--threads 4 ', '--threads 1 ').replace('OMP_NUM_THREADS=4 MKL_NUM_THREADS=4', 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1')
            .replace('$(seq 0 30)', '$(seq 0 14)').replace('--pregen-shard $i/31 ', '--pregen-shard $i/15 '))


exec(base)
exec(body)

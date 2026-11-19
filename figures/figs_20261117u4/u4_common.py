"""2026-11-17 u4 (copy of figs_20261117u3/u3_common.py; only OUT and SUF changed). u3 docstring: copy of figs_20261117u2/u2_common.py; only OUT, SUF and the inventory path changed, data access identical.
u2 docstring: shared data access, jackknife lookup and provenance recording for the *_u2.py figure scripts.
Read-only on all data; writes only into figs_20261117u2/ (provenance json in prov/).

Chain-file lookup (basename -> path), identical priority to the u scripts (fig_scaling files()):
  REV/longchain_min, him_figdata_20261116 longchain_min (job-sorted), him_figdata_20261117 longchain_min,
  him_figdata_20261116 pilot_bigL, him_figdata_20261117 pilot_bigL, iqcb_figdata_20261116/longchain_min,
  iqcb_figdata_20261117/longchain_min; first hit wins.  All copies of one basename were checked (inventory_u2.json):
  they agree in n, criteria_met, acceptance and tau except pilot (n=1000) vs long chain, where the long chain wins (as in u).
Trace files: next to the chosen json, else next to any copy with the same seed and n (him_figdata_20261117 carries the
traces of the L>=48 transfer chains), else figs/data/traces_extra.

Jackknife (jackknife_20261117, method of jk_cells.py): tau error = delete-one-chain jackknife of the chain mean
(tau_absM_jk_chain_mean; the plotted value stays the paper statistic, median over 4 chains of max(Sokal,BM)/2);
acceptance error = 20-block-per-chain jackknife of the accept flags when traces exist, else delete-one-chain.
Cells not in the jackknife cache are computed here with jk_cells.analyze (same code)."""
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

REV = os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision'
OUT = REV + '/figs_20261117u4'
U2 = REV + '/figs_20261117u2'
UST = REV + '/figs_20261117u'
JK = REV + '/jackknife_20261117'
sys.path.insert(0, UST)
import nhmc_style as NS  # noqa: E402,F401
sys.path.insert(0, JK)
import jk_cells  # noqa: E402

SUF = '_20261117u4'
H16 = REV + '/him_figdata_20261116/himout'
H17 = REV + '/him_figdata_20261117/himout'
IQ16 = REV + '/iqcb_figdata_20261116'
IQ17 = REV + '/iqcb_figdata_20261117'
EXTRA = REV + '/figs/data/traces_extra'

DIRS = ([REV + '/longchain_min'] + sorted(glob.glob(H16 + '/*/revision_runs/longchain_min'))
        + sorted(glob.glob(H17 + '/*/revision_runs/longchain_min')) + sorted(glob.glob(H16 + '/*/revision_runs/pilot_bigL'))
        + sorted(glob.glob(H17 + '/*/revision_runs/pilot_bigL')) + [IQ16 + '/longchain_min', IQ17 + '/longchain_min'])
_FILES = {}
for _d in DIRS:
    for _f in glob.glob(_d + '/*.json'):
        _FILES.setdefault(os.path.basename(_f), _f)
INV = json.load(open(U2 + '/inventory_u2.json'))


def pdf(name):
    return f'{OUT}/{name}{SUF}.pdf'


def find(base):
    return _FILES.get(base)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def job_of(p):
    m = re.search(r'/himout/([^/]+)/', p)
    if m:
        return 'HIM ' + m[1]
    if '/iqcb_figdata' in p:
        return 'iqcb'
    return 'local'


def traces_path(jf):
    if jf is None:
        return None
    b = jf[:-5] + '_traces.npz'
    if os.path.exists(b):
        return b
    d = json.load(open(jf))
    for c in INV.get(os.path.basename(jf), []):
        if c['seed'] == d.get('seed') and c['n'] == d.get('n_transitions') and c['traces']:
            return c['path'].replace('/Users/mxq/phi4_bench', os.environ.get('NHMC_ROOT', '.'))[:-5] + '_traces.npz'
    e = os.path.join(EXTRA, os.path.basename(b))
    return e if os.path.exists(e) else None


# ------------------------------------------------------------------ jackknife cells
_JKC = {}
for _g in ('E2', 'buf', 'T16', 'T4', 'T8', 'T12', 'E2tr16'):
    _p = f'{JK}/cells_{_g}.json'
    if os.path.exists(_p):
        _JKC.update(json.load(open(_p)))
_NEW = {}


def cellinfo(jf):
    """jackknife record of one chain json (cache from jackknife_20261117 if it is the same chain, else computed)."""
    if jf is None:
        return None
    key = os.path.basename(jf)[:-5]
    d = json.load(open(jf))
    r = _JKC.get(key)
    if r and r.get('n_transitions') == d.get('n_transitions') and abs(r['acc'] - np.mean([c['acceptance'] for c in d['chains']])) < 1e-12:
        need_tr = traces_path(jf) is not None and r.get('acc_jk_block') is None
        if not need_tr:
            r = dict(r)
            r['jk_source'] = 'jackknife_20261117/cells cache'
            return r
    if key in _NEW:
        return _NEW[key]
    tp = traces_path(jf)
    src = jf
    if tp is not None and not os.path.exists(jf[:-5] + '_traces.npz'):
        src = tp[:-len('_traces.npz')] + '.json'           # same chain (same seed, n), copy that carries the traces
    r = jk_cells.analyze(src)
    r['jk_source'] = 'computed here (jk_cells.analyze)'
    _NEW[key] = r
    return r


def chain_summary(jf):
    """paper statistics plus jackknife errors and bookkeeping for one chain json."""
    if jf is None:
        return None
    d = json.load(open(jf))
    cs = d['chains']
    acc = float(np.mean([c['acceptance'] for c in cs]))
    tau = float(np.median([max(c['absM']['tau_sokal'], c['absM'].get('tau_batch_means') or 0) / 2 for c in cs]))
    r = cellinfo(jf)
    npost = (int(d['n_transitions']) - int(d.get('burn', 0))) * len(cs)
    nacc = r.get('n_acc') if r else None
    acc_err = r.get('acc_jk_block') if r and r.get('acc_jk_block') is not None else (r.get('acc_jk_chain') if r else None)
    pilot = '/pilot_bigL' in jf or int(d['n_transitions']) <= 1000
    if nacc is not None and nacc < 30:                     # few accepts: Poisson error
        acc_err = np.sqrt(max(nacc, 1)) / npost
    return dict(path=jf, acc=acc, tau=tau, acc_err=acc_err, tau_err=r.get('tau_absM_jk_chain_mean') if r else None,
                conv=bool(d.get('criteria_met')), n=int(d['n_transitions']), burn=int(d.get('burn', 0)), nch=len(cs),
                npost=npost, nacc=nacc, pilot=pilot, ul95=3.0 / npost, seed=d.get('seed'),
                tau_sign=r.get('tau_sign') if r else None, tau_sign_err=r.get('tau_sign_jk_chain_mean') if r else None,
                jk_source=r.get('jk_source') if r else None)


# ------------------------------------------------------------------ provenance
class Prov:
    def __init__(self, fig):
        self.fig = fig
        self.rows = []
        self.notes = []
        self.seen = set()

    def add(self, panel, path, rule='', note=''):
        key = (panel, path, rule)
        if key in self.seen:
            return
        self.seen.add(key)
        row = dict(panel=panel, path=path, rule=rule, note=note)
        if path and os.path.isfile(path):
            row.update(md5=md5(path), mtime=time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path))),
                       job=job_of(path))
            if path.endswith('.json'):
                try:
                    d = json.load(open(path))
                except Exception:  # noqa: BLE001
                    d = None
                if isinstance(d, dict) and d.get('chains'):
                    row.update(n=d.get('n_transitions'), n_chains=len(d['chains']), criteria_met=d.get('criteria_met'),
                               burn=d.get('burn'), model=d.get('model_id'))
                elif isinstance(d, list) and d and isinstance(d[0], dict) and 'n' in d[0]:
                    row.update(n=d[0].get('n'))
                elif isinstance(d, dict) and 'n' in d:
                    row.update(n=d.get('n'))
        self.rows.append(row)

    def note(self, s):
        self.notes.append(s)

    def dump(self, extra=None):
        os.makedirs(OUT + '/prov', exist_ok=True)
        json.dump(dict(fig=self.fig, rows=self.rows, notes=self.notes, extra=extra or {}),
                  open(f'{OUT}/prov/{self.fig}.json', 'w'), indent=1, default=str)


# ------------------------------------------------------------------ Var(log w) at kappa = 0.27 (stage scan, scaling_var c)
KSCAN_RULE = ('seed rule (u2): seed 0 from iqcb phi4_buffer (GPU pipeline: L*_k27_K{T}_s0, and the T=16 buffer models '
              'L*_k0.27_buf_s0); seeds 1, 2 from HIM kscan_cpu; if a seed-0 model exists only on HIM (CPU) it is used; '
              'T=8 seed 0 exists twice (iqcb 2.1564, HIM 2.1411): iqcb is used; iqcb L16_k027_buf_s1/s2 (GPU seeds 1, 2) '
              'are excluded by the rule; lf10 (10 leapfrog substeps) runs are not part of the T scan; '
              'eval_N262144.json var_W (262144 forward paths)')


def _ev(p):
    d = json.load(open(p))
    return d[0] if isinstance(d, list) else d


def kscan_points():
    """(L, T) -> list of dict(seed, src, path, var, new) under KSCAN_RULE, plus excluded candidates."""
    cand = []
    for p in sorted(glob.glob(H17 + '/*/revision_runs/kscan_cpu/L*_k27_K*_s*/eval_N262144.json')):
        m = re.match(r'L(\d+)_k27_K(\d+)_s(\d)$', p.split('/')[-2])
        if m:
            cand.append(dict(L=int(m[1]), T=int(m[2]), seed=int(m[3]), src='HIM kscan_cpu (CPU)', path=p))
    for p in sorted(glob.glob(IQ17 + '/phi4_buffer/L*_k27_K*_s*/eval_N262144.json')):
        m = re.match(r'L(\d+)_k27_K(\d+)_s(\d)$', p.split('/')[-2])
        if m:
            cand.append(dict(L=int(m[1]), T=int(m[2]), seed=int(m[3]), src='iqcb phi4_buffer (GPU)', path=p))
    seen = set()
    for root in (IQ17 + '/phi4_buffer', IQ16 + '/models_phi4_buffer', IQ16 + '/runs_phi4_buffer'):
        for p in sorted(glob.glob(root + '/L*_k0.27_buf_s0/eval_N262144.json')) + sorted(glob.glob(root + '/L*_k027_buf_s[12]/eval_N262144.json')):
            m = re.match(r'L(\d+)_k0\.?27_buf_s(\d)$', p.split('/')[-2])
            if m and (int(m[1]), int(m[2])) not in seen:
                seen.add((int(m[1]), int(m[2])))
                cand.append(dict(L=int(m[1]), T=16, seed=int(m[2]), src='iqcb phi4_buffer (GPU, T=16 buffer model)', path=p))
    for c in cand:
        c['var'] = float(_ev(c['path'])['var_W'])
    cells, excl = {}, []
    for c in cand:
        gpu = c['src'].startswith('iqcb')
        if c['seed'] == 0:
            other = [o for o in cand if o is not c and (o['L'], o['T'], o['seed']) == (c['L'], c['T'], 0)]
            if not gpu and any(o['src'].startswith('iqcb') for o in other):
                c['why'] = 'seed 0 also on iqcb (GPU): iqcb copy used, this HIM CPU model excluded'
                excl.append(c)
                continue
        elif gpu:
            c['why'] = 'GPU seed %d: rule takes seeds 1, 2 from HIM kscan_cpu only' % c['seed']
            excl.append(c)
            continue
        cells.setdefault((c['L'], c['T']), []).append(c)
    # what u read (fig_scaling_20261117u: HIM kscan_cpu of 20261116 + iqcb T16 buffer s0; iqcb K-scan regex never matched)
    old = set()
    for p in glob.glob(H16 + '/*/revision_runs/kscan_cpu/*/eval_N262144.json'):
        m = re.match(r'L(\d+)_k27_K(\d+)_s(\d)$', p.split('/')[-2])
        if m:
            old.add((int(m[1]), int(m[2]), int(m[3]), 'HIM'))
    for (L, T), v in cells.items():
        for c in v:
            c['new'] = not ((L, T, c['seed'], 'HIM') in old and not c['src'].startswith('iqcb')) and not (T == 16 and c['seed'] == 0)
        v.sort(key=lambda c: c['seed'])
    return dict(cells=cells, excluded=excl, rule=KSCAN_RULE)

"""Shared helpers of the table scripts (Zenodo record, 2026-11-18). All paths are relative to NHMC_ROOT."""
import json
import os

import numpy as np

ROOT = os.environ.get('NHMC_ROOT', '.')
REV = ROOT + '/results_sync/revision'


def load(f):
    d = json.load(open(f))
    return d[0] if isinstance(d, list) else d


def tau(c, obs='absM'):
    """tau_int = max(Sokal, batch means) / 2 of one chain (the chain files store g = 2 tau_int)"""
    o = c[obs]
    return max(o['tau_sokal'], o.get('tau_batch_means') or 0) / 2


def cell(f):
    """(acceptance mean over chains, median tau_int(|m|), median tau_int(sign), criteria_met, args) of one chain file"""
    d = load(f)
    cs = d['chains']
    return dict(acc=float(np.mean([c['acceptance'] for c in cs])), tau=float(np.median([tau(c) for c in cs])),
                tau_sign=float(np.median([tau(c, 'sign') for c in cs])) if 'sign' in cs[0] else None,
                ok=bool(d.get('criteria_met')), args=d.get('args', {}), file=os.path.relpath(f, ROOT))


def first(paths):
    for p in paths:
        if os.path.exists(p):
            return cell(p)
    return None


def g2(x):
    """two significant figures as in the paper tables: one decimal below 10, integer above"""
    return f'{x:.1f}' if x < 10 else f'{x:.0f}'

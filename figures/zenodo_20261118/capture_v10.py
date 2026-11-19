"""Redraw one figure script and capture its plotted data (Zenodo check, 2026-11-18).

Usage (NHMC_ROOT must point to the extracted record):
    python capture_v10.py <outdir> <figure script> [script arguments]

The script runs in its own directory with nhmc_style.save wrapped: every figure is written to <outdir>/pdf/<file>.pdf
(not into the script directory) and its plotted data to <outdir>/figdata/<figure>.json, in the format of
figs_20261117u4/capture_u4.py: per axes the position, limits, scales, labels, Line2D vertices (with marker, colours,
gid), LineCollection segments (error bars), PolyCollection vertices (fill_between) and patch vertices (axhspan / hist).
The script's own provenance file (prov/<figure>.json, path and md5 of every input) is written as usual."""
import json
import os
import pathlib
import re
import runpy
import sys

import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

outd = pathlib.Path(sys.argv[1]).resolve()
script = pathlib.Path(sys.argv[2]).resolve()
sys.argv = [str(script)] + sys.argv[3:]
ROOT = os.environ['NHMC_ROOT']
sys.path.insert(0, ROOT + '/results_sync/revision/figs_20261117u')
sys.path.insert(0, str(script.parent))
os.chdir(script.parent)
import nhmc_style as NS  # noqa: E402

_orig = NS.save
(outd / 'figdata').mkdir(parents=True, exist_ok=True)
(outd / 'pdf').mkdir(parents=True, exist_ok=True)


def grab(fig):
    axes = []
    for a in fig.axes:
        d = dict(pos=[round(v, 4) for v in a.get_position().bounds], xlim=list(a.get_xlim()), ylim=list(a.get_ylim()),
                 xscale=a.get_xscale(), yscale=a.get_yscale(), xlabel=a.get_xlabel(), ylabel=a.get_ylabel(),
                 lines=[], ebars=[], polys=[], patches=[])
        for l in a.get_lines():
            x = np.asarray(l.get_xdata(), float)
            y = np.asarray(l.get_ydata(), float)
            d['lines'].append(dict(x=x.tolist(), y=y.tolist(), marker=str(l.get_marker()), ls=str(l.get_linestyle()),
                                   color=matplotlib.colors.to_hex(l.get_color()),
                                   mfc=str(l.get_markerfacecolor()), mec=str(l.get_markeredgecolor()), ms=l.get_markersize(),
                                   z=l.get_zorder(), gid=l.get_gid()))
        for c in a.collections:
            if isinstance(c, LineCollection):
                d['ebars'].append(dict(gid=c.get_gid(), segs=[np.asarray(s_, float).tolist() for s_ in c.get_segments()]))
            elif isinstance(c, PolyCollection):
                d['polys'].append(dict(gid=c.get_gid(), paths=[np.asarray(p.vertices, float).tolist() for p in c.get_paths()]))
        for p in a.patches:
            if isinstance(p, Patch):
                v = p.get_patch_transform().transform(p.get_path().vertices)
                d['patches'].append(dict(gid=p.get_gid(), kind=type(p).__name__, v=np.asarray(v, float).tolist()))
        axes.append(d)
    return axes


def save(fig, path):
    p = pathlib.Path(str(path))
    name = re.sub(r'_2026111\d+u\d*$', '', p.stem)
    json.dump(grab(fig), open(outd / 'figdata' / f'{name}.json', 'w'))
    _orig(fig, str(outd / 'pdf' / p.name))
    print('captured', name, '->', p.name)


NS.save = save

TRACE = os.environ.get('NHMC_TRACE')        # optional: log every file under NHMC_ROOT opened for reading (OPEN), every path looked up but absent (MISSING),
if TRACE:                                   # and every glob pattern with its number of matches
    import builtins
    import glob as _glob
    _log = open(TRACE, 'a')
    _isf, _ex, _isd, _open, _glb, _pex = os.path.isfile, os.path.exists, os.path.isdir, builtins.open, _glob.glob, pathlib.Path.exists

    def _chk(p, ok):
        p = str(p)
        if not ok and p.startswith(ROOT):
            _log.write(f'MISSING\t{os.path.relpath(p, ROOT)}\n')
        return ok

    os.path.isfile = lambda p: _chk(p, _isf(p))
    os.path.exists = lambda p: _chk(p, _ex(p))
    pathlib.Path.exists = lambda self, **k: _chk(self, _pex(self, **k))

    def _open2(f, *a, **k):
        if isinstance(f, (str, pathlib.Path)):
            if _chk(f, _ex(f)) and str(f).startswith(ROOT) and not any(c in (a[0] if a else k.get('mode', 'r')) for c in 'wax+'):
                _log.write(f'OPEN\t{os.path.relpath(str(f), ROOT)}\n')
        return _open(f, *a, **k)
    builtins.open = _open2

    def _glob2(pat, *a, **k):
        r = _glb(pat, *a, **k)
        if str(pat).startswith(ROOT):
            _log.write(f'GLOB\t{os.path.relpath(str(pat), ROOT)}\t{len(r)}\n')
        return r
    _glob.glob = _glob2
    _pglob = pathlib.Path.glob

    def _pglob2(self, pat, **k):
        r = list(_pglob(self, pat, **k))
        if str(self).startswith(ROOT):
            _log.write(f'GLOB\t{os.path.relpath(str(self), ROOT)}/{pat}\t{len(r)}\n')
        return iter(r)
    pathlib.Path.glob = _pglob2
runpy.run_path(str(script), run_name='__main__')

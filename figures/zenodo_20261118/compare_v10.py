"""Compare redrawn figures with reference figures (Zenodo check, 2026-11-18).

    python compare_v10.py data  <a.json> <b.json>     plotted data of two captures (capture_v10.py / capture_u4.py format)
    python compare_v10.py pixel <a.pdf>  <b.pdf> [dpi] rendered pixels of two PDFs (needs PyMuPDF)
    python compare_v10.py prov  <a.json> <b.json>     input files (path relative to NHMC_ROOT / phi4_bench) and md5

data : every axes, line, error-bar segment, polygon and patch in drawing order; reports the largest absolute and relative
       difference of the coordinates (NaN = NaN) and any change of count, limits, scales or labels.
pixel: both PDFs rendered at the same dpi in RGB; reports page size, the fraction of pixels whose channels differ by more
       than 8/255 and the largest difference.
Exit status 0 if identical within 1e-9 relative (data), 0 differing pixels (pixel), same inputs and md5 (prov)."""
import json
import re
import sys

import numpy as np


def _arr(v):
    return np.asarray(v, float)


def _diff(a, b):
    a, b = _arr(a), _arr(b)
    if a.shape != b.shape:
        return np.inf, np.inf
    m = ~(np.isnan(a) & np.isnan(b))
    if not m.any():
        return 0.0, 0.0
    d = np.abs(a[m] - b[m])
    d = np.where(np.isnan(d), np.inf, d)
    s = np.maximum(np.abs(a[m]), np.abs(b[m]))
    rel = np.where(s > 0, d / np.where(s > 0, s, 1), d)
    return float(d.max()), float(rel.max())


def data(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    issues, mabs, mrel, n = [], 0.0, 0.0, 0
    if len(A) != len(B):
        return [f'axes count {len(A)} vs {len(B)}'], np.inf, np.inf, 0
    for i, (a, b) in enumerate(zip(A, B)):
        for k in ('xscale', 'yscale', 'xlabel', 'ylabel'):
            if a[k] != b[k]:
                issues.append(f'ax{i} {k}: {a[k]!r} vs {b[k]!r}')
        for k in ('xlim', 'ylim'):
            da, dr = _diff(a[k], b[k])
            if dr > 1e-9:
                issues.append(f'ax{i} {k}: {a[k]} vs {b[k]}')
        pairs = []
        for k in ('lines', 'ebars', 'polys', 'patches'):
            la, lb = a.get(k, []), b.get(k, [])
            if len(la) != len(lb):
                issues.append(f'ax{i} {k} count {len(la)} vs {len(lb)}')
                continue
            for x, y in zip(la, lb):
                if k == 'lines':
                    pairs += [(x['x'], y['x']), (x['y'], y['y'])]
                    for s in ('marker', 'ls', 'color', 'mfc', 'mec'):
                        if x[s] != y[s]:
                            issues.append(f'ax{i} line style {s}: {x[s]} vs {y[s]}')
                elif k == 'ebars':
                    if len(x['segs']) != len(y['segs']):
                        issues.append(f'ax{i} error-bar segment count')
                    pairs += list(zip(x['segs'], y['segs']))
                elif k == 'polys':
                    if len(x['paths']) != len(y['paths']):
                        issues.append(f'ax{i} poly path count')
                    pairs += list(zip(x['paths'], y['paths']))
                else:
                    pairs.append((x['v'], y['v']))
        for u, v in pairs:
            da, dr = _diff(u, v)
            n += _arr(u).size
            mabs, mrel = max(mabs, da), max(mrel, dr)
    return issues, mabs, mrel, n


def pixel(fa, fb, dpi=100):
    import pymupdf
    out = []
    da, db = pymupdf.open(fa), pymupdf.open(fb)
    if len(da) != len(db):
        return [f'pages {len(da)} vs {len(db)}'], 1.0, 255
    frac, mx = 0.0, 0
    for pa, pb in zip(da, db):
        ra, rb_ = pa.rect, pb.rect
        if abs(ra.width - rb_.width) > 0.01 or abs(ra.height - rb_.height) > 0.01:
            out.append(f'page size {ra.width:.2f}x{ra.height:.2f} vs {rb_.width:.2f}x{rb_.height:.2f} pt')
        ia = pa.get_pixmap(dpi=dpi, alpha=False)
        ib = pb.get_pixmap(dpi=dpi, alpha=False)
        A = np.frombuffer(ia.samples, np.uint8).reshape(ia.h, ia.w, ia.n).astype(int)
        B = np.frombuffer(ib.samples, np.uint8).reshape(ib.h, ib.w, ib.n).astype(int)
        if A.shape != B.shape:
            h, w = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
            out.append(f'pixmap {A.shape[:2]} vs {B.shape[:2]} (compared on the common {h}x{w})')
            A, B = A[:h, :w], B[:h, :w]
        d = np.abs(A - B).max(axis=2)
        frac = max(frac, float((d > 8).mean()))
        mx = max(mx, int(d.max()))
    return out, frac, mx


def _rel(p):
    p = str(p)
    for pre in ('/Users/mxq/phi4_bench/',):
        if p.startswith(pre):
            return p[len(pre):]
    m = re.search(r'(results_sync/.*|models/.*|cluster/.*|revision/.*)$', p)
    return m.group(1) if m else p


def prov(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    ra = {(_rel(r['path']), r.get('panel')): r.get('md5') for r in A['rows'] if r.get('path')}
    rb_ = {(_rel(r['path']), r.get('panel')): r.get('md5') for r in B['rows'] if r.get('path')}
    only_a = sorted(set(ra) - set(rb_))
    only_b = sorted(set(rb_) - set(ra))
    diff = sorted(k for k in set(ra) & set(rb_) if ra[k] != rb_[k])
    return only_a, only_b, diff, len(ra)


def prov_traces_only(fa, fb):
    """as prov, but md5 differences of *_traces.npz files are expected (rb column removed in the record)"""
    oa, ob, df, n = prov(fa, fb)
    return oa, ob, [k for k in df if not k[0].endswith('_traces.npz')], n, len(df)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'data':
        iss, ma, mr, n = data(sys.argv[2], sys.argv[3])
        print(f'data: {n} coordinates, max abs diff {ma:.3g}, max rel diff {mr:.3g}, {len(iss)} other differences')
        for s in iss[:20]:
            print('  ', s)
        sys.exit(0 if (mr <= 1e-9 and not iss) else 1)
    if mode == 'pixel':
        iss, fr, mx = pixel(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 100)
        print(f'pixel: fraction of differing pixels {fr:.2e}, max channel difference {mx}')
        for s in iss:
            print('  ', s)
        sys.exit(0 if (fr == 0 and not iss) else 1)
    if mode == 'prov':
        oa, ob, df, n, ntr = prov_traces_only(sys.argv[2], sys.argv[3])
        print(f'prov: {n} inputs; only in a {len(oa)}, only in b {len(ob)}, md5 differs {ntr} '
              f'({ntr - len(df)} trace files without rb, {len(df)} other)')
        for s in (oa[:5] + ob[:5] + df[:10]):
            print('  ', s)
        sys.exit(0 if not (oa or ob or df) else 1)

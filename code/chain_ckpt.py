"""Checkpoint / resume for the chain-comparison scripts (phi4_compare, dw_compare, lj13_compare).

Layout: <out>.ckpt/ next to the output json, one file per (method, run) tag:
  <tag>.pt        in-progress state of the running chain (or SNIS chunk loop), rewritten every --ckpt-min
                  minutes and when SIGTERM / SIGUSR1 arrives;
  <tag>.done.pt   summary of a finished run plus the RNG states at its end (a resumed process skips the run and
                  restores those states, so later runs see exactly the streams of an uninterrupted process).
Every file carries a hash of the arguments (without --ckpt-min) and of the model files; a mismatch stops the
script instead of resuming. Writes are atomic (tmp file + fsync + os.replace). The directory is removed after
the final json has been written.
Signals: SIGTERM saves at the end of the current transition / chunk and exits with code 143; SIGUSR1 saves and
continues. Before the first chain step they are only recorded; a pending SIGTERM is acted on at the next step or
after the next finished run.
RNG states (torch CPU, CUDA if initialised, numpy, python) are saved on the CPU and restored with .cpu(), so a
resumed run is bitwise identical to an uninterrupted one on the same device and thread count.
Test hook: CHAIN_CKPT_TEST_KILL='<tag>:<n>' raises SIGTERM in the process after the n-th completed step of <tag>;
'<tag>:done' raises it when the finished run <tag> is stored (i.e. during its summary).
"""
import hashlib
import json
import os
import pathlib
import random
import shutil
import signal
import sys
import time

import numpy as np
import torch

VERSION = 'chain-ckpt-v1'
EXIT_CODE = 143
TEST_ENV = 'CHAIN_CKPT_TEST_KILL'


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def args_hash(args, files=(), code=(), exclude=('ckpt_min',)):
    """files: data / model files (path and content); code: source files (content only, so the same code under
    another mount point or directory is accepted)."""
    d = {k: v for k, v in sorted(dict(args).items()) if k not in exclude}
    h = hashlib.sha256((VERSION + json.dumps(d, sort_keys=True, default=str)).encode())
    for f in files:
        h.update(str(f).encode())
        h.update(file_sha256(f).encode() if os.path.isfile(f) else b'missing')
    for c in sorted(file_sha256(f) for f in code if os.path.isfile(f)):
        h.update(c.encode())
    return h.hexdigest()


def code_files(namespace):
    """Source files of the calling script and of every module it imported from outside the standard library /
    site-packages (sampler, chainstats, chain_ckpt, kernels): a code change between kill and resume refuses the
    resume instead of splicing two code versions into one result."""
    import inspect
    files = {namespace.get('__file__')}
    for v in list(namespace.values()):
        m = v if inspect.ismodule(v) else sys.modules.get(getattr(v, '__module__', None) or '')
        f = getattr(m, '__file__', None) if m is not None else None
        if f and f.endswith('.py') and 'site-packages' not in f and not f.startswith(sys.base_prefix):
            files.add(f)
    files.discard(None)
    return sorted(os.path.realpath(f) for f in files)


def rng_state():
    s = dict(torch=torch.get_rng_state().cpu(), numpy=np.random.get_state(), python=random.getstate())
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        s['cuda'] = [t.cpu() for t in torch.cuda.get_rng_state_all()]
    return s


def set_rng_state(s):
    torch.set_rng_state(s['torch'].cpu())
    np.random.set_state(s['numpy'])
    random.setstate(s['python'])
    if 'cuda' in s:
        torch.cuda.set_rng_state_all([t.cpu() for t in s['cuda']])


def to_cpu(obj):
    """Detached CPU copies of every tensor in a nested dict / list (numpy arrays are copied)."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def atomic_json(obj, path, **kw):
    """json.dump to path via tmp + rename (a killed write never leaves a truncated json that a task skip guard
    would take as finished)."""
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, **kw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Checkpointer:
    def __init__(self, out, ahash, minutes=20.0):
        self.dir = pathlib.Path(str(out) + '.ckpt')
        self.hash = ahash
        self.period = float(minutes) * 60.0
        self.last = time.time()
        self.pending = None
        self.n_saves = 0
        self._kill = None
        spec = os.environ.get(TEST_ENV, '')
        if spec:
            tag, n = spec.rsplit(':', 1)
            self._kill = (tag, n if n == 'done' else int(n))
        out = pathlib.Path(out)                     # leftovers of writes killed mid-way
        for f in list(self.dir.glob('*.tmp*')) + list(out.parent.glob(out.name + '.tmp*')):
            f.unlink(missing_ok=True)
        for name in ('SIGTERM', 'SIGUSR1'):
            if hasattr(signal, name):
                try:
                    signal.signal(getattr(signal, name), self._handler)
                except (ValueError, OSError):       # not in the main thread
                    pass

    def _handler(self, signum, frame):
        if self.pending is None or signum == getattr(signal, 'SIGTERM', None):
            self.pending = signum
        try:
            print(f'[ckpt] signal {signum} received; saving at the next step', file=sys.stderr, flush=True)
        except Exception:                           # never raise from a signal handler
            pass

    def _test_kill(self, tag, n):
        if self._kill is not None and self._kill == (tag, n):
            self._kill = None
            if hasattr(signal, 'raise_signal'):
                signal.raise_signal(signal.SIGTERM)
            else:
                self.pending = signal.SIGTERM

    def exit_if_term(self):
        """Call where nothing is in progress (before a run starts): exit at once on a pending SIGTERM."""
        if self.pending is not None and self.pending == getattr(signal, 'SIGTERM', None):
            print('[ckpt] exiting after SIGTERM (between runs)', flush=True)
            sys.exit(EXIT_CODE)

    @staticmethod
    def tag(meth, r):
        return meth.replace(':', '') + f'_r{r}'

    def _path(self, tag, done=False):
        return self.dir / (f'{tag}.done.pt' if done else f'{tag}.pt')

    def _write(self, path, obj):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
        with open(tmp, 'wb') as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _read(self, path):
        if not path.exists():
            return None
        obj = torch.load(path, map_location='cpu', weights_only=False)
        if obj.get('hash') != self.hash:
            raise SystemExit(f'[ckpt] {path} was written with different arguments or model files '
                             f'(args hash mismatch); refusing to resume. Delete {self.dir} to restart from scratch.')
        return obj

    # ------------------------------------------------------------------ in-progress state
    def load(self, tag):
        obj = self._read(self._path(tag))
        if obj is None:
            return None
        print(f"[ckpt] resuming {tag} after step {obj['state']['step']} from {self._path(tag)}", flush=True)
        return obj['state']

    def save(self, tag, state):
        self._write(self._path(tag), dict(hash=self.hash, tag=tag, state=state, saved=time.time()))
        self.last = time.time()
        self.n_saves += 1

    def step(self, tag, n_done, make_state, force=False):
        """Call after every completed step (n_done steps done in total). make_state() -> dict with key 'step'.
        force=True (the last step of a run) always saves, so a kill during the summary loses no transitions."""
        self._test_kill(tag, n_done)
        sig = self.pending
        if sig is None and not force and not (self.period > 0 and time.time() - self.last >= self.period):
            return
        st = make_state()
        st['rng'] = rng_state()
        self.save(tag, st)
        print(f'[ckpt] saved {tag} after step {n_done}', flush=True)
        if sig is not None:
            self.pending = None
            if sig == getattr(signal, 'SIGTERM', None):
                print(f'[ckpt] exiting after SIGTERM ({tag}, step {n_done})', flush=True)
                sys.exit(EXIT_CODE)

    # ------------------------------------------------------------------ finished runs
    def load_done(self, tag):
        """Summary of a finished run, or None. Restores the RNG states at the end of that run."""
        obj = self._read(self._path(tag, done=True))
        if obj is None:
            return None
        set_rng_state(obj['rng'])
        print(f'[ckpt] {tag} already finished, loaded from {self._path(tag, done=True)}', flush=True)
        return obj['result']

    def done(self, tag, result):
        self._test_kill(tag, 'done')
        self._write(self._path(tag, done=True), dict(hash=self.hash, tag=tag, result=result, rng=rng_state()))
        p = self._path(tag)
        if p.exists():
            p.unlink()
        if self.pending == getattr(signal, 'SIGTERM', None):
            print(f'[ckpt] exiting after SIGTERM (after finished run {tag})', flush=True)
            sys.exit(EXIT_CODE)
        self.pending = None

    def finish(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def add_args(ap):
    ap.add_argument('--ckpt-min', '--ckpt_min', dest='ckpt_min', type=float, default=20.0,
                    help='checkpoint every this many minutes (0: only at SIGTERM / SIGUSR1); <out>.ckpt/ is resumed '
                         'automatically when the arguments match')


def public_args(a):
    """vars(a) without the checkpoint option (the output json 'args' stays as before)."""
    return {k: v for k, v in vars(a).items() if k != 'ckpt_min'}

#!/usr/bin/env python3
"""Relocate LJ13 chain checkpoints to another mount point (2026-11-15).
chain_ckpt.args_hash includes the path strings of --run-dir/--refs/--out and of the model files, so a checkpoint written
under /media/scratch/mqian refuses to resume under another prefix even when model, data and code are identical.
  hash  : print the args hash lj13_compare.py would use for the given command line (runs the script up to the
          Checkpointer and stops), plus the sha256 of every code file that enters the hash.
  check : compare a stored checkpoint hash with a given hash.
  rewrite: copy a checkpoint directory to a NEW directory with the stored hash replaced, only if the stored hash equals
          --old (i.e. the run was written by exactly this command line under the old prefix). The source is not touched."""
import hashlib, json, os, runpy, shutil, sys

def cmd_hash(script, argv):
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)) + '/..')
    import chain_ckpt as ck
    cap = {}
    class Stop(Exception):
        pass
    def fake(out, ahash, minutes=20.0):
        cap['h'] = ahash
        raise Stop
    ck.Checkpointer = fake
    real = ck.code_files
    ck.code_files = lambda ns: cap.setdefault('code', real(ns))
    sys.argv = [script] + argv
    try:
        runpy.run_path(script, run_name='__main__')
    except Stop:
        pass
    code = {os.path.basename(f): ck.file_sha256(f)[:16] for f in cap.get('code', [])}
    print(json.dumps(dict(hash=cap.get('h'), code=code)))

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'hash':
        cmd_hash(sys.argv[2], sys.argv[3:])
    elif mode in ('check', 'rewrite'):
        import torch
        src, old = sys.argv[2], sys.argv[3]
        new = sys.argv[4] if mode == 'rewrite' else None
        dst = sys.argv[5] if mode == 'rewrite' else None
        files = sorted(f for f in os.listdir(src) if f.endswith('.pt'))
        ok = True
        for f in files:
            o = torch.load(os.path.join(src, f), map_location='cpu', weights_only=False)
            print(f, 'stored', o.get('hash')[:16], 'match' if o.get('hash') == old else 'MISMATCH')
            ok &= o.get('hash') == old
        if mode == 'rewrite':
            if not ok:
                sys.exit('not rewriting: stored hash differs from --old')
            os.makedirs(dst, exist_ok=False)
            for f in files:
                o = torch.load(os.path.join(src, f), map_location='cpu', weights_only=False)
                o['hash'] = new
                torch.save(o, os.path.join(dst, f))
            print('rewritten', len(files), 'files ->', dst)

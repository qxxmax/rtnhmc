#!/usr/bin/env python3
"""Copy of exp_logz.py for the CMCD / DBS samplers (--kind cmcd --model ...); the fresh full reverse record of a
state is m.reverse_to(x, 0) plus log q0(u), the same full-path work as CmcdStepper.start.
Free energy from NHMC chains: forward work of fresh proposals and reverse work of the chain states.

Work convention of paths.py: W(path) = S(x_T) + sum_t (log qF_t - log qR_t) + log q0(u), so E_Q[e^{-W}] = Z (forward
proposal paths) and E_P[e^{+W}] = 1 / Z under the target path law P (x ~ pi, record drawn by the reverse kernel).
Per transition of an IMH / RT / pRT chain (same transition code as exp_chains.py, chains start at forward endpoints)
the current state x gets a fresh full reverse record (rev_partial to stage 0) whose work W_R(x) is stored; for RT the
record drawn by the transition itself is a full reverse record of x and is used directly (no extra cost). n_fwd fresh
forward paths give W_F. The estimators (SNIS = forward, X-rev = reverse from chain X, X + SNIS = Bennett / Meng-Wong
bridge) are computed by free_energy.estimates from the saved arrays.
--rev-every k (IMH / pRT): the fresh reverse record is drawn only after every k-th transition (consecutive
chain states are strongly correlated, so the thinned records lose little; RT keeps its free per-transition record).
Output npz: WR [n / k, C] (reverse work of the state after transitions k, 2k, ...), WF [n_fwd], acc [n, C], absM [n, C],
rev_every; json: args."""
import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_chains as LC  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='cmcd_phi4 model.pt (CMCD or DBS)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--kappa', type=float, required=True)
    ap.add_argument('--method', required=True, help='imh | rt | rtp:s')
    ap.add_argument('--n-chains', type=int, default=4)
    ap.add_argument('--n', type=int, default=20000, help='transitions per chain')
    ap.add_argument('--n-fwd', type=int, default=32768)
    ap.add_argument('--fwd-batch', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=20261107)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--rev-every', type=int, default=1, help='IMH / pRT: fresh reverse record after every k-th transition')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    st = LC.CmcdStepper(a.model, a.kappa, dev)
    name, s = (a.method.split(':')[0], int(a.method.split(':')[1])) if ':' in a.method else (a.method, 0)
    C, n = a.n_chains, a.n
    t0 = time.time()
    WF = []
    for i in range(0, a.n_fwd, a.fwd_batch):
        _, W = st.start(min(a.fwd_batch, a.n_fwd - i))
        WF.append(W.double().cpu().numpy())
    WF = np.concatenate(WF)
    x, W = st.start(C)
    for _ in range(100):
        bad = ~torch.isfinite(W)
        if not bad.any():
            break
        x1, W1 = st.start(int(bad.sum()))
        x[bad], W[bad] = x1, W1
    k = 1 if name == 'rt' else a.rev_every
    WR = np.zeros((n // k, C))
    acc = np.zeros((n, C), bool)
    absM = np.zeros((n, C), np.float32)
    for it in range(n):
        y, Wx, Wy = st.propose(name, s, x, W)
        la = torch.nan_to_num((Wx - Wy).double(), nan=-math.inf, posinf=math.inf, neginf=-math.inf)
        ac = torch.log(torch.rand_like(la)) < la
        x = torch.where(ac.view(-1, *([1] * (x.dim() - 1))), y, x)
        if name == 'imh':
            W = torch.where(ac, Wy, W)
        if name == 'rt':                 # propose() returns works without log q0(u) (it cancels in the ratio): draw a full record
            u0, wr = st.m.reverse_to(x, 0, st.c)
            wr = wr + st.m.prior_logpdf(u0, st.c)
        elif (it + 1) % k == 0:
            u0, wr = st.m.reverse_to(x, 0, st.c)
            wr = wr + st.m.prior_logpdf(u0, st.c)
        if (it + 1) % k == 0:
            WR[(it + 1) // k - 1] = wr.double().cpu().numpy()
        acc[it] = ac.cpu().numpy()
        absM[it] = st.obs(x)[0].cpu().numpy()
        if (it + 1) % 2000 == 0:
            print(f'{it + 1} transitions, acc {acc[:it + 1].mean():.3f}, {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(a.out[:-5] + '.npz', WR=WR, WF=WF, acc=acc, absM=absM, rev_every=k)
    json.dump(dict(args=vars(a), V=st.L * st.L, L=st.L, alg=st.id, T=st.T, seconds=time.time() - t0,
                   note='WR: reverse work of the state after each transition; WF: forward work of fresh proposals'),
              open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

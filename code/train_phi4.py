#!/usr/bin/env python3
"""
phi4 kappa-conditional NHMC with **explicit κ in drift/sigma/learned force**.

This replaces the implicit setup in train_cond_nhmc_learned.py (kappa only in S while
networks see t alone). Uses kernels.create_cond_kernel + CondZ2Wrapper.

Example:
  python train_phi4.py --kernel nhmc_learned --L 8 \\
      --kappa-min 0.20 --kappa-max 0.30 --num-steps 16 --use-z2 \\
      --save-dir runs/cond_phi4_kappa_explicit_z2_L8_k020-030
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from phi4 import phi4_action
from kernels import create_cond_kernel, base_draw
from paths import rev_path, rt_step, draw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--kernel",
        type=str,
        default="nhmc_learned",
        choices=["nhmc_learned", "nhmc_resid", "nhmc_resid2"],
    )
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--L2", type=int, default=0, help="second lattice extent: lattice L x L2 (default 0 = square)")
    p.add_argument("--lam", type=float, default=0.022)
    p.add_argument("--kappa-min", type=float, default=0.20)
    p.add_argument("--kappa-max", type=float, default=0.30)
    p.add_argument("--num-steps", type=int, default=16)
    p.add_argument("--n-leapfrog", type=int, default=5)
    p.add_argument("--kappa-per-sample", action="store_true",
                   help="vatd-style: an independent kappa for every sample in the batch")
    p.add_argument("--kappa-crit", type=float, default=None,
                   help="if set, mix uniform kappa with N(kappa_crit, crit_width) (fraction crit_frac)")
    p.add_argument("--crit-frac", type=float, default=0.4)
    p.add_argument("--crit-width", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=0, help="linear LR warm-up epochs before cosine decay")
    p.add_argument("--fixed-pflip", type=float, default=None,
                   help="Z2 wrapper: fix the per-stage flip probability (no gradient to the discrete move)")
    p.add_argument("--save-epochs", type=int, nargs="+", default=None,
                   help="also save ckpt_e<epoch>.pt at these epochs (0 = before training)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--L-choices", type=int, nargs="+", default=None,
                   help="lattice-conditional training: sample L per batch; loss is divided by V")
    p.add_argument("--T-choices", type=int, nargs="+", default=None,
                   help="sample the number of stages per batch from this list (nhmc_cont)")
    p.add_argument("--odd-nets", action="store_true", help="Z2-equivariant nets: odd force/means, even widths")
    p.add_argument("--buffer", action="store_true",
                   help="add a forward-KL term on reverse records from a buffer of round-trip MH chain states")
    p.add_argument("--buffer-start", type=int, default=1000, help="epoch at which chains start and the term switches on")
    p.add_argument("--buffer-chains", type=int, default=128, help="round-trip chains (split over the buffer kappas)")
    p.add_argument("--buffer-every", type=int, default=2, help="advance every chain by one RT step every N epochs")
    p.add_argument("--buffer-burn", type=int, default=20, help="RT steps after chain initialisation")
    p.add_argument("--buffer-size", type=int, default=20000, help="FIFO capacity (total over kappas)")
    p.add_argument("--buffer-weight", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-lr", type=float, default=1e-5)
    p.add_argument("--var-weight", type=float, default=0.01)
    p.add_argument("--loss", choices=["kl", "logvar"], default="kl",
                   help="kl: mean(W) + var_weight*Var(W) (default); logvar: Var(W) only (pathwise, on-policy)")
    p.add_argument("--eval-interval", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=1024)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--channels", type=int, nargs="+", default=[32, 32])
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-z2", action="store_true", help="Wrap with CondZ2Wrapper")
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to warm-start from (transfer learning).",
    )
    p.add_argument(
        "--init-strict",
        action="store_true",
        help="Require all checkpoint keys to match (default: allow missing/extra).",
    )
    p.add_argument("--ckpt-minutes", type=float, default=30.0,
                   help="save the exact training state (last.pt) every N minutes; restarted automatically if present")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue in save-dir from best.pt/kernel_final.pt using train_log epoch.",
    )
    return p.parse_args()


def _max_epoch_from_train_log(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    m = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("epoch"):
                continue
            try:
                ep = int(line.split(",", 1)[0])
                m = max(m, ep)
            except ValueError:
                continue
    return m


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    L = args.L
    L2 = args.L2 or L
    assert L2 == L or not args.L_choices, "--L2 with --L-choices is not supported"
    V = L * L2
    channels = tuple(args.channels)

    kappa_mid = (args.kappa_min + args.kappa_max) / 2
    kernel = create_cond_kernel(
            args.kernel,
            embed_dim=args.embed_dim,
            channels=channels,
            n_leapfrog=args.n_leapfrog,
            lam=args.lam,
            num_steps=args.num_steps,
            use_z2=args.use_z2,
            odd_nets=args.odd_nets,
        ).to(device)
    if args.fixed_pflip is not None:
        assert hasattr(kernel, "flip_logits"), "--fixed-pflip needs the Z2 wrapper (--use-z2)"
        p_ = min(max(args.fixed_pflip, 1e-12), 1 - 1e-12)
        with torch.no_grad():
            kernel.flip_logits.fill_(math.log(p_ / (1 - p_)))
        kernel.flip_logits.requires_grad_(False)

    n_params = sum(p.numel() for p in kernel.parameters())
    print("=" * 75)
    print(
        f"phi4 explicit-kappa cond | kernel={args.kernel} z2={args.use_z2} "
        f"params={n_params:,}"
    )
    print(f"L={L} lam={args.lam} kappa in [{args.kappa_min}, {args.kappa_max}]")
    print(f"steps={args.num_steps} n_lf={args.n_leapfrog} batch={args.batch_size}")

    os.makedirs(args.save_dir, exist_ok=True)

    def _filter_compatible_state_dict(model_sd, ckpt_sd):
        out, skipped = {}, []
        for k, v in ckpt_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                out[k] = v
            else:
                why = ("missing in model" if k not in model_sd
                       else f"shape {tuple(v.shape)} vs "
                            f"{tuple(model_sd[k].shape)}")
                skipped.append(f"{k} ({why})")
        return out, skipped

    train_log_path = os.path.join(args.save_dir, "train_log.csv")
    start_epoch = 1
    log_mode = "w"
    weights_loaded = False
    last_path = os.path.join(args.save_dir, "last.pt")
    last = torch.load(last_path, map_location=device, weights_only=False) if os.path.isfile(last_path) else None
    if last is not None:
        kernel.load_state_dict(last["kernel"])
        weights_loaded = True
        start_epoch = last["epoch"] + 1
        log_mode = "a"
        for fn in ("train_log.csv", "eval_log.csv", "buffer_log.csv"):     # drop rows written after the checkpoint
            fp = os.path.join(args.save_dir, fn)
            if os.path.isfile(fp):
                rows = open(fp).read().splitlines()
                keep = [rows[0]] + [r for r in rows[1:] if r.split(",")[0].isdigit() and int(r.split(",")[0]) <= last["epoch"]]
                open(fp, "w").write("\n".join(keep) + "\n")
        print(f"EXACT RESUME: {last_path} epoch={last['epoch']} -> starting at {start_epoch}")
    elif args.resume:
        ck_best = os.path.join(args.save_dir, "best.pt")
        ck_final = os.path.join(args.save_dir, "kernel_final.pt")
        ckpt = None
        if os.path.isfile(ck_best):
            ckpt = ck_best
        elif os.path.isfile(ck_final):
            ckpt = ck_final
        if ckpt is not None:
            sd = torch.load(ckpt, map_location=device)
            if args.init_strict:
                miss, unexp = kernel.load_state_dict(sd, strict=True)
            else:
                compat, skipped = _filter_compatible_state_dict(
                    kernel.state_dict(), sd)
                miss, unexp = kernel.load_state_dict(compat, strict=False)
                if skipped:
                    print(f"RESUME: dropped {len(skipped)} incompatible "
                          f"keys: {skipped}")
            print(f"RESUME: loaded {ckpt} missing={len(miss)} unexpected={len(unexp)}")
            weights_loaded = True
            last_done = _max_epoch_from_train_log(train_log_path)
            start_epoch = last_done + 1
            log_mode = "a" if last_done > 0 else "w"
            print(f"RESUME: last train_log epoch={last_done} -> starting at {start_epoch}")
            if start_epoch > args.epochs:
                print(f"RESUME: already finished ({last_done} >= {args.epochs}), exiting.")
                return
        else:
            print("RESUME: no checkpoint in save-dir; training from scratch.")
    if args.init_from and not weights_loaded:
        sd = torch.load(args.init_from, map_location=device)
        if args.init_strict:
            miss, unexp = kernel.load_state_dict(sd, strict=True)
        else:
            compat, skipped = _filter_compatible_state_dict(
                kernel.state_dict(), sd)
            miss, unexp = kernel.load_state_dict(compat, strict=False)
            if skipped:
                print(f"INIT FROM: dropped {len(skipped)} incompatible "
                      f"keys: {skipped}")
        print(f"INIT FROM: {args.init_from}  missing={len(miss)} unexpected={len(unexp)}")
    print("=" * 75)

    optimizer = torch.optim.Adam([q for q in kernel.parameters() if q.requires_grad], lr=args.lr)
    if args.save_epochs and 0 in args.save_epochs and start_epoch == 1:
        torch.save(kernel.state_dict(), os.path.join(args.save_dir, "ckpt_e0.pt"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.target_lr
    )
    if args.warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warm, torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.epochs - args.warmup, 1), eta_min=args.target_lr)],
            milestones=[args.warmup])
    if last is not None:
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
    else:
        for _ in range(start_epoch - 1):
            scheduler.step()

    cfg = {
        "kernel": args.kernel,
        "variant": args.kernel,
        "use_z2": args.use_z2,
        "kappa_cond": "explicit",
        "L": L,
        "L2": L2,
        "lam": args.lam,
        "kappa_min": args.kappa_min,
        "kappa_max": args.kappa_max,
        "num_steps": args.num_steps,
        "n_leapfrog": args.n_leapfrog,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "embed_dim": args.embed_dim,
        "channels": list(channels),
        "train_script": "train_cond_phi4_kappa_explicit.py",
        "buffer": dict(on=args.buffer, start=args.buffer_start, chains=args.buffer_chains, every=args.buffer_every,
                       burn=args.buffer_burn, size=args.buffer_size, weight=args.buffer_weight) if args.buffer else None,
        "fixed_pflip": args.fixed_pflip, "seed": args.seed, "odd_nets": args.odd_nets,
        "loss": args.loss, "var_weight": args.var_weight,
    }
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Single-κ training (kappa_min == kappa_max): avoid repeating the same κ 11× in eval.
    if abs(args.kappa_max - args.kappa_min) < 1e-12:
        eval_kappas = [round(float(args.kappa_min), 4)]
    else:
        eval_kappas = [
            round(args.kappa_min + i * (args.kappa_max - args.kappa_min) / 10, 4)
            for i in range(11)
        ]

    train_log = open(train_log_path, log_mode, buffering=1)
    if log_mode == "w":
        train_log.write("epoch,loss,fv,std,kappa,lr,gpu_mem,samp_per_sec\n")

    eval_log_path = os.path.join(args.save_dir, "eval_log.csv")
    eval_log = open(eval_log_path, log_mode, buffering=1)
    if log_mode == "w":
        eval_log.write(
            "epoch,kappa,fv,ess,abs_mag,frac_pos,var_w,"
            "abs_mag_rw,m2_rw,chi_rw,wall_sec\n"
        )

    # Round-trip MH buffer: the RT kernel is pi-invariant for any parameters (Theorem 2), so chain
    # states are target samples throughout training; reverse records from them give an unbiased
    # reparametrised gradient of the path forward KL  KL(P||Q) = -E_P[W] - log Z.
    if args.buffer:
        assert not args.L_choices and not args.T_choices, "--buffer needs fixed L and T"
        buf_kappas = eval_kappas
        n_per = max(args.buffer_chains // len(buf_kappas), 1)
        cap = max(args.buffer_size // len(buf_kappas), 4 * args.batch_size)
        chains, buf = {}, {}
        n_rt_paths = 0
        buf_log = open(os.path.join(args.save_dir, "buffer_log.csv"), log_mode, buffering=1)
        if log_mode == "w":
            buf_log.write("epoch,kappa_b,loss_b,mean_Wrev,var_Wrev,rt_acc,buf_frac_pos,buf_n,rt_paths_total\n")
        rt_acc_last = float("nan")

    best_ess_mean = 0.0
    t_start = time.time()
    t_prev = 0.0
    if last is not None:
        best_ess_mean = last["best_ess_mean"]
        t_prev = last.get("wall", 0.0)
        if args.buffer:
            chains = {float(k_): v_.to(device) for k_, v_ in last["chains"].items()}
            buf = {float(k_): v_.to(device) for k_, v_ in last["buf"].items()}
            n_rt_paths, rt_acc_last = last["n_rt_paths"], last["rt_acc_last"]
            buf_kappas = [float(k_) for k_ in buf_kappas]
        torch.set_rng_state(last["rng_torch"].cpu())
        if torch.cuda.is_available() and last.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([s_.cpu() for s_ in last["rng_cuda"]])
        np.random.set_state(last["rng_numpy"])
    t_ckpt = time.time()

    def save_last(ep):
        st = dict(epoch=ep, kernel=kernel.state_dict(), optimizer=optimizer.state_dict(),
                  scheduler=scheduler.state_dict(), best_ess_mean=best_ess_mean, wall=t_prev + time.time() - t_start,
                  rng_torch=torch.get_rng_state(),
                  rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  rng_numpy=np.random.get_state())
        if args.buffer:
            st.update(chains={k_: v_.cpu() for k_, v_ in chains.items()}, buf={k_: v_.cpu() for k_, v_ in buf.items()},
                      n_rt_paths=n_rt_paths, rt_acc_last=rt_acc_last)
        torch.save(st, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
    shape = (args.batch_size, 1, L, L2)

    for epoch in range(start_epoch, args.epochs + 1):
        kernel.train()
        t0 = time.time()

        if args.kappa_per_sample:
            kv = np.random.uniform(args.kappa_min, args.kappa_max, args.batch_size)
            if args.kappa_crit is not None:
                crit = np.random.rand(args.batch_size) < args.crit_frac
                kv[crit] = np.clip(np.random.normal(args.kappa_crit, args.crit_width, crit.sum()),
                                   args.kappa_min, args.kappa_max)
            kappa = torch.tensor(kv, dtype=torch.float32, device=device)
        else:
            kappa = float(np.random.uniform(args.kappa_min, args.kappa_max))
        kappa_log = float(kappa.mean()) if torch.is_tensor(kappa) else kappa

        L_ep = int(np.random.choice(args.L_choices)) if args.L_choices else L
        shape_ep = (args.batch_size, 1, L_ep, L_ep) if args.L_choices else shape
        x0, logP = base_draw(kernel, shape_ep, device, kappa=kappa)

        T_ep = int(np.random.choice(args.T_choices)) if args.T_choices else args.num_steps
        x = x0
        for step in range(1, T_ep + 1):
            x, lf, lb = kernel.forward_step(x, step, T_ep, kappa=kappa)
            logP = logP + lf - lb

        phi = x.squeeze(1)
        S = phi4_action(phi, lam=args.lam, kappa=kappa)
        W = logP + S

        loss = W.var() if args.loss == "logvar" else W.mean() + args.var_weight * W.var()
        if args.L_choices:
            loss = loss / (L_ep * L_ep)

        if args.buffer and epoch >= args.buffer_start:
            if not chains:
                for kb in buf_kappas:
                    xc, _ = draw(kernel, n_per, (L, L2), args.num_steps, kb, args.lam, device)
                    for _ in range(args.buffer_burn):
                        xc, _ = rt_step(kernel, xc, args.num_steps, kb, args.lam)
                    chains[kb], buf[kb] = xc, xc.clone()
                n_rt_paths += 2 * n_per * len(buf_kappas) * args.buffer_burn
            if epoch % args.buffer_every == 0:
                accs = []
                for kb in buf_kappas:
                    chains[kb], a_ = rt_step(kernel, chains[kb], args.num_steps, kb, args.lam)
                    buf[kb] = torch.cat([buf[kb], chains[kb]])[-cap:]
                    accs.append(a_.float().mean().item())
                rt_acc_last = float(np.mean(accs))
                n_rt_paths += 2 * n_per * len(buf_kappas)
            kb = float(buf_kappas[np.random.randint(len(buf_kappas))])
            xb = buf[kb][torch.randint(buf[kb].size(0), (args.batch_size,), device=buf[kb].device)]
            _, Wr = rev_path(kernel, xb, args.num_steps, kb, args.lam)
            loss_b = -Wr.mean() + args.var_weight * Wr.var()
            loss = loss + args.buffer_weight * loss_b
            n_rt_paths += args.batch_size
            if epoch % args.log_interval == 0:
                fpos = float(np.mean([(buf[k_].mean(dim=(1, 2, 3)) > 0).float().mean().item() for k_ in buf_kappas]))
                buf_log.write(f"{epoch},{kb},{loss_b.item()},{Wr.mean().item()},{Wr.var().item()},{rt_acc_last},"
                              f"{fpos},{sum(b_.size(0) for b_ in buf.values())},{n_rt_paths}\n")

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(kernel.parameters(), 5.0)
        optimizer.step()
        if args.save_epochs and epoch in args.save_epochs:
            torch.save(kernel.state_dict(), os.path.join(args.save_dir, f"ckpt_e{epoch}.pt"))
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        gpu_mem = (
            torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        )

        if epoch % args.log_interval == 0:
            fv = W.mean().item() / V
            std = W.std().item()
            sps = args.batch_size / dt
            eps_str = ""
            base_mod = kernel.base if args.use_z2 else kernel
            if hasattr(base_mod, "epsilon"):
                eps_str = f"eps={base_mod.epsilon.item():.4f}  "
            print(
                f"[{epoch:6d}/{args.epochs}] loss={loss.item():10.4f}  "
                f"F/V={fv:8.4f}  std={std:6.2f}  kappa={kappa_log:.3f}  "
                f"lr={lr:.1e}  {eps_str}"
                f"mem={gpu_mem:.0f}MB  {sps:.0f} samp/s"
            )
            train_log.write(
                f"{epoch},{loss.item()},{fv},{std},{kappa_log},{lr},{gpu_mem},{sps}\n"
            )
            train_log.flush()

        if epoch % args.eval_interval == 0:
            kernel.eval()
            wall = time.time() - t_start
            ess_list = []
            with torch.no_grad():
                for ek in eval_kappas:
                    all_W, all_M = [], []
                    for _ in range(args.eval_samples // args.batch_size):
                        x0, logP = base_draw(kernel, shape, device, kappa=ek)
                        x = x0
                        for step in range(1, args.num_steps + 1):
                            x, lf, lb = kernel.forward_step(
                                x, step, args.num_steps, kappa=ek
                            )
                            logP = logP + lf - lb
                        phi = x.squeeze(1)
                        S = phi4_action(phi, lam=args.lam, kappa=ek)
                        W = logP + S
                        M = phi.sum(dim=(1, 2)) / V
                        all_W.append(W.cpu())
                        all_M.append(M.cpu())

                    all_W = torch.cat(all_W)
                    all_M = torch.cat(all_M)

                    log_w = -(all_W - all_W.mean())
                    exp_w = torch.exp(log_w - log_w.max())
                    log_ess = (
                        2 * torch.log(exp_w.mean())
                        - torch.log((exp_w**2).mean())
                    )
                    ess = torch.exp(log_ess).item()
                    ess_list.append(ess)

                    abs_mag = torch.abs(all_M).mean().item()
                    frac_plus = (all_M > 0).float().mean().item()
                    fv = all_W.mean().item() / V
                    var_w = all_W.var().item()

                    w_norm = exp_w / exp_w.sum().clamp_min(1e-30)
                    abs_M_t = torch.abs(all_M)
                    m2_t = all_M**2
                    abs_mag_rw = (w_norm * abs_M_t).sum().item()
                    m2_rw = (w_norm * m2_t).sum().item()
                    chi_rw = V * (m2_rw - abs_mag_rw**2)

                    eval_log.write(
                        f"{epoch},{ek},{fv},{ess},{abs_mag},{frac_plus},{var_w},"
                        f"{abs_mag_rw},{m2_rw},{chi_rw},{wall}\n"
                    )
                    print(
                        f"  kappa={ek:.2f}  F/V={fv:.4f}  ESS={ess:.4f}  "
                        f"|M|={abs_mag:.4f}  |M|_rw={abs_mag_rw:.4f}  "
                        f"chi={chi_rw:.3f}  +:{frac_plus:.0%}"
                    )

                eval_log.flush()
                mean_ess = float(np.mean(ess_list))
                print(f"  >> Mean ESS = {mean_ess:.4f}")

                if mean_ess > best_ess_mean:
                    best_ess_mean = mean_ess
                    torch.save(kernel.state_dict(), os.path.join(args.save_dir, "best.pt"))
                    print(f"    -> Best saved (mean ESS={mean_ess:.4f})")

        if time.time() - t_ckpt >= args.ckpt_minutes * 60:
            save_last(epoch)
            t_ckpt = time.time()

    save_last(args.epochs)
    torch.save(kernel.state_dict(), os.path.join(args.save_dir, "kernel_final.pt"))
    if args.buffer:
        buf_log.close()
    train_log.close()
    eval_log.close()
    print(f"\nDone. Best mean ESS: {best_ess_mean:.4f}. Saved under {args.save_dir}")


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.save_dir is None:
        z = "z2" if args.use_z2 else "noz2"
        args.save_dir = (
            f"runs/cond_phi4_kappa_explicit_{z}_k{args.kappa_min:.2f}-{args.kappa_max:.2f}_L{args.L}"
        )
    train(args)


if __name__ == "__main__":
    main()

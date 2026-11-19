# Methods: modelling and analysis choices

Conventions shared by all scripts in `code/` and `figures/`. Section and equation names refer to the paper
(arXiv:2607.15682).

## Target and observables

- Action `S(phi) = sum_x [(1 - 2 lambda) phi_x^2 + lambda phi_x^4 - 2 kappa sum_mu phi_x phi_{x+mu}]`, periodic
  L x L (or N_x x 8 for the rectangular lattices), lambda = 0.022 (`phi4.py`).
- Magnetisation m = (1/V) sum_x phi_x; chains record |m|, sign(m) and S/V per transition (`phi4.observables`).
- Reference log Z: thermodynamic integration of the Wolff / HMC hop density on a coupling grid 0 ... 0.30, step 0.005
  (0.001 on 0.25-0.29 for L = 12, 16), composite Simpson, single-site integral by quadrature
  (`free_energy.ti`, `exp_ti_logz.py`); AIS (`exp_ais_logz.py`) as a cross-check.

## Path model

- Stage t of T: Z2 flip with probability rho_t (fixed at 0.5 for the paper models), momentum p ~ q_F(. | x, t/T),
  five leapfrog substeps with step eps_t and force
  `F_t = (beta_t + g_t)(-grad S) - (1 - beta_t) phi / s^2 + f_theta(phi, t/T, kappa)`, reverse law q_R(-p' | x', (t+1)/T)
  (`kernels.CondNHMCResidual2Kernel`, `kernels.CondZ2Wrapper`). Base law N(0, s^2) with learned s.
  `odd_nets`: force and momentum means odd, log-widths even in phi.
- Work `W = log q0(u) + sum_t [log q_F,t - log q_R,t] + S(x_T)`, identical for forward paths and reverse records
  (`paths.fwd_partial`, `paths.rev_partial`); E_Q[e^{-W}] = Z. Partial work W_s: stages s+1..T only, no log q0.
- The kappa-conditional L = 8 model (`nhmc_learned`, kappa in [0.20, 0.30]) is used for the thermodynamics table and the
  conditional-flow figure.
- TF32 is switched off in all chain and evaluation scripts, so that the reverse stage inverts the forward stage to
  float32 precision.

## Training (`train_phi4.py`)

- Loss `mean(W) + 0.01 Var(W)` on fresh forward paths (reverse KL up to log Z), batch 48, Adam, learning rate 1e-3
  with cosine decay to 1e-5, gradient clipping at norm 5, 20000 epochs (native models).
- Round-trip buffer (from epoch 1000): 64 round-trip MH chains of the current model, advanced one step every second
  epoch, FIFO of 20000 states; reverse records from buffer states add `-mean(W_R) + 0.01 Var(W_R)` with weight 1
  (forward KL of the path law). The RT chains are pi-invariant for any parameters, so buffer states are target samples
  throughout training.
- Checkpoint used by all chains: `kernel_final.pt`; `best.pt` is the checkpoint with the largest evaluation ESS.

## Updates

- path-IMH: fresh forward path; the chain carries the work of the path that produced its state.
  RT: reverse path to u_0 and forward path from the same u_0. pRT(s): the same through u_s, s = 3T/4 (s = 12 at
  T = 16, return depth d = T - s = 4) unless stated otherwise. Accept with `min(1, exp(W_x - W_y))`; NaN = reject.
- Force evaluations (-grad S) per chain and update, counted in the code: each stage has six kicks with one -grad S
  each, so a path costs 6T; path-IMH 6T, RT 12T, pRT 12d (T = 16: 96, 192, 48).
- Chains start at endpoints of fresh forward paths (no reference configurations); the IMH record of the start is the
  work of that path. Chains of one set share one random stream seeded from
  sha256(seed | model id | kappa | method | 'long') and are independent because their noise is disjoint.
- IMH chains can be run from pre-generated proposal streams (`exp_chains.py --imh-pregen`): one torch seed per
  (chain, batch of proposals), numpy uniforms for the accept recursion; identical in law to the step loop.

## The single tau rule (`tau.py`)

- Burn-in: the first 10 % of each chain.
- Per chain: g = max(Sokal self-consistent window with c = 5, batch means with 50 batches), g = 1 + 2 sum_t rho(t);
  tau_int = g / 2 (the chain files store g as `tau_sokal`, `tau_batch_means`).
- Chain set (four chains unless stated otherwise): median of tau_int over chains; error = delete-one-chain jackknife.
- Length rule: start with `n_start` transitions (1e4), double up to `n_max` until every chain satisfies, for |m| and for
  sign(m): post-burn length >= 1000 g, >= 20 x the longest run of rejections, Sokal window < post-burn length / 50, and
  the sign is not frozen (`tau.chain_criteria`). Sets that do not meet it give lower bounds on tau_int (arrows in the
  figures, no error bars).
- Acceptance: mean after burn-in; error from a block jackknife with 20 blocks per chain (Poisson error below 30
  acceptances). HMC / Wolff references: Madras-Sokal error with window 5g.

## Conditional work variance (`exp_wtail_var.py`)

- sigma_s^2 = E_u[Var(W_s | x_s = u)]: n/R intermediate fields x_s drawn on forward paths, each continued R = 16 times
  to x_T; the mean of the within-group variances (n = 16384 continuation paths, seed 11, batch 4096). Error: jackknife
  over 64 groups of intermediate fields. Prediction of the mean acceptance: erfc(sigma_s / 2).
- Transfer to larger lattices: the per-site sigma_s^2 of the native L = 16 model times the target volume.

## Free energies (`free_energy.py`, `exp_logz.py`)

- SNIS: log mean_F e^{-W_F} over forward paths (65520 of 65536 per cell enter the estimates).
- One-way reverse estimate from chain X (X = IMH, RT, pRT): -log mean_X e^{+W_R}, W_R a fresh full reverse record of
  each chain state (for RT the record drawn by the transition itself; for IMH / pRT optionally every k-th state).
- Bridge (BAR, Meng-Wong form with Bennett weights): iterate
  `Z = [mean_F w / (s_R w + s_F Z)] / [mean_X 1 / (s_R w + s_F Z)]`, w = e^{-W}, s_F = n_F / (n_F + n_R*),
  n_R* = chain size / max(tau(W_R), tau(|m|)).
- Errors: jackknife over 20 blocks (chain and forward sample cut into blocks, block i removed from both together).
- Weight diagnostics: Kish ESS_w / N = (sum w)^2 / (N sum w^2), Pareto k-hat of the largest min(N/5, 3 sqrt N) weights
  (`free_energy.summarize`).

## Scaling fits

- sigma_alpha^2 = 4 [erfc^{-1}(acceptance)]^2, fitted per coupling as sigma_alpha^2 ~ sigma_bar^2 V + sigma_fs^2;
  Var(log w) versus T at kappa = 0.27 (L = 4-16, T = 4-64) gives the T^{-0.8} decrease; tau_int(|m|) ~ C (1/alpha - 1/2)
  with C = exp(median log ratio) over the native-model points (C = 1.18 for full round trips).
  Fit scripts: `figures/jackknife_20261117/jk_fits.py`, `figures/figs_20261117u4/fig_scaling_u4.py`,
  `figures/figs_20261118u5/fig_scaling_tau_u5.py`.

## CMCD / DBS, DW, LJ13

- CMCD and DBS (`baselines/cmcd_dbs/`): PyTorch port of the UDB recipe (overdamped Euler-Maruyama paths, T = 80 at
  L = 8 and T = 160 at L = 16, Z2 flips), same chain driver (`exp_chains.py --kind cmcd`) and the same tau rule;
  inference-time refinement x2, x4, x8 in `refine_cmcd.py` and `longchain_refined.py`.
- DW (many-well, `baselines/dw/`) and LJ13 (`baselines/lj13/`): native chain code of each system, driven through
  `exp_chains.py --kind dw / lj` with the same length rule; LJ13 references from parallel tempering
  (`lj13_pt_reference.py`). The LJ13 test / validation configurations of Koehler, Klein and Noe (2020) are not
  redistributed.

# 2D phi^4 baselines: HMC, Wolff (Brower-Tamayo), Metropolis

Same action as NHMC: S = sum_x [phi^2 + lam (phi^2-1)^2] - 2 kappa sum_<xy> phi_x phi_y, lam = 0.022.

- `spin/phi4.py`: `Phi4_2D` with `hmc_step` (leapfrog), `metropolis_sweep`, `wolff_cluster`
  (embedded Ising cluster, p_add = 1 - exp(-4 kappa phi_x phi_y); one "wolff" iteration = 1 Metropolis sweep + n_cluster flips).
- `spin/run_spin.py`: CLI, writes schema json + npz traces; tau_int by the Gamma method (`common/wolff_tau.py`).
- `spin/_smoke_spin.py`: Ising exact / metro-vs-wolff / phi4 HMC-metro-wolff cross-checks.

Run (CPU, 1 core), from `spin/`:

    OMP_NUM_THREADS=1 python run_spin.py --theory phi4 --L 8 --params kappa=0.3 lam=0.022 --algo wolff --n-samples 20000 --burn 2000 --out-dir OUT
    ... --algo hmc --dt 0.1 --n-leap 10

Check at L = 8, kappa = 0.30 (`check_L8/`, 2e4 samples): <|m|> HMC 2.0884(28), tau 3.3; Wolff 2.0853(49), tau 9.7;
reference (results_sync/revision/phi4/hmc_k030.json) 2.0873. Both within 0.5 sigma.
In the broken phase the cluster degenerates to a global flip, so Wolff gives no |m| speed-up there (tau ~ Metropolis);
near kappa_c (0.25) it is ~7x faster than Metropolis.

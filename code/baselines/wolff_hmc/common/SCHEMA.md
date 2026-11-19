# algo_bench result schema (v1, 2026-08-01)

Every benchmark run writes ONE json + ONE npz under
`algo_bench\results\{theory}{dim}d\L{L}_{algorithm}[_tag].json/.npz`.

## JSON fields (required unless noted)

```json
{
  "schema": 1,
  "theory": "u1 | su2 | su3 | phi4 | ising | xy | schwinger | wilson4d_fermion",
  "dim": 2,
  "shape": [12, 12],
  "params": {"beta": 8.0},              // theory couplings (kappa, lambda, m, ...)
  "algorithm": "hb | wolff | hmc | pt | apt | sps | nhmc",
  "algo_params": {},                    // step sizes, n_leap, K, ladder, ckpt path...
  "device": "cpu",
  "seed": 20260801,
  "n_samples": 8000,                    // recorded measurements after burn-in
  "burn": 1000,
  "cost": {
    "wall_s": 0.0,
    "sweeps": 0,                        // local-update sweeps (hb/wolff/metropolis)
    "grad_evals": 0,                    // force evaluations (hmc/nhmc; 0 if n/a)
    "model_evals": 0                    // NN forward passes (sps/nhmc; 0 if n/a)
  },
  "observables": {
    "plaq": {"mean": 0.0, "err": 0.0, "tau_int": 1.0, "exact": null, "dev_pct": null}
  },
  "extra": {}                           // round_trips, acceptance, ESS for IS methods...
}
```

## Conventions

- float64 everywhere; `torch.set_num_threads(8)`.
- Errors: `common\wolff_tau.py` `stat_error` (Wolff automatic windowing) on the
  measurement series. For IS-based samplers (sps) report SNIS ESS in `extra`
  and use bootstrap errors; set `tau_int` to null.
- Observables per theory:
  - gauge (u1/su2/su3 any dim): `plaq`, `w2x2`, `w4x4` (2D only), `q`+`chi_top` (2D U(1) only).
  - spin (phi4/ising/xy): `abs_m`, `chi`, `binder`, `energy`.
  - fermion: `plaq`, `psibar_psi` (chiral condensate or proxy), optional `pion_corr` in npz.
- Exact references when available: 2D U(1) finite-volume character expansion
  (see `apt_u1_reference.py` and `_check_finiteV_wilson.py` in 2dsps_u1_standalone;
  NOTE: use FINITE-VOLUME Wilson refs at beta > 8); 2D Ising Onsager; SU(2)/SU(3)
  2D exact plaquette in `_reference\baselines_2026\`.
- npz: raw measurement series (one array per observable, plus per-chain arrays
  where applicable).
- GPU: check `common\gpu_gate.py` `free_mb() >= 3000` before ANY cuda allocation;
  poll-wait, never kill other processes. GPU jobs run serially within each module.
- Detached launches: PowerShell `Start-Process -WindowStyle Hidden` with
  `-RedirectStandardOutput/-RedirectStandardError` (never `cmd /c start /b`
  with `>` redirects: it blocks the parent shell via inherited handles).
- Fermion Dirac operators: NEVER build dense meshgrid matrices (V x V blows up
  memory); implement hopping via torch.roll stencils / even-odd
  preconditioning (see snf4dfermion\hmc_eo.py), CG in float64 with fixed
  iteration cap, and chunk pseudofermion batches to respect the 3000 MB gate.

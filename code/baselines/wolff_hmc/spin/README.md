# `spin\` — 2D Ising, XY and phi4 samplers

Wolff single cluster, Metropolis, and (phi4 only) HMC on 2D lattices, with
`common\SCHEMA.md` v1 output and Wolff-windowed autocorrelation errors.
3D is out of scope for this module.

| file | contents |
|---|---|
| `lattice.py` | periodic neighbour table, checkerboard mask, and the shared deque-BFS single-cluster growth routine |
| `ising.py` | 2D Ising: checkerboard Metropolis, Wolff, **exact finite-torus energy** (Kaufman / Ferdinand-Fisher) |
| `xy.py` | 2D XY: checkerboard Metropolis, Wolff with a random reflection axis |
| `phi4.py` | 2D phi4: HMC, site Metropolis, Brower-Tamayo embedded-cluster Wolff |
| `run_spin.py` | CLI driver -> SCHEMA json + npz |
| `_smoke_spin.py` | validation suite (see below) |

Everything is float64 numpy on CPU; `torch.set_num_threads(8)` is set in
`run_spin.py` per the schema convention even though the kernels are numpy.

## CLI

```
python run_spin.py --theory ising --L 32 --params T=2.269 --algo wolff \
                   --n-samples 20000 --burn 2000
python run_spin.py --theory xy    --L 16 --params T=0.8   --algo metro \
                   --n-samples 40000 --burn 4000
python run_spin.py --theory phi4  --L 16 --L2 8 --params kappa=0.3 lam=0.022 \
                   --algo hmc --n-samples 10000 --burn 1000 --dt 0.1 --n-leap 10
```

`--algo hmc` is phi4 only.  `--L2` gives the second extent (default `--L`),
which is what makes the 16x8 geometry of the free-energy project available.

## Output location

`common\SCHEMA.md` mandates `results\{theory}{dim}d\L{L}_{algorithm}[_tag]`, so
runs land in `results\ising2d\`, `results\xy2d\`, `results\phi42d\` rather than
a single `results\spin\` folder; that keeps the spin runs discoverable by the
same report generator that reads the gauge modules.  `--out-dir` overrides it.
The stem is extended to `L{L}x{L2}` for anisotropic lattices and always carries
a parameter tag (`T2.269`, `k0.3`).

## Conventions

Ising and XY use `E = -sum_{<xy>} s_x . s_y` with `J = 1`, `beta = 1/T`, and
report `energy = <E>/V` (ground state `-2`).  phi4 uses the action of the
free-energy project,

```
S[phi] = sum_x [ -2 kappa sum_mu phi_x phi_{x+mu} + phi_x^2 + lambda (phi_x^2 - 1)^2 ]
```

and reports `energy = <S>/V`.  Expanding the quartic term shows this equals the
`free_energy_reweighting.py` / `run_observables.py` action plus a constant
`lambda` per site, so `extra.u_density_ref_conv = energy - lambda` is the number
that compares directly to the reference `obs_*.json` files.

All three theories report the same four observables, built from the per-site
magnetisation `m` (a vector for XY, in which case `m^2` means `|m_vec|^2`):

```
abs_m  = <|m|>
chi    = V * ( <m^2> - <|m|>^2 )        (no beta factor)
binder = 1 - <m^4> / (3 <m^2>^2)
energy = as above
```

`abs_m` and `energy` are plain series means and use
`common\wolff_tau.stat_error`.  `chi` and `binder` are nonlinear in several
means, so they use a blocked jackknife with block length `4 * tau_int` (>= 20
blocks); the `tau_int` field reported for them is that of the `m^2` series.

## Cluster algorithms

All three share one embedded-Ising construction (`lattice.grow_cluster`): a
sign variable `eps_x = +-1` whose conditional weight is an inhomogeneous
ferromagnetic Ising model `exp(sum_<xy> c w_x w_y eps_x eps_y)`, hence bond
probability `p = 1 - exp(-2 c w_x w_y)` between aligned neighbours.

| theory | `eps_x` | `w_x` | `c` | `p` |
|---|---|---|---|---|
| Ising | `s_x` | 1 | `beta` | `1 - exp(-2 beta)` |
| XY | `sign(s_x . r)` | `\|s_x . r\|` | `beta` | `1 - exp(-2 beta (s_x.r)(s_y.r))` |
| phi4 | `sign(phi_x)` | `\|phi_x\|` | `2 kappa` | `1 - exp(-4 kappa phi_x phi_y)` |

XY draws a fresh reflection axis `r = (cos a, sin a)`, `a ~ U[0, pi)`, each
cluster and flips the component along `r` (`theta -> 2a + pi - theta`), leaving
the perpendicular component frozen.  The phi4 bond probability is derived in
full in the `phi4.py` docstring; the key point is that the site potential
`phi^2 + lambda(phi^2-1)^2` depends on `|phi|` only, so the sign variables see
exactly the hopping term `exp(2 kappa sum_<xy> |phi_x||phi_y| eps_x eps_y)`.

The phi4 cluster move changes signs but never magnitudes, so it is not ergodic
alone.  One `--algo wolff` iteration for phi4 is therefore **one Metropolis
sweep followed by `--n-cluster` cluster flips** (Brower-Tamayo).  Deep in the
broken phase the bond probability saturates and the cluster degenerates to a
global sign flip, which restores tunnelling between the two vacua but leaves
all magnitude decorrelation to the Metropolis sweep — visible as
`extra.mean_cluster_frac -> 1` and `tau_int` equal to the pure-Metropolis value.

## Cost accounting

`cost.sweeps` counts one sweep-equivalent per update:

- `metro`: 1 per lattice sweep (`V` attempted single-site moves).
- `wolff` (Ising/XY): **1 per single-cluster flip**, as specified for this
  module.  A cluster touches `<|C|>` sites rather than `V`, so this convention
  flatters Wolff whenever `<|C|> < V`.  The site-normalised count is reported
  alongside it as `extra.site_sweep_equiv` (total flipped sites / `V`) together
  with `extra.mean_cluster_frac = <|C|>/V`, so a work-matched comparison is
  always available: `tau_int_work = tau_int * <|C|>/V`.
- `wolff` (phi4): `1 + n_cluster` per iteration.
- `hmc`: `sweeps = 0`; `cost.grad_evals = trajectories * (n_leap + 1)`.

## Validation (`_smoke_spin.py`)

```
python _smoke_spin.py          # full statistics, ~4 min on CPU
python _smoke_spin.py --fast   # reduced statistics, structure only
```

0. The Ferdinand-Fisher finite-torus energy is checked against brute-force
   enumeration of the 4x4 lattice (agreement to ~1e-10) and, at L=256, against
   the Onsager thermodynamic-limit expression.  `python ising.py` runs this
   check standalone.
1. Ising L=16 at T=2.0 and 2.5: energy against the **exact L=16 torus value**,
   `abs_m` against the Onsager spontaneous magnetisation (below T_c) or a broad
   finite-size window (above), and Metropolis vs Wolff within 4 sigma.
2. Ising L=16 at T_c: the headline `tau_int(|m|)` ratio, Metropolis vs Wolff.
3. XY L=16 at T=0.8: Metropolis vs Wolff within 4 sigma.
4. phi4 16x8 at lambda=0.022, kappa in {0.25, 0.3}: HMC vs Metropolis vs Wolff
   pairwise within 4 sigma on `abs_m` and `energy`, and each within 4 sigma of
   `free_energy_hmc_phi4_latest\obs_16x8_k{0.25,0.3}.json` (after the constant
   `lambda` shift).

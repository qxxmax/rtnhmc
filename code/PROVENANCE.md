# PROVENANCE (internal; remove before publishing)

Source of every file of this repository, for the bitwise-equivalence check on the cluster (same seeds, tiny runs:
original script vs repository script). Source paths are relative to `/Users/mxq/phi4_bench/` (the author's Mac copy,
state of 2026-11-19). The cluster copies that produced the paper data (`/media/scratch/mqian/revision_code`,
`/lustre/him/him-mam/mqian/revision_code`) should be compared with these sha256 values first; where they differ, the
equivalence check must use the cluster copy.

Rule followed: library modules contain verbatim copies of existing functions and classes (text identical, including
docstrings and comments); experiment scripts are the original scripts with only import statements, module names,
machine-specific paths and docstring text changed. Every change is listed in Section 4.

## 1. sha256 of the core sources

| source | sha256 |
|---|---|
| `nhmc_code/action.py` | `f9af8bfb5748c2a0030273a348a729f59968f50a30d786de1ae887d6b0dff435` |
| `nhmc_code/networks.py` | `7178c81309a4893da1f6642318259cc1aa6b10bc137aab61c1b98c161547c508` |
| `nhmc_code/cond_networks.py` | `0b1b20b612843ad1513f8ad639aa2699695c2531b60886eaa19868ba79163143` |
| `nhmc_code/cond_kernels.py` | `4dfa8194c0e215d47d6143253bfdb4315c2b911242ec706d3f6b4c2c25225e34` |
| `nhmc_code/nhmc_paths.py` | `b02967369e0420b4e7311bc2ae0ad43b097231596c5bcc02227ac5a11adb6af2` |
| `nhmc_code/mech_generic.py` | `ec3774029d1d6942db3a51428322a4646b0a67d4be5dcccd243b31ac67be5dc3` |
| `nhmc_code/eval_generic.py` | `44ee44d8dea20ee5d4502400b03cf28f601f89bd594d5c05af82ce0e8662b4c9` |
| `nhmc_code/train_cond_phi4_kappa_explicit.py` | `ca97b48731d80dce52d084607d4872f9f369c8234f727a75a8d19e95ebcb3d48` |
| `revision/longchain/longchain.py` | `a75a72b72cb2ee3e9d6aea552aeb4d69b5b0ba6a3d4d9d810e10c014c111e98d` |
| `revision/longchain/phi4_gpu.py` | `4b616eddb78715a6ed9deea377bc5d2147d196ebe5e680052fafbbaf94087a6a` |
| `revision/longchain/logz_chains.py` | `b2267c4e3d01a801ffb2f50fcdd9d5dafea8e098c3894c49f0963950b58f0762` |
| `revision/longchain/logz_chains_udb.py` | `b42a588157b3db5fcad35d547271e269395d1f2823c4e179d32556f72feeb3b3` |
| `revision/longchain/logz_estimators.py` | `8345d463ab890f016662c6976fd418daff514b514d2954e552e58d5c143824bf` |
| `revision/longchain/raw_snis_obs.py` | `5ed81b0f54d0c069532a1d60ce9429ae2a9791480c49588fb652fb08737a2f1d` |
| `revision/longchain/diag.py` | `26429346b98d8e7a5141810e88053dace4231e2c92c72f9c2693ed40c4307efc` |
| `revision/chainstats.py` | `eb37b478e1ee463b9b4e9c38f5993590bc2e74efea378849c6942686c73647dd` |
| `revision/chain_ckpt.py` | `3ca758651e55799090235db9519c9fa5bbdf669a15ae57c3d5c6988f5deea530` |
| `results_sync/revision/jackknife_20261117/jk_lib.py` | `4af15f4bbf14dc09c9f108d079ecb462c04ee8b61ac9fd10ed2456d986779716` |
| `results_sync/revision/jackknife_20261117/jk_cells.py` | `41a332a97ba8ec4e519ec7e87775f9bf2d0699d96ab29b3f64c2a481f636225f` |
| `results_sync/revision/jackknife_20261117/cluster/wtail_var2_jk.py` | `65ec5289b983bdcd12a4ec8d672761a23d59555b8fb1d388449f02f49d95f6ec` |
| `results_sync/revision/ti_wolff_hmc/ti_logz.py` | `994d9dc23769aff8a1d400631fd2714a67ef5b79da06fca951531857fcdd4183` |
| `phi4_logz_ref.py` | `58a954e172c041c2be5a0f0ef4391f7b76a8c6aad6c58fbfbfa0726b9f6d5a38` |
| `results_sync/revision/tasks/nehmc_eval.py` | `1f233eb60a062a035bb28fab46d1f5920fb501de34260d324c0b2bd7de688869` |
| `results_sync/revision/tasks/nehmc_grid.py` | `f78d5027c210e97c8c760424a4c67693dd98089779685daba5b35f2382c60f86` |
| `results_sync/revision/tasks/proto_train.py` | `30525b4ce1b631423e0b9e23e7c56a16559332e673f42ac434baaea81301b62d` |
| `results_sync/revision/bench_timing_20261118/code/bench_sample.py` | `692857d74d318cc32ab860be2a3b84ac63f582d39336958cb776566a35e81214` |

## 2. Library modules: function / class -> source

| new file | object(s) | source file : object(s) (lines) | verbatim? |
|---|---|---|---|
| `code/phi4.py` | `_kappa_to_broadcastable`, `phi4_action_density`, `phi4_action`, `phi4_force` | `nhmc_code/action.py` (8-49) | yes |
| `code/phi4.py` | `observables` | `revision/longchain/phi4_gpu.py` : `observables` (131-135) | yes |
| `code/networks.py` | `CyclicConv2d` | `nhmc_code/networks.py` (44-58) | yes |
| `code/networks.py` | `TimeKappaEmbedding`, `CondDriftNet`, `CondSigmaNet` | `nhmc_code/cond_networks.py` (15-101) | yes |
| `code/kernels.py` | `_phys_force`, `_broadcast_kappa` | `nhmc_code/cond_kernels.py` (26-47) | yes |
| `code/kernels.py` | `CondNeuralHMCLearnedKernel`, `CondNHMCResidualKernel`, `base_draw`, `CondNHMCResidual2Kernel` | `nhmc_code/cond_kernels.py` (140-344) | yes |
| `code/kernels.py` | `_Odd`, `_Even` | `nhmc_code/cond_kernels.py` (356-375) | yes |
| `code/kernels.py` | `CondZ2Wrapper`, `base_logprob` | `nhmc_code/cond_kernels.py` (868-927) | yes |
| `code/kernels.py` | `COND_KERNEL_REGISTRY`, `COND_KERNEL_LATTICE_REGISTRY` | `nhmc_code/cond_kernels.py` (969-988) | no: trimmed to the kernels used (Section 4) |
| `code/kernels.py` | `create_cond_kernel` | `nhmc_code/cond_kernels.py` (991-1037) | yes |
| `code/kernels.py` | `load_any` | `nhmc_code/mech_generic.py` (22-30) | yes |
| `code/kernels.py` | `load_model` | `revision/longchain/phi4_gpu.py` (101-128) | no: two in-function imports removed |
| `code/paths.py` | `fwd_path`, `rev_path`, `rt_step`, `draw` | `nhmc_code/nhmc_paths.py` (12-44) | yes |
| `code/paths.py` | `rev_partial`, `fwd_partial` | `revision/longchain/phi4_gpu.py` (139-157) | yes |
| `code/paths.py` | `paths` | `nhmc_code/eval_generic.py` (20-27) | yes |
| `code/updates.py` | `_accept`, `start_states`, `path_imh_step`, `truncated_round_trip_step`, `round_trip_step` | NEW: the transition of `revision/longchain/longchain.py` (`PhiStepper.start` / `propose` 96-108 and the accept lines of `main` 672-681) written as functions; same calls in the same order | new |
| `code/tau.py` | `sokal_single`, `batch_means_tau`, `holding_runs` | `revision/longchain/longchain.py` (43-80) | yes |
| `code/tau.py` | `chain_criteria` | `revision/longchain/longchain.py` (269-294) | yes |
| `code/tau.py` | `jk_err`, `jk_delete_one`, `sokal_g`, `bm_g` | `results_sync/revision/jackknife_20261117/jk_lib.py` (32-72) | yes except the docstrings of `sokal_g`, `bm_g` |
| `code/tau.py` | `tau_seg` | `results_sync/revision/jackknife_20261117/jk_cells.py` (26-29) | yes |
| `code/tau.py` | `tau_int_chains` | NEW: the paper statistic of `jk_cells.analyze` (max(Sokal, BM)/2 per chain, median, delete-one-chain jackknife) computed from traces with `sokal_single` / `batch_means_tau` | new |
| `code/free_energy.py` | `lme`, `tau_of`, `bar_logz`, `EST_VERSION`, `estimates`, `__main__` | `revision/longchain/logz_estimators.py` (21-98, 100-103) | yes except one docstring phrase; `__main__` imports json, sys locally |
| `code/free_energy.py` | `pareto_khat`, `summarize` | `nhmc_code/eval_generic.py` (30-59) | yes |
| `code/free_energy.py` | `site_quadrature` | `phi4_logz_ref.py` (39-51) | yes |
| `code/free_energy.py` | `LAM`, `LOGZ1`, `FINE_LO`, `FINE_HI`, `weights`, `series`, `combine`, `nodes`, `ti` | `results_sync/revision/ti_wolff_hmc/ti_logz.py` (29-46, 60-105) | yes |
| `code/chainstats.py` | whole file | `revision/chainstats.py` | yes |
| `code/chain_ckpt.py` | whole file | `revision/chain_ckpt.py` | yes |
| `code/diag.py` | whole file | `revision/longchain/diag.py` | yes except line 1 of the docstring |

Dropped as unused by any paper script: `cond_kernels` classes `CondGeneralizedKernel`, `CondNeuralHMCKernel`,
`CondNHMCContinuousKernel`, `CondNHMCCont2Kernel`, `CondNHMCCont3Kernel`, `CondNHMCRG*`, all `*Lattice*` kernels and
`CondZ2LatticeWrapper`, helpers `_interp_knots`, `_khat2`, `_spec_apply`, `_spec_quad`; all other classes of
`cond_networks.py` and `networks.py`; `phi4_gpu.py` functions `hmc_z2_step`, `stream_seed`, `run_chain`,
`tau_global`, `summarise_chain`, `burn_check`, `max_window_ratio`, `combine`, `load_ref`, `mode_*`, `main`
(its CLI is not used by any paper task line; `longchain.py` only uses `load_model`, `base_draw`, `fwd_partial`,
`rev_partial`, `observables`, `save_json`).

## 3. Experiment scripts

| new file | source | changes |
|---|---|---|
| `code/exp_chains.py` | `revision/longchain/longchain.py` | functions of Section 2 moved to `tau.py` (imported); `import phi4_gpu as G` -> `import kernels, paths, phi4` and `G.load_model / G.base_draw / G.fwd_partial / G.rev_partial / G.observables / G.save_json` -> `kernels.load_model / kernels.base_draw / paths.fwd_partial / paths.rev_partial / phi4.observables / save_json` (`save_json` copied verbatim from `phi4_gpu.py` 82-87); sys.path block (lines 34-40, incl. `D:/cc/...` default of `$CMCD_TORCH`) replaced by `code/` and `code/baselines/cmcd_dbs`; `DwStepper` / `LjStepper` directories `HERE.parent/'dw'`, `HERE.parent/'lj13'` -> `HERE/'baselines'/...`; docstring lines 2, 6, 7 |
| `code/exp_logz.py` | `revision/longchain/logz_chains.py` | `import longchain as LC` -> `import exp_chains as LC`; `G = LC.G` -> `G = LC.paths`; docstring names and dates |
| `code/exp_logz_udb.py` | `revision/longchain/logz_chains_udb.py` | same import change; unused `G = LC.G` removed; docstring |
| `code/exp_raw_snis.py` | `revision/longchain/raw_snis_obs.py` | same import change; docstring / comment dates |
| `code/exp_forward_eval.py` | `nhmc_code/eval_generic.py` | `paths`, `pareto_khat`, `summarize` moved to `paths.py` / `free_energy.py` (imported); `from action import ...`, `from cond_kernels import ...` -> `kernels`; docstring lines 4, 6 |
| `code/train_phi4.py` | `nhmc_code/train_cond_phi4_kappa_explicit.py` | imports `action` -> `phi4`, `cond_kernels` -> `kernels`, `nhmc_paths` -> `paths`; `--kernel` choices trimmed to `nhmc_learned`, `nhmc_resid`, `nhmc_resid2`; docstring lines 6, 9. The value `"train_script": "train_cond_phi4_kappa_explicit.py"` written to config.json is unchanged |
| `code/exp_wtail_var.py` | `results_sync/revision/jackknife_20261117/cluster/wtail_var2_jk.py` | docstring rewritten (cluster paths and job notes removed); imports `action` -> `phi4`, `cond_kernels` -> `kernels` |
| `code/exp_nehmc.py`, `code/exp_nehmc_grid.py`, `code/train_protocol.py` | `results_sync/revision/tasks/nehmc_eval.py`, `nehmc_grid.py`, `proto_train.py` | imports `action` -> `phi4`, `eval_generic.summarize` -> `free_energy.summarize`; docstring names and dates |
| `code/exp_ti_logz.py` | `results_sync/revision/ti_wolff_hmc/ti_logz.py` (1-22, 47-59, 107-210) | functions of Section 2 imported from `free_energy.py`; `HERE`, `RS` from `$NHMC_ROOT` (`HERE = $NHMC_ROOT/results_sync/revision/ti_wolff_hmc`); outputs `ti_logz.csv`, `ti_hop_consistency.csv`, `TI_logZ.md` written to the working directory (`OUTDIR`) instead of `HERE`; docstring date |
| `code/exp_ais_logz.py` | `phi4_logz_ref.py` | `site_quadrature` imported from `free_energy.py` instead of defined here |
| `code/exp_timing.py` | `results_sync/revision/bench_timing_20261118/code/bench_sample.py` | `sys.path.insert(0, os.environ['LONGCHAIN_DIR'])` removed, `import longchain as LC` -> `import exp_chains as LC`; docstring |
| `code/build_notebook.py`, `tests/test_reversibility.py`, `tests/test_tau_rule.py`, `code/METHODS.md` | NEW | - |

## 4. Non-verbatim changes (complete list)

1. `kernels.py`: registries trimmed to `nhmc_learned`, `nhmc_resid`, `nhmc_resid2` (lattice registry empty); other names
   now raise `ValueError` in `create_cond_kernel` (unchanged function body). `load_model`: removed
   `from mech_generic import load_any` and `from cond_kernels import create_cond_kernel` (both now in the same module).
   New module docstring and import block (`networks`, `phi4` instead of `cond_networks`, `action`).
2. `networks.py`, `phi4.py`, `paths.py`, `tau.py`, `free_energy.py`: new module docstrings and import blocks;
   `free_energy.py` gains `import torch` (needed by `pareto_khat`, `summarize`).
3. `tau.py`: docstrings of `sokal_g` and `bm_g` reworded (they referred to `longchain`); new `tau_int_chains`.
4. `free_energy.py`: `bar_logz` docstring: "(default since 2026-11-10)" removed.
5. `diag.py`: docstring line 1 reworded.
6. `updates.py`: new (Section 2).
7. Experiment scripts: as in Section 3.
8. `baselines/cmcd_dbs/cmcd_phi4.py` lines 60-68: `CLONE` = `code/` (or `$NHMC_CODE`) instead of the repository root,
   `sys.path.insert(0, CLONE)`, `from phi4 import phi4_action, phi4_force` (was `from action import ...`),
   `RESULTS = $NHMC_ROOT/results_sync` (was `CLONE/results_sync`); comment lines reworded.
9. `baselines/cmcd_dbs/rt_cmcd_phi4.py`: `sys.path.insert(0, os.path.join(C.CLONE, 'revision'))` ->
   `sys.path.insert(0, C.CLONE)`; the `gpu_common` path for `diag` -> `C.CLONE`.
10. `baselines/cmcd_dbs/longchain_refined.py`: `code/` added to `sys.path`, `import longchain as LCH` ->
    `import exp_chains as LCH`.
11. `baselines/cmcd_dbs/tests/test_resume_rtcmcd.py`: default model path -> `$NHMC_ROOT/models/cmcd_torch_L8/...`.
12. `baselines/dw/dw_compare.py`, `dw_hmc_flip.py`, `baselines/lj13/lj13_compare.py`, `lj13_hmc.py`,
    `lj13_pt_reference.py`: `sys.path.insert(0, str(HERE.parent))` -> `str(HERE.parents[1])` (`code/`, for
    `chainstats` / `chain_ckpt`); LJ13: default `$LJ_CODE` -> `HERE/'particle'` (copies of
    `particle/code/{eval_lj_cluster_nhmc, lj13_chain_diagnostics, lj13_nhmc, particle_targets, neural_nhmc_dw,
    equiv_drift}.py`, verbatim).
13. `baselines/wolff_hmc/README.md`: removed the line naming the original folder (`D:/thermocode/algo_bench`) and the
    line about `results_algo_bench_20260801/` (not included).
14. `figures/**/*.py`: the rewrite rule of the Zenodo package (`make_zenodo_package.sh rewrite_paths`):
    `'/Users/mxq/phi4_bench...` -> `os.environ.get('NHMC_ROOT', '.') + '...`, the same for f-strings and double quotes,
    `/Users/mxq/Documents/...` -> `MANUSCRIPT_FIGURE_NOT_INCLUDED`; `import os` added where missing (inserted before
    the first import statement; the Zenodo package prepends it to the file, so the line position differs).
    In `u2_common.py`, `u3_common.py`, `u4_common.py`, `u5_common.py` `traces_path` maps the absolute paths stored in
    `inventory_u2.json`: `c['path'].replace('/Users/mxq/phi4_bench', os.environ.get('NHMC_ROOT', '.'))` (the Zenodo
    copies have an equivalent change made on iqcb; its exact text was not available locally - diff before publishing).
    `fig5_conditional_flow_u4.py`: unused `import fitz` removed. `bench_timing_20261118/make_table.py`: `ROOT` ->
    `results/timing/` of this repository.
15. Not copied (superseded or not used by a paper figure / table): `figs_20261117u4/{capture_u4, compare_u4}.py`,
    `figs_20261118u5/{capture_u5, compare_u5}.py` (replaced by `figures/zenodo_20261118/{capture_v10, compare_v10}.py`),
    `results_sync/revision/figs/table_udb_compare.py` (cluster-path original of `make_tab_udb_compare.py`),
    `revision/phi4/{phi4_compare, phi4_hmc_z2, e1_collect}.py`, `revision/e2/*`, `revision/gpu_common/{alpha_test,
    alpha_dw}.py`, `revision/cmcd_torch/alpha_cmcd.py`, `nhmc_code/{rt_imh_chains, p0_mixing, mech_generic (main),
    cand_ce_phi4, budget3, omega_k_conv, pflip_kconv, refine_eval, smc_eval, stage_diag, mismatch_sweep, eval_cont,
    _full_is_eval, test_rt_L2, test_cont_L2}.py`, `revision/longchain/{chain_diagnostics, test_kinds_dw_lj,
    cmcd_phi4 (duplicate)}.py`, `results_sync/revision/tasks/{imh_alpha, refine_eval2, refine_eval3, wtail_var}.py` are
    kept only as as-executed copies in `code/tasks/cluster/`.
16. Moved: `code/tasks/cluster/` = verbatim copy of `results_sync/revision/tasks/` without the seven scripts promoted
    to `code/` (`refine_cmcd`, `longchain_refined`, `diag_gf`, `nehmc_eval`, `nehmc_grid`, `proto_train`,
    `wtail_var2`); `code/tasks/cluster/him_20261118/` = the two HIM line files of 2026-11-18 (from the Zenodo relay);
    `code/tasks/jackknife/` = `jackknife_20261117/cluster/{lines_evalW_him.txt, lines_wtail_jk_him.txt,
    submit_him.sh}`; `code/tasks/lj13/lj13_buf.txt` = `lj_buffer/lj13_buf.txt`; `code/tasks/timing/` = the timing
    benchmark driver, configs, `train_log_wall.py`, `make_results.py`; `code/tasks/analysis_cluster/` =
    `item4_floor_pg_him.py`, `item4_eq_w2_him.py`, `jk_cells_remote.py` (machine paths); `code/tasks/cmcd_dbs/` =
    `launch_repro.sh`, `train_snapshots*.sh`.

## 5. Suggested bitwise-equivalence checks (cluster, CPU, tiny)

- `exp_chains.py --kind phi --run-dir <L8_k0.27_buf_s0> --ckpt kernel_final.pt --kappa 0.27 --method rt --n-start 200
  --n-max 200 --device cpu --out a.json` vs `longchain.py` with the same arguments: identical `*_traces.npz` and json
  (`args.out` aside). Repeat for `imh`, `imh --imh-pregen`, `rtp:12`, `--kind cmcd`, `--kind dw`, `--kind lj`.
- `exp_logz.py` vs `logz_chains.py`, `exp_raw_snis.py` vs `raw_snis_obs.py`, `exp_forward_eval.py` vs
  `eval_generic.py`, `exp_wtail_var.py` vs `wtail_var2_jk.py` (small `--n`), `train_phi4.py` vs
  `train_cond_phi4_kappa_explicit.py` (`--epochs 20 --buffer --buffer-start 5`, compare `kernel_final.pt` tensors).
- `tests/test_reversibility.py` (includes the identity `updates.py` == `exp_chains.py` transition).

## 6. File copies (sha256 of the source; `modified` = see Section 4)

| file | source | status | sha256 (source) |
|---|---|---|---|
| `code/baselines/cmcd_dbs/cmcd_phi4.py` | `revision/cmcd_torch/cmcd_phi4.py` | modified | `c54e9ca7eb109d03f84338ed1824494ef8d22e4d448dcfe5f6dd87c5dd4f095d` |
| `code/baselines/cmcd_dbs/rt_cmcd_phi4.py` | `revision/cmcd_torch/rt_cmcd_phi4.py` | modified | `1158811c4709fd77e56fecce0f362b4855edc5a7e81a4fcf02a64aa1abcd7091` |
| `code/baselines/cmcd_dbs/cmcd_stage.py` | `revision/cmcd_torch/cmcd_stage.py` | verbatim | `e0e2e523153be1cfdc72c1dee794c1ca001d4be62566d43f96b58964c1090481` |
| `code/baselines/cmcd_dbs/summ_cmcd.py` | `revision/cmcd_torch/summ_cmcd.py` | verbatim | `31dd25644478558e4259f57ea3e5376ae688ed8af1e73e1bd6efa32800a04335` |
| `code/baselines/cmcd_dbs/tests/l2_exact.py` | `revision/cmcd_torch/tests/l2_exact.py` | verbatim | `ae1e119f91cbef55351ee8fe198c4df0384bd0701528fb88bc95f26ebe2e80ad` |
| `code/baselines/cmcd_dbs/tests/test_jax_formulas.json` | `revision/cmcd_torch/tests/test_jax_formulas.json` | verbatim | `e65ac7a0df29197661fd44794531571638a891c16bb229c7393605274b469d23` |
| `code/baselines/cmcd_dbs/tests/test_jax_formulas.py` | `revision/cmcd_torch/tests/test_jax_formulas.py` | verbatim | `df6a4effeedf426c9e43fa178fe0202bd66c5ee8d582afbed5a7c78ee2ef88a9` |
| `code/baselines/cmcd_dbs/tests/test_resume.py` | `revision/cmcd_torch/tests/test_resume.py` | verbatim | `19463b6176f597484b754928d0f5d8d3f2548ba46a1a2d0fb7514a00bbf4d02e` |
| `code/baselines/cmcd_dbs/tests/test_resume_cpu.json` | `revision/cmcd_torch/tests/test_resume_cpu.json` | verbatim | `a8fcd209519ec7cb1d308ef9fd06dfd24fffab942a54caa5e49668c8758a9855` |
| `code/baselines/cmcd_dbs/tests/test_resume_cuda.json` | `revision/cmcd_torch/tests/test_resume_cuda.json` | verbatim | `2c7b7a5c3dc3a0e7dd1d8ce29da2b5e313ebd945be40ccaa8ca2342997aee7e0` |
| `code/baselines/cmcd_dbs/tests/test_resume_rtcmcd.py` | `revision/cmcd_torch/tests/test_resume_rtcmcd.py` | modified | `5b847f0ce2d04460579ccef5955281994f7ce9d4753b414917ce22677920d84e` |
| `code/baselines/cmcd_dbs/tests/test_rt_exact.json` | `revision/cmcd_torch/tests/test_rt_exact.json` | verbatim | `e3e4e25a07bc1b73aeda51ceaffc9904ab6e72231516a1e89a233e73ee17d1ab` |
| `code/baselines/cmcd_dbs/tests/test_rt_exact.py` | `revision/cmcd_torch/tests/test_rt_exact.py` | verbatim | `f7414fec73475caec5aa8226e472052ab62e0b58a63fe032e883faf5aad5c88b` |
| `code/baselines/cmcd_dbs/tests/test_savefinal_resume.py` | `revision/cmcd_torch/tests/test_savefinal_resume.py` | verbatim | `62f8fc58f8a2070e14a3b573339dcdcadd2c5548999ec4f1a2acc0791eb3522e` |
| `code/baselines/cmcd_dbs/tests/test_weight_identity.json` | `revision/cmcd_torch/tests/test_weight_identity.json` | verbatim | `a508a99dcbc186f02f3d40213dff911863bc9be1f1c6c114f02a164f8c927002` |
| `code/baselines/cmcd_dbs/tests/test_weight_identity.py` | `revision/cmcd_torch/tests/test_weight_identity.py` | verbatim | `7b84d9c0aa3470e0b6d55aaf33bc96378bd391d789c1bfcb18ddf90344dc5bb7` |
| `code/baselines/cmcd_dbs/refine_cmcd.py` | `results_sync/revision/tasks/refine_cmcd.py` | verbatim | `a72c22de025a9539767cca1739b8aa663cb6d4809dd05defd152543a450451bc` |
| `code/baselines/cmcd_dbs/longchain_refined.py` | `results_sync/revision/tasks/longchain_refined.py` | modified | `df138eab26ef3a993ff8f33a26803f7f5923f42fe23d2a548d11a81803de920b` |
| `code/baselines/cmcd_dbs/diag_gf.py` | `results_sync/revision/tasks/diag_gf.py` | verbatim | `30fb50884acf65ae3133c6721ff3a75c694945d9c282d3dadbf05221c5805c3c` |
| `code/tasks/cmcd_dbs/launch_repro.sh` | `revision/cmcd_torch/launch_repro.sh` | verbatim | `47945564fe470017b88ca426b90b6813dcfafa3381a9b097abbec871d14b33ec` |
| `code/tasks/cmcd_dbs/train_snapshots.sh` | `revision/cmcd_torch/train_snapshots.sh` | verbatim | `a24d47f99edd40a49090e84718aa548300d2d7a1d8ba3dc9bb0efd43633d94b4` |
| `code/tasks/cmcd_dbs/train_snapshots_cluster.sh` | `revision/cmcd_torch/train_snapshots_cluster.sh` | verbatim | `03f2d05b43f5b4246ce9d235fbd20fab362f7994c04d0d8620a1c6213995de74` |
| `code/baselines/dw/dw_compare.py` | `revision/dw/dw_compare.py` | modified | `68ed18df61a9ac79c90ad5b14db4ae97a30cf6a83a8245e6715c9866fbd44667` |
| `code/baselines/dw/dw_flow.py` | `revision/dw/dw_flow.py` | verbatim | `8c8a8f17f711cca4f8ab60e9b7e59882a861ebe9eef724f787f574c33aaaecff` |
| `code/baselines/dw/dw_hmc_flip.py` | `revision/dw/dw_hmc_flip.py` | modified | `d09b99bebe6bbfcb1e8c033d53e347f30160d2fcb524241fc90299a58aa1375f` |
| `code/baselines/dw/chain_stats.py` | `revision/dw/chain_stats.py` | verbatim | `186b969b4590c166727c35c05759543a5e6d2b6cc25bfa0cf0638234b7473671` |
| `code/baselines/dw/dw_v2/dw_chains.py` | `revision/dw/dw_v2/dw_chains.py` | verbatim | `c5841c0250fb425d7cc3ab31275ab52609147fcf6db92168f8fd4f1690d181ed` |
| `code/baselines/dw/dw_v2/eval_block_dw_nhmc.py` | `revision/dw/dw_v2/eval_block_dw_nhmc.py` | verbatim | `ef2f991238a27f67459190491dc5071086d596f4ad2e5e66a4eb8f96b4e93371` |
| `code/baselines/dw/dw_v2/eval_fixed_flip_ablation.py` | `revision/dw/dw_v2/eval_fixed_flip_ablation.py` | verbatim | `7ff70594258d3784ee35a2bc93dde7f97a916cae133e51e89953a8c587c37117` |
| `code/baselines/dw/dw_v2/make_dw_imh_figure.py` | `revision/dw/dw_v2/make_dw_imh_figure.py` | verbatim | `9f9d1a6d9a47e0359a16a2c61928af7b8cadf3600d4da491924ada2385bd4436` |
| `code/baselines/dw/dw_v2/neural_nhmc_dw.py` | `revision/dw/dw_v2/neural_nhmc_dw.py` | verbatim | `82afd51dc9b8fad5042bfd423468d7e8a531662a58183deafa944d18321f5ef5` |
| `code/baselines/dw/dw_v2/pisgrad_net_torch.py` | `revision/dw/dw_v2/pisgrad_net_torch.py` | verbatim | `0134e70fd5c0feb9c681e203fa6b62993cfc81752d48995115e39994e094a468` |
| `code/baselines/dw/dw_v2/test_block_dw_nhmc.py` | `revision/dw/dw_v2/test_block_dw_nhmc.py` | verbatim | `f5adf81f1b7ff421d968dddf585fdbc0eae3ef8efa826795d5cdbb8191d9de10` |
| `code/baselines/dw/dw_v2/test_dw_chains.py` | `revision/dw/dw_v2/test_dw_chains.py` | verbatim | `2419aa81025fd0f21be6f44f02a88a16cc60e071777af53a7f49cb1f3a71fbbf` |
| `code/baselines/dw/dw_v2/train_block_dw_nhmc.py` | `revision/dw/dw_v2/train_block_dw_nhmc.py` | verbatim | `1f347aaec0232b07d0f70aa8d29354ba63fdd9546476614cece4884c46299e4e` |
| `code/baselines/lj13/lj13_compare.py` | `revision/lj13/lj13_compare.py` | modified | `191595e9bea3db934590a3b43c5a9af7f45c54a68dc6471d5986d691d7572485` |
| `code/baselines/lj13/lj13_hmc.py` | `revision/lj13/lj13_hmc.py` | modified | `d95a06fe63c3e1694947d06375d46d9430a9fabdb7c5b2ec6bd3d2c668d6282d` |
| `code/baselines/lj13/lj13_pt_reference.py` | `revision/lj13/lj13_pt_reference.py` | modified | `c977b2010b81f912e586399a5fc0072e9a10f115a7aa1f8122222d72f539cea7` |
| `code/baselines/lj13/train_lj_rtbuf.py` | iqcb-02 `/media/scratch/mqian/lj_buffer/code_rtlocal/train_lj_rtbuf.py` (the version the cluster runs used; the Mac copy 0181f030 is the older code_bak_20261012) | verbatim | `447ec47322b3…` |
| `code/baselines/lj13/particle/ce_stage.py` | iqcb-02 `/media/scratch/mqian/lj_buffer/code_rtlocal/ce_stage.py` | verbatim | `3eb2aabee45b…` |
| `code/baselines/lj13/particle/equiv_drift.py` | `particle/code/equiv_drift.py` | verbatim | `622dbf10dcbe0343b6948fc8699e117aa65ca0775532640b9cd04b4bbc51d33b` |
| `code/baselines/lj13/particle/eval_lj_cluster_nhmc.py` | `particle/code/eval_lj_cluster_nhmc.py` | verbatim | `fb04cfb04554bfe4a82eb0d60dcf8ca8e0f803c8bd4eca007a8a955cd78f8148` |
| `code/baselines/lj13/particle/lj13_chain_diagnostics.py` | `particle/code/lj13_chain_diagnostics.py` | verbatim | `d90b9692ef9f571a880a39cd824852ad0fd8509936f7d367c0e545b1dd06727a` |
| `code/baselines/lj13/particle/lj13_nhmc.py` | `particle/code/lj13_nhmc.py` | verbatim | `8ffd9e47db739c3a863c3e633ab12e5ba0765639f1a97b52234bf7f618c3f17f` |
| `code/baselines/lj13/particle/neural_nhmc_dw.py` | `particle/code/neural_nhmc_dw.py` | verbatim | `6cb80b41b15e350fb60ba0425221db8207d7dc6b2834230454671d2c16c683da` |
| `code/baselines/lj13/particle/particle_targets.py` | `particle/code/particle_targets.py` | verbatim | `644f19a71e22d51d86be479343d3357c4aa539ea513ec15dc400d344b8bbeca5` |
| `code/baselines/wolff_hmc/README.md` | `revision/baselines/spin_wolff_hmc/README.md` | modified | `12d690c59beaa9898059eae988f02cb57fce4b84039b2a94dbbd99e9ac0d6d31` |
| `code/baselines/wolff_hmc/spin/ising.py` | `revision/baselines/spin_wolff_hmc/spin/ising.py` | verbatim | `6272f98d5262a15a79b446950639476cf9e9b6d20e74abda8be1f77f14ce62c3` |
| `code/baselines/wolff_hmc/spin/xy.py` | `revision/baselines/spin_wolff_hmc/spin/xy.py` | verbatim | `4710f57d0c0f6ec2a8448344e1f32cc9fa94e746bf7c24c137c0ca21442be2bb` |
| `code/baselines/wolff_hmc/spin/_smoke_spin.py` | `revision/baselines/spin_wolff_hmc/spin/_smoke_spin.py` | verbatim | `a120a26230f95d8545f9ee39dd6a96846c81437d63505cb1e857f3816272340f` |
| `code/baselines/wolff_hmc/spin/README.md` | `revision/baselines/spin_wolff_hmc/spin/README.md` | verbatim | `760cef148dd19196262272316ac125a8802a36a16c065ad949bb458806bb4219` |
| `code/baselines/wolff_hmc/spin/lattice.py` | `revision/baselines/spin_wolff_hmc/spin/lattice.py` | verbatim | `9ff9bcb0ea7bb12857cd61f3b067020bd500fafdf944e35e867666cd1c562212` |
| `code/baselines/wolff_hmc/spin/run_spin.py` | `revision/baselines/spin_wolff_hmc/spin/run_spin.py` | verbatim | `5227dfffedbfb467108ab00e8bb9747951b047e26ebc8d3d4ea514042db6f1fe` |
| `code/baselines/wolff_hmc/spin/phi4.py` | `revision/baselines/spin_wolff_hmc/spin/phi4.py` | verbatim | `70fa9eda7fdfc5f75b87fffc55ea03cbe13fc37eed4c00f58da5d996cf5c3e5f` |
| `code/baselines/wolff_hmc/common/SCHEMA.md` | `revision/baselines/spin_wolff_hmc/common/SCHEMA.md` | verbatim | `b5d16223c433b47de08c69ac296d1dd0709e6863a38b3f9001ba9b14b9950d82` |
| `code/baselines/wolff_hmc/common/__init__.py` | `revision/baselines/spin_wolff_hmc/common/__init__.py` | verbatim | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `code/baselines/wolff_hmc/common/wolff_tau.py` | `revision/baselines/spin_wolff_hmc/common/wolff_tau.py` | verbatim | `d543d9f3c994613d0a4e5580e8d3d4fd25d3f4dd2167aff7d59c3147235ce4d6` |
| `figures/bench_timing_20261118/make_table.py` | `results_sync/revision/bench_timing_20261118/code/make_table.py` | modified | `cd4745fb53c5a7455a8018d188e41665a8a8a3259e56f233d158154a57393255` |
| `figures/figs_20261117u2/u2_common.py` | `results_sync/revision/figs_20261117u2/u2_common.py` | modified | `a9db7fb0b6a7d7b0efadd55cf2f52192875039bd7888a26051c0018b2f1c38d9` |
| `figures/figs_20261117u2/phi4_figs_u2.py` | `results_sync/revision/figs_20261117u2/phi4_figs_u2.py` | modified | `443ed8c471eccaefd1a47b17387ecdf6f07fb32adbefc17dabbf52c65f055c37` |
| `figures/figs_20261117u2/make_figs_u2.py` | `results_sync/revision/figs_20261117u2/make_figs_u2.py` | modified | `eda1bc5ee1fa83b7f5657089d3619572792c60f72efa55a8db6011b4b9a7982f` |
| `figures/figs_20261117u2/make_fig14_acf_u2.py` | `results_sync/revision/figs_20261117u2/make_fig14_acf_u2.py` | modified | `922081aa7b0c4539036afeafb5d2a6183ceafa831a57de5208705bc34092f53c` |
| `figures/figs_20261117u3/make_stage_scan_u3.py` | `results_sync/revision/figs_20261117u3/make_stage_scan_u3.py` | modified | `26623c85c19348514bfede99ef1f50718421ca7e4bdbf646d2d0789de4fb19c0` |
| `figures/figs_20261117u3/u3_common.py` | `results_sync/revision/figs_20261117u3/u3_common.py` | modified | `a71ef75e689b8ff54bc7be0ebc67088ddf7436feadf07e9870804e58ed20d054` |
| `figures/figs_20261117u3/phi4_figs_u3.py` | `results_sync/revision/figs_20261117u3/phi4_figs_u3.py` | modified | `5de8d158ad357c5c5209b02b02c65a0bc72c2ea0f4ce45b0de9535312617cc57` |
| `figures/figs_20261117u3/fig_phi4_main_u3.py` | `results_sync/revision/figs_20261117u3/fig_phi4_main_u3.py` | modified | `c8768da41322a42cbcccc208d554022274314f0a9aaee5ac0f3f61cf8b726f9d` |
| `figures/figs_20261117u3/make_figs_u3.py` | `results_sync/revision/figs_20261117u3/make_figs_u3.py` | modified | `cd6f514c9813382e786e372e5b284117bb110e4a7f8f69419564a301b761f72e` |
| `figures/figs_20261117u4/make_figs_u4.py` | `results_sync/revision/figs_20261117u4/make_figs_u4.py` | modified | `8dccba596c639d9ef38f7aae31075c794129522f8fb1cb39e858616eb2bbad20` |
| `figures/figs_20261117u4/fig_scaling_accept_u4.py` | `results_sync/revision/figs_20261117u4/fig_scaling_accept_u4.py` | modified | `65a9624f6c8c164bdc370c7208ff91d8509d1212448752e151015098c5c0cdf2` |
| `figures/figs_20261117u4/fig_scaling_u4.py` | `results_sync/revision/figs_20261117u4/fig_scaling_u4.py` | modified | `484d9459bac09b904fb6aab5081d73c68e366d6dd15f4743788f84af8081474a` |
| `figures/figs_20261117u4/u4_common.py` | `results_sync/revision/figs_20261117u4/u4_common.py` | modified | `4a228df3bf136ca3504dbaeef3c5be9d4dab2292eb6c941ff19822d0c250e584` |
| `figures/figs_20261117u4/make_bar_figs_u4.py` | `results_sync/revision/figs_20261117u4/make_bar_figs_u4.py` | modified | `494bef8aa2a8f0b98b75e5a2867f4bccb4217dc48a14fcdcdcf338a80c6648ad` |
| `figures/figs_20261117u4/phi4_figs_u4.py` | `results_sync/revision/figs_20261117u4/phi4_figs_u4.py` | modified | `91f754d90b498aef256342f0d9d606ba375f0f6256fa776e9eaad7e85d24d8b1` |
| `figures/figs_20261117u4/fig5_conditional_flow_u4.py` | `results_sync/revision/figs_20261117u4/fig5_conditional_flow_u4.py` | modified | `491f4550be91d925dfac61735bd67c3552c07556fee5785882d7728f8d078ca1` |
| `figures/ti_wolff_hmc/ti_logz.py` | `results_sync/revision/ti_wolff_hmc/ti_logz.py` | verbatim | `994d9dc23769aff8a1d400631fd2714a67ef5b79da06fca951531857fcdd4183` |
| `figures/figs_20261117u/nhmc_style.py` | `results_sync/revision/figs_20261117u/nhmc_style.py` | verbatim | `38774bf9cfb9f713742430e5931003f6c20c793a94d1141010c1be585e36a76c` |
| `figures/jackknife_20261117/jk_udb.py` | `results_sync/revision/jackknife_20261117/jk_udb.py` | verbatim | `636e641fa57f5222953e25272580ebb1fc5f77af26f21f157d5311b48edec1ed` |
| `figures/jackknife_20261117/jk_dwlj.py` | `results_sync/revision/jackknife_20261117/jk_dwlj.py` | verbatim | `6b80083461209371c69e013e7be746dbd334cf960236f41c57163cb4cca897f6` |
| `figures/jackknife_20261117/make_summary.py` | `results_sync/revision/jackknife_20261117/make_summary.py` | modified | `94a2310cc763ff3ebc85dbd78886fcbd50f81864b00fd5cb9d3875c8973fda02` |
| `figures/jackknife_20261117/jk_snis.py` | `results_sync/revision/jackknife_20261117/jk_snis.py` | verbatim | `17c721c5caab9e3a51d4db41f2a43f9cb496ed5c18e6eac1a6e77f161cad8227` |
| `figures/jackknife_20261117/jk_jump.py` | `results_sync/revision/jackknife_20261117/jk_jump.py` | verbatim | `cca200fdf7bd592d80bb0c2cc2c2821dedc67393a3b610e30397b809137113ca` |
| `figures/jackknife_20261117/jk_cells.py` | `results_sync/revision/jackknife_20261117/jk_cells.py` | verbatim | `41a332a97ba8ec4e519ec7e87775f9bf2d0699d96ab29b3f64c2a481f636225f` |
| `figures/jackknife_20261117/jk_lib.py` | `results_sync/revision/jackknife_20261117/jk_lib.py` | modified | `4af15f4bbf14dc09c9f108d079ecb462c04ee8b61ac9fd10ed2456d986779716` |
| `figures/jackknife_20261117/jk_fits.py` | `results_sync/revision/jackknife_20261117/jk_fits.py` | verbatim | `c5863ec89bf93e72ca9dbd4add45027becafc533c6fa509dbe98288d6ce1104d` |
| `figures/jackknife_20261117/jk_stage.py` | `results_sync/revision/jackknife_20261117/jk_stage.py` | verbatim | `6a8aeb79d8b698156fab6e422b73e41c688099c286bef5ea0d3e25cff3fc1462` |
| `figures/unify_20261118/item6_ref_tau.py` | `results_sync/revision/unify_20261118/item6_ref_tau.py` | modified | `fe40eb31b81292f4bb598d2c09c2e6a31c4a85323db1ac0e66fad0842ec4a519` |
| `figures/unify_20261118/item4_lj13_local.py` | `results_sync/revision/unify_20261118/item4_lj13_local.py` | modified | `482d7d9615e5a7c210653eabb3985751bb3fb5c0d6ffb33aadf2adb15dafe544` |
| `figures/unify_20261118/ucommon.py` | `results_sync/revision/unify_20261118/ucommon.py` | verbatim | `af870aaf1dffbb5ff16ac02d615954a8c8cb1f16ab6003f3113d5ac996025722` |
| `figures/unify_20261118/item12_bar.py` | `results_sync/revision/unify_20261118/item12_bar.py` | modified | `5e7634dfc032bc3803f109980830658d45d8e2988fa2d07bba98ddac4719586a` |
| `figures/unify_20261118/item5_ess.py` | `results_sync/revision/unify_20261118/item5_ess.py` | modified | `1a1a17daf50268d26fdad68218db7e1fd5feb4dce3e9b531eebcda49cbba941c` |
| `figures/unify_20261118/item3_fig14.py` | `results_sync/revision/unify_20261118/item3_fig14.py` | modified | `f9762f8b42f34b4f582b8f363f9349659f375c91bc9f295f1730366f719efee0` |
| `figures/unify_20261118/make_index.py` | `results_sync/revision/unify_20261118/make_index.py` | modified | `faaf50efe6d4e07ab5bda832f3e1ee10ee8dfe235bbc0e5538ed2dab1166b4b8` |
| `figures/unify_20261118/item4_floor.py` | `results_sync/revision/unify_20261118/item4_floor.py` | modified | `d5ea3cca78a7116704f9752c8dad3c9eee2a827880b2ddd9827db18ff8ec6c97` |
| `figures/unify_20261118/item4_merge.py` | `results_sync/revision/unify_20261118/item4_merge.py` | modified | `a8a69d011c46d84b76f43ceace9dc776441bf627d87a2d1e730db9c38fc80a26` |
| `figures/figs_20261119u6/logz_stats_u6.py` | `results_sync/revision/figs_20261119u6/logz_stats_u6.py` | verbatim | `2b8ce7a9d42c84acf681650eeda90648ab8c9bedd2ec95e24e9f337eb5a0cfa7` |
| `figures/figs_20261119u6/fig_logz_u6.py` | `results_sync/revision/figs_20261119u6/fig_logz_u6.py` | modified | `b1365d58270a88515ba44a79d6ddf5adb9e4f960b45cd70c54112763607caefb` |
| `figures/figs_train_20261118/make_train_curves.py` | `results_sync/revision/figs_train_20261118/make_train_curves.py` | modified | `24afb96895ae2fc263ccf66f1952aed3c623ab8ed7d058f473cbac3c2d743b38` |
| `figures/figs_20261118u5/fig_scaling_tau_u5.py` | `results_sync/revision/figs_20261118u5/fig_scaling_tau_u5.py` | modified | `661bef21ef4b44782de78a4704f04a78f96d8777f1d04fa29d5c38930cad24ee` |
| `figures/figs_20261118u5/u5_common.py` | `results_sync/revision/figs_20261118u5/u5_common.py` | modified | `820c621523168edeef92ab9959ca75103fc149927554400c513c8444e5a5bf84` |
| `figures/figs_20261118u5/dw_lj_figs_u5.py` | `results_sync/revision/figs_20261118u5/dw_lj_figs_u5.py` | modified | `6976009bdcdc40051546990f7e9da089f27cc3e2d9a83da6d9eb6495af6a8af7` |
| `figures/figs_20261118u5/make_bar_prefix_u5.py` | `results_sync/revision/figs_20261118u5/make_bar_prefix_u5.py` | modified | `14b8939bcd12b8f777d8df749c395d78ccb4d5d73c680975e99eba8282e1e60c` |
| `figures/audits/chain_audit_20261106/chain_audit.py` | `results_sync/revision/audits/chain_audit_20261106/chain_audit.py` | verbatim | `e1c37c4e492bde5ca49f2b4e9e2ff22687d2ee848c5e8a6dbf28a389b57e4586` |
| `figures/jackknife_20261117/cluster/eval_W_dump.py` | `results_sync/revision/jackknife_20261117/cluster/eval_W_dump.py` | verbatim | `e37003a78c3a6d53812de0bf48150f8a9b78878f2456105ab0729ac3efe5dc89` |
| `figures/zenodo_20261118/capture_v10.py` | `zenodo_prep_20261118/final/tools/capture_v10.py` | verbatim | `07b1f5dea0c8e3f6eb5b326773be947a19328525f05c85a1cda9b90ec2801780` |
| `figures/zenodo_20261118/redraw_all.sh` | `zenodo_prep_20261118/final/tools/redraw_all.sh` | verbatim | `213f3e9d790f7759f5455435d983b502c8b35539be3d470afd26895bc72f0c60` |
| `figures/zenodo_20261118/compare_v10.py` | `zenodo_prep_20261118/final/tools/compare_v10.py` | verbatim | `e51a87525a480341ba4a1feaacc3038dbf9298201b3d71f0908a0eecd9ec10aa` |
| `figures/zenodo_20261118/tables/make_tab_udb_refine.py` | `zenodo_prep_20261118/final/tools/tables/make_tab_udb_refine.py` | verbatim | `6e62bfec1d3cb373718425e54632bc868c092d6f805b6380474a9ef916fa3ed0` |
| `figures/zenodo_20261118/tables/tab_common.py` | `zenodo_prep_20261118/final/tools/tables/tab_common.py` | verbatim | `f2d4ff7200ec3fd28d9fd8a8404d9fdde2d4946b87fd583d869c394cdb926cdc` |
| `figures/zenodo_20261118/tables/make_tab_thermo.py` | `zenodo_prep_20261118/final/tools/tables/make_tab_thermo.py` | verbatim | `8e04022a0c185fc039985ccce76c4fe8d85cd2e8dbe4ef96784314fb466d91f6` |
| `figures/zenodo_20261118/tables/make_tab_udb_compare.py` | `zenodo_prep_20261118/final/tools/tables/make_tab_udb_compare.py` | verbatim | `27b74605b035626d6abfbe835854c35aa5ba84eac4c055a7436ba0e663172bf3` |
| `figures/zenodo_20261118/tables/make_tab_tau_per_size.py` | `zenodo_prep_20261118/final/tools/tables/make_tab_tau_per_size.py` | verbatim | `c42dc30f2480ae46907aaa656442e981a692593f2e1088bda93dffd1475e780d` |
| `code/tasks/timing/train_log_wall.py` | `results_sync/revision/bench_timing_20261118/code/train_log_wall.py` | modified | `5f86c946b0e964e2662c3ba49a54462b1a9cf77672da78dc610685f6f20b2e7f` |
| `code/tasks/timing/make_results.py` | `results_sync/revision/bench_timing_20261118/code/make_results.py` | verbatim | `7e0c3221dd71bf37b5f5e38a90b1fc30c6122353bad6438fafb59c11b557b234` |
| `code/tasks/analysis_cluster/item4_floor_pg_him.py` | `results_sync/revision/unify_20261118/floor_pg_20261118/item4_floor_pg_him.py` | verbatim | `34757a59c69b3d569378750ad996856ee1852437d02e86c8d279bb784ae4424d` |
| `code/tasks/analysis_cluster/item4_eq_w2_him.py` | `results_sync/revision/unify_20261118/item4_eq_w2_him.py` | verbatim | `7375bfdc177f1e4bad3e34b14de00aea7624278f5e26775a5589fd61560c9e9b` |
| `code/tasks/analysis_cluster/jk_cells_remote.py` | `results_sync/revision/jackknife_20261117/cluster/jk_cells_remote.py` | verbatim | `9bc9697d013ceed3a4b48290d57a759781deceb0c8179a87d3005c2f49727803` |
| `tests/l2_exact.py` | `revision/cmcd_torch/tests/l2_exact.py` | verbatim | `ae1e119f91cbef55351ee8fe198c4df0384bd0701528fb88bc95f26ebe2e80ad` |

## Notes from the HIM equivalence test (2026-11-20, jobs 94801, 94807)
- All original-vs-repo comparisons bitwise identical (training, chains, free energy, forward evaluation, CMCD/DBS, DW, LJ13, TI).
- The cluster copies of `logz_estimators.py` (HIM, iqcb revision_code) are the pre-2026-11-10 v1 (swapped bridge weights); every
  paper BAR number was computed with the Mac v2 (EST_VERSION = 2, checked in the figure caches and udb_bar_summary.json), which is the
  version shipped here.
- iqcb `longchain.py` lacks `--pregen-wait` and walks pregen shards in a different order (chain values unchanged); HIM copy = Mac copy.

## Test rerun (2026-11-20, jobs 94845, 94846, 94851)
- `tests/test_tau_rule.py`: 5/5 pass (BM over 16 AR(1) chains; Sokal-only median within 10 % of g/2).
- `tests/test_reversibility.py`: the chain test now starts from exact draws (rejection on the quadrature box, path-IMH record
  from a reverse path); from forward-path starts the random kernel (path-IMH acceptance 0.01-0.04) does not equilibrate
  within the test length. 5/5 pass: exact draws max|z| 0.38; path-IMH 1.38, RT 0.73, pRT 2.03; no-MH control |z| 6614.
- `code/baselines/hmc_z2/phi4_hmc_z2.py`: `revision/phi4/phi4_hmc_z2.py` with `observables`, `summarise`, `KEYS` copied
  verbatim from `revision/phi4/phi4_compare.py` and the imports pointed at `code/`; output JSON identical to the original
  (L = 4, kappa = 0.27, 2 runs, same seed; timing fields excluded).
- Notebook with the bundled `model/kernel_final.pt` (MD5 9f68f751...): executed, 490 s on 4 cores.

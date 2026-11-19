"""Collect the jackknife outputs (cells_*.json, fits_jk_{paper,buf}.json, snis_jk.json, stage_jk.json, udb_jk.json) into
jackknife_summary.json and the Chinese jackknife_summary.md, with the ZH manuscript locations (v10, 2026-11-17).
Error conventions (see the scripts): tau = paper median, error = delete-one-chain jackknife of the chain mean ('jk_chain');
acc error = block jackknife on the accept flags (20 blocks/chain) if traces exist, else delete-one-chain; pilot runs with
< 30 accepted transitions: Poisson sqrt(n_acc)/N instead (0 accepted -> 95 % upper limit 3/N)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jk_lib import OUT  # noqa: E402

J = lambda n: json.load(open(os.path.join(OUT, n)))
CELLS = {}
for g in ('E2', 'buf', 'T16'):
    CELLS.update(J(f'cells_{g}.json'))
FP, FB = J('fits_jk_paper.json')['paper'], J('fits_jk_buf.json')['buf']
REF = J('fits_jk_paper.json')['ref_tau']
SN, ST, UD = J('snis_jk.json'), J('stage_jk.json'), J('udb_jk.json')
Z = 'sections/{}_zh.tex'
A = 'appendices/v10_app_{}_zh.tex'


def cell(L, k, m, pre='E2'):
    n = f'{pre}_L{L}_k0{k}_{m}'
    c = CELLS.get(n) or (CELLS.get(f'buf_L{L}_k0{k}_{m}') if pre == 'E2' else None)
    return c


def acc_err(c):
    npost = (c['n_transitions'] - c['burn']) * c['n_chains']
    if c.get('n_acc') is not None and c['n_acc'] < 30:
        return (np.sqrt(c['n_acc']) if c['n_acc'] > 0 else 3.0) / npost, 'Poisson' if c['n_acc'] > 0 else '95%UL'
    if c.get('acc_jk_block') is not None:
        return c['acc_jk_block'], 'block-jk'
    return c['acc_jk_chain'], 'chain-jk'


def fmt(v, e, nd=None):
    if v is None:
        return '—'
    if e is None or not np.isfinite(e) or e == 0:
        return f'{v:.4g}'
    if nd is None:
        nd = max(0, -int(np.floor(np.log10(abs(e)))) + 1)
        nd = min(nd, 6)
    return f'{v:.{nd}f} ± {e:.{nd}f}'


S = dict(conventions=__doc__, cells={}, fits_paper=FP, fits_buf=FB, ref_tau=REF, stage=ST)
md = []
w = md.append
w('# NHMC 论文定量结果的 jackknife 误差（2026-11-17）\n')
w('数据只读；脚本与输出都在 `results_sync/revision/jackknife_20261117/`。全部在本机运行，每个脚本运行时间小于 30 秒。\n')
w('**误差口径**')
w('- τ_int：沿用正文统计量，即 4 条链 max(Sokal, BM)/2 的中位数。误差取逐一删链 jackknife 下链均值的误差（记作 jk_chain）。')
w('  - 4 条链时，中位数本身的删一 jackknife 极不稳定，只作对照，存于 JSON 的 `*_jk_chain` 与 `stat_med`。')
w('  - 有 trace 时，另给"每链切 4 段、共 16 段"的分段 jackknife 作交叉检查。')
w('- 接受率：有 accept-flag trace 时，用 20 块/链的分块 jackknife；没有 trace 时逐一删链。')
w('  - 接受数少于 30 的试跑用 Poisson 误差；0 次接受给 95% 上限 3/N。')
w('- 拟合量的误差分两部分。')
w('  - **stat**：各输入点统计误差的 jackknife 传播（逐点逐链替换，各点方差相加）。')
w('  - **pts**：逐一删数据点的 jackknife（C 删格点，逐 κ 拟合与有效指数删 L）。')
w('- "paper" 为与 `fig_scaling_20261116.py` 完全相同的数据选择。"buf" 在其基础上补入漏读的 `buf_*`（L12、L6 的 κ=.27/.30），对应 REVIEW_code B1。\n')

# ------------------------------------------------------------------ 1 key per-cell numbers
w('## 1. 正文引用的单格点 τ_int 与接受率\n')
w('| 量 | 稿中值 | jackknife 值（paper 选择） | 收敛 | 稿件位置 |')
w('|---|---|---|---|---|')
KEY = [
    ('L16 κ.27 RT τ(|m|)', 16, '27', 'rt', 'tau', '9.6', f"{Z.format('section_I')}:16, {Z.format('section_IV')}:80, {Z.format('section_S')}:113, {Z.format('section_VII')}:11-13, {A.format('C')}:168, {A.format('D')}:31"),
    ('L16 κ.27 path-IMH τ(|m|)', 16, '27', 'imhG', 'tau', '180', f"{Z.format('section_I')}:16, {Z.format('section_VII')}:11, {A.format('D')}:31"),
    ('L16 κ.27 pRT τ(|m|)', 16, '27', 'rtp12', 'tau', '18', f"{Z.format('section_VII')}:12, {A.format('C')}:168, {A.format('D')}:31"),
    ('L8 κ.27 RT τ(|m|)', 8, '27', 'rt', 'tau', '1.9', f"{Z.format('section_IV')}:80, {A.format('D')}:31"),
    ('L16 κ.24 pRT τ', 16, '24', 'rtp12', 'tau', '2.8', f"{Z.format('section_VII')}:12"),
    ('L16 κ.24 RT τ', 16, '24', 'rt', 'tau', '4.5', f"{Z.format('section_VII')}:12"),
    ('L16 κ.30 pRT τ', 16, '30', 'rtp12', 'tau', '1.4', f"{Z.format('section_VII')}:12, {Z.format('section_IV')}:79"),
    ('L16 κ.30 RT τ', 16, '30', 'rt', 'tau', '3.7', f"{Z.format('section_VII')}:12"),
    ('L16 κ.27 s=2 τ', 16, '27', 'rtp2', 'tau', '≈7.5', f"{Z.format('section_VII')}:13"),
    ('L16 κ.27 s=4 τ', 16, '27', 'rtp4', 'tau', '≈7.5', f"{Z.format('section_VII')}:13"),
    ('L16 κ.27 s=6 τ', 16, '27', 'rtp6', 'tau', '≈7.5', f"{Z.format('section_VII')}:13"),
    ('L16 κ.27 RT 接受率', 16, '27', 'rt', 'acc', '0.14', f"{Z.format('abstract')}:2, {Z.format('section_IV')}:44"),
    ('L16 κ.27 path-IMH 接受率', 16, '27', 'imhG', 'acc', '0.13', f"{Z.format('section_IV')}:44"),
]
S['key_cells'] = []
for lab, L, k, m, q, paper, loc in KEY:
    c = cell(L, k, m)
    if not c:
        continue
    if q == 'tau':
        v, e = c['tau_absM'], c['tau_absM_jk_chain_mean']
        extra = f" (分段 {fmt(c.get('tau_absM_seg_median'), c.get('tau_absM_seg_median_jk'))})" if c.get('tau_absM_seg_median') else ''
    else:
        v, (e, how) = c['acc'], acc_err(c)
        extra = f' ({how})'
    S['key_cells'].append(dict(label=lab, paper=paper, value=v, err=e, converged=c['converged'], loc=loc, file=c['file']))
    w(f"| {lab} | {paper} | {fmt(v, e)}{extra} | {'是' if c['converged'] else '**否（下界）**'} | {loc} |")
kish = SN['E2_L16_k027']
w(f"| L16 κ.27 Kish ESS/N（65536 条正向路径，64 块 jk） | 0.012 / ≈1% | {fmt(kish['kish'], kish['kish_jk'])} | — | {Z.format('abstract')}:2, {Z.format('section_IV')}:44 |")
c1, c2 = cell(16, '27', 'rt'), cell(16, '27', 'imhG')
d = c1['acc'] - c2['acc']; e = np.hypot(acc_err(c1)[0], acc_err(c2)[0])
w(f"| L16 κ.27 接受率差 RT−IMH | 0.14 vs 0.13 | {fmt(d, e)} | — | {Z.format('section_IV')}:44 |")
h, eh = REF['hmc_L16']['tau'], REF['hmc_L16']['err']
r = h / c1['tau_absM']; er = r * np.hypot(eh / h, c1['tau_absM_jk_chain_mean'] / c1['tau_absM'])
w(f"| L16 κ.27 HMC/RT τ 比 | ≈7 | {fmt(r, er)} | — | {Z.format('section_VII')}:14 |")
w('')
# 9-cell IMH vs RT, pRT/RT
w('**九组 (L=8,12,16 × κ=.24,.27,.30) 的比值**（`section_VII_zh.tex:11-12`）\n')
w('| L | κ | acc(RT)−acc(IMH) | τ(IMH)/τ(RT) | acc(pRT)/acc(RT) | τ(pRT)/τ(RT) |')
w('|---|---|---|---|---|---|')
S['nine'] = []
for L in (8, 12, 16):
    for k in ('24', '27', '30'):
        rt, im, pr = cell(L, k, 'rt'), cell(L, k, 'imhG') or cell(L, k, 'imh'), cell(L, k, 'rtp12')
        if not (rt and im and pr):
            continue
        da = rt['acc'] - im['acc']; eda = np.hypot(acc_err(rt)[0], acc_err(im)[0])
        rti = im['tau_absM'] / rt['tau_absM']; erti = rti * np.hypot(im['tau_absM_jk_chain_mean'] / im['tau_absM'], rt['tau_absM_jk_chain_mean'] / rt['tau_absM'])
        ra = pr['acc'] / rt['acc']; era = ra * np.hypot(acc_err(pr)[0] / pr['acc'], acc_err(rt)[0] / rt['acc'])
        rtp = pr['tau_absM'] / rt['tau_absM']; ertp = rtp * np.hypot(pr['tau_absM_jk_chain_mean'] / pr['tau_absM'], rt['tau_absM_jk_chain_mean'] / rt['tau_absM'])
        S['nine'].append(dict(L=L, k=k, dacc=(da, eda), tau_imh_over_rt=(rti, erti), acc_prt_over_rt=(ra, era), tau_prt_over_rt=(rtp, ertp),
                              imh_converged=im['converged']))
        w(f"| {L} | 0.{k} | {fmt(da, eda)} | {fmt(rti, erti)}{'' if im['converged'] else ' (IMH 下界)'} | {fmt(ra, era)} | {fmt(rtp, ertp)} |")
w('')

# ------------------------------------------------------------------ 2 table tau_per_size
w('## 2. 附录 D 表 `tab:tau_per_size`（`v10_app_D_zh.tex:31`）：τ_int(|m|) / τ_int(sign) ± jk\n')
w('NHMC 的误差为删链 jackknife（链均值）。HMC/Wolff |m| 的误差有两种来源。')
w('- L12/16 κ.27（以及 κ.26）有 1e6 条 trace：做 16 段 jackknife。')
w('- 其余格点：用 Madras–Sokal 公式 τ·sqrt(2(2W+1)/N)，取 W≈5g。')
w('- HMC/Wolff 的 sign 列来自 2e5 条记录，未在此处重算。\n')
w('| κ | 方法 | L=4 | L=8 | L=12 | L=16 |')
w('|---|---|---|---|---|---|')
S['tau_table'] = {}
G = json.load(open(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/figs/data/wh_tau_grid.json'))
for k in ('24', '27', '30'):
    for alg in ('hmc', 'wolff'):
        row = []
        for L in (4, 8, 12, 16):
            v = [x for x in G.values() if x['alg'] == alg and x['L'] == L and round(x['kappa'] * 100) == int(k)][0]
            t = v['tau_absm'] / 2
            if k == '27':
                e = REF[f'{alg}_L{L}']['err']
            else:
                e = t * np.sqrt(2 * (2 * 5 * 2 * t + 1) / v['n'])
            row.append(fmt(t, e))
            S['tau_table'][f'{alg}_L{L}_k{k}_absM'] = (t, e)
        w(f"| 0.{k} | {alg.upper() if alg == 'hmc' else 'Wolff'} (|m|) | " + ' | '.join(row) + ' |')
    for m, lab in (('rt', 'RT'), ('rtp12', 'pRT'), ('imhG', 'path-IMH')):
        row = []
        for L in (4, 8, 12, 16):
            c = cell(L, k, m) or (cell(L, k, 'imh') if m == 'imhG' else None)
            if not c:
                row.append('—'); continue
            s = f"{fmt(c['tau_absM'], c['tau_absM_jk_chain_mean'])} / {fmt(c['tau_sign'], c['tau_sign_jk_chain_mean'])}"
            if not c['converged']:
                s += ' (下界)'
            row.append(s)
            S['tau_table'][f'{m}_L{L}_k{k}'] = dict(absM=(c['tau_absM'], c['tau_absM_jk_chain_mean']), sign=(c['tau_sign'], c['tau_sign_jk_chain_mean']),
                                                    converged=c['converged'])
        w(f'| 0.{k} | {lab} | ' + ' | '.join(row) + ' |')
w('')

# ------------------------------------------------------------------ 3 fits
w('## 3. 拟合量（第 V 节、附录 D）\n')
w('格式：值 ± 总误差。总误差为 stat 与 pts 两项的平方和，括号内依次列出 [stat, pts]。\n')
w('| 量 | 稿中值 | paper 选择 | 含 buf_*（74 点） | 稿件位置 |')
w('|---|---|---|---|---|')


def F(key, d):
    r = d.get(key)
    if not r or not isinstance(r.get('value'), float):
        return str(r and r.get('value'))
    s = fmt(r['value'], r['total'])
    return s + f" [{r['stat']:.2g}, {r['pts']:.2g}]" if r.get('pts') else s + f" [{r['stat']:.2g}, —]"


FITROWS = [
    ('C', 'C', '1.18', f"{Z.format('section_I')}:16, {Z.format('section_S')}:79,102, {Z.format('section_VII')}:18, {A.format('D')}:17"),
    ('C_npts', '点数', '70', f"{Z.format('section_S')}:79, {A.format('D')}:17"),
    ('C_rms', 'C 的对数残差 rms', '≈15%', f"{Z.format('section_S')}:79, {A.format('D')}:17"),
    ('sig2_peak', 'σ̄² 峰值（κ=0.265）', '0.0173', f"{Z.format('section_S')}:57, {A.format('D')}:17"),
    ('sig2_rt_k20', 'σ̄²(κ=.20)', '0.0060', f"{Z.format('section_S')}:57, {Z.format('section_V')}:64, {A.format('D')}:17"),
    ('sig2_rt_k27', 'σ̄²(κ=.27)', '0.0165', f"{A.format('D')}:17"),
    ('s0_rt_k27', 'σ0²(κ=.27)', '≈0.16', f"{A.format('D')}:17"),
    ('sig2_rt_k30', 'σ̄²(κ=.30)', '0.0082', f"{A.format('D')}:17"),
    ('s0_rt_k30', 'σ0²(κ=.30)', '≈0.25（破缺相）', f"{A.format('D')}:17"),
    ('s0_broken_mean_k275_30', 'σ0² 均值 κ=.275–.30', '≈0.25', f"{A.format('D')}:17"),
    ('prt_over_rt_sig2_k27', 'σ̄²(pRT)/σ̄²(RT), κ=.27', '约 1/4', f"{Z.format('section_S')}:57"),
    ('efold_volume_k27', 'e 倍体积 4/σ̄²(κ=.27)', '≈240', f"{Z.format('section_S')}:85, {A.format('D')}:17"),
    ('maxres_rt', '逐 κ 拟合最大残差', '0.024', f"{A.format('D')}:6,17"),
    ('medres_rt_of_kmax', '各 κ 最大残差的中位数', '0.009', f"{A.format('D')}:6"),
    ('accept_native_maxdev', '7 模型 |预测−实测| 最大', '≤0.022', f"{Z.format('section_S')}:34"),
    ('exp_rt_k27', 'RT 有效指数（L=8–16）', 'L^2.3', f"{Z.format('section_IV')}:80, {Z.format('section_S')}:112, {A.format('D')}:17"),
    ('exp_rtp12_k27', 'pRT 有效指数', '（未引用）', ''),
    ('exp_imhG_k27', 'path-IMH 有效指数', '（未引用）', ''),
    ('exp_hmc_k27', 'HMC 有效指数（L=8,12,16）', 'L^1.5', f"{Z.format('section_IV')}:80, {Z.format('section_S')}:112, {A.format('D')}:17"),
    ('exp_wolff_k27', 'Wolff 有效指数', 'L^0.4', f"{Z.format('section_IV')}:80, {Z.format('section_S')}:112, {A.format('D')}:17"),
    ('crossover_L_fit', 'RT/HMC 交点（HMC 三点幂律）', 'L≈29', f"{Z.format('section_S')}:113, {A.format('D')}:17"),
    ('crossover_L_anchor15', '交点（HMC=65.4(L/16)^1.5）', 'L≈29', f"{Z.format('section_S')}:113, {A.format('D')}:17"),
    ('hmc_over_rt_L16_k27', 'HMC/RT τ, L16 κ.27', '≈7（65 对 9.6）', f"{Z.format('section_S')}:113, {Z.format('section_VII')}:14"),
    ('trtau_meas_L24_k26', 'L24 κ.26 τ 实测', '40.6', f"{A.format('D')}:17"),
    ('trtau_pred_L24_k26', 'L24 κ.26 τ 预测', '40.2（代码 39.85）', f"{A.format('D')}:17"),
    ('trtau_meas_L24_k27', 'L24 κ.27 τ 实测（未收敛，下界）', '126', f"{A.format('D')}:17,61"),
    ('trtau_pred_L24_k27', 'L24 κ.27 τ 预测', '50', f"{A.format('D')}:17"),
    ('trtau_ratio_L24_k27', 'L24 κ.27 实测/预测', '2.5 倍', f"{Z.format('section_S')}:80, {A.format('D')}:17"),
    ('trfit_g_k20_rt', '迁移拟合 σ̄², κ=.20 RT', '0.0058（对 0.0060）', f"{Z.format('section_V')}:64"),
    ('trfit_g_k24_rt', '迁移拟合 σ̄², κ=.24 RT', '0.0102（对 0.0109）', f"{Z.format('section_V')}:64"),
]
for key, lab, paper, loc in FITROWS:
    w(f'| {lab} | {paper} | {F(key, FP)} | {F(key, FB)} | {loc} |')
w('')
w(f"RT 有效指数所用的 L：paper 选择为 {FP['exp_rt_k27_Ls']['value']}，缺 L12。原因是 L12 κ=.27 只有 buf_* 文件，被 native() 漏读。buf 选择为 {FB['exp_rt_k27_Ls']['value']}。")
w(f"交点误差的分解（paper 选择，pts 部分）：{ {k: round(v, 2) for k, v in FP['_crossover_parts'].items()} }。\n")
w('**pRT 的 C = τ/(1/α−1/2)**（`section_S_zh.tex:88-89`；稿中写 κ=.20/.24/.30 时 C≈1.2–2，κ=.27 时从 L4 的 3.5 增至 L16 的 11）\n')
w('| κ | L=4 | L=6 | L=8 | L=10 | L=12 | L=16 |')
w('|---|---|---|---|---|---|---|')
for k in ('20', '24', '27', '30'):
    w(f'| 0.{k} | ' + ' | '.join(F(f'C_prt_L{L}_k{k}', FB) if f'C_prt_L{L}_k{k}' in FB else '—' for L in (4, 6, 8, 10, 12, 16)) + ' |')
w('')

# acceptance law
w('## 4. 接受率公式的检验（`section_S_zh.tex:34,36`，图 fig_scaling_accept）\n')
w('偏差 = 实测 − erfc(σ_s/2)，误差只含实测接受率的 jackknife。σ_s² 无法 jackknife，因为 wtail2 JSON 只存了组内方差的总均值。')
w('σ_p 是 σ_s² 抽样误差的高斯近似：每个 σ_s² 由 1024 个中间场 × R=16 条延续路径估计，相对误差约 sqrt(2/15)/sqrt(1024) ≈ 1.1%，再传播到预测接受率。σ_p 仅作参考，不是 jackknife。\n')
w('| 模型 | 更新 | 实测 − 预测 | σ_meas | σ_p（近似） | z |')
w('|---|---|---|---|---|---|')
import glob, re  # noqa: E401,E402
from scipy.special import erfc  # noqa: E402
CV = {}
for f in glob.glob(os.environ.get('NHMC_ROOT', '.') + '/results_sync/revision/him_figdata_20261116/himout/*/revision_runs/wtail2_20261114/*.json'):
    L_, k_ = re.match(r'L(\d+)_k0(\d+)', os.path.basename(f)).groups()
    CV[(int(L_), k_)] = {int(s): v['cond_var'] for s, v in json.load(open(f))['stages'].items()}
S['accept_law'] = []
for key in sorted(k for k in FP if k.startswith('accdev_')):
    L_, k_, m_ = re.match(r'accdev_L(\d+)_k(\d+)_(\w+)', key).groups()
    s2 = CV[(int(L_), k_)][0 if m_ == 'rt' else 12]
    sp = abs((erfc(np.sqrt(s2 * 1.011) / 2) - erfc(np.sqrt(s2 * 0.989) / 2)) / 2)
    c = cell(int(L_), k_, m_)
    em = acc_err(c)[0]
    v = FP[key]['value']
    z = v / np.hypot(em, sp)
    S['accept_law'].append(dict(model=f'L{L_}_k{k_}', m=m_, dev=v, err_meas=em, err_pred_approx=sp, z=z))
    w(f"| L{L_} κ.{k_} | {m_} | {v:+.4f} | {em:.4f} | {sp:.4f} | {z:+.1f} |")
w('')
w('**迁移接受率：实测/预测**（预测为 L16 的 σ_s²/256 × V，无拟合参数）\n')
w('| κ | 更新 | L | 接受数 | 实测接受率 | 实测/预测 |')
w('|---|---|---|---|---|---|')
S['accept_transfer'] = []
for k in ('24', '27', '30'):
    for m in ('rt', 'rtp12'):
        for L in (24, 32, 48, 64, 96):
            c = CELLS.get(f'T16_L{L}_k0{k}_{m}')
            if not c:
                continue
            e, how = acc_err(c)
            s = 0 if m == 'rt' else 12
            p = erfc(np.sqrt(CV[(16, k)][s] * L * L / 256) / 2)
            S['accept_transfer'].append(dict(L=L, k=k, m=m, n_acc=c['n_acc'], acc=c['acc'], err=e, how=how, pred=p, ratio=c['acc'] / p, ratio_err=e / p))
            rs = f'< {e / p:.3g}（上限）' if c['n_acc'] == 0 else fmt(c['acc'] / p, e / p)
            w(f"| 0.{k} | {m} | {L} | {c['n_acc']} | {'< ' + format(e, '.2g') if c['n_acc'] == 0 else fmt(c['acc'], e)} ({how}) | {rs} |")
w('')

# ------------------------------------------------------------------ 5 stage scan, Var(log w)
w('## 5. 路径段数标度与 Var(log w)（附录 D 第 8 行、图 fig_stage_scan，`section_S_zh.tex:51,58,60`）\n')
s = ST
w('| 量 | 稿中值 | jackknife 值 | 说明 |')
w('|---|---|---|---|')
for key, lab, paper in (('slope_T_L8_fig', 'T 指数（图中硬编码种子，L=8）', '−0.81 / T^{-0.8}'),
                        ('slope_T_L8_files_L8_T4-32', 'T 指数（同步文件，L=8，T=4–32）', '—'),
                        ('slope_T_L4', 'T 指数 L=4（含正文 T64 值）', '−0.81 到 −0.82'),
                        ('slope_T_L6', 'T 指数 L=6', '−0.81 到 −0.82'),
                        ('slope_T_L8', 'T 指数 L=8（文件 + 正文 T64）', '−0.81 到 −0.82')):
    r = s[key]
    w(f"| {lab} | {paper} | {fmt(r['value'], np.hypot(r['jk_seed'], r['jk_T']))} [seed {r['jk_seed']:.3f}, 删 T {r['jk_T']:.3f}] | 删一种子 / 删一 T |")
g = s['global_fit']
w(f"| 全局拟合 V 指数 | 0.74 | {fmt(g['V_exp'], np.hypot(g['V_exp_jk_seed'], g['V_exp_jk_cell']))} | 14 个 (L,T) 格 |")
w(f"| 全局拟合 T 指数 | −0.81 | {fmt(g['T_exp'], np.hypot(g['T_exp_jk_seed'], g['T_exp_jk_cell']))} | 同上 |")
w(f"| 全局拟合对数残差 rms | 2.7% | {100 * g['log_rms']:.1f}% | 同上 |")
r = s['T_propto_V_exp']
w(f"| T ∝ V^(1/b) | V^1.2 / V^1.23 | {fmt(r['value'], r['err'])} | 由图斜率 |")
r = s['slopes_T16']
w(f"| T=16 斜率 L4→8 | 0.0165 | {fmt(r['slope_4_8'], r['slope_4_8_err'])} | 种子 jk |")
w(f"| T=16 斜率 L8→16 | 0.0165 | {fmt(r['slope_8_16'], r['slope_8_16_err'])} | L16 单模型：65536 路径 jk |")
w(f"| 常数项 | 0.16 | {fmt(r['intercept'], r['intercept_err'])} | |")
for nm, lab, paper in (('E2_L16_k027', 'L16 κ.27 Var(log w)（raw_snis 65536）', '4.37（eval 262144）'),
                       ('E2_L8_k027', 'L8 κ.27 Var(log w)', '1.22'), ('E2_L4_k027', 'L4 κ.27 Var(log w)', '0.43')):
    q = SN[nm]
    w(f"| {lab} | {paper} | {fmt(q['var_logw'], q['var_logw_jk'])} | 64 块 jk |")
w('')
w('注：图 fig_stage_scan 的 T=64（L8 为 0.382，L4 为 0.139/0.143/0.144）与 T=8 种子 0（2.156，文件中为 2.141）在同步目录里没有来源（REVIEW C1），这里按正文值引入，并标出。\n')

# ------------------------------------------------------------------ 6 Kish / UDB
w('## 6. Kish ESS 与 CMCD/DBS 对照表（附录 G `tab:udb_compare`）\n')
w('表中的长链（rtp:60）与 N=102400 评估只在集群上。本地 `logz_udb_20261116/*.npz` 保存的是同一批模型的独立样本：4×20000 RT 链与 65536 条正向路径。下面用它们给出误差代理，并与表中数值对照。\n')
w('| 模型 | κ | 表中 Var | jk Var | 表中 ESS | jk ESS | 表中 RT acc/τ | jk RT acc / τ |')
w('|---|---|---|---|---|---|---|---|')
S['udb'] = UD
for key, r in sorted(UD.items()):
    if key.startswith('nhmc') or 'L8' not in key:
        continue
    t = r.get('table', {})
    w(f"| {key.split('_')[0].upper()} | 0.{key[-2:]} | {t.get('var', 0):.3f} | {fmt(r['var_logw'], r['var_logw_jk'])} | {t.get('ess', 0):.3f} | {fmt(r['kish'], r['kish_jk'])} | "
      f"{t.get('rt_acc', 0):.3f} / {t.get('rt_tau', 0):.2f} | {fmt(r['acc'], r['acc_jk'])} / {fmt(r['tau'], r['tau_jk_chain_mean'])} |")
w('')
JU, DL = J('jump_jk.json'), J('dwlj_jk.json')
S['jump'] = JU; S['dwlj'] = DL
w('## 7. 被接受更新的 |m| 跳幅（`section_S_zh.tex:82,90`，`section_IV_zh.tex:44`，图 new_fig_acceptance (b)）\n')
w('J/sd 为每次被接受更新的平均 |Δ|m|| 除以 |m| 的标准差。独立样本对应 2/√π ≈ 1.128。J/indep 为同一跳幅除以独立样本对的 E|X−X′|，后者取精确的 Gini 平均差。误差为分块 jackknife（20 块/链）。\n')
w('| L | κ | RT J/sd | RT J/indep | pRT J/sd | pRT J/indep | IMH J/indep |')
w('|---|---|---|---|---|---|---|')
for L in (8, 16):
    for k in ('24', '27', '30'):
        g = lambda m, q: fmt(JU[f'L{L}_k{k}_{m}'][q], JU[f'L{L}_k{k}_{m}'][q + '_jk']) if f'L{L}_k{k}_{m}' in JU else '—'
        w(f"| {L} | 0.{k} | {g('rt', 'J_sd')} | {g('rt', 'J_indep')} | {g('rtp12', 'J_sd')} | {g('rtp12', 'J_indep')} | {g('imhG', 'J_indep')} |")
w('')
w('## 8. 附录 F：MW 与 LJ13（删一 run / 删一种子 jackknife）\n')
w('| 系统 | 方法 | 量 | jackknife 值 | 种子/run 范围 | 稿中 |')
w('|---|---|---|---|---|---|')
for key, v in DL['MW'].items():
    r = v['mode_change_rate']
    w(f"| {key.split('_')[0].upper().replace('DW', 'MW')} | {key.split('_', 1)[1]} | 模式切换率/步 | {fmt(r['mean'], r['jk'])} (n={r['n']}) | {r['min']:.4g}–{r['max']:.4g} | `v10_app_F_zh.tex:16-19`：0.15–0.2；HMC 5e-4；HMC+flip ≈0.3 |")
for lab in ('IMH', 'RT', 'pRT'):
    v = DL['LJ13'][lab]
    for q, nm, paper in (('energy_w2', '能量 W2', 'IMH 0.55–0.63；RT/pRT 0.38–0.45'), ('dist_w2', '距离 W2', '0.007–0.010'), ('eq_w2', '几何 W2', '1.58–1.59；IMH 1.61–1.63')):
        r = v[q]
        w(f"| LJ13 | {lab} | {nm} | {fmt(r['mean'], r['jk']) if r['n'] > 1 else format(r['mean'], '.4f') + '（单种子，无 jk）'} (n={r['n']}) | {r['min']:.3f}–{r['max']:.3f} | `v10_app_F_zh.tex:31-34`：{paper} |")
fl = DL['LJ13'].get('floor', {})
if fl:
    w(f"| LJ13 | 参考下限（20 次重复，sd） | 能量 / 距离 / 几何 W2 | {fl['energy_w2']['mean']:.3f} ± {fl['energy_w2']['sd']:.3f} / {fl['dist_w2']['mean']:.4f} ± {fl['dist_w2']['sd']:.4f} / {fl['eq_w2']['mean']:.3f} ± {fl['eq_w2']['sd']:.3f} | — | 0.42±0.10，0.006±0.003，1.446±0.005 |")
w('')
w('## 9. 正文引用但链未收敛的格点（τ 是下界，jackknife 误差不适用）\n')
w('| 文件 | τ(|m|) | 接受率 | 稿件 |')
w('|---|---|---|---|')
for n in ('T16_L24_k027_rt', 'T16_L24_k027_rtp12', 'T16_L32_k027_rt', 'T16_L32_k027_rtp12', 'T16_L32_k024_rt', 'T16_L32_k026_rt'):
    c = CELLS.get(n)
    if c:
        w(f"| {n} | {c['tau_absM']:.4g} | {c['acc']:.4g} | {'是' if c['converged'] else '**未收敛**'} |")
unc = sorted(n for n, c in CELLS.items() if 'converged' in c and not c['converged'] and (n.startswith('E2_') or n.startswith('buf_') or n.startswith('T16_L24') or n.startswith('T16_L32')))
w(f"\n未收敛的 E2/buf/T16(L24,32) 文件共 {len(unc)} 个：{', '.join(unc)}\n")
S['unconverged'] = unc
json.dump(S, open(os.path.join(OUT, 'jackknife_summary.json'), 'w'), indent=1, default=float)
head = open(os.path.join(OUT, 'summary_head_zh.txt')).read() if os.path.exists(os.path.join(OUT, 'summary_head_zh.txt')) else ''
open(os.path.join(OUT, 'jackknife_summary.md'), 'w').write(head + '\n'.join(md) + '\n')
print('ok', len(md), 'lines')

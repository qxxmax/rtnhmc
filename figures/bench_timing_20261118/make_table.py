#!/usr/bin/env python3
"""results.json -> table_draft.md (ZH + EN table, appendix-B subsection draft ZH + EN)."""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'timing')
R = json.load(open(os.path.join(ROOT, 'results.json')))
C = R['configs']


def f(x, nd=3):
    if x is None:
        return '--'
    if x >= 100:
        return f'{x:.0f}'
    return f'{x:#.{nd}g}'


def ms(x):
    return '--' if x is None else f(1e3 * x)


def sec(cn, lab):
    s = C[cn].get('main', {}).get('sampling', {})
    for k, v in s.items():
        if k == lab or (lab in ('prt_d4', 'prt_dT4') and k == 'prt_d4=dT4'):
            return v['sec_per_update']['median']
    return None


def row(cn, mark=''):
    c = C[cn]
    m = c.get('main', {})
    T = c['T']
    tr = m.get('train', {})
    a = tr.get('A', {}).get('median') if tr.get('A') else None
    b = tr.get('B', {}).get('cycle') if tr.get('B') else None
    fp = m.get('forward_path', {})
    d4 = sec(cn, 'prt_d4') if T > 4 else None
    dT4 = sec(cn, 'prt_dT4')
    n_imh, n_rt = 6 * T, 12 * T
    n_q, n_4 = 12 * (T // 4), (48 if T > 4 else None)
    cnt = f'{n_imh} / {n_rt} / {n_q}' + (f' / {n_4}' if n_4 and T != 16 else '')
    return (f"| {c['L']} | {T}{mark} | {f(a)} / {f(b)} | {ms(sec(cn, 'imh'))} / {ms(sec(cn, 'rt'))} / {ms(dT4)} / "
            f"{'--' if T == 4 else ('=' if T == 16 else ms(d4))} | {ms(fp.get('batch1024_s_per_path'))} "
            f"({ms(fp.get('batch4_s_per_path'))}) | {cnt} |")


BLOCKS = [('vol', ['nat_L4_T16', 'nat_L8_T16', 'nat_L12_T16', 'nat_L16_T16']),
          ('T8', ['ks_L8_T4', 'ks_L8_T8', 'ks_L8_T16', 'ks_L8_T32', 'ks_L8_T64']),
          ('T16', ['ks_L16_T8', 'nat_L16_T16', 'ks_L16_T32', 'ks_L16_T64'])]


def table(lang):
    if lang == 'zh':
        head = ('| L | T | 训练 s/epoch（缓冲前 / 缓冲阶段） | 采样 ms/更新（path-IMH / RT / pRT d=T/4 / pRT d=4） | '
                '正向路径 ms/条，批 1024（批 4） | 每次更新每条链的 ∇S 次数（IMH / RT / pRT d=T/4 / pRT d=4） |')
        tit = {'vol': '体积扫描（原生模型，T=16）', 'T8': 'T 扫描，L=8', 'T16': 'T 扫描，L=16'}
    else:
        head = ('| L | T | training s/epoch (before buffer / buffer phase) | sampling ms/update (path-IMH / RT / '
                'pRT d=T/4 / pRT d=4) | forward path ms/path, batch 1024 (batch 4) | grad-S evaluations per chain per '
                'update (IMH / RT / pRT d=T/4 / pRT d=4) |')
        tit = {'vol': 'Volume scan (native models, T = 16)', 'T8': 'T scan, L = 8', 'T16': 'T scan, L = 16'}
    out = [head, '|' + '---|' * 6]
    for key, names in BLOCKS:
        out.append(f'| **{tit[key]}** | | | | | |')
        for n in names:
            out.append(row(n, '*' if (key == 'T16' and n == 'nat_L16_T16') else ''))
    return '\n'.join(out)


LOGMAP = [  # benchmark config -> (source, model dir) of the real training run
    ('nat_L4_T16', 'iqcb', 'L4_k0.27_buf_s0'), ('nat_L8_T16', 'iqcb', 'L8_k0.27_buf_s0'),
    ('nat_L12_T16', 'iqcb', 'L12_k0.27_buf_s0'), ('nat_L16_T16', 'iqcb', 'L16_k0.27_buf_s0'),
    ('ks_L8_T4', 'him', 'L8_k27_K4_s0'), ('ks_L8_T8', 'him', 'L8_k27_K8_s0'), ('ks_L8_T8', 'iqcb', 'L8_k27_K8_s0'),
    ('ks_L8_T16', 'him', 'L8_k27_K16_s1'), ('ks_L8_T32', 'him', 'L8_k27_K32_s1'), ('ks_L8_T32', 'iqcb', 'L8_k27_K32_s0'),
    ('ks_L8_T64', 'him', 'L8_k27_K64_s1'), ('ks_L8_T64', 'iqcb', 'L8_k27_K64_s0'),
    ('ks_L16_T8', 'him', 'L16_k27_K8_s0'), ('ks_L16_T32', 'him', 'L16_k27_K32_s0'), ('ks_L16_T64', 'iqcb', 'L16_k27_K64_s0')]
HIM_THREADS = {'93231': 16, '93310': 16, '93718': 16, '93865': 16, '94375': 16, '93237': 4, '93867': 4}


def logs_table(lang):
    W = json.load(open(os.path.join(ROOT, 'train_log_wall.json')))
    if lang == 'zh':
        out = ['| 模型 | L | T | 设备 | 完成周期 | 由 samp/s 累加 (h) | 修正偶数周期后 (h) | eval_log 墙钟 (h) | 续跑次数 | 本基准推算 20000 周期 (h) |']
    else:
        out = ['| model | L | T | device | epochs done | sum from samp/s (h) | even-epoch corrected (h) | eval_log wall (h) | resumes | this benchmark, 20000 epochs (h) |']
    out.append('|' + '---|' * 10)
    for cn, src, md in LOGMAP:
        r = W[src].get(md)
        if r is None:
            continue
        b = C[cn]['main']['train']['B']
        corr = r['pre_buffer_h'] + r['buffer_phase_h'] * b['cycle'] / b['median_even']
        if src == 'iqcb':
            devs = 'iqcb GPU (L40S, shared)' if lang == 'en' else 'iqcb GPU（L40S，共享）'
        else:
            job = r['source'].split('/')[1]
            devs = f"HIM CPU, {HIM_THREADS.get(job, '?')} thr (job {job})" if lang == 'en' else f"HIM CPU，{HIM_THREADS.get(job, '?')} 线程（作业 {job}）"
        wall = r['eval_wall_last_h']
        wall_s = '--' if wall is None else f"{wall:.1f}" + ('' if not r['n_resumes'] else ('†' ))
        out.append(f"| {md} | {r['L']} | {r['T']} | {devs} | {r['epochs_logged']} | {r['train_h_from_sps']:.1f} | {corr:.1f} | "
                   f"{wall_s} | {r['n_resumes'] if r['n_resumes'] is not None else 'n/a'} | {C[cn]['main']['train_20000_epochs_h_est']:.1f} |")
    return '\n'.join(out)


def main():
    dev = R['device'] or {}
    rp = {n: C[n].get('repeat', {}) for n in ('nat_L8_T16', 'nat_L16_T16')}
    rep_lines = []
    for n, e in rp.items():
        if e.get('sampling'):
            a = {k: v['sec_per_update']['median'] for k, v in C[n]['main']['sampling'].items()}
            b = {k: v['sec_per_update']['median'] for k, v in e['sampling'].items()}
            rel = max(abs(b[k] / a[k] - 1) for k in a if k in b)
            ta = C[n]['main']['train']; tb = e.get('train', {})
            relt = max(abs(tb[p]['median'] / ta[p]['median'] - 1) for p in 'AB' if tb.get(p) and ta.get(p)) if tb else None
            rep_lines.append(f'{n}: sampling max |rel. diff| {rel:.3f}, training {relt if relt is None else round(relt, 3)}')
    txt = f"""# phi4 training and sampling timings (appendix B draft, 2026-11-18)

Device: {dev.get('cpu')} ({dev.get('sockets')} x {dev.get('cores_per_socket')} cores, {dev.get('threads_per_core')} thread/core), HIMSTER3 node {dev.get('host_line', '').split()[1] if dev.get('host_line') else '?'}, job 94496; PyTorch {dev.get('torch')}, 4 threads (OMP/MKL/torch), pinned to 4 cores with taskset, all configurations run one after another. kappa = 0.27, lambda = 0.022, 5 leapfrog substeps per stage, 4 chains per method.
Repeat at the end of the job (drift check): {'; '.join(rep_lines) if rep_lines else 'n/a'}.
Data: results.json (per-config medians, mean, sd, q10/q90; plot-ready series under "series"). Raw: raw/main, raw/repeat (copies of HIM bench_timing_20261118/out).

## 中文版

{table('zh')}

注：每次更新指 4 条链（同一张量中批处理）各完成一次转移，取 20 次热化后 200 次更新的中位数。pRT d=T/4 即 s=3T/4（正文与 T 扫描链的设置）；pRT d=4 即 s=T-4；T=16 时两者都是 s=12（表中记为 "="）；T=4 时 d=4 即完整往返（"--"）。正向路径为基分布抽样、T 个正向阶段与作用量，括号内为批 4 时的每条路径耗时。训练为批 48 的单个周期，第 1–10 周期热化，第 11–60 周期取中位数；“缓冲前”对应正式训练的第 1–1000 周期，“缓冲阶段”对应第 1001–20000 周期，按两周期取平均（偶数周期另外推进 64 条 RT 链一步）。∇S 次数按代码计：每阶段 5 个 drift、6 次 kick，每次 kick 计算一次 −∇S 和一次力残差网络，另有 4 个动量网络；path-IMH 为 6T，RT 为 12T，pRT 为 12d。* 为原生 L=16 模型，与体积扫描同一行。

## English version

{table('en')}

Notes: one update = one transition of 4 chains held in one batched tensor; median of 200 updates after 20 warm-up updates. pRT d = T/4 is s = 3T/4 (setting of the paper and of the T-scan chains); pRT d = 4 is s = T - 4; at T = 16 both are s = 12 ("="); at T = 4, d = 4 is the full RT ("--"). Forward path = base draw, T forward stages and the action; in brackets the per-path time at batch 4. Training: one epoch at batch 48, epochs 1-10 warm-up, median of epochs 11-60; "before buffer" corresponds to epochs 1-1000 of a training run and "buffer phase" to epochs 1001-20000, averaged over two epochs (even epochs also advance the 64 RT chains by one step). grad-S counts are exact for the code: per stage 5 drifts and 6 kicks, each kick one -grad S and one force-residual network, plus 4 momentum networks; path-IMH 6T, RT 12T, pRT 12d. * native L = 16 model, same row as in the volume scan.

## 附注：已有训练日志中的实际训练时间（设备不统一，只作参考）

{logs_table('zh')}

由 samp/s 累加：train_log.csv 每 200 个周期记录一次单个周期的 samp/s，按段乘以周期数累加，不含每 2000 个周期一次的评估。记录的周期都是偶数周期，缓冲阶段的偶数周期包含 RT 链推进，因此累加值偏高；“修正后”用本基准测得的同一 (L,T) 的“两周期平均 / 偶数周期”比值修正缓冲阶段（比值取自 CPU，对 GPU 一批只是近似）。eval_log 墙钟为最后一次评估时距进程启动的时间，含评估；† 表示该模型续跑过，墙钟只覆盖最后一段。iqcb 一批为共享 GPU，HIM 一批为 CPU（4 或 16 线程，节点上同时有其他作业）。最后一列为本基准（4 线程）按 1000 个缓冲前周期加 19000 个缓冲阶段周期推算，不含评估与启动。

## Note: actual training times from the existing logs (different devices; for reference only)

{logs_table('en')}

Sum from samp/s: train_log.csv records the samp/s of a single epoch every 200 epochs; segments are summed, the evaluation every 2000 epochs is not included. All logged epochs are even and in the buffer phase even epochs include the RT-chain step, so the sum is too high; "corrected" rescales the buffer phase by the ratio (two-epoch average / even epoch) measured here for the same (L, T) (ratio from the CPU benchmark; approximate for the GPU runs). eval_log wall = time since process start at the last evaluation, including evaluations; † = the run was resumed, the wall time covers only the last segment. iqcb runs: shared GPU; HIM runs: CPU (4 or 16 threads, other jobs on the node). Last column: this benchmark (4 threads), 1000 epochs before the buffer plus 19000 buffer-phase epochs, without evaluation and start-up.

## 附录 B 小节草稿（中文）

\\subsection{{训练与采样耗时}}
表~\\ref{{tab:timing}} 的数字均在同一台设备上测得：HIMSTER3 的一个计算节点（{dev.get('cpu')}），PyTorch {dev.get('torch')}，每个配置固定 4 个线程并绑定到同样的 4 个核，各配置依次运行。模型取 $\\kappa=0.27$，每阶段五个 leapfrog 子步；体积扫描使用 $T=16$ 的原生模型，$T$ 扫描使用 $L=8$、$T=4$–$64$ 与 $L=16$、$T=8$–$64$ 的模型。
训练耗时为每批 $48$ 条路径的一个训练周期，热化 $10$ 个周期后取 $50$ 个周期的中位数，分别给出加入缓冲区反向项之前和之后的值；缓冲阶段每两个周期推进一次 RT 链，表中取两个周期的平均。计时从对应模型的已训练参数起步。
采样耗时为 $4$ 条链批处理时一次更新的耗时，热化 $20$ 次后取 $200$ 次更新的中位数，方法为 path-IMH、完整往返与截断往返；截断往返给出 $d=T/4$（正文设置，$T=16$ 时 $s=12$）与固定 $d=4$ 两种回退深度。正向路径一栏为生成一条完整正向路径的耗时，即 SNIS 与预生成 path-IMH 提议所用的一步，按每批 $1024$ 条生成并取每条的平均。
最后一栏为每次更新每条链计算 $-\\nabla S$ 的次数，与硬件无关：每阶段五个 drift、六次 kick，故 path-IMH 为 $6T$，完整往返为 $12T$，截断往返为 $12d$；每次 kick 另含一次力残差网络，每阶段另有四个动量网络。

## Appendix B subsection draft (English)

\\subsection{{Training and sampling time}}
All numbers in Table~\\ref{{tab:timing}} were measured on one device, a HIMSTER3 compute node ({dev.get('cpu')}), with PyTorch {dev.get('torch')}, 4 threads pinned to the same 4 cores for every configuration, and the configurations run one after another. The models are at $\\kappa=0.27$ with five leapfrog substeps per stage; the volume scan uses the native $T=16$ models, the $T$ scan the models with $L=8$, $T=4$--$64$ and $L=16$, $T=8$--$64$.
Training time is the time of one epoch with a batch of $48$ paths, the median of $50$ epochs after $10$ warm-up epochs, given before and after the buffer term is switched on; in the buffer phase the RT chains advance every second epoch, and the table gives the average over two epochs. The timed runs start from the trained parameters of the corresponding model.
Sampling time is the time of one update of $4$ chains processed as one batch, the median of $200$ updates after $20$ warm-up updates, for path-IMH, the full round trip and the truncated round trip; the truncated round trip is given for $d=T/4$ (the setting of the main text, $s=12$ at $T=16$) and for a fixed $d=4$. The forward-path column gives the time to generate one complete forward path, the step used by SNIS and by pre-generated path-IMH proposals, generated in batches of $1024$ and given per path.
The last column gives the number of $-\\nabla S$ evaluations per chain and update, which does not depend on the hardware: each stage has five drifts and six kicks, so path-IMH needs $6T$, the full round trip $12T$ and the truncated round trip $12d$; each kick also evaluates the force-residual network once, and each stage evaluates four momentum networks.
"""
    open(os.path.join(ROOT, 'table_draft.md'), 'w').write(txt)
    print('wrote table_draft.md')


if __name__ == '__main__':
    main()

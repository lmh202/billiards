"""
evaluate_parallel.py - 并行化的 Agent 评估脚本（最小化对原有逻辑的改动）

说明：
- 本脚本把原来的逐局顺序评估改为多进程并行评估，每个进程独立运行若干局。
- 单局逻辑与 `evaluate.py` 保持一致（Env/Agent 初始化和交互流程不变），只是在进程级别并行化。
- 为尽量利用显存，提供 `--use-gpu` 和 `--gpu-devices` 选项：
  - 如果 Agent 内部使用 PyTorch 并自动选择 GPU（或读取 `CUDA_VISIBLE_DEVICES`），该选项会为每个子进程设置不同的 `CUDA_VISIBLE_DEVICES`。
  - 注意：此项不会修改 Agent 内部代码——仅设置环境变量，能够在不改逻辑下让支持 GPU 的模型利用显存。

使用示例：
    python evaluate_parallel.py --n-games 120 --n-workers 4
    python evaluate_parallel.py --n-games 32 --n-workers 2 --use-gpu --gpu-devices 0,1

作者：自动化优化脚本
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict


def _run_games_worker(worker_id: int, n_games: int, seed: int, target_choices, use_gpu: bool, gpu_device: str, quiet: bool=False) -> Dict[str, int]:
    """在子进程中运行 n_games 局，并返回统计结果（与 evaluate.py 保持一致的统计口径）。

    注意：该函数会在子进程内导入项目模块（PoolEnv/agents），确保每个子进程环境是独立的。
    """
    # 可选：为子进程设置显卡（如果需要）
    if use_gpu and gpu_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    # 在子进程中导入依赖（避免主进程导入导致不可序列化的大对象）
    from utils import set_random_seed
    from poolenv import PoolEnv
    from agents import BasicAgent, BasicAgentPro, NewAgent

    set_random_seed(enable=False, seed=seed)

    env = PoolEnv()
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}

    # 轮换先后手与球型与 evaluate.py 保持一致
    players = [BasicAgent(), NewAgent()]

    # 如果需要静默输出，则重定向 stdout/stderr 到 null（比替换 print 更安全）
    if quiet:
        import sys
        # 使用 utf-8 并忽略无法编码的字符，避免 Windows 控制台编码问题
        devnull = open(os.devnull, 'w', encoding='utf-8', errors='ignore')
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull

    try:
        for i in range(n_games):
            env.reset(target_ball=target_choices[i % len(target_choices)])
            while True:
                player = env.get_curr_player()
                obs = env.get_observation(player)
                if player == 'A':
                    action = players[i % 2].decision(*obs)
                else:
                    action = players[(i + 1) % 2].decision(*obs)
                step_info = env.take_shot(action)
                done, info = env.get_done()
                if done:
                    if info['winner'] == 'SAME':
                        results['SAME'] += 1
                    elif info['winner'] == 'A':
                        results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
                    else:
                        results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
                    break
    finally:
        if quiet:
            try:
                import sys
                sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
                devnull.close()
            except Exception:
                pass

    return results


def merge_results(accum: Dict[str, int], part: Dict[str, int]):
    for k, v in part.items():
        accum[k] = accum.get(k, 0) + v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-games', type=int, default=120)
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-gpu', action='store_true', default=True, help='尝试为子进程分配 GPU（仅设置环境变量）')
    parser.add_argument('--gpu-devices', type=str, default=None,
                        help='逗号分隔的 GPU id 列表，例如 "0,1,2"（当 --use-gpu 时生效）')
    parser.add_argument('--quiet', action='store_true', help='在子进程中静默打印以提升吞吐（会屏蔽 print 输出）')
    args = parser.parse_args()

    n_games = args.n_games
    n_workers = max(1, args.n_workers)
    seed = args.seed
    use_gpu = args.use_gpu

    if args.gpu_devices:
        gpu_list = [d.strip() for d in args.gpu_devices.split(',') if d.strip() != '']
    else:
        gpu_list = []

    # 将总局数等分到每个 worker（尽量均匀）
    base = n_games // n_workers
    extras = n_games % n_workers
    per_worker = [base + (1 if i < extras else 0) for i in range(n_workers)]

    target_choices = ['solid', 'solid', 'stripe', 'stripe']

    aggregated = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}

    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        start_seed = seed
        for wid in range(n_workers):
            wg = per_worker[wid]
            if wg == 0:
                continue
            # 为每个 worker 分配一个 GPU id（如果提供）或 None
            gpu_device = None
            if use_gpu and gpu_list:
                gpu_device = gpu_list[wid % len(gpu_list)]
            future = ex.submit(_run_games_worker, wid, wg, start_seed + wid, target_choices, use_gpu, gpu_device, args.quiet)
            futures.append(future)

        for fut in as_completed(futures):
            part = fut.result()
            merge_results(aggregated, part)

    # 计算分数
    aggregated['AGENT_A_SCORE'] = aggregated['AGENT_A_WIN'] * 1 + aggregated['SAME'] * 0.5
    aggregated['AGENT_B_SCORE'] = aggregated['AGENT_B_WIN'] * 1 + aggregated['SAME'] * 0.5

    print('\n最终结果：', aggregated)


if __name__ == '__main__':
    main()

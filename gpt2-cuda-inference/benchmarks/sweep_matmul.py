#!/usr/bin/env python3

import subprocess
import shutil
import os
import sys
import time
from datetime import datetime

REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_MATMUL    = os.path.join(REPO_ROOT, "variants", "matmul-block-size-sweep", "matmul.cuh")
KERNEL_MATMUL   = os.path.join(REPO_ROOT, "kernels", "matmul.cuh")
KERNEL_BACKUP   = os.path.join(REPO_ROOT, "kernels", "matmul.cuh.bak")
BINARY          = os.path.join(REPO_ROOT, "test_gpt2")
MAKE_TARGET     = "test_gpt2"

NUM_RUNS = 3

CONFIGS = [
    (64,    4),
    (64,    8),
    (128,   4),
    (128,   8),
    (128,   16),
    (256,   4),
    (256,   8),
    (256,   16),
    (512,   4),
    (512,   8),
    (512,   16),
    (1024,  4),
    (1024,  8),
]

def smem_bytes(block_size, u):
    S = block_size // u
    return S * u * 4  # floats

def compile_config(block_size, u):
    extra = f"-DBLOCK_SIZE={block_size} -DU_TILE={u}"
    cmd = [
        "make", "-C", REPO_ROOT, MAKE_TARGET,
        f"CFLAGS=-O3 -arch=sm_86 -std=c++17 -rdc=true -g -lineinfo {extra}",
        "--always-make", "-s"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr

def run_binary():
    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            [BINARY], capture_output=True, text=True, timeout=180,
            cwd=REPO_ROOT
        )
        t1 = time.perf_counter()
    except subprocess.TimeoutExpired:
        return []
    if result.returncode != 0:
        print(f"\n  BINARY FAILED: {result.stderr[:200]}")
        return []
    return [(t1 - t0) * 1000.0]

def avg(lst):
    return sum(lst) / len(lst) if lst else float("inf")

def main():
    print("=" * 70)
    print("  matmul-block-size-sweep")
    print(f"  Repo root : {REPO_ROOT}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Configs   : {len(CONFIGS)}    Runs/config: {NUM_RUNS}")
    print("=" * 70)

    if os.path.exists(KERNEL_MATMUL):
        shutil.copy2(KERNEL_MATMUL, KERNEL_BACKUP)
        print(f"  Backed up kernels/matmul.cuh → matmul.cuh.bak\n")

    shutil.copy2(SWEEP_MATMUL, KERNEL_MATMUL)
    print(f"  Installed matmul-block-size-sweep/matmul.cuh → kernels/matmul.cuh\n")

    results = []

    try:
        for (bs, u) in CONFIGS:
            if bs % u != 0:
                print(f"  SKIP  BLOCK_SIZE={bs:4d}  U={u:2d}  (not divisible)")
                continue
            if bs > 1024:
                print(f"  SKIP  BLOCK_SIZE={bs:4d}  U={u:2d}  (exceeds 1024 thread limit)")
                continue
            S = bs // u
            smem = smem_bytes(bs, u)
            if smem > 48 * 1024:
                print(f"  SKIP  BLOCK_SIZE={bs:4d}  U={u:2d}  (smem={smem}B > 48KB)")
                continue

            print(f"  Testing  BLOCK_SIZE={bs:4d}  U={u:2d}  S={S:4d}  smem={smem//1024}KB", end="  ", flush=True)

            ok, err = compile_config(bs, u)
            if not ok:
                print(f"COMPILE ERROR")
                if err:
                    print(f"    {err[:300]}")
                continue

            all_times = []
            for _ in range(NUM_RUNS):
                times = run_binary()
                if times:
                    all_times.append(times[0])

            if not all_times:
                print("NO TIMING OUTPUT")
                continue

            mean = avg(all_times)
            runs_str = "  ".join(f"{t:.3f}" for t in all_times)
            print(f"avg={mean:.3f} ms  [{runs_str}]")
            results.append((mean, bs, u, S, smem, all_times))

    finally:
        if os.path.exists(KERNEL_BACKUP):
            shutil.copy2(KERNEL_BACKUP, KERNEL_MATMUL)
            os.remove(KERNEL_BACKUP)
            print(f"\n  Restored original kernels/matmul.cuh")

    print("\n")
    print("=" * 70)
    print("  SWEEP RESULTS  (sorted fastest → slowest)")
    print("=" * 70)
    print(f"  {'BLOCK_SIZE':>10} {'U':>4} {'S':>4} {'smem':>6}  {'Avg ms':>9}  Runs")
    print("  " + "-" * 60)

    results.sort(key=lambda x: x[0])
    for rank, (mean, bs, u, S, smem, all_times) in enumerate(results):
        marker = " ← BEST" if rank == 0 else ""
        runs_str = "  ".join(f"{t:.3f}" for t in all_times)
        print(f"  {bs:>10} {u:>4} {S:>4} {smem//1024:>5}KB  {mean:>9.3f}  [{runs_str}]{marker}")

    print("=" * 70)

    if results:
        best_mean, best_bs, best_u, best_S, _, _ = results[0]
        worst_mean = results[-1][0]
        speedup = worst_mean / best_mean if best_mean > 0 else 0

        print(f"\n  Best config : BLOCK_SIZE={best_bs}  U={best_u}  S={best_S}")
        print(f"  Best avg    : {best_mean:.3f} ms")
        print(f"  Worst avg   : {worst_mean:.3f} ms")
        print(f"  Sweep range : {speedup:.2f}x difference between best and worst")
        print(f"\n  To use best config permanently, add to your Makefile or compile cmd:")
        print(f"    -DBLOCK_SIZE={best_bs} -DU_TILE={best_u}")

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

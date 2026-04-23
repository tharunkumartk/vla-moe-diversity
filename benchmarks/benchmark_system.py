#!/usr/bin/env python
"""
Comprehensive system benchmark for lerobot VLA training/eval.

Sections:
  1. hardware    – GPU / CPU / RAM specs + FP16 throughput
  2. dataloader  – DataLoader batches/sec sweep over num_workers
  3. inference   – Policy select_action latency sweep over batch sizes
  4. libero_env  – LIBERO env.step throughput sweep over n_envs (Sync & Async)

Total runtime: ~15–25 min (can skip sections with --skip).

Usage:
  bash benchmarks/run_benchmark.sh
  # or directly (after env is activated):
  CUDA_VISIBLE_DEVICES=2 python benchmarks/benchmark_system.py [--skip inference libero_env]
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Config (overridable via env vars)
# ──────────────────────────────────────────────────────────────────────────────
SCR = os.environ.get("SCR", "/scratch/gpfs/EYSENBACH/ij9461")
POLICY_PATH = os.environ.get("POLICY_PATH", f"{SCR}/huggingface/smolvla_base")
DATASET_ROOT = os.environ.get(
    "TRAIN_DATASET_ROOT",
    f"{SCR}/hf_cache_user/lerobot/hub/datasets--HuggingFaceVLA--libero"
    "/snapshots/cc29b569e0c32cd8d492757c7f2e076de90c7ba5",
)
DATASET_REPO_ID = os.environ.get("TRAIN_DATASET_REPO_ID", "HuggingFaceVLA/libero")
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}

SEP = "=" * 72
SUBSEP = "-" * 72


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def row(label: str, value: str) -> None:
    print(f"  {label:<35s} {value}")


def timing_row(label: str, ms: float, pct: float | None = None) -> None:
    pct_str = f"  ({pct:5.1f}%)" if pct is not None else ""
    print(f"  {label:<35s} {ms:8.2f} ms/step{pct_str}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Hardware
# ──────────────────────────────────────────────────────────────────────────────

def bench_hardware() -> None:
    header("1. HARDWARE INFO")

    row("Python", sys.version.split()[0])
    row("PyTorch", torch.__version__)
    row("CUDA available", str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            total_gb = p.total_memory / 1e9
            print(f"\n  GPU {i}: {p.name}")
            row("  VRAM total", f"{total_gb:.1f} GB")
            row("  Compute capability", f"{p.major}.{p.minor}")
            row("  Multiprocessors", str(p.multi_processor_count))
            row("  Max threads / SM", str(p.max_threads_per_multi_processor))
        print()

    row("CPU logical cores", str(os.cpu_count()))

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line or "MemAvailable" in line:
                    k, v = line.split(":")
                    kb = int(v.strip().split()[0])
                    row(k.strip(), f"{kb / 1e6:.1f} GB")
    except Exception:
        pass

    # FP16 matmul throughput estimate
    if torch.cuda.is_available():
        print(f"\n  {SUBSEP}")
        print("  FP16 matmul throughput (warm-up then timed)")
        dev = "cuda"
        B, M, K, N = 128, 1024, 1024, 1024
        a = torch.randn(B, M, K, dtype=torch.float16, device=dev)
        b = torch.randn(B, K, N, dtype=torch.float16, device=dev)
        for _ in range(10):
            torch.bmm(a, b)
        torch.cuda.synchronize()
        REPS = 50
        t0 = time.perf_counter()
        for _ in range(REPS):
            torch.bmm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tflops = 2 * B * M * K * N * REPS / elapsed / 1e12
        row(f"  FP16 BMM {B}×{M}×{K}→{B}×{M}×{N}", f"{tflops:.1f} TFLOPS")

        # BF16
        a16 = a.to(torch.bfloat16)
        b16 = b.to(torch.bfloat16)
        for _ in range(10):
            torch.bmm(a16, b16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            torch.bmm(a16, b16)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tflops_bf = 2 * B * M * K * N * REPS / elapsed / 1e12
        row("  BF16 (same)", f"{tflops_bf:.1f} TFLOPS")

        del a, b, a16, b16
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# 2. DataLoader throughput
# ──────────────────────────────────────────────────────────────────────────────

def bench_dataloader(
    batch_size: int = 64,
    n_batches: int = 100,
    episodes: list[int] | None = None,
) -> None:
    header("2. DATALOADER THROUGHPUT  (num_workers sweep)")
    print(f"  batch_size={batch_size}, measuring {n_batches} batches each")
    print()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        print(f"  [SKIP] Cannot import LeRobotDataset: {e}")
        return

    print("  Loading dataset (first load may be slow)...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        dataset = LeRobotDataset(
            repo_id=DATASET_REPO_ID,
            root=DATASET_ROOT,
            episodes=episodes or list(range(200)),  # subset for speed
            download_videos=False,
        )
    except Exception as e:
        print(f"\n  [SKIP] Dataset load failed: {e}")
        return
    print(f" done ({time.perf_counter()-t0:.1f}s, {len(dataset)} frames)")

    from torch.utils.data import DataLoader

    best_nw, best_throughput = 0, 0.0
    for nw in [2, 4, 8, 10, 12, 16, 20, 24, 30, 36]:
        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=nw,
                pin_memory=(torch.cuda.is_available()),
                drop_last=True,
                prefetch_factor=2 if nw > 0 else None,
                persistent_workers=(nw > 0),
            )
            # warmup
            it = iter(loader)
            for _ in range(min(5, n_batches)):
                next(it)

            t0 = time.perf_counter()
            count = 0
            it = iter(loader)
            for _ in range(n_batches):
                next(it)
                count += 1
            elapsed = time.perf_counter() - t0
            samples_s = count * batch_size / elapsed
            ms_batch = elapsed / count * 1000

            marker = "  <-- current" if nw == 16 else ""
            print(f"  num_workers={nw:2d}: {samples_s:7.0f} samples/s  |  {ms_batch:6.1f} ms/batch{marker}")

            if samples_s > best_throughput:
                best_throughput = samples_s
                best_nw = nw
        except Exception as e:
            print(f"  num_workers={nw:2d}: FAILED ({e})")

    print(f"\n  --> Recommended num_workers: {best_nw}  ({best_throughput:.0f} samples/s)")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Policy inference
# ──────────────────────────────────────────────────────────────────────────────

def bench_inference(
    batch_sizes: list[int] | None = None,
    n_reps: int = 20,
) -> None:
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64]
    header("3. POLICY INFERENCE THROUGHPUT  (batch_size sweep)")
    print(f"  policy: {POLICY_PATH}")
    print(f"  reps per size: {n_reps} (policy.reset() called each rep to force forward pass)")
    print()

    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA device")
        return

    # ---- load policy ----
    try:
        from lerobot.policies.factory import make_policy
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg

        print("  Loading policy...", end="", flush=True)
        t0 = time.perf_counter()
        _env_cfg = LiberoEnvCfg(task="libero_10")
        policy = make_policy(cfg=PreTrainedConfig.from_pretrained(POLICY_PATH), env_cfg=_env_cfg, rename_map=RENAME_MAP)
        policy = policy.cuda().eval()
        print(f" done ({time.perf_counter()-t0:.1f}s)")
    except Exception as e:
        print(f"\n  [SKIP] Policy load failed: {e}")
        return

    # ---- get one real observation from LIBERO ----
    print("  Building one LIBERO env to get real observation...", end="", flush=True)
    try:
        import gymnasium as gym
        from lerobot.envs.libero import create_libero_envs
        from lerobot.envs.utils import preprocess_observation, add_envs_task
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg

        env_cfg = LiberoEnvCfg(task="libero_10")
        envs_dict = create_libero_envs(
            task="libero_10",
            n_envs=1,
            env_cls=gym.vector.SyncVectorEnv,
            init_states=False,
        )
        env = list(list(envs_dict.values())[0].values())[0]
        obs, _ = env.reset()

        obs = preprocess_observation(obs)
        obs = add_envs_task(env, obs)

        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy.config
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=POLICY_PATH,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {"rename_map": RENAME_MAP},
            },
        )
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)
        env.close()
        print(" done")
    except Exception as e:
        print(f"\n  [SKIP] LIBERO env setup failed: {e}")
        return

    # ---- sweep batch sizes ----
    tensor_keys = [k for k, v in obs.items() if isinstance(v, torch.Tensor)]
    print(f"  obs keys after preprocessing: {sorted(obs.keys())}")
    print(f"  tensor keys: {sorted(tensor_keys)}")
    print()
    print(f"  {'batch':>6}  {'ms/step':>9}  {'steps/s':>9}  {'GPU MB':>8}  {'throughput':>12}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*12}")

    def _repeat_obs(o, B: int):
        if isinstance(o, dict):
            return {k: _repeat_obs(v, B) for k, v in o.items()}
        if isinstance(o, torch.Tensor):
            reps = [B] + [1] * (o.dim() - 1)
            return o.repeat(reps)
        return o

    for B in batch_sizes:
        try:
            obs_b = _repeat_obs(obs, B)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # warmup
            policy.reset()
            with torch.inference_mode():
                policy.select_action(obs_b)
            torch.cuda.synchronize()

            # timed
            t0 = time.perf_counter()
            for _ in range(n_reps):
                policy.reset()
                with torch.inference_mode():
                    policy.select_action(obs_b)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            ms = elapsed / n_reps * 1000
            sps = n_reps / elapsed
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            imgs_s = B * sps
            print(f"  {B:>6}  {ms:>9.1f}  {sps:>9.1f}  {peak_mb:>8.0f}  {imgs_s:>8.0f} env/s")
        except torch.cuda.OutOfMemoryError:
            print(f"  {B:>6}  OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {B:>6}  ERROR: {e}")

    del policy
    torch.cuda.empty_cache()
    gc.collect()


# ──────────────────────────────────────────────────────────────────────────────
# 4. LIBERO env step throughput
# ──────────────────────────────────────────────────────────────────────────────

def bench_libero_env(
    n_envs_list: list[int] | None = None,
    n_steps: int = 60,
    suite: str = "libero_10",
    task_id: int = 0,
) -> None:
    if n_envs_list is None:
        n_envs_list = [1, 4, 8, 16, 20, 30, 40]
    header("4. LIBERO ENV STEP THROUGHPUT  (n_envs sweep)")
    print(f"  suite={suite}, task_id={task_id}, n_steps={n_steps}")
    print(f"  Random actions (no policy). Timing env.step() only.")
    print()

    try:
        import gymnasium as gym
        from lerobot.envs.libero import create_libero_envs, _get_suite, _make_env_fns
    except ImportError as e:
        print(f"  [SKIP] Cannot import LIBERO: {e}")
        return

    ACTION_DIM = 7

    def _run_bench(env_cls_name: str, env_cls, n: int) -> float | None:
        try:
            envs_dict = create_libero_envs(
                task=suite,
                n_envs=n,
                env_cls=env_cls,
                init_states=False,
                gym_kwargs={"task_ids": [task_id]},
            )
            env = list(list(envs_dict.values())[0].values())[0]
            env.reset()

            # warmup
            for _ in range(5):
                actions = np.random.uniform(-1, 1, size=(n, ACTION_DIM)).astype(np.float32)
                env.step(actions)

            t0 = time.perf_counter()
            for _ in range(n_steps):
                actions = np.random.uniform(-1, 1, size=(n, ACTION_DIM)).astype(np.float32)
                env.step(actions)
            elapsed = time.perf_counter() - t0
            env.close()

            total_steps = n * n_steps
            sps = total_steps / elapsed
            ms = elapsed / n_steps * 1000
            return elapsed, ms, sps
        except Exception as e:
            print(f"    {env_cls_name:5s} n_envs={n:2d}: FAILED ({type(e).__name__}: {e})")
            return None

    print(f"  {'n_envs':>7}  {'vec_type':>10}  {'ms/batch':>10}  {'env.steps/s':>13}  {'speedup vs 1':>13}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*13}  {'-'*13}")

    baseline_sps = None
    for n in n_envs_list:
        # SyncVectorEnv
        result = _run_bench("Sync", gym.vector.SyncVectorEnv, n)
        if result is not None:
            elapsed, ms, sps = result
            if baseline_sps is None:
                baseline_sps = sps / n  # per-env baseline
            speedup = sps / (baseline_sps * n) if baseline_sps else float("nan")
            note = " <-- current" if n == 20 else ""
            print(f"  {n:>7}  {'Sync':>10}  {ms:>10.1f}  {sps:>13.0f}  {speedup:>12.2f}x{note}")

    # Try AsyncVectorEnv with forkserver context (avoids fork-in-multithreaded deadlock)
    import functools
    async_cls = functools.partial(gym.vector.AsyncVectorEnv, context="forkserver")
    print()
    print("  AsyncVectorEnv/forkserver (parallelizes env steps across CPU cores):")
    for n in [4, 8, 16, 20, 30, 40]:
        result = _run_bench("Async", async_cls, n)
        if result is not None:
            elapsed, ms, sps = result
            speedup = sps / (baseline_sps * n) if baseline_sps else float("nan")
            print(f"  {n:>7}  {'Async':>10}  {ms:>10.1f}  {sps:>13.0f}  {speedup:>12.2f}x")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="System benchmark for lerobot VLA")
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=["hardware", "dataloader", "inference", "libero_env"],
        default=[],
        help="Sections to skip",
    )
    args = parser.parse_args()
    skip = set(args.skip or [])

    print(SEP)
    print("  lerobot VLA – system benchmark")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"  SCR = {SCR}")
    print(SEP)

    if "hardware" not in skip:
        bench_hardware()
    if "dataloader" not in skip:
        bench_dataloader()
    if "inference" not in skip:
        bench_inference()
    if "libero_env" not in skip:
        bench_libero_env()

    print(f"\n{SEP}")
    print("  Benchmark complete.")
    print("  Share this full output to get targeted optimization recommendations.")
    print(SEP)


if __name__ == "__main__":
    main()

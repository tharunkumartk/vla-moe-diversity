#!/usr/bin/env python
"""Masked-expert ablation evaluation for LIBERO.

For every (subset, task, rollout) triple this script runs:
  - one rollout with the full policy   (condition: "original")
  - one rollout per masked expert index (condition: "masked_expert_<i>")

The router still runs normally; we zero out the chosen expert's probability
mass and renormalise the remaining weights so routing is self-consistent.

Results are flushed to JSON after every task for resilience to preemption.
A grouped bar chart (success rate per subset, one bar group per condition)
is written at the end.

Usage:
  python -m lerobot.scripts.eval_expert_masking \\
    --policy.path <path> \\
    --output_dir  <dir> \\
    --masked_experts 3 \\
    --rollouts_per_task 3
"""

from __future__ import annotations

import argparse
import json
import logging
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import _get_suite, create_libero_envs
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


DEFAULT_SUBSETS = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
DEFAULT_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.wrist_image": "observation.images.camera2",
    "observation.images.image2": "observation.images.camera2",
}


# ---------------------------------------------------------------------------
# Router masking hook
# ---------------------------------------------------------------------------

def _install_masking_hook(policy: PreTrainedPolicy):
    """Patch SeparateExpertResidualMoE.route to support runtime expert masking.

    After calling this, set `moe._masked_experts = {3}` to mask expert 3,
    or `moe._masked_experts = set()` to disable masking (original behaviour).

    The patch is applied at the class level so it affects the single inference
    instance without touching any checkpoint state.

    Returns the moe module so the caller can update `_masked_experts`.
    """
    from lerobot.policies.smolvla.moe import SeparateExpertResidualMoE, _compute_router_statistics

    vlm_with_expert = getattr(policy.model, "vlm_with_expert", None)
    if vlm_with_expert is None:
        raise RuntimeError(
            "Policy has no vlm_with_expert — is this a separate-experts MoE model?"
        )
    moe = vlm_with_expert.separate_expert_moe
    if moe is None:
        raise RuntimeError(
            "vlm_with_expert.separate_expert_moe is None — model is not a separate-expert MoE"
        )

    moe._masked_experts: set[int] = set()
    _original_route = SeparateExpertResidualMoE.route

    def _masked_route(self, x):
        masked: set[int] = getattr(self, "_masked_experts", set())
        if not masked:
            return _original_route(self, x)

        # Replicate route() but zero out masked experts before top-k selection.
        pooled = x.mean(dim=1)
        router_logits = F.linear(pooled, self.router.weight.to(x.dtype))
        # noisy_routing only applies during training; skip at eval time.
        router_probs = F.softmax(router_logits, dim=-1)

        # Zero out masked experts and renormalise so remaining probs sum to 1.
        router_probs_masked = router_probs.clone()
        for expert_idx in masked:
            router_probs_masked[:, expert_idx] = 0.0
        router_probs_masked = router_probs_masked / (
            router_probs_masked.sum(dim=-1, keepdim=True) + 1e-9
        )

        topk_weights, topk_indices = torch.topk(router_probs_masked, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        tokens_per_expert, load_balance_loss = _compute_router_statistics(
            topk_indices=topk_indices,
            router_probs=router_probs_masked,
            num_experts=self.num_experts,
        )
        routing = {"topk_indices": topk_indices, "topk_weights": topk_weights}
        aux = {
            "load_balance_loss": load_balance_loss,
            "router_logits": router_logits,
            "tokens_per_expert": tokens_per_expert,
        }
        return routing, aux

    SeparateExpertResidualMoE.route = _masked_route
    return moe


def _set_mask(moe, masked_experts: set[int]) -> None:
    moe._masked_experts = masked_experts


def _condition_name(masked_experts: set[int]) -> str:
    if not masked_experts:
        return "original"
    return "masked_expert_" + "_".join(str(e) for e in sorted(masked_experts))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Masked-expert ablation evaluation for LIBERO."
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subsets", default=",".join(DEFAULT_SUBSETS))
    parser.add_argument(
        "--rollouts_per_task",
        type=int,
        default=3,
        help="Number of rollouts per (task, condition) pair.",
    )
    parser.add_argument(
        "--masked_experts",
        type=int,
        nargs="+",
        default=[],
        help=(
            "Expert indices to ablate, one experiment per index. "
            "E.g. --masked_experts 3 1 2 runs three ablations."
        ),
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy-device", dest="policy_device", default="cuda")
    parser.add_argument(
        "--policy-use-amp", dest="policy_use_amp", action="store_true", default=True
    )
    parser.add_argument(
        "--no-policy-use-amp", dest="policy_use_amp", action="store_false"
    )
    parser.add_argument(
        "--camera_name", default="agentview_image,robot0_eye_in_hand_image"
    )
    parser.add_argument("--control_mode", default="relative")
    parser.add_argument("--episode_length", type=int, default=None)
    parser.add_argument("--rename_map", default=json.dumps(DEFAULT_RENAME_MAP))
    parser.add_argument(
        "--save_videos",
        action="store_true",
        default=False,
        help="Save an MP4 for every rollout under output_dir/<subset>/task_<id>/<condition>/rollout_<idx>.mp4",
    )
    parser.add_argument(
        "--results_filename",
        default="results.json",
        help="Name of the JSON results file within output_dir (default: results.json).",
    )
    parser.add_argument(
        "--skip_original",
        action="store_true",
        default=False,
        help="Skip the unmasked 'original' condition and only run the masked expert conditions.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Policy + environment loading
# ---------------------------------------------------------------------------

def _load_policy(args: argparse.Namespace):
    cli_overrides = [
        f"--device={args.policy_device}",
        f"--use_amp={str(args.policy_use_amp).lower()}",
    ]
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path, cli_overrides=cli_overrides)
    policy_cfg.pretrained_path = Path(args.policy_path)

    env_cfg = LiberoEnvConfig(
        task=DEFAULT_SUBSETS[0],
        task_ids=[0],
        camera_name=args.camera_name,
        init_states=True,
        episode_length=args.episode_length,
        control_mode=args.control_mode,
        max_parallel_tasks=1,
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=json.loads(args.rename_map))
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": json.loads(args.rename_map)},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )
    return policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor


def _make_task_env(args: argparse.Namespace, subset: str, task_id: int) -> gym.vector.VectorEnv:
    envs = create_libero_envs(
        task=subset,
        n_envs=1,
        camera_name=args.camera_name,
        init_states=True,
        gym_kwargs={
            "task_ids": [task_id],
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array",
        },
        env_cls=gym.vector.SyncVectorEnv,
        control_mode=args.control_mode,
        episode_length=args.episode_length,
    )
    return envs[subset][task_id]


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def _extract_success(info: dict, num_envs: int) -> list[bool]:
    if "final_info" in info:
        fi = info["final_info"]
        if isinstance(fi, dict) and "is_success" in fi:
            values = fi["is_success"]
            if hasattr(values, "tolist"):
                return [bool(x) for x in values.tolist()]
            return [bool(values)]
    return [False] * num_envs


def _render_frame(env: gym.vector.VectorEnv) -> np.ndarray:
    if isinstance(env, gym.vector.SyncVectorEnv):
        return env.envs[0].render()
    return env.call("render")[0]


def _run_rollout(
    *,
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seed: int,
    use_amp: bool,
    save_videos: bool = False,
) -> dict[str, Any]:
    policy.reset()
    observation, _ = env.reset(seed=[seed])
    total_reward = 0.0
    done = np.array([False], dtype=bool)
    max_steps = max(env.call("_max_episode_steps"))
    device = get_safe_torch_device(policy.config.device, log=False)
    last_success = False
    num_steps = 0
    frames: list[np.ndarray] = [_render_frame(env)] if save_videos else []

    for step_idx in range(max_steps):
        observation_t = preprocess_observation(observation)
        observation_t = add_envs_task(env, observation_t)
        observation_t = env_preprocessor(observation_t)
        observation_t = preprocessor(observation_t)

        amp_ctx = (
            torch.autocast(device_type=device.type)
            if use_amp and device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with torch.inference_mode(), amp_ctx:
            action = policy.select_action(observation_t)

        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        action_np = action.to("cpu").numpy()

        observation, reward, terminated, truncated, info = env.step(action_np)
        done = terminated | truncated | done
        successes = _extract_success(info, env.num_envs)
        total_reward += float(reward[0])
        last_success = bool(successes[0])
        num_steps = step_idx + 1

        if save_videos:
            frames.append(_render_frame(env))

        if bool(done[0]):
            break

    fps = int(env.unwrapped.metadata.get("render_fps", 20))
    return {
        "success": last_success,
        "num_steps": num_steps,
        "total_reward": round(total_reward, 4),
        "seed": seed,
        "frames": frames,
        "fps": fps,
    }


def _save_video(video_path: Path, frames: list[np.ndarray], fps: int) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(video_path), frames, fps=fps)


# ---------------------------------------------------------------------------
# JSON writing (atomic write via rename) + resume loading
# ---------------------------------------------------------------------------

def _load_existing_results(json_path: Path) -> dict[str, Any]:
    """Load results dict from an existing JSON file, if present and valid."""
    if not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        existing = data.get("results", {})
        logging.info("Loaded existing results from %s", json_path)
        return existing
    except Exception as exc:
        logging.warning("Could not load %s (%s) — starting fresh", json_path, exc)
        return {}


def _already_done_rollout_idxs(results: dict, subset: str, task_key: str, cond_name: str) -> set[int]:
    """Return the set of rollout_idx values already recorded for this (subset, task, condition)."""
    rollouts = (
        results
        .get(subset, {})
        .get(task_key, {})
        .get("conditions", {})
        .get(cond_name, {})
        .get("rollouts", [])
    )
    return {r["rollout_idx"] for r in rollouts if "rollout_idx" in r}


def _flush_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Success-rate helpers
# ---------------------------------------------------------------------------

def _sr(rollouts: list[dict]) -> float:
    if not rollouts:
        return 0.0
    return round(100.0 * sum(r["success"] for r in rollouts) / len(rollouts), 2)


def _subset_sr(subset_data: dict, condition: str) -> float:
    rates = []
    for key, val in subset_data.items():
        if not key.startswith("task_"):
            continue
        cond = val.get("conditions", {}).get(condition, {})
        rollouts = cond.get("rollouts", [])
        if rollouts:
            rates.append(_sr(rollouts))
    return round(float(np.mean(rates)), 2) if rates else 0.0


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def _make_bar_chart(
    results: dict,
    condition_names: list[str],
    subsets: list[str],
    output_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available — skipping bar chart")
        return

    # success rate per (condition, subset)
    sr: dict[str, list[float]] = {c: [] for c in condition_names}
    for subset in subsets:
        subset_data = results.get(subset, {})
        for cond in condition_names:
            sr[cond].append(_subset_sr(subset_data, cond))

    x = np.arange(len(subsets))
    n = len(condition_names)
    width = min(0.7 / n, 0.25)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(subsets)), 6))

    for i, cond in enumerate(condition_names):
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            sr[cond],
            width,
            label=cond.replace("_", " "),
            color=colors[i],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, rate in zip(bars, sr[cond]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{rate:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

    pretty_labels = [s.replace("libero_", "LIBERO-").upper() for s in subsets]
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels, fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_xlabel("LIBERO Suite", fontsize=12)
    ax.set_title("Expert Masking Ablation — LIBERO Success Rates", fontsize=14)
    ax.set_ylim(0, 118)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("Bar chart saved → %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    init_logging()
    register_third_party_plugins()
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor = _load_policy(args)
    logging.info("Policy loaded from %s", args.policy_path)

    moe = _install_masking_hook(policy)
    logging.info("Masking hook installed on SeparateExpertResidualMoE.route")

    # Conditions: original first (unless skipped), then one per masked expert
    conditions_masks: list[set[int]] = ([] if args.skip_original else [set()]) + [
        {i} for i in args.masked_experts
    ]
    condition_names: list[str] = [_condition_name(m) for m in conditions_masks]
    logging.info("Conditions: %s", condition_names)

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    json_path = output_dir / args.results_filename
    results: dict[str, Any] = _load_existing_results(json_path)

    run_data: dict[str, Any] = {
        "config": {
            "policy_path": args.policy_path,
            "subsets": subsets,
            "rollouts_per_task": args.rollouts_per_task,
            "masked_experts": args.masked_experts,
            "conditions": condition_names,
            "seed": args.seed,
            "save_videos": args.save_videos,
        },
        "results": results,
        "global_summary": {},
    }

    for subset_idx, subset in enumerate(subsets):
        suite = _get_suite(subset)
        num_tasks = min(10, len(suite.tasks))
        results.setdefault(subset, {})
        logging.info("=== Subset %s  (%d tasks) ===", subset, num_tasks)

        for task_id in range(num_tasks):
            task_key = f"task_{task_id}"
            results[subset].setdefault(task_key, {"task_id": task_id, "conditions": {}})

            task_had_new_work = False

            for cond_idx, (mask, cond_name) in enumerate(
                zip(conditions_masks, condition_names)
            ):
                done_idxs = _already_done_rollout_idxs(results, subset, task_key, cond_name)
                todo_idxs = [i for i in range(args.rollouts_per_task) if i not in done_idxs]

                if not todo_idxs:
                    logging.info(
                        "  subset=%-16s task=%02d  cond=%-24s  all %d rollouts already done — skipping",
                        subset, task_id, cond_name, args.rollouts_per_task,
                    )
                    continue

                task_had_new_work = True
                _set_mask(moe, mask)

                # Start from whatever rollouts already exist for this condition.
                existing_rollouts = (
                    results[subset][task_key]["conditions"]
                    .get(cond_name, {})
                    .get("rollouts", [])
                )
                rollout_records: list[dict] = list(existing_rollouts)

                for rollout_idx in todo_idxs:
                    # Same seed across conditions for the same (task, rollout) pair
                    # so that initial env states are paired — enables per-pair analysis.
                    seed = (
                        args.seed
                        + subset_idx * 100_000
                        + task_id * 1_000
                        + rollout_idx
                    )
                    env = _make_task_env(args, subset, task_id)
                    try:
                        record = _run_rollout(
                            env=env,
                            policy=policy,
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            seed=seed,
                            use_amp=args.policy_use_amp,
                            save_videos=args.save_videos,
                        )
                    finally:
                        env.close()

                    record["rollout_idx"] = rollout_idx

                    if args.save_videos and record["frames"]:
                        video_path = (
                            output_dir / subset / f"task_{task_id:02d}" / cond_name
                            / f"rollout_{rollout_idx:02d}.mp4"
                        )
                        _save_video(video_path, record["frames"], record["fps"])
                        record["video_path"] = str(video_path)

                    # Strip frames from record before JSON serialisation.
                    record.pop("frames", None)
                    rollout_records.append(record)
                    logging.info(
                        "  subset=%-16s task=%02d  cond=%-24s  rollout=%d → %s  steps=%d",
                        subset,
                        task_id,
                        cond_name,
                        rollout_idx,
                        "SUCCESS" if record["success"] else "FAIL",
                        record["num_steps"],
                    )

                # Sort by rollout_idx so order is deterministic regardless of resume order.
                rollout_records.sort(key=lambda r: r["rollout_idx"])
                n_success = sum(r["success"] for r in rollout_records)
                results[subset][task_key]["conditions"][cond_name] = {
                    "rollouts": rollout_records,
                    "num_rollouts": len(rollout_records),
                    "num_successes": n_success,
                    "success_rate": _sr(rollout_records),
                }

            # Flush after every task that had any new work.
            if task_had_new_work:
                _flush_json(json_path, run_data)
            sr_line = "  ".join(
                f"{c}={results[subset][task_key]['conditions'].get(c, {}).get('success_rate', float('nan')):.0f}%"
                for c in condition_names
            )
            logging.info("  [task %d complete] %s", task_id, sr_line)

        # Per-subset summary across all tasks
        subset_summary: dict[str, Any] = {}
        for cond_name in condition_names:
            task_rates = [
                results[subset][f"task_{t}"]["conditions"][cond_name]["success_rate"]
                for t in range(num_tasks)
                if f"task_{t}" in results[subset]
            ]
            subset_summary[cond_name] = {
                "mean_success_rate": round(float(np.mean(task_rates)), 2),
                "task_success_rates": task_rates,
                "num_tasks": len(task_rates),
            }
        results[subset]["subset_summary"] = subset_summary
        _flush_json(json_path, run_data)

        sr_summary = "  ".join(
            f"{c}={subset_summary[c]['mean_success_rate']:.1f}%"
            for c in condition_names
        )
        logging.info("=== %s summary: %s ===", subset, sr_summary)

    # Global summary across all subsets
    global_summary: dict[str, Any] = {}
    for cond_name in condition_names:
        per_subset = {}
        for subset in subsets:
            ss = results.get(subset, {}).get("subset_summary", {})
            per_subset[subset] = ss.get(cond_name, {}).get("mean_success_rate", 0.0)
        all_rates = list(per_subset.values())
        global_summary[cond_name] = {
            "per_subset_success_rate": per_subset,
            "overall_mean_success_rate": round(float(np.mean(all_rates)), 2),
        }
    run_data["global_summary"] = global_summary
    _flush_json(json_path, run_data)

    logging.info("=== GLOBAL SUMMARY ===")
    for cond_name in condition_names:
        logging.info(
            "  %s: overall=%.1f%%  per-subset=%s",
            cond_name,
            global_summary[cond_name]["overall_mean_success_rate"],
            global_summary[cond_name]["per_subset_success_rate"],
        )

    _make_bar_chart(results, condition_names, subsets, output_dir / "masked_expert_comparison.png")
    logging.info("All done. Results at %s", output_dir)


if __name__ == "__main__":
    main()

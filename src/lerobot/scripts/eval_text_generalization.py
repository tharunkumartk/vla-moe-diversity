#!/usr/bin/env python
"""Evaluate VLA generalization to custom text commands.

Accepts a list of (subset, task_id, new_command) triples and runs rollouts
where the policy receives `new_command` instead of the environment's natural
task description. Everything else (physics, init states, success criteria)
is unchanged so you can visually inspect whether the policy obeys the new text.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import types
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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


DEFAULT_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.wrist_image": "observation.images.camera2",
    "observation.images.image2": "observation.images.camera2",
}


# ---------------------------------------------------------------------------
# Routing recorder (unchanged from eval_moe_routing)
# ---------------------------------------------------------------------------

class RoutingRecorder:
    def __init__(self, policy: PreTrainedPolicy):
        self.policy = policy
        self.current_generation_calls: list[dict[str, Any]] = []
        self.current_chunk_summary: dict[str, Any] | None = None
        self.generation_summaries: list[dict[str, Any]] = []
        self.generation_index = -1
        self.layer_names = self._build_layer_name_map()
        self.effective_top_k = int(getattr(policy.config, "moe_top_k", 1) or 1)

    def _build_layer_name_map(self) -> dict[int, str]:
        layer_names: dict[int, str] = {}
        vlm_with_expert = getattr(self.policy.model, "vlm_with_expert", None)
        if vlm_with_expert is None:
            return layer_names
        for name, module in vlm_with_expert.named_modules():
            layer_names[id(module)] = name
        return layer_names

    def begin_generation(self) -> None:
        self.current_generation_calls = []

    def record_ffn(self, module: torch.nn.Module, aux: dict[str, Any]) -> None:
        router_logits = aux.get("router_logits")
        tokens_per_expert = aux.get("tokens_per_expert")
        if router_logits is None or tokens_per_expert is None:
            return
        router_probs = F.softmax(router_logits.detach().float(), dim=-1)
        top_k = getattr(module, "top_k", min(1, router_probs.shape[-1]))
        topk_weights, topk_indices = torch.topk(router_probs, top_k, dim=-1)
        call = {
            "variant": "ffn_moe",
            "layer_name": self.layer_names.get(id(module), module.__class__.__name__),
            "usage": tokens_per_expert.detach().float().cpu().tolist(),
            "router_prob_mean": router_probs.mean(dim=0).cpu().tolist(),
            "topk_indices_sample": topk_indices[: min(8, topk_indices.shape[0])].cpu().tolist(),
            "topk_weights_sample": topk_weights[: min(8, topk_weights.shape[0])].cpu().tolist(),
        }
        self.current_generation_calls.append(call)

    def record_separate(self, module: torch.nn.Module, routing: dict[str, Any], aux: dict[str, Any]) -> None:
        del module
        router_logits = aux.get("router_logits")
        tokens_per_expert = aux.get("tokens_per_expert")
        if router_logits is None or tokens_per_expert is None:
            return
        router_probs = F.softmax(router_logits.detach().float(), dim=-1)
        call = {
            "variant": "separate_experts",
            "usage": tokens_per_expert.detach().float().cpu().tolist(),
            "router_prob_mean": router_probs.mean(dim=0).cpu().tolist(),
            "topk_indices": routing["topk_indices"].detach().cpu().tolist(),
            "topk_weights": routing["topk_weights"].detach().float().cpu().tolist(),
        }
        self.current_generation_calls.append(call)

    def finalize_generation(self) -> dict[str, Any] | None:
        if not self.current_generation_calls:
            return self.current_chunk_summary
        self.generation_index += 1
        variant = self.current_generation_calls[0]["variant"]
        num_experts = len(self.current_generation_calls[0]["usage"])
        usage = np.mean([np.asarray(call["usage"], dtype=float) for call in self.current_generation_calls], axis=0)
        router_prob_mean = np.mean(
            [np.asarray(call["router_prob_mean"], dtype=float) for call in self.current_generation_calls],
            axis=0,
        )
        top_order = np.argsort(-router_prob_mean).tolist()
        sparse_top_k = min(self.effective_top_k, num_experts)
        sparse_ids = top_order[:sparse_top_k]
        sparse_weights = router_prob_mean[sparse_ids]
        sparse_weights = sparse_weights / max(float(sparse_weights.sum()), 1e-9)
        dense_display_weights = np.zeros(num_experts, dtype=float)
        dense_display_weights[sparse_ids] = sparse_weights
        top_experts = [
            {"expert_id": int(expert_idx), "weight": float(router_prob_mean[expert_idx])}
            for expert_idx in top_order[: min(num_experts, 5)]
        ]
        summary: dict[str, Any] = {
            "generation_index": self.generation_index,
            "variant": variant,
            "num_calls": len(self.current_generation_calls),
            "num_experts": num_experts,
            "expert_usage": usage.tolist(),
            "router_prob_mean": router_prob_mean.tolist(),
            "top_experts": top_experts,
            "display_expert_ids": [int(expert_idx) for expert_idx in sparse_ids],
            "display_expert_weights": [float(weight) for weight in sparse_weights.tolist()],
            "display_expert_weights_dense": dense_display_weights.tolist(),
            "raw_calls": self.current_generation_calls,
        }
        if variant == "ffn_moe":
            layer_usage: dict[str, list[list[float]]] = defaultdict(list)
            for call in self.current_generation_calls:
                layer_usage[str(call["layer_name"])].append(call["usage"])
            summary["per_layer_usage"] = {
                layer_name: np.mean(np.asarray(usages, dtype=float), axis=0).tolist()
                for layer_name, usages in layer_usage.items()
            }
        else:
            summary["topk_sequences"] = [
                {"topk_indices": call["topk_indices"], "topk_weights": call["topk_weights"]}
                for call in self.current_generation_calls
            ]
        self.current_chunk_summary = summary
        self.generation_summaries.append(summary)
        return summary


def _patch_policy_for_routing(policy: PreTrainedPolicy, recorder: RoutingRecorder) -> None:
    from lerobot.policies.smolvla.moe import MoELayer, SeparateExpertResidualMoE

    original_moe_forward = MoELayer.forward
    original_separate_route = SeparateExpertResidualMoE.route
    original_get_action_chunk = policy._get_action_chunk

    def wrapped_moe_forward(self, x, collect_expert_outputs=False):
        output, aux = original_moe_forward(self, x, collect_expert_outputs=collect_expert_outputs)
        if not self.training:
            recorder.record_ffn(self, aux)
        return output, aux

    def wrapped_separate_route(self, x):
        routing, aux = original_separate_route(self, x)
        if not self.training:
            recorder.record_separate(self, routing, aux)
        return routing, aux

    def wrapped_get_action_chunk(self, batch, noise=None, **kwargs):
        recorder.begin_generation()
        try:
            return original_get_action_chunk(batch, noise, **kwargs)
        finally:
            recorder.finalize_generation()

    MoELayer.forward = wrapped_moe_forward
    SeparateExpertResidualMoE.route = wrapped_separate_route
    policy._get_action_chunk = types.MethodType(wrapped_get_action_chunk, policy)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VLA generalization to custom text commands. "
            "Supply a list of (subset, task_id, new_command) triples via --task_overrides. "
            "The policy receives the custom command instead of the environment's natural description."
        )
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--task_overrides",
        required=True,
        help=(
            'JSON list of override entries. Each entry must have "subset", "task_id", and "command". '
            'Example: \'[{"subset": "libero_goal", "task_id": 2, "command": "pick up the red bowl"}]\''
        ),
    )
    parser.add_argument("--rollouts_per_task", type=int, default=5)
    parser.add_argument("--videos_per_task", type=int, default=5)
    parser.add_argument(
        "--video_selection_mode",
        choices=("successes_first", "first_seen", "mixed"),
        default="successes_first",
    )
    parser.add_argument("--overlay_update_hz", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy-device", dest="policy_device", default="cuda")
    parser.add_argument("--policy-use-amp", dest="policy_use_amp", action="store_true", default=True)
    parser.add_argument("--no-policy-use-amp", dest="policy_use_amp", action="store_false")
    parser.add_argument("--camera_name", default="agentview_image,robot0_eye_in_hand_image")
    parser.add_argument("--control_mode", default="relative")
    parser.add_argument("--episode_length", type=int, default=None)
    parser.add_argument("--use_async_envs", action="store_true")
    parser.add_argument("--rename_map", default=json.dumps(DEFAULT_RENAME_MAP))
    return parser.parse_args()


def _parse_task_overrides(raw: str) -> list[dict[str, Any]]:
    overrides = json.loads(raw)
    if not isinstance(overrides, list):
        raise ValueError("--task_overrides must be a JSON list")
    for entry in overrides:
        if not isinstance(entry, dict):
            raise ValueError(f"Each override must be a JSON object, got: {entry!r}")
        for key in ("subset", "task_id", "command"):
            if key not in entry:
                raise ValueError(f"Override entry missing required key '{key}': {entry!r}")
        entry["task_id"] = int(entry["task_id"])
    return overrides


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _make_env_cfg(args: argparse.Namespace, subset: str, task_id: int) -> LiberoEnvConfig:
    return LiberoEnvConfig(
        task=subset,
        task_ids=[task_id],
        camera_name=args.camera_name,
        init_states=True,
        episode_length=args.episode_length,
        control_mode=args.control_mode,
        max_parallel_tasks=1,
    )


def _load_policy_and_processors(
    args: argparse.Namespace,
    first_subset: str,
    first_task_id: int,
) -> tuple[PreTrainedPolicy, LiberoEnvConfig, Any, Any, Any, Any, RoutingRecorder]:
    cli_overrides = [f"--device={args.policy_device}", f"--use_amp={str(args.policy_use_amp).lower()}"]
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path, cli_overrides=cli_overrides)
    policy_cfg.pretrained_path = Path(args.policy_path)

    env_cfg = _make_env_cfg(args, first_subset, first_task_id)
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
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    recorder = RoutingRecorder(policy)
    _patch_policy_for_routing(policy, recorder)
    return policy, env_cfg, env_preprocessor, env_postprocessor, preprocessor, postprocessor, recorder


def _make_task_env(args: argparse.Namespace, subset: str, task_id: int) -> gym.vector.VectorEnv:
    env_cls = gym.vector.AsyncVectorEnv if args.use_async_envs else gym.vector.SyncVectorEnv
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
        env_cls=env_cls,
        control_mode=args.control_mode,
        episode_length=args.episode_length,
    )
    return envs[subset][task_id]


def _extract_render_frame(env: gym.vector.VectorEnv) -> np.ndarray:
    if isinstance(env, gym.vector.SyncVectorEnv):
        return env.envs[0].render()
    return env.call("render")[0]


def _extract_success(info: dict[str, Any], num_envs: int) -> list[bool]:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "is_success" in final_info:
            values = final_info["is_success"]
            if hasattr(values, "tolist"):
                return [bool(x) for x in values.tolist()]
            return [bool(values)]
    return [False] * num_envs


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _run_rollout(
    *,
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    recorder: RoutingRecorder,
    seed: int,
    subset: str,
    task_id: int,
    rollout_index: int,
    use_amp: bool,
    text_command_override: str,
) -> dict[str, Any]:
    policy.reset()
    observation, _ = env.reset(seed=[seed])
    frames: list[np.ndarray] = [_extract_render_frame(env)]
    step_records: list[dict[str, Any]] = []
    total_reward = 0.0
    done = np.array([False], dtype=bool)
    max_steps = max(env.call("_max_episode_steps"))

    device = get_safe_torch_device(policy.config.device, log=False)
    for step_idx in range(max_steps):
        observation_t = preprocess_observation(observation)
        observation_t = add_envs_task(env, observation_t)
        # Override the natural task description with the user-specified command.
        observation_t["task"] = [text_command_override] * len(observation_t["task"])
        observation_t = env_preprocessor(observation_t)
        observation_t = preprocessor(observation_t)

        amp_context = (
            torch.autocast(device_type=device.type)
            if use_amp and device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            action = policy.select_action(observation_t)
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        action_np = action.to("cpu").numpy()

        observation, reward, terminated, truncated, info = env.step(action_np)
        done = terminated | truncated | done
        successes = _extract_success(info, env.num_envs)
        total_reward += float(reward[0])
        frames.append(_extract_render_frame(env))

        current_summary = recorder.current_chunk_summary or {}
        queue_len = len(policy._queues[ACTION]) if hasattr(policy, "_queues") else 0
        chunk_step_index = max(policy.config.n_action_steps - 1 - queue_len, 0)
        step_record = {
            "subset": subset,
            "task_id": task_id,
            "rollout_index": rollout_index,
            "text_command_override": text_command_override,
            "env_step": step_idx,
            "reward": float(reward[0]),
            "success": bool(successes[0]),
            "done": bool(done[0]),
            "generation_index": current_summary.get("generation_index"),
            "chunk_step_index": int(chunk_step_index),
            "variant": current_summary.get("variant"),
            "expert_usage": current_summary.get("expert_usage"),
            "router_prob_mean": current_summary.get("router_prob_mean"),
            "top_experts": current_summary.get("top_experts"),
            "display_expert_ids": current_summary.get("display_expert_ids"),
            "display_expert_weights": current_summary.get("display_expert_weights"),
            "display_expert_weights_dense": current_summary.get("display_expert_weights_dense"),
        }
        step_records.append(step_record)

        if bool(done[0]):
            break

    return {
        "subset": subset,
        "task_id": task_id,
        "rollout_index": rollout_index,
        "text_command_override": text_command_override,
        "seed": seed,
        "success": bool(step_records[-1]["success"]) if step_records else False,
        "num_steps": len(step_records),
        "total_reward": total_reward,
        "frames": frames,
        "steps": step_records,
        "generations": list(recorder.generation_summaries),
        "variant": step_records[-1]["variant"] if step_records else None,
        "num_experts": len(step_records[-1]["expert_usage"]) if step_records and step_records[-1]["expert_usage"] else 0,
        "fps": int(env.unwrapped.metadata["render_fps"]),
    }


# ---------------------------------------------------------------------------
# Video / artifact helpers (unchanged from eval_moe_routing)
# ---------------------------------------------------------------------------

def _select_rollouts(results: list[dict[str, Any]], limit: int, mode: str) -> set[int]:
    if limit <= 0:
        return set()
    if limit >= len(results):
        return {result["rollout_index"] for result in results}
    ordered = list(results)
    if mode == "first_seen":
        return {result["rollout_index"] for result in ordered[:limit]}
    if mode == "successes_first":
        ranked = sorted(ordered, key=lambda result: (not result["success"], result["rollout_index"]))
        return {result["rollout_index"] for result in ranked[:limit]}
    successes = [result for result in ordered if result["success"]]
    failures = [result for result in ordered if not result["success"]]
    ranked: list[dict[str, Any]] = []
    while successes or failures:
        if successes:
            ranked.append(successes.pop(0))
        if failures:
            ranked.append(failures.pop(0))
    return {result["rollout_index"] for result in ranked[:limit]}


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def _overlay_frame(frame: np.ndarray, step_info: dict[str, Any] | None, fps: int) -> np.ndarray:
    del fps
    panel_height = 200
    height, width = frame.shape[:2]
    canvas = Image.new("RGB", (width, height + panel_height), (20, 24, 32))
    canvas.paste(Image.fromarray(frame), (0, panel_height))
    draw = ImageDraw.Draw(canvas)
    font = _font()

    if not step_info:
        draw.text((16, 16), "No routing info yet", fill=(240, 240, 240), font=font)
        return np.asarray(canvas)

    variant = step_info.get("variant", "unknown")
    title = (
        f"{step_info['subset']} task={step_info['task_id']} rollout={step_info['rollout_index']} "
        f"step={step_info['env_step']} chunk_step={step_info['chunk_step_index']}"
    )
    subtitle = "Separate action experts" if variant == "separate_experts" else "FFN MoE aggregate histogram"
    # Show the overridden text command prominently.
    override_text = f"CMD: {step_info.get('text_command_override', '')}"
    draw.text((16, 8), title, fill=(245, 245, 245), font=font)
    draw.text((16, 30), subtitle, fill=(180, 190, 210), font=font)
    draw.text((16, 52), override_text, fill=(255, 220, 80), font=font)

    display_ids = step_info.get("display_expert_ids") or []
    display_weights = step_info.get("display_expert_weights") or []
    dense_display_weights = step_info.get("display_expert_weights_dense")
    num_experts = len(step_info.get("expert_usage") or [])
    if dense_display_weights is None:
        dense_display_weights = [0.0] * num_experts
        for expert_id, weight in zip(display_ids, display_weights, strict=False):
            if 0 <= int(expert_id) < num_experts:
                dense_display_weights[int(expert_id)] = float(weight)
    max_usage = max(max(dense_display_weights), 1e-6) if dense_display_weights else 1.0
    bar_left = 16
    bar_top = 82
    bar_width = max(width - 32, 1)
    usable_height = 56
    baseline_y = bar_top + usable_height
    draw.line((bar_left, baseline_y, bar_left + bar_width, baseline_y), fill=(120, 128, 142), width=1)
    per_bar = max(bar_width // max(len(dense_display_weights), 1), 1)
    for idx, usage in enumerate(dense_display_weights):
        x0 = bar_left + idx * per_bar
        x1 = x0 + max(per_bar - 8, 1)
        draw.rectangle((x0, bar_top, x1, baseline_y), outline=(90, 98, 112), width=1)
        h = int(usable_height * float(usage) / max_usage)
        if h > 0:
            y0 = bar_top + (usable_height - h)
            color = (80 + (idx * 37) % 160, 120 + (idx * 29) % 120, 210)
            draw.rectangle((x0 + 1, y0, x1 - 1, baseline_y - 1), fill=color)
        label_y = baseline_y + 4 + (idx % 2) * 14
        draw.text((x0, label_y), f"E{idx}", fill=(230, 230, 230), font=font)

    top_line = ", ".join(
        f"E{expert_id}={weight:.2f}"
        for expert_id, weight in zip(display_ids, display_weights, strict=False)
    )
    draw.text((16, 174), f"Top-k experts: {top_line}", fill=(245, 245, 245), font=font)
    return np.asarray(canvas)


def _write_overlay_video(
    video_path: Path,
    frames: list[np.ndarray],
    steps: list[dict[str, Any]],
    fps: int,
    overlay_update_hz: float,
) -> None:
    overlay_frames = []
    frames_per_update = max(int(round(fps / overlay_update_hz)), 1) if overlay_update_hz > 0 else 1
    for frame_idx, frame in enumerate(frames):
        step_info = None
        if steps:
            if frame_idx == 0:
                selected_step_idx = 0
            else:
                selected_step_idx = min(((frame_idx - 1) // frames_per_update) * frames_per_update, len(steps) - 1)
            step_info = steps[selected_step_idx]
        overlay_frames.append(_overlay_frame(frame, step_info, fps))
    imageio.mimsave(video_path, overlay_frames, fps=fps)


def _write_rollout_csv(csv_path: Path, steps: list[dict[str, Any]], num_experts: int) -> None:
    fieldnames = [
        "env_step", "reward", "success", "done",
        "generation_index", "chunk_step_index", "variant",
    ] + [f"expert_weight_{idx}" for idx in range(num_experts)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps:
            row = {key: step.get(key) for key in fieldnames[:7]}
            dense = step.get("display_expert_weights_dense") or [0.0] * num_experts
            for idx in range(num_experts):
                row[f"expert_weight_{idx}"] = dense[idx] if idx < len(dense) else 0.0
            writer.writerow(row)


def _save_rollout_artifacts(rollout_dir: Path, result: dict[str, Any], overlay_update_hz: float) -> Path:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    trace_path = rollout_dir / "routing_trace.json"
    csv_path = rollout_dir / "expert_usage.csv"
    video_path = rollout_dir / "overlay.mp4"

    metadata = {k: v for k, v in result.items() if k not in {"frames"}}
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    num_experts = int(result.get("num_experts") or 0)
    _write_rollout_csv(csv_path, result["steps"], num_experts)
    _write_overlay_video(
        video_path, result["frames"], result["steps"],
        fps=result["fps"], overlay_update_hz=overlay_update_hz,
    )
    return video_path


def _materialize_selected_videos(task_dir: Path, results: list[dict[str, Any]], selected_rollout_ids: set[int]) -> list[str]:
    selected_dir = task_dir / "selected_videos"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_paths: list[str] = []
    for result in results:
        if result["rollout_index"] not in selected_rollout_ids:
            continue
        source = Path(result["video_path"])
        if not source.exists():
            continue
        destination = selected_dir / f"{source.parent.name}.mp4"
        shutil.copy2(source, destination)
        selected_paths.append(str(destination))
    return selected_paths


def _cleanup_unselected_videos(results: list[dict[str, Any]], selected_rollout_ids: set[int]) -> None:
    for result in results:
        if result["rollout_index"] in selected_rollout_ids:
            continue
        video_path = result.get("video_path")
        if video_path and Path(video_path).exists():
            Path(video_path).unlink()


def _safe_dir_name(text: str, max_len: int = 60) -> str:
    """Turn an arbitrary string into a safe directory name component."""
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in text)
    safe = "_".join(safe.split())
    return safe[:max_len]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    init_logging()
    register_third_party_plugins()
    args = _parse_args()

    task_overrides = _parse_task_overrides(args.task_overrides)
    if not task_overrides:
        logging.warning("--task_overrides list is empty, nothing to evaluate.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "task_overrides.json", "w", encoding="utf-8") as f:
        json.dump(task_overrides, f, indent=2)

    set_seed(args.seed)

    first = task_overrides[0]
    policy, _, env_preprocessor, env_postprocessor, preprocessor, postprocessor, recorder = (
        _load_policy_and_processors(args, first["subset"], first["task_id"])
    )
    logging.info("Loaded policy from %s", args.policy_path)

    all_results: list[dict[str, Any]] = []

    for override_index, override in enumerate(task_overrides):
        subset = override["subset"]
        task_id = int(override["task_id"])
        command = override["command"]

        logging.info(
            "Override %d/%d: subset=%s task_id=%d command=%r",
            override_index + 1, len(task_overrides), subset, task_id, command,
        )

        task_dir_name = f"{subset}__task{task_id:02d}__{_safe_dir_name(command)}"
        task_dir = output_dir / task_dir_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save the override metadata so the directory is self-documenting.
        with open(task_dir / "override_info.json", "w", encoding="utf-8") as f:
            json.dump({"subset": subset, "task_id": task_id, "command": command}, f, indent=2)

        task_results: list[dict[str, Any]] = []

        for rollout_index in range(args.rollouts_per_task):
            seed = args.seed + override_index * 10_000 + rollout_index
            recorder.generation_summaries = []
            recorder.current_chunk_summary = None
            env = _make_task_env(args, subset, task_id)
            try:
                result = _run_rollout(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    recorder=recorder,
                    seed=seed,
                    subset=subset,
                    task_id=task_id,
                    rollout_index=rollout_index,
                    use_amp=args.policy_use_amp,
                    text_command_override=command,
                )
            finally:
                env.close()

            rollout_dir = task_dir / f"rollout_{rollout_index:03d}"
            video_path = _save_rollout_artifacts(rollout_dir, result, args.overlay_update_hz)
            result["video_path"] = str(video_path)
            task_results.append(result)
            all_results.append(result)
            logging.info(
                "  rollout=%d seed=%d success=%s steps=%d",
                rollout_index, seed, result["success"], result["num_steps"],
            )

        selected_rollout_ids = _select_rollouts(task_results, limit=args.videos_per_task, mode=args.video_selection_mode)
        _cleanup_unselected_videos(task_results, selected_rollout_ids)
        selected_video_paths = _materialize_selected_videos(task_dir, task_results, selected_rollout_ids)

        task_summary = {
            "subset": subset,
            "task_id": task_id,
            "command": command,
            "num_rollouts": len(task_results),
            "num_successes": int(sum(r["success"] for r in task_results)),
            "success_rate": float(np.mean([r["success"] for r in task_results]) * 100.0),
            "avg_episode_length": float(np.mean([r["num_steps"] for r in task_results])),
            "selected_videos": selected_video_paths,
        }
        with open(task_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(task_summary, f, indent=2)
        logging.info(
            "  success_rate=%.1f%% (%d/%d)",
            task_summary["success_rate"],
            task_summary["num_successes"],
            task_summary["num_rollouts"],
        )

    # Overall manifest
    manifest = {
        "policy_path": args.policy_path,
        "task_overrides": task_overrides,
        "rollouts_per_task": args.rollouts_per_task,
        "videos_per_task": args.videos_per_task,
        "per_task_results": [
            {
                "subset": r["subset"],
                "task_id": r["task_id"],
                "command": r["text_command_override"],
                "success": r["success"],
            }
            for r in all_results
        ],
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logging.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()

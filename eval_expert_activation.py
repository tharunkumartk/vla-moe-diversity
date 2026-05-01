#!/usr/bin/env python
"""
Expert activation tracking eval for VLA-MoE models on LIBERO.

Runs rollouts on LIBERO task suites, tracks MoE expert routing at each
action-chunk step, and produces:
  - Videos with expert overlay (ALL experts shown, top-k highlighted)
  - Per-task overall expert distribution bar chart
  - Per-task 2x2 grid (episode split into 4 quarters, one histogram each)
  - Aggregate JSON stats

Usage:
  python eval_expert_activation.py \\
    --checkpoint_path /path/to/checkpoints/035000/pretrained_model \\
    --output_dir /path/to/eval_outputs \\
    --n_videos 10 \\
    --n_rollouts_per_task 1 \\
    --task_suites libero_goal libero_object libero_spatial libero_10 \\
    --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))

OVERLAY_H = 180   # pixel height of the expert-activation panel above each frame
VIDEO_FPS = 8
TASK_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
# rename_map used at training time
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera2",
}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VLA-MoE expert activation eval on LIBERO")
    p.add_argument("--checkpoint_path", required=True,
                   help="Path to pretrained_model/ directory")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_videos", type=int, default=10,
                   help="Max videos saved per task suite (one per task)")
    p.add_argument("--n_rollouts_per_task", type=int, default=1)
    p.add_argument("--task_suites", nargs="+",
                   default=["libero_goal", "libero_object", "libero_spatial", "libero_10"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─── Policy loading ───────────────────────────────────────────────────────────

def load_policy(checkpoint_path: str, device: str):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy = policy.to(device)
    policy.eval()
    return policy


def load_preprocessors(checkpoint_path: str):
    """Load policy obs normalizer and action un-normalizer from checkpoint."""
    try:
        from lerobot.processor import PolicyProcessorPipeline
        pre = PolicyProcessorPipeline.from_pretrained(
            checkpoint_path, prefix="policy_preprocessor"
        )
        post = PolicyProcessorPipeline.from_pretrained(
            checkpoint_path, prefix="policy_postprocessor"
        )
        return pre, post
    except Exception as exc:
        print(f"  [warn] could not load preprocessors: {exc}")
        return None, None


# ─── Expert Tracker ───────────────────────────────────────────────────────────

class ExpertTracker:
    """
    Monkeypatches predict_action_chunk and the MoE routing functions to capture
    expert activations.

    For separate-expert models: captures (topk_indices, topk_weights, router_probs)
    once per action-chunk prediction (one routing decision for the whole sequence).

    For per-layer FFN MoE models: captures tokens_per_expert (L, E) per
    action-chunk prediction, averaged across the flow-matching denoising loop.

    `episode_timeline` holds one activation dict per env step (same dict is
    repeated across steps that share the same action chunk).
    """

    def __init__(self, policy):
        self.is_separate = policy.config.separate_experts
        self.num_experts = policy.config.moe_num_experts
        self.top_k = policy.config.moe_top_k
        self._buf: list = []
        self.current_activation: dict | None = None
        self.episode_timeline: list = []
        self._install(policy)

    # ── hook installation ──────────────────────────────────────────────────

    def _install(self, policy):
        model = policy.model.vlm_with_expert
        tracker = self

        if self.is_separate:
            orig_route = model.separate_expert_moe.route

            def hooked_route(x):
                routing, aux = orig_route(x)
                tracker._buf.append({
                    "topk_indices": routing["topk_indices"].detach().cpu().numpy(),
                    "topk_weights": routing["topk_weights"].detach().cpu().numpy(),
                    "router_probs": F.softmax(
                        aux["router_logits"], dim=-1
                    ).detach().cpu().numpy(),
                })
                return routing, aux

            model.separate_expert_moe.route = hooked_route

        else:
            orig_fes = model._forward_expert_stack

            def hooked_fes(*args, **kwargs):
                result = orig_fes(*args, **kwargs)
                _, _, moe_aux = result
                if moe_aux:
                    layer_fracs = np.stack([
                        a["tokens_per_expert"].detach().cpu().float().numpy()
                        for a in moe_aux
                    ])  # (L, E)
                    tracker._buf.append(layer_fracs)
                return result

            model._forward_expert_stack = hooked_fes

        # Wrap predict_action_chunk to flush buffer after each chunk
        orig_pac = policy.predict_action_chunk

        def hooked_pac(*args, **kwargs):
            tracker._buf.clear()
            result = orig_pac(*args, **kwargs)
            tracker._flush_chunk()
            return result

        policy.predict_action_chunk = hooked_pac

    # ── internal buffer management ─────────────────────────────────────────

    def _flush_chunk(self):
        if not self._buf:
            return
        if self.is_separate:
            # Average router_probs across denoising steps; take last topk decision
            mean_probs = np.stack(
                [d["router_probs"] for d in self._buf]
            ).mean(0)  # (B, E)
            last = self._buf[-1]
            self.current_activation = {
                "type": "separate",
                "topk_indices": last["topk_indices"],   # (B, top_k)
                "topk_weights": last["topk_weights"],   # (B, top_k)
                "router_probs": mean_probs,             # (B, E)
            }
        else:
            mean_fracs = np.stack(self._buf).mean(0)   # (L, E)
            self.current_activation = {
                "type": "perlayer",
                "layer_fracs": mean_fracs,
            }
        self._buf.clear()

    # ── public API ─────────────────────────────────────────────────────────

    def record_frame(self):
        """Append current activation (may be None) to the episode timeline."""
        self.episode_timeline.append(self.current_activation)

    def reset_episode(self):
        self._buf.clear()
        self.current_activation = None
        self.episode_timeline = []


# ─── Overlay rendering ────────────────────────────────────────────────────────

def _fig_to_rgb(fig, target_w: int) -> np.ndarray:
    """Render a matplotlib figure to (OVERLAY_H, target_w, 3) uint8."""
    fig.canvas.draw()
    raw = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fw, fh = fig.canvas.get_width_height()
    raw = raw.reshape(fh, fw, 3)
    plt.close(fig)
    if fw != target_w or fh != OVERLAY_H:
        pil = PILImage.fromarray(raw).resize((target_w, OVERLAY_H), PILImage.LANCZOS)
        return np.array(pil)
    return raw


def render_overlay_separate(activation, frame_w: int, step: int,
                             num_experts: int, top_k: int) -> np.ndarray:
    """
    Horizontal bar chart with ALL experts shown.
    Top-k active ones are highlighted in orange; rest are gray.
    Bar height = router probability for that expert.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(frame_w / dpi, OVERLAY_H / dpi), dpi=dpi)
    fig.patch.set_facecolor("#111122")
    ax.set_facecolor("#111122")

    if activation is None:
        ax.text(0.5, 0.5, "— awaiting first action chunk —",
                color="#666688", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.axis("off")
        return _fig_to_rgb(fig, frame_w)

    top_indices = activation["topk_indices"][0].tolist()   # list of top-k expert ids
    top_weights = activation["topk_weights"][0]            # (top_k,) raw weights

    # Normalise top-k weights so the visible bars sum to 1
    w_sum = float(top_weights.sum()) + 1e-9
    norm_weights = top_weights / w_sum                     # (top_k,)

    # Build height array: active experts get their normalised weight, rest are 0
    heights = np.zeros(num_experts, dtype=np.float32)
    for idx, w in zip(top_indices, norm_weights):
        heights[idx] = float(w)

    ax.bar(range(num_experts), heights, color="#f4a261", width=0.72, edgecolor="none")

    # Sort by weight descending for the title annotation
    pairs = sorted(zip(top_indices, norm_weights.tolist()), key=lambda x: -x[1])
    top_str = "  |  ".join(f"E{idx} ({w:.2f})" for idx, w in pairs)

    ax.set_xlim(-0.6, num_experts - 0.4)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f"E{i}" for i in range(num_experts)],
                       color="#ccccdd", fontsize=max(6, 8 - num_experts // 4))
    ax.tick_params(colors="#ccccdd", length=2)
    ax.spines[:].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title(
        f"Step {step}   Active top-{top_k}: {top_str}",
        color="#eeeeee", fontsize=8, pad=3, loc="left",
    )

    fig.tight_layout(pad=0.3)
    return _fig_to_rgb(fig, frame_w)


def render_overlay_perlayer(activation, frame_w: int, step: int,
                             num_experts: int) -> np.ndarray:
    """
    Left: heatmap (layers × experts), color = fraction of tokens routed there.
    Right: bar chart aggregated across layers (shows dominant expert overall).
    """
    dpi = 100
    fig, (ax_h, ax_b) = plt.subplots(
        1, 2, figsize=(frame_w / dpi, OVERLAY_H / dpi), dpi=dpi,
        gridspec_kw={"width_ratios": [4, 1]},
    )
    fig.patch.set_facecolor("#111122")
    ax_h.set_facecolor("#111122")
    ax_b.set_facecolor("#111122")

    if activation is None:
        ax_h.text(0.5, 0.5, "— awaiting first action chunk —",
                  color="#666688", ha="center", va="center",
                  transform=ax_h.transAxes, fontsize=9)
        ax_h.axis("off"); ax_b.axis("off")
        return _fig_to_rgb(fig, frame_w)

    fracs = activation["layer_fracs"]  # (L, E)
    L, E = fracs.shape

    ax_h.imshow(fracs, aspect="auto", cmap="hot", vmin=0, vmax=1,
                interpolation="nearest")
    ax_h.set_xticks(range(E))
    ax_h.set_xticklabels([f"E{i}" for i in range(E)], color="#ccccdd", fontsize=7)
    stride = max(1, L // 6)
    ax_h.set_yticks(range(0, L, stride))
    ax_h.set_yticklabels([f"L{i}" for i in range(0, L, stride)],
                         color="#ccccdd", fontsize=6)
    ax_h.tick_params(colors="#ccccdd", length=2)
    ax_h.spines[:].set_visible(False)
    ax_h.set_title(f"Step {step}  — per-layer token routing",
                   color="#eeeeee", fontsize=8, pad=2, loc="left")

    # Aggregate bar: average across layers
    mean_e = fracs.mean(0)  # (E,)
    top_idx = int(np.argmax(mean_e))
    b_colors = ["#f4a261" if i == top_idx else "#556677" for i in range(E)]
    ax_b.bar(range(E), mean_e, color=b_colors, width=0.72)
    ax_b.set_xticks(range(E))
    ax_b.set_xticklabels([f"E{i}" for i in range(E)], color="#ccccdd", fontsize=7)
    ax_b.set_ylim(0, 1)
    ax_b.spines[:].set_visible(False)
    ax_b.yaxis.set_visible(False)
    ax_b.tick_params(colors="#ccccdd", length=2)
    ax_b.set_title("avg", color="#eeeeee", fontsize=7, pad=2)

    fig.tight_layout(pad=0.3)
    return _fig_to_rgb(fig, frame_w)


def composite_frame(env_frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Stack overlay (OVERLAY_H rows) above the env_frame."""
    w = env_frame.shape[1]
    if overlay.shape[1] != w:
        ov = PILImage.fromarray(overlay).resize((w, OVERLAY_H), PILImage.LANCZOS)
        overlay = np.array(ov)
    # Clamp height to OVERLAY_H
    overlay = overlay[:OVERLAY_H] if overlay.shape[0] >= OVERLAY_H else np.pad(
        overlay, ((0, OVERLAY_H - overlay.shape[0]), (0, 0), (0, 0))
    )
    return np.vstack([overlay, env_frame])


# ─── Aggregate plots ──────────────────────────────────────────────────────────

def _avg_probs(timeline: list, num_experts: int, is_separate: bool) -> np.ndarray:
    """Mean activation probability per expert over a timeline slice."""
    valid = [a for a in timeline if a is not None]
    if not valid:
        return np.zeros(num_experts)
    if is_separate:
        return np.stack([a["router_probs"][0] for a in valid]).mean(0)
    else:
        return np.stack([a["layer_fracs"].mean(0) for a in valid]).mean(0)


def _draw_hist_bar(ax, avg: np.ndarray, num_experts: int, title: str):
    hi = int(np.argmax(avg))
    colors = ["#f4a261" if i == hi else "#6c63ff" for i in range(num_experts)]
    ax.bar(range(num_experts), avg, color=colors, edgecolor="none")
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f"E{i}" for i in range(num_experts)], fontsize=8)
    ax.set_ylim(0, max(float(avg.max()) * 1.25, 0.05))
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_overall_dist(timeline, num_experts, is_separate, label, path):
    avg = _avg_probs(timeline, num_experts, is_separate)
    fig, ax = plt.subplots(figsize=(max(5, num_experts * 0.9), 3))
    _draw_hist_bar(ax, avg, num_experts, f"{label} — full episode")
    ax.set_ylabel("Avg router probability")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_temporal_grid(timeline, num_experts, is_separate, label, path):
    """2x2 grid: one histogram per quarter of the episode."""
    n = len(timeline)
    if n == 0:
        return
    q = max(1, n // 4)
    chunks = [timeline[i * q: min((i + 1) * q, n)] for i in range(4)]
    titles = ["Q1 (0–25%)", "Q2 (25–50%)", "Q3 (50–75%)", "Q4 (75–100%)"]

    fig, axes = plt.subplots(2, 2, figsize=(max(7, num_experts * 1.2), 5), sharey=False)
    fig.suptitle(f"{label} — expert usage by episode quarter", fontsize=11)
    for ax, chunk, title in zip(axes.flat, chunks, titles):
        avg = _avg_probs(chunk, num_experts, is_separate)
        _draw_hist_bar(ax, avg, num_experts, title)
        ax.set_ylabel("Avg prob", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─── Obs preprocessing ────────────────────────────────────────────────────────

def preprocess_raw_obs(raw_obs: dict, task_instruction: str | None,
                       policy, preprocessor, device: str) -> dict:
    """
    Convert LiberoEnv raw obs dict into the tensor dict that SmolVLAPolicy
    expects (matching training-time preprocessing).

    raw_obs from LiberoEnv has the form:
        {"pixels": {"image": H×W×3 uint8, "image2": H×W×3 uint8}, ...}
    (or {"pixels": {"image": ...}} if only one camera)

    After this function the dict contains:
        observation.images.camera1  (1, C, H, W) float32 in [0, 1]
        observation.images.camera2  (1, C, H, W) float32 in [0, 1]   [if present]
        observation.language_tokens (1, L) long
        observation.language_attention_mask (1, L) long
        observation.state           (1, D) float32                    [if present]
    """
    batch: dict = {}

    # ── images ────────────────────────────────────────────────────────────
    pixels = raw_obs.get("pixels", {})
    # Map LiberoEnv camera slots → training rename_map targets
    cam_map = {
        "image": "observation.images.camera1",
        "image2": "observation.images.camera2",
        "wrist_image": "observation.images.camera2",
    }
    for cam_key, policy_key in cam_map.items():
        if cam_key in pixels:
            img = pixels[cam_key]  # (H, W, 3) uint8
            t = torch.from_numpy(img.copy()).float() / 255.0  # (H, W, 3)
            t = t.permute(2, 0, 1).unsqueeze(0)              # (1, 3, H, W)
            batch[policy_key] = t.to(device)

    # ── proprioceptive state ───────────────────────────────────────────────
    if "robot_state" in raw_obs:
        rs = raw_obs["robot_state"]
        parts = []
        for sub in ["joints", "eef", "gripper"]:
            if sub in rs:
                for v in rs[sub].values():
                    parts.append(
                        torch.from_numpy(np.asarray(v, dtype=np.float32).ravel())
                    )
        if parts:
            state = torch.cat(parts).unsqueeze(0).to(device)  # (1, D)
            batch["observation.state"] = state
    elif "agent_pos" in raw_obs:
        batch["observation.state"] = torch.from_numpy(
            np.asarray(raw_obs["agent_pos"], dtype=np.float32)
        ).unsqueeze(0).to(device)

    # ── language tokens ───────────────────────────────────────────────────
    if task_instruction is not None:
        try:
            vlm_proc = policy.model.vlm_with_expert.processor
            enc = vlm_proc(
                text=task_instruction,
                return_tensors="pt",
                padding="max_length",
                max_length=48,
                truncation=True,
            )
            batch["observation.language_tokens"] = enc["input_ids"].to(device)
            batch["observation.language_attention_mask"] = enc["attention_mask"].to(device)
        except Exception as exc:
            print(f"  [warn] language tokenization failed: {exc}")

    # ── apply policy obs normalizer ───────────────────────────────────────
    if preprocessor is not None:
        try:
            batch = preprocessor(batch)
        except Exception as exc:
            # Normalizer may fail on keys it didn't see; carry on with raw
            pass

    return batch


def get_rgb_frame(raw_obs: dict) -> np.ndarray:
    """Extract a (H, W, 3) uint8 RGB array from the raw env observation."""
    pixels = raw_obs.get("pixels", {})
    for key in ("image", "image2"):
        if key in pixels:
            img = pixels[key]
            return img if img.dtype == np.uint8 else (img * 255).clip(0, 255).astype(np.uint8)
    # fallback
    return np.zeros((256, 256, 3), dtype=np.uint8)


# ─── Rollout ──────────────────────────────────────────────────────────────────

def run_rollout(policy, env, tracker: ExpertTracker, preprocessor,
                task_instruction: str | None, record_video: bool,
                max_steps: int, device: str):
    """Run one episode. Returns (frames, timeline, success)."""
    raw_obs, _ = env.reset()
    policy.reset()
    tracker.reset_episode()

    frames: list = []
    success = False

    for step in range(max_steps):
        raw_frame = get_rgb_frame(raw_obs)
        tracker.record_frame()

        if record_video:
            act = tracker.current_activation
            if tracker.is_separate:
                ov = render_overlay_separate(
                    act, raw_frame.shape[1], step,
                    tracker.num_experts, tracker.top_k,
                )
            else:
                ov = render_overlay_perlayer(
                    act, raw_frame.shape[1], step, tracker.num_experts,
                )
            frames.append(composite_frame(raw_frame, ov))

        batch = preprocess_raw_obs(raw_obs, task_instruction, policy, preprocessor, device)

        with torch.inference_mode():
            action = policy.select_action(batch)

        if isinstance(action, torch.Tensor):
            action_np = action.squeeze(0).cpu().numpy()
        else:
            action_np = np.asarray(action).squeeze()

        raw_obs, reward, terminated, truncated, _ = env.step(action_np)

        if float(reward) > 0:
            success = True
        if terminated or truncated:
            break

    return frames, list(tracker.episode_timeline), success


# ─── Suite evaluation ─────────────────────────────────────────────────────────

def run_suite(policy, preprocessor, suite_name: str, tracker: ExpertTracker,
              n_rollouts_per_task: int, n_videos: int,
              output_dir: Path, device: str, seed: int):
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    bench = benchmark.get_benchmark_dict()
    task_suite = bench[suite_name]()
    n_tasks = len(task_suite.tasks)
    max_steps = TASK_MAX_STEPS.get(suite_name, 300)

    suite_dir = output_dir / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    videos_saved = 0

    for task_id in range(n_tasks):
        task = task_suite.tasks[task_id]
        task_name = task.name
        instruction = getattr(task, "language_instruction", None) or task_name.replace("_", " ")

        task_dir = suite_dir / f"task_{task_id:02d}_{task_name[:40]}"
        task_dir.mkdir(parents=True, exist_ok=True)

        for rollout_idx in range(n_rollouts_per_task):
            record_video = (videos_saved < n_videos)

            env = LiberoEnv(
                task_suite=task_suite,
                task_id=task_id,
                task_suite_name=suite_name,
                camera_name="agentview_image,robot0_eye_in_hand_image",
                obs_type="pixels",
                episode_index=rollout_idx,
                n_envs=1,
            )

            try:
                frames, timeline, success = run_rollout(
                    policy, env, tracker, preprocessor, instruction,
                    record_video, max_steps, device,
                )
            finally:
                env.close()

            status = "SUCCESS" if success else "fail"
            print(f"  [{suite_name}] task {task_id:02d} r{rollout_idx}: "
                  f"{status} ({len(timeline)} steps)")

            all_results.append({
                "task_id": task_id,
                "task_name": task_name,
                "rollout": rollout_idx,
                "success": success,
                "steps": len(timeline),
            })

            # Save video
            if record_video and frames:
                vid_path = task_dir / f"rollout_{rollout_idx:02d}_{status}.mp4"
                imageio.mimsave(str(vid_path), frames, fps=VIDEO_FPS)
                videos_saved += 1
                print(f"    → video saved: {vid_path.name}")

            # Per-task aggregate plots (on the last rollout for this task)
            if rollout_idx == n_rollouts_per_task - 1 and timeline:
                label = f"{suite_name} / task {task_id:02d}"
                save_overall_dist(
                    timeline, tracker.num_experts, tracker.is_separate,
                    label, task_dir / "expert_dist_full.png",
                )
                save_temporal_grid(
                    timeline, tracker.num_experts, tracker.is_separate,
                    label, task_dir / "expert_dist_quarters.png",
                )

                # Serialise timeline to JSON
                serial = []
                for a in timeline:
                    if a is None:
                        serial.append(None)
                    elif a["type"] == "separate":
                        serial.append({
                            "type": "separate",
                            "topk_indices": a["topk_indices"].tolist(),
                            "topk_weights": a["topk_weights"].tolist(),
                            "router_probs": a["router_probs"].tolist(),
                        })
                    else:
                        serial.append({
                            "type": "perlayer",
                            "layer_fracs": a["layer_fracs"].tolist(),
                        })
                (task_dir / "activations.json").write_text(json.dumps(serial))

    # Suite-level summary
    sr = sum(r["success"] for r in all_results) / len(all_results) if all_results else 0.0
    (suite_dir / "results.json").write_text(
        json.dumps({"success_rate": sr, "n_rollouts": len(all_results),
                    "results": all_results}, indent=2)
    )
    print(f"  [{suite_name}] done — success rate: {sr:.1%} ({len(all_results)} rollouts)")
    return all_results


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading policy from: {args.checkpoint_path}")
    policy = load_policy(args.checkpoint_path, args.device)
    preprocessor, _ = load_preprocessors(args.checkpoint_path)

    cfg = policy.config
    model_type = "separate_experts" if cfg.separate_experts else "per-layer FFN MoE"
    print(f"  Model type : {model_type}")
    print(f"  Experts    : {cfg.moe_num_experts}   top-k: {cfg.moe_top_k}")

    tracker = ExpertTracker(policy)
    all_results: dict = {}

    for suite in args.task_suites:
        print(f"\n{'='*64}")
        print(f"Task suite: {suite}")
        results = run_suite(
            policy, preprocessor, suite, tracker,
            n_rollouts_per_task=args.n_rollouts_per_task,
            n_videos=args.n_videos,
            output_dir=out,
            device=args.device,
            seed=args.seed,
        )
        all_results[suite] = results

    summary = {
        suite: {
            "success_rate": (
                sum(r["success"] for r in rs) / len(rs) if rs else 0.0
            ),
            "n_rollouts": len(rs),
        }
        for suite, rs in all_results.items()
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "="*64)
    print("=== Summary ===")
    for suite, s in summary.items():
        print(f"  {suite}: {s['success_rate']:.1%} ({s['n_rollouts']} rollouts)")
    print(f"\nAll outputs saved to: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import functools
import logging
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.libero import _get_suite, _select_task_ids, create_libero_envs_grouped
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all, rollout
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)


def _aggregate_eval_info(eval_infos: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple eval_policy_all outputs into one result dict."""
    acc_keys = ("sum_rewards", "max_rewards", "successes", "video_paths")
    group_acc: dict[str, dict[str, list[float] | list[str]]] = defaultdict(
        lambda: {k: [] for k in acc_keys}
    )
    overall: dict[str, list[float] | list[str]] = {k: [] for k in acc_keys}
    per_task_infos: list[dict[str, Any]] = []

    def _append_value(group: str, key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, list):
            group_acc[group][key].extend(value)
            overall[key].extend(value)
        else:
            group_acc[group][key].append(value)
            overall[key].append(value)

    for info in eval_infos:
        per_task_infos.extend(info.get("per_task", []))
        for task_info in info.get("per_task", []):
            group = task_info["task_group"]
            metrics = task_info["metrics"]
            _append_value(group, "sum_rewards", metrics.get("sum_rewards"))
            _append_value(group, "max_rewards", metrics.get("max_rewards"))
            _append_value(group, "successes", metrics.get("successes"))
            _append_value(group, "video_paths", metrics.get("video_paths", []))

    def _nanmean(values: list[float]) -> float:
        if not values:
            return float("nan")
        return float(torch.tensor(values, dtype=torch.float32).nanmean().item())

    groups_aggregated = {}
    for group, acc in group_acc.items():
        group_sum = acc["sum_rewards"]  # type: ignore[assignment]
        group_max = acc["max_rewards"]  # type: ignore[assignment]
        group_success = acc["successes"]  # type: ignore[assignment]
        group_videos = acc["video_paths"]  # type: ignore[assignment]
        groups_aggregated[group] = {
            "avg_sum_reward": _nanmean(group_sum),  # type: ignore[arg-type]
            "avg_max_reward": _nanmean(group_max),  # type: ignore[arg-type]
            "pc_success": _nanmean(group_success) * 100 if group_success else float("nan"),  # type: ignore[arg-type]
            "n_episodes": len(group_sum),
            "video_paths": list(group_videos),
        }

    overall_sum = overall["sum_rewards"]  # type: ignore[assignment]
    overall_max = overall["max_rewards"]  # type: ignore[assignment]
    overall_success = overall["successes"]  # type: ignore[assignment]
    overall_videos = overall["video_paths"]  # type: ignore[assignment]
    overall_agg = {
        "avg_sum_reward": _nanmean(overall_sum),  # type: ignore[arg-type]
        "avg_max_reward": _nanmean(overall_max),  # type: ignore[arg-type]
        "pc_success": _nanmean(overall_success) * 100 if overall_success else float("nan"),  # type: ignore[arg-type]
        "n_episodes": len(overall_sum),
        "video_paths": list(overall_videos),
    }
    return {"per_task": per_task_infos, "per_group": groups_aggregated, "overall": overall_agg}


def _eval_libero_sequential(
    env_cfg,
    cfg: TrainPipelineConfig,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    step_id: str,
) -> dict[str, Any]:
    """Build/evaluate/close one LIBERO task at a time to avoid host-memory spikes."""
    suite_names = [s.strip() for s in str(env_cfg.task).split(",") if s.strip()]
    eval_infos: list[dict[str, Any]] = []
    n_total = sum(
        len(_select_task_ids(len(_get_suite(s).tasks), env_cfg.task_ids)) for s in suite_names
    )
    n_done = 0

    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        selected_task_ids = _select_task_ids(len(suite.tasks), env_cfg.task_ids)
        for task_id in selected_task_ids:
            logging.info(
                "[eval] starting  suite=%-20s  task_id=%d  (%d/%d)",
                suite_name, task_id, n_done + 1, n_total,
            )
            one_task_env_cfg = dataclasses.replace(env_cfg, task=suite_name, task_ids=[task_id])
            one_task_envs = make_env(
                one_task_env_cfg, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
            )
            try:
                task_eval_info = eval_policy_all(
                    envs=one_task_envs,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                    max_parallel_tasks=1,
                )
                task_pc = task_eval_info["overall"].get("pc_success", float("nan"))
                task_n = task_eval_info["overall"].get("n_episodes", 0)
                logging.info(
                    "[eval] finished  suite=%-20s  task_id=%d  pc_success=%.1f%%  n_episodes=%d  (%d/%d)",
                    suite_name, task_id, task_pc, task_n, n_done + 1, n_total,
                )
                eval_infos.append(task_eval_info)
            except Exception:
                logging.exception(
                    "[eval] FAILED on suite=%s task_id=%d — re-raising", suite_name, task_id
                )
                raise
            finally:
                close_envs(one_task_envs)
            n_done += 1

    aggregated = _aggregate_eval_info(eval_infos)
    return aggregated


def _split_by_task(
    rollout_data: dict,
    suite_name: str,
    task_ids: list[int],
    sub_env_task_ids: list[int],
    n_episodes_per_task: int,
) -> list[dict]:
    """Split a mixed-task rollout into per-task eval_info dicts for _aggregate_eval_info.

    rollout_data has tensors of shape (B, T) where B = len(sub_env_task_ids).
    Returns a list of one-task info dicts, one per unique task_id.
    """
    reward = rollout_data["reward"]   # (B, T)
    success = rollout_data["success"] # (B, T)
    done = rollout_data["done"]       # (B, T)

    T = done.shape[1]
    done_indices = torch.argmax(done.int(), dim=1)  # (B,)
    mask = (torch.arange(T) <= done_indices.unsqueeze(1) + 1).int()  # (B, T)

    sum_rewards_all = (reward * mask).sum(dim=1)   # (B,)
    max_rewards_all = (reward * mask).max(dim=1).values  # (B,)
    successes_all = ((success * mask).sum(dim=1) > 0)    # (B,) bool

    eval_infos = []
    for tid in task_ids:
        indices = [i for i, t in enumerate(sub_env_task_ids) if t == tid]
        metrics = {
            "sum_rewards": sum_rewards_all[indices].tolist(),
            "max_rewards": max_rewards_all[indices].tolist(),
            "successes": successes_all[indices].tolist(),
            "video_paths": [],
        }
        task_info = {"task_group": suite_name, "task_id": tid, "metrics": metrics}
        eval_infos.append({"per_task": [task_info], "per_group": {}, "overall": {}})
    return eval_infos


def _eval_libero_parallel(
    env_cfg,
    cfg: "TrainPipelineConfig",
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    step_id: str,
) -> dict:
    """Cross-task parallel LIBERO eval.

    Builds one mixed VectorEnv per task-group (tasks_per_batch × n_episodes_per_task sub-envs),
    runs a single rollout covering all tasks in the group simultaneously, then splits results
    per task before aggregating.  Reduces wall-clock time roughly by tasks_per_batch.
    """
    import gymnasium as gym

    n_eps = cfg.eval.n_episodes_per_task or cfg.eval.n_episodes
    tasks_per_batch = cfg.eval.tasks_per_batch

    suite_names = [s.strip() for s in str(env_cfg.task).split(",") if s.strip()]
    eval_infos: list[dict] = []

    env_cls = (
        functools.partial(gym.vector.AsyncVectorEnv, context="forkserver")
        if cfg.eval.use_async_envs
        else gym.vector.SyncVectorEnv
    )

    # Build task_ids filter to pass through gym_kwargs (create_libero_envs_grouped pops it)
    gym_kwargs = dict(getattr(env_cfg, "gym_kwargs", None) or {})
    if env_cfg.task_ids is not None:
        gym_kwargs["task_ids"] = list(env_cfg.task_ids)

    grouped = create_libero_envs_grouped(
        task=",".join(suite_names),
        tasks_per_batch=tasks_per_batch,
        n_episodes_per_task=n_eps,
        gym_kwargs=gym_kwargs,
        camera_name=getattr(env_cfg, "camera_name", "agentview_image,robot0_eye_in_hand_image"),
        init_states=True,
        env_cls=env_cls,
        control_mode=getattr(env_cfg, "control_mode", "relative"),
        episode_length=getattr(env_cfg, "episode_length", None),
    )

    for suite_name, groups in grouped.items():
        for group_info in groups:
            vec = group_info["env"]
            task_ids = group_info["task_ids"]
            sub_env_task_ids = group_info["sub_env_task_ids"]
            try:
                rollout_data = rollout(
                    env=vec,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seeds=list(range(cfg.seed, cfg.seed + vec.num_envs)) if cfg.seed is not None else None,
                )
                task_eval_infos = _split_by_task(
                    rollout_data=rollout_data,
                    suite_name=suite_name,
                    task_ids=task_ids,
                    sub_env_task_ids=sub_env_task_ids,
                    n_episodes_per_task=n_eps,
                )
                eval_infos.extend(task_eval_infos)
            finally:
                vec.close()

    return _aggregate_eval_info(eval_infos)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        mixed_precision = "bf16" if getattr(cfg.policy, "use_amp", False) and not force_cpu else "no"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
            mixed_precision=mixed_precision,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    use_lazy_libero_eval = False
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        if cfg.env.type == "libero":
            use_lazy_libero_eval = True
            logging.info("Using lazy LIBERO eval env construction (sequential task eval).")
        else:
            eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
        resume=cfg.resume,
    )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    processor_pretrained_path = cfg.policy.pretrained_path
    if (
        getattr(cfg.policy, "use_relative_actions", False)
        and processor_pretrained_path is not None
        and not cfg.resume
    ):
        logging.warning(
            "use_relative_actions=true with pretrained processors can skip relative transforms if "
            "the checkpoint processors do not define them. Building processors from current policy config."
        )
        processor_pretrained_path = None

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if processor_pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=processor_pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        # Detect partial local cache: if fewer frames are loaded than the full metadata expects,
        # restrict the sampler to only the episodes actually present on disk. Without this,
        # EpisodeAwareSampler would generate absolute frame indices from full metadata
        # (e.g. 260013) that are out of bounds for the loaded hf_dataset (e.g. 239814 rows).
        if (
            dataset.episodes is None
            and dataset.reader.hf_dataset is not None
            and len(dataset.reader.hf_dataset) < dataset.meta.total_frames
        ):
            episode_indices_to_use = sorted({
                int(ep_idx) for ep_idx in dataset.reader.hf_dataset.unique("episode_index")
            })
            logging.warning(
                "Partial local cache detected (%d/%d frames). Restricting sampler to %d available episodes.",
                len(dataset.reader.hf_dataset),
                dataset.meta.total_frames,
                len(episode_indices_to_use),
            )
        else:
            episode_indices_to_use = dataset.episodes
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=episode_indices_to_use,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        unwrapped = accelerator.unwrap_model(policy)
        if hasattr(unwrapped, "update_step"):
            unwrapped.update_step(step)
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        # Allow save_freq=0 to disable periodic checkpointing without crashing.
        is_saving_step = (cfg.save_freq > 0 and step % cfg.save_freq == 0) or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                _eval_t0 = time.perf_counter()
                with torch.no_grad(), accelerator.autocast():
                    if use_lazy_libero_eval:
                        _eval_fn = (
                            _eval_libero_parallel
                            if cfg.eval.tasks_per_batch > 1
                            else _eval_libero_sequential
                        )
                        eval_info = _eval_fn(
                            env_cfg=cfg.env,
                            cfg=cfg,
                            policy=accelerator.unwrap_model(policy),
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            step_id=step_id,
                        )
                    else:
                        eval_info = eval_policy_all(
                            envs=eval_env,  # dict[suite][task_id] -> vec_env
                            policy=accelerator.unwrap_model(policy),
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            n_episodes=cfg.eval.n_episodes,
                            videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                            max_episodes_rendered=4,
                            start_seed=cfg.seed,
                            max_parallel_tasks=cfg.env.max_parallel_tasks,
                        )
                # overall metrics (suite-agnostic)
                overall_metrics = eval_info["overall"]
                overall_metrics.setdefault("eval_s", time.perf_counter() - _eval_t0)

                # per-suite and overall summary — always logged so results survive a wandb failure
                logging.info(
                    "[eval] COMPLETE  step=%d  overall_pc_success=%.1f%%  n_episodes=%d  eval_s=%.0fs",
                    step,
                    overall_metrics.get("pc_success", float("nan")),
                    overall_metrics.get("n_episodes", 0),
                    overall_metrics["eval_s"],
                )
                for suite, suite_info in eval_info.get("per_group", {}).items():
                    logging.info(
                        "[eval]   suite=%-20s  pc_success=%.1f%%  n_episodes=%d",
                        suite,
                        suite_info.get("pc_success", float("nan")),
                        suite_info.get("n_episodes", 0),
                    )

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = overall_metrics["eval_s"]
                eval_tracker.avg_sum_reward = overall_metrics["avg_sum_reward"]
                eval_tracker.pc_success = overall_metrics["pc_success"]
                if wandb_logger:
                    wandb_log_dict = eval_tracker.to_dict()
                    # Flatten per-suite eval metrics so each suite gets dedicated W&B curves.
                    for suite, suite_metrics in eval_info.get("per_group", {}).items():
                        suite_key = str(suite).replace(" ", "_")
                        for metric_name in ("avg_sum_reward", "avg_max_reward", "pc_success", "n_episodes"):
                            metric_value = suite_metrics.get(metric_name)
                            if isinstance(metric_value, int | float):
                                wandb_log_dict[f"{suite_key}/{metric_name}"] = metric_value
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    if eval_info["overall"].get("video_paths"):
                        wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()

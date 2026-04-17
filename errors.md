# SLURM Debug Log: Variants, Errors, Hypotheses, Fixes

This file summarizes what was tried across `new_slurms`, what failed, why it likely failed, and what was changed.

## Scope

- Primary scripts touched:
  - `new_slurms/finetune_residual_moe_learned_gate.slurm`
  - `new_slurms/finetune_residual_moe_zeroconv.slurm`
  - `new_slurms/finetune_baseline_reinit_expert.slurm`
  - `new_slurms/finetune_baseline_original_expert.slurm` (new)
  - `new_slurms/finetune_baseline_reinit_expert_l90_objdisjoint_eval_object.slurm`
  - `new_slurms/finetune_baseline_original_expert_l90_objdisjoint_eval_object.slurm` (new)
  - `new_slurms/finetune_residual_moe_learned_gate_l90_objdisjoint_eval_object.slurm`
  - `new_slurms/finetune_residual_moe_zeroconv_l90_objdisjoint_eval_object.slurm`
  - plus the other `new_slurms` variants standardized in parallel.

- Core code touched:
  - `src/lerobot/scripts/lerobot_train.py`
  - `src/lerobot/datasets/dataset_reader.py`

---

## 1) Early failure: LIBERO `init_files` path missing

### Observed error

- `FileNotFoundError` for LIBERO assets under an old env path like:
  - `.../.conda/envs/lerobot/.../pruned_init`

### What was suspected

- `~/.libero/config.yaml` pointed to stale paths from a previous environment install.
- SLURM job was using a different active env than the one that generated `~/.libero/config.yaml`.

### Fix applied

- Added per-job LIBERO config generation in all relevant `new_slurms`:
  - `export LIBERO_CONFIG_PATH="$SCR/.libero"`
  - Python block calling `libero.libero.get_default_path_dict()` and writing config under scratch.

### Status

- This class of path error was addressed by forcing env-local/scratch-local LIBERO config each run.

---

## 2) Early cancellation: short time limit

### Observed error

- SLURM job canceled due to time limit (`00:05:00` / `00:03:00` style limits in some scripts).

### What was suspected

- Not a code bug; scheduler walltime too short.

### Fix applied

- Updated key jobs to longer time limits (notably `30:00:00` where requested).

### Status

- Time-limit failures reduced for targeted scripts.

---

## 3) Disk quota exceeded despite scratch usage

### Observed error

- `OSError: [Errno 122] Disk quota exceeded` under:
  - `~/.local/share/wandb/artifacts/staging/...`

### What was suspected

- W&B artifacts were still staging in home directory even when output/checkpoints were on scratch.

### Fix applied

- Standardized W&B env + flags in SLURMs:
  - `WANDB_MODE=offline`
  - `WANDB_DIR=$SCR/wandb`
  - `WANDB_CACHE_DIR=$SCR/.cache/wandb`
  - `WANDB_DATA_DIR=$SCR/.local/share/wandb`
  - `--wandb.disable_artifact=true`
  - `--wandb.mode=offline`

### Status

- Home-directory W&B staging path risk mitigated.

---

## 4) Non-objdisjoint runs failed fast on HF offline lookup

### Observed error

- `huggingface_hub.errors.OfflineModeIsEnabled` during dataset revision lookup (e.g. `libero_90/refs`).
- Happened when switching non-objdisjoint jobs to `libero_90` + multiple eval suites.

### What was suspected

- `HF_HUB_OFFLINE=1` prevented a code path that still wanted remote revision metadata (`get_safe_version` / refs).

### Fix attempt A

- Temporarily switched non-objdisjoint scripts to `HF_HUB_OFFLINE=0`.
- Added explicit:
  - `--dataset.root=/scratch/.../datasets--HuggingFaceVLA--libero/snapshots/<rev>`
  - `--dataset.repo_id=HuggingFaceVLA/libero`

### Result of attempt A

- Bypassed `OfflineModeIsEnabled`, but introduced DNS/network dependency on compute nodes.

---

## 5) New fast failure with network/DNS

### Observed error

- `httpx.ConnectError: [Errno -2] Name or service not known`
- Indicates compute node could not resolve/reach HF endpoint.

### What was suspected

- Cluster GPU nodes should be treated as effectively offline for HF API calls.
- Any fallback path requiring `list_repo_refs` / remote metadata is fragile.

### Fix applied (code + SLURM)

- Patched `src/lerobot/datasets/dataset_reader.py`:
  - In local-cache validation with `episodes=None`, accept locally available episodes instead of forcing "all metadata episodes must exist" and triggering download/sync fallback.
  - Added warning log when local cache is partial.
- Reverted all `new_slurms` back to:
  - `HF_HUB_OFFLINE=1`

### Intended behavior now

- Prefer local `dataset.root` cache and do not require HF network access on node startup.

---

## 6) OOM during multi-suite evaluation

### Observed error

- SLURM `OUT_OF_MEMORY` in runs evaluating multiple suites/tasks.

### What was suspected

- Even with `env.max_parallel_tasks=1`, eval was constructing many envs up front across suites/tasks.
- Memory spike was from environment construction pattern, not just `eval.batch_size`.

### Fix applied

- Patched `src/lerobot/scripts/lerobot_train.py` to evaluate LIBERO sequentially:
  - Build env(s) for one task
  - Evaluate
  - Close env(s)
  - Move to next task
- Avoids large up-front env allocation across all tasks.

### Status

- Architectural fix applied to reduce host RAM pressure during eval.

---

## 7) SLURM argument and variant changes made

### Standardization done across `new_slurms`

- `set -euo pipefail` + `export PS1=${PS1:-}` ordering.
- Conda/module setup normalized.
- Scratch-centered dirs normalized (`TMPDIR`, `PIP_CACHE_DIR`, W&B dirs).
- LIBERO config generated per job in scratch.
- `--save_freq=5000` standardized.
- Non-objdisjoint training/eval target adjusted where requested:
  - train on local-backed `HuggingFaceVLA/libero` root snapshot
  - eval task list like `libero_10,libero_object,libero_goal`.

### New script variants created

- `finetune_baseline_original_expert.slurm`
  - cloned from reinit baseline variant
  - changed to `--policy.reinit_expert_mlps=false`
- `finetune_baseline_original_expert_l90_objdisjoint_eval_object.slurm`
  - cloned from l90 objdisjoint reinit baseline variant
  - changed to `--policy.reinit_expert_mlps=false`

---

## 8) High-level status by family

- Objdisjoint family: historically the more stable path.
- Non-objdisjoint family: primary instability was offline-vs-online HF metadata calls; latest changes aim to keep it fully local/offline.
- Residual MoE variants (`learned_gate`, `zeroconv`) and baseline variants both received the same infra fixes (LIBERO config, W&B scratch, offline handling, save frequency).

---

## 9) Remaining uncertainty

- Latest code+SLURM fixes are in place, but full end-to-end success still depends on current node state + integrity/completeness of local cached dataset root at runtime.
- If failures continue, first check should be startup logs for:
  - any remaining hub/network call attempt
  - local dataset root consistency warnings
  - memory behavior during first eval boundary.


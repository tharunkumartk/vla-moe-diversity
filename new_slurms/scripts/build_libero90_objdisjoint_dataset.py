#!/usr/bin/env python3
"""Build LIBERO-90 object-disjoint subset dataset for training.

Keeps LIBERO-90 tasks whose object types do NOT overlap LIBERO-Object object vocabulary.
Uses audit CSV + BDDL language mapping to identify target task-language strings, then
selects episodes from a source LeRobot dataset that match those tasks.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

LIBERO_OBJECT_VOCAB = {
    "alphabet_soup",
    "basket",
    "bbq_sauce",
    "butter",
    "chocolate_pudding",
    "cream_cheese",
    "floor",
    "ketchup",
    "milk",
    "orange_juice",
    "salad_dressing",
    "tomato_sauce",
}


def parse_language_from_bddl(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\(:language\s+([^)]+)\)", text)
    if not m:
        raise ValueError(f"No :language in {path}")
    return m.group(1).strip().lower()


def get_keep_task_ids(matrix_csv: Path) -> list[str]:
    with matrix_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    meta_cols = {"suite", "task_id", "goal_predicates", "goal_template"}
    obj_cols = [c for c in rows[0].keys() if c not in meta_cols]

    keep = []
    for r in rows:
        if r["suite"] != "libero_90":
            continue
        overlap = any(c in LIBERO_OBJECT_VOCAB and r[c] == "1" for c in obj_cols)
        if not overlap:
            keep.append(r["task_id"])
    return sorted(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-csv", required=True)
    ap.add_argument("--bddl-libero90-dir", required=True)
    ap.add_argument("--source-repo-id", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--source-revision", default=None)
    ap.add_argument("--target-root", required=True)
    args = ap.parse_args()

    matrix_csv = Path(args.matrix_csv)
    bddl_dir = Path(args.bddl_libero90_dir)
    target_root = Path(args.target_root)

    keep_task_ids = get_keep_task_ids(matrix_csv)
    if not keep_task_ids:
        raise RuntimeError("No keep tasks found from matrix CSV")

    keep_languages = set()
    for task_id in keep_task_ids:
        keep_languages.add(parse_language_from_bddl(bddl_dir / f"{task_id}.bddl"))

    ds = LeRobotDataset(
        args.source_repo_id,
        root=args.source_root,
        revision=args.source_revision,
    )

    keep_episode_indices: list[int] = []
    for row in ds.meta.episodes:
        ep = int(row["episode_index"])
        tasks = row.get("tasks")
        if not tasks:
            continue
        # For LIBERO this is usually a single string list.
        task_l = str(tasks[0]).strip().lower()
        if task_l in keep_languages:
            keep_episode_indices.append(ep)

    keep_episode_indices = sorted(set(keep_episode_indices))
    if not keep_episode_indices:
        raise RuntimeError("No episodes matched keep task languages")

    target_root.mkdir(parents=True, exist_ok=True)
    out = split_dataset(ds, {"train": keep_episode_indices}, output_dir=target_root)

    out_ds = out["train"]
    print("Built object-disjoint LIBERO-90 subset")
    print("source_repo:", args.source_repo_id)
    print("source_root:", args.source_root)
    print("target_root:", str(target_root / "train"))
    print("kept_task_count:", len(keep_task_ids))
    print("kept_episode_count:", len(keep_episode_indices))
    print("out_total_episodes:", out_ds.meta.total_episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

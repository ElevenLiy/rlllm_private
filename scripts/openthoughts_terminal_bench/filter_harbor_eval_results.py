#!/usr/bin/env python3
"""Filter Harbor eval tasks by success rate, mapping UIDs back to task directories.

The eval_results.json uses random uuid4 keys (generated per-run), not task directory
names. This script reconstructs the UID→task_directory mapping by correlating the
ordered UIDs in each batch file with the sorted task directory list (which matches
the registration order in register_harbor_eval_dataset.py).
"""
import argparse
import glob
import json
import os


def _get_sorted_task_dirs(task_root_base: str) -> list[str]:
    return sorted(
        (d for d in os.listdir(task_root_base)
         if os.path.isdir(os.path.join(task_root_base, d)) and d.isdigit()),
        key=int,
    )


def _build_uid_to_task_map(eval_results_dir: str, task_dirs: list[str], n_rollouts: int) -> dict[str, str]:
    """Map each UID to its task directory name using positional correspondence."""
    batch_files = sorted(
        glob.glob(os.path.join(eval_results_dir, "eval_results_batch*.json")),
        key=lambda p: int(os.path.basename(p).split("batch")[1].split(".")[0]),
    )
    if not batch_files:
        raise FileNotFoundError(f"No eval_results_batch*.json found in {eval_results_dir}")

    uid_to_task = {}
    task_offset = 0
    for bf in batch_files:
        with open(bf) as f:
            batch = json.load(f)
        uids = batch["uids"]
        unique_uids = []
        for i in range(0, len(uids), n_rollouts):
            uid = uids[i]
            if not unique_uids or unique_uids[-1] != uid:
                unique_uids.append(uid)
        for j, uid in enumerate(unique_uids):
            idx = task_offset + j
            if idx >= len(task_dirs):
                raise ValueError(
                    f"UID index {idx} exceeds task count {len(task_dirs)} "
                    f"(batch file: {os.path.basename(bf)}, position {j})"
                )
            uid_to_task[uid] = task_dirs[idx]
        task_offset += len(unique_uids)

    return uid_to_task


def main():
    parser = argparse.ArgumentParser(
        description="Filter Harbor eval tasks by success rate with UID→task directory mapping"
    )
    parser.add_argument("eval_results", help="Path to eval_results.json")
    parser.add_argument("--min-rate", type=float, default=0.125)
    parser.add_argument("--max-rate", type=float, default=0.875)
    parser.add_argument("--task-root", default=None,
                        help="Harbor-Dataset root (default: $TB_TASKS_ROOT or /data/users/liyu/seta-env/Harbor-Dataset)")
    parser.add_argument("--output", "-o", default="harbor_filtered_tasks_12_85.json")
    args = parser.parse_args()

    task_root_base = args.task_root or os.environ.get(
        "TB_TASKS_ROOT", "/data/users/liyu/seta-env/Harbor-Dataset"
    )

    with open(args.eval_results) as f:
        data = json.load(f)

    per_task = data["per_task_rewards"]
    eval_results_dir = os.path.dirname(os.path.abspath(args.eval_results))

    task_dirs = _get_sorted_task_dirs(task_root_base)
    n_rollouts = len(next(iter(per_task.values()))["all_rewards"])

    uid_to_task = _build_uid_to_task_map(eval_results_dir, task_dirs, n_rollouts)

    unmapped = [uid for uid in per_task if uid not in uid_to_task]
    if unmapped:
        print(f"WARNING: {len(unmapped)} UIDs could not be mapped to task directories")

    filtered = []
    all_rates = []
    for uid, info in per_task.items():
        sr = info["success_rate"]
        all_rates.append(sr)
        if args.min_rate <= sr <= args.max_rate and uid in uid_to_task:
            filtered.append({
                "task_name": uid_to_task[uid],
                "success_rate": sr,
                "rewards": info["all_rewards"],
            })

    filtered.sort(key=lambda x: int(x["task_name"]))

    output = {
        "filter_params": {"min_rate": args.min_rate, "max_rate": args.max_rate},
        "task_root_base": task_root_base,
        "total_tasks_evaluated": len(per_task),
        "filtered_count": len(filtered),
        "summary": {
            "all_zero": sum(1 for r in all_rates if r == 0),
            "all_pass": sum(1 for r in all_rates if r == 1.0),
            "in_range": len(filtered),
            "avg_success_rate": sum(all_rates) / max(len(all_rates), 1),
        },
        "tasks": filtered,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Total tasks evaluated: {len(per_task)}")
    print(f"  Mapped UIDs → task dirs: {len(uid_to_task)}")
    print(f"  0% success (too hard):     {output['summary']['all_zero']}")
    print(f"  100% success (too easy):   {output['summary']['all_pass']}")
    print(f"  [{args.min_rate*100:.1f}%-{args.max_rate*100:.1f}%] (good for RL): {len(filtered)}")
    print(f"  Avg success rate:          {output['summary']['avg_success_rate']:.1%}")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()

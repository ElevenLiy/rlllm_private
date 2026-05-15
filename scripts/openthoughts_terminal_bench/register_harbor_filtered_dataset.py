#!/usr/bin/env python3
"""Register a filtered subset of Harbor tasks for RL training.

Reads a JSON file produced by filter_harbor_eval_results.py and registers
the filtered tasks with a 90/10 train/test split.
"""
import json
import os

from rllm.data.dataset import DatasetRegistry


def _build_examples(tasks: list[str], dataset_name: str, data_source: str,
                    task_root_base: str, repeats: int) -> list[dict]:
    examples = []
    for _ in range(max(repeats, 1)):
        for task_name in tasks:
            examples.append({
                "task_name": task_name,
                "task_root": f"{task_root_base.rstrip('/')}/{task_name}",
                "dataset_name": dataset_name,
                "data_source": data_source,
            })
    return examples


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json = os.path.join(script_dir, "harbor_filtered_tasks_12_85.json")

    filtered_json = os.environ.get("TB_FILTERED_TASKS_JSON", default_json)
    registry_name = os.environ.get("TB_RLLM_DATASET_NAME", "harbor_filtered_12_85")
    source_dataset = os.environ.get("TB_SOURCE_DATASET_NAME", "harbor-dataset")
    data_source = os.environ.get("TB_DATA_SOURCE", "terminal_bench_direct")
    train_ratio = float(os.environ.get("TB_TRAIN_RATIO", "0.9"))
    train_repeats = int(os.environ.get("TB_REPEAT_TRAIN", "1"))
    test_repeats = int(os.environ.get("TB_REPEAT_TEST", "1"))

    with open(filtered_json) as f:
        data = json.load(f)

    task_root_base = os.environ.get("TB_TASKS_ROOT", data["task_root_base"])
    task_names = [t["task_name"] for t in data["tasks"]]

    if not task_names:
        raise RuntimeError(f"No tasks found in {filtered_json}")

    split_idx = int(len(task_names) * train_ratio)
    train_tasks = task_names[:split_idx]
    test_tasks = task_names[split_idx:]

    if not test_tasks:
        test_tasks = train_tasks[-10:]

    train_examples = _build_examples(train_tasks, source_dataset, data_source, task_root_base, train_repeats)
    test_examples = _build_examples(test_tasks, source_dataset, data_source, task_root_base, test_repeats)

    DatasetRegistry.register_dataset(registry_name, train_examples, "train")
    DatasetRegistry.register_dataset(registry_name, test_examples, "test")
    print(
        f"Registered {registry_name}: "
        f"{len(train_tasks)} train tasks (x{train_repeats}={len(train_examples)}), "
        f"{len(test_tasks)} test tasks (x{test_repeats}={len(test_examples)}), "
        f"task_root_base={task_root_base}, "
        f"source={filtered_json}"
    )


if __name__ == "__main__":
    main()

import os

from rllm.data.dataset import DatasetRegistry


def _build_examples(tasks: list[str], dataset_name: str, data_source: str, task_root_base: str, repeats: int) -> list[dict]:
    examples = []
    for _ in range(max(repeats, 1)):
        for task_name in tasks:
            examples.append(
                {
                    "task_name": task_name,
                    "task_root": f"{task_root_base.rstrip('/')}/{task_name}",
                    "dataset_name": dataset_name,
                    "data_source": data_source,
                }
            )
    return examples


def main() -> None:
    registry_name = os.environ.get("TB_RLLM_DATASET_NAME", "openthoughts_nl2bash")
    source_dataset = os.environ.get("TB_SOURCE_DATASET_NAME", "openthoughts-extracted-tasks")
    data_source = os.environ.get("TB_DATA_SOURCE", "terminal_bench_direct")
    task_root_base = os.environ.get("TB_TASKS_ROOT", "/data/openthoughts-extracted-tasks")
    train_repeats = int(os.environ.get("TB_REPEAT_TRAIN", "1"))
    test_repeats = int(os.environ.get("TB_REPEAT_TEST", "1"))
    train_ratio = float(os.environ.get("TB_TRAIN_RATIO", "0.9"))

    # Auto-discover all task_* directories
    task_names = sorted(
        d for d in os.listdir(task_root_base)
        if os.path.isdir(os.path.join(task_root_base, d)) and d.startswith("task_")
    )

    if not task_names:
        raise RuntimeError(f"No task_* directories found in {task_root_base}")

    # Split into train/test
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
        f"task_root_base={task_root_base}"
    )


if __name__ == "__main__":
    main()

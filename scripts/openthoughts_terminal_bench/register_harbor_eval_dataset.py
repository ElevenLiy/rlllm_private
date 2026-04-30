import os

from rllm.data.dataset import DatasetRegistry


def _build_examples(tasks: list[str], dataset_name: str, data_source: str, task_root_base: str) -> list[dict]:
    examples = []
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
    registry_name = os.environ.get("TB_RLLM_DATASET_NAME", "harbor_eval_all")
    source_dataset = os.environ.get("TB_SOURCE_DATASET_NAME", "harbor-dataset")
    data_source = os.environ.get("TB_DATA_SOURCE", "terminal_bench_direct")
    task_root_base = os.environ.get("TB_TASKS_ROOT", "/data/users/liyu/seta-env/Harbor-Dataset")

    task_names = sorted(
        (d for d in os.listdir(task_root_base) if os.path.isdir(os.path.join(task_root_base, d)) and d.isdigit()),
        key=int,
    )

    if not task_names:
        raise RuntimeError(f"No numeric task directories found in {task_root_base}")

    all_examples = _build_examples(task_names, source_dataset, data_source, task_root_base)

    DatasetRegistry.register_dataset(registry_name, all_examples, "test")

    dummy_train_size = int(os.environ.get("TB_DUMMY_TRAIN_SIZE", "32"))
    dummy_train = all_examples[:dummy_train_size]
    DatasetRegistry.register_dataset(registry_name, dummy_train, "train")

    print(
        f"Registered {registry_name}: "
        f"{len(all_examples)} test tasks (all for eval), "
        f"{len(dummy_train)} dummy train tasks, "
        f"task_root_base={task_root_base}"
    )


if __name__ == "__main__":
    main()

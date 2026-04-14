import os

# Force tb environment's libstdc++ to be loaded first for all subprocesses
# This fixes flashinfer GLIBCXX_3.4.32 compatibility issue
tb_lib_path = "/root/anaconda3/envs/tb/lib"
current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if current_ld_path:
    os.environ["LD_LIBRARY_PATH"] = f"{tb_lib_path}:{current_ld_path}"
else:
    os.environ["LD_LIBRARY_PATH"] = tb_lib_path

# Also set LIBRARY_PATH for compile-time linking (flashinfer JIT compilation)
current_lib_path = os.environ.get("LIBRARY_PATH", "")
if current_lib_path:
    os.environ["LIBRARY_PATH"] = f"{tb_lib_path}:{current_lib_path}"
else:
    os.environ["LIBRARY_PATH"] = tb_lib_path

import hydra

from rllm.agents.tool_agent import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

from terminal_bench_direct_env import (
    TERMINAL_BENCH_OPENHANDS_SYSTEM_PROMPT,
    TERMINAL_BENCH_TOOL_MAP,
    TerminalBenchDirectEnv,
)


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    dataset_name = os.environ.get("TB_RLLM_DATASET_NAME", "terminal_bench_direct_smoke")
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    if train_dataset is None or test_dataset is None:
        raise RuntimeError(f"{dataset_name} dataset is missing. Run register_terminal_bench_direct_dataset.py first.")

    trainer = AgentTrainer(
        agent_class=ToolAgent,
        agent_args={
            "system_prompt": TERMINAL_BENCH_OPENHANDS_SYSTEM_PROMPT,
            "parser_name": "qwen",
            "tool_map": TERMINAL_BENCH_TOOL_MAP,
        },
        env_class=TerminalBenchDirectEnv,
        env_args={
            "step_timeout": int(os.environ.get("TB_STEP_TIMEOUT", "120")),
            "verifier_timeout": int(os.environ.get("TB_VERIFIER_TIMEOUT", "1800")),
            "max_steps": int(os.environ.get("TB_MAX_STEPS", "8")),
            "keep_container": os.environ.get("TB_KEEP_CONTAINER", "false"),
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()

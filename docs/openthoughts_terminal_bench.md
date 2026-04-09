# OpenThoughts / Terminal-Bench 复现说明

本文档补充 [README.md](../README.md)，只说明当前仓库的安装、路径和启动方式。

## 1. 数据集与入口

### 1.1 OpenThoughts nl2bash

- 数据集名：`openthoughts_nl2bash`
- 任务根目录：`/data/openthoughts-extracted-tasks`
- 默认模型路径：`/data/models/Qwen3___5-9B`
- 主启动脚本：`scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`
- 数据注册入口：`scripts/openthoughts_terminal_bench/register_openthoughts_dataset.py`
- 训练入口：`scripts/openthoughts_terminal_bench/train_terminal_bench_direct_rllm.py`
- 环境实现：`scripts/openthoughts_terminal_bench/terminal_bench_direct_env.py`

### 1.2 terminal-bench

- 任务根目录：`/data/terminal-bench-2`
- 相关环境实现：`scripts/openthoughts_terminal_bench/terminal_bench_direct_env.py`

## 2. 默认训练口径

| 参数 | 值 |
|---|---|
| 模型 | Qwen3.5-9B |
| 模型路径 | `/data/models/Qwen3___5-9B` |
| 算法 | GRPO |
| 数据集 | `openthoughts_nl2bash` |
| train tasks | 655 |
| test tasks | 73 |
| data.train_batch_size | 32 |
| actor_rollout_ref.rollout.n | 8 |
| n_parallel_agents | 256 |
| data.max_prompt_length | 8192 |
| data.max_response_length | 24576 |
| 总序列长度 | 32768 |
| actor_rollout_ref.actor.ulysses_sequence_parallel_size | 2 |
| actor_rollout_ref.model.use_remove_padding | True |
| attn_implementation | sdpa |
| rollout mode | async (vLLM) |
| trainer.test_freq | 10 |
| trainer.logger | console + tensorboard |
| TB_KUBE_CONTROL_MAX_PARALLEL | 256 |

## 3. 安装方式

统一从共享 `/data` 中的 wheelhouse 安装，并使用仓内 `verl`：

```bash
pip install --no-index \
  --find-links /data/wheelhouse/openthoughts_terminal_bench_py312 \
  -r requirements/openthoughts_terminal_bench.lock.txt
pip install --no-build-isolation --no-deps -e third_party/verl
pip install --no-build-isolation --no-deps -e .
```

## 4. 共享路径

| 资源 | 路径 |
|---|---|
| wheelhouse | `/data/wheelhouse/openthoughts_terminal_bench_py312` |
| OpenThoughts 数据 | `/data/openthoughts-extracted-tasks` |
| terminal-bench 数据 | `/data/terminal-bench-2` |
| Qwen3.5-9B 模型 | `/data/models/Qwen3___5-9B` |
| kubeconfig | `/data/k8s_access/kubeconfig` |
| kubectl | `/data/k8s_access/kubectl.real` |

## 5. K8s 环境变量

```bash
export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG=/data/k8s_access/kubeconfig
export TB_KUBECTL_BIN=/data/k8s_access/kubectl.real
export TB_RUN_OWNER=${TB_RUN_OWNER:-yourname}
export TB_KUBE_NAMESPACE=${TB_KUBE_NAMESPACE:-terminal-bench-${TB_RUN_OWNER}-$(date +%m%d%H%M%S)}
export TB_KUBE_READY_TIMEOUT=1800
export TB_KUBE_CONTROL_MAX_PARALLEL=256
```

## 6. 启动 OpenThoughts 训练

```bash
export TB_TASKS_ROOT=/data/openthoughts-extracted-tasks
export MODEL_PATH=/data/models/Qwen3___5-9B

bash scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh
```

## 7. TensorBoard 与日志

| 项目 | 路径 |
|---|---|
| TensorBoard event dir | `/root/work/tensorboard_log/rllm-openthoughts-nl2bash/<run_name>` |
| 共享 TensorBoard | `/data/rllm_tensorboard/` |
| 训练机运行日志 | `/root/work/launch_*.log` |

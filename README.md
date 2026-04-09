# rllm-private

这个仓库用于复现当前的 `OpenThoughts nl2bash` 和 `terminal-bench` 相关训练链路。

## 分支说明

- `main`：当前默认复现分支。

```bash
git clone git@github.com:to1a/rllm-private.git
cd rllm-private
git branch --show-current
```

## 最小启动

其他机器接手时，只使用共享 `/data` 里的 wheelhouse、模型、数据和 K8s 连接材料。

```bash
git clone git@github.com:to1a/rllm-private.git
cd rllm-private

pip install --no-index \
  --find-links /data/wheelhouse/openthoughts_terminal_bench_py312 \
  -r requirements/openthoughts_terminal_bench.lock.txt

pip install --no-build-isolation --no-deps -e third_party/verl
pip install --no-build-isolation --no-deps -e .

export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG=/data/k8s_access/kubeconfig
export TB_KUBECTL_BIN=/data/k8s_access/kubectl.real
export TB_TASKS_ROOT=/data/openthoughts-extracted-tasks
export MODEL_PATH=/data/models/Qwen3___5-9B

bash scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh
```

共享 `/data` 上当前默认依赖的资源：

- wheelhouse：`/data/wheelhouse/openthoughts_terminal_bench_py312`
- OpenThoughts 任务数据：`/data/openthoughts-extracted-tasks`
- Terminal-Bench 任务数据：`/data/terminal-bench-2`
- Qwen3.5-9B 模型：`/data/models/Qwen3___5-9B`
- K8s 连接材料：`/data/k8s_access/kubeconfig` 和 `/data/k8s_access/kubectl.real`

## 代码入口

- 主说明文档：`docs/openthoughts_terminal_bench.md`
- 当前 OpenThoughts 启动脚本：`scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`
- 仓内训练入口：`scripts/openthoughts_terminal_bench/train_terminal_bench_direct_rllm.py`
- 仓内数据注册入口：`scripts/openthoughts_terminal_bench/register_openthoughts_dataset.py`
- 仓内环境实现：`scripts/openthoughts_terminal_bench/terminal_bench_direct_env.py`
- 仓内固定 `verl`：`third_party/verl/`

## 当前默认训练口径

- 模型：`Qwen3.5-9B`
- 数据集：`openthoughts_nl2bash`
- 默认任务路径：`/data/openthoughts-extracted-tasks`
- 默认模型路径：`/data/models/Qwen3___5-9B`
- train tasks：`655`
- test tasks：`73`
- `data.train_batch_size=32`
- `actor_rollout_ref.rollout.n=8`
- `n_parallel_agents=256`
- `data.max_prompt_length=8192`
- `data.max_response_length=24576`
- 总长度：`32768`
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size=2`
- `actor_rollout_ref.model.use_remove_padding=True`
- `attn_implementation=sdpa`
- `trainer.test_freq=10`
- `trainer.logger=[console,tensorboard]`
- `TB_KUBE_CONTROL_MAX_PARALLEL=256`

## K8s 连接材料

最小环境变量示例：

```bash
export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG=/data/k8s_access/kubeconfig
export TB_KUBECTL_BIN=/data/k8s_access/kubectl.real
export TB_RUN_OWNER=${TB_RUN_OWNER:-yourname}
export TB_KUBE_NAMESPACE=${TB_KUBE_NAMESPACE:-terminal-bench-${TB_RUN_OWNER}-$(date +%m%d%H%M%S)}
export TB_KUBE_READY_TIMEOUT=1800
```

注意：`kubeconfig` 带集群访问权限，按凭证文件对待，不要随意外传。

## 关于 `verl`

统一使用仓内的 `third_party/verl/`，不要使用机器上其他来源不明的 `verl`。

# rllm-private

这个仓库是当前私有协作镜像，目前跑了两个数据集方向：

- `OpenThoughts nl2bash`
- `terminal-bench`

其中：

- `OpenThoughts nl2bash` 是当前主要复现路径，也是目前 `main` 分支默认对应的训练口径。
- `terminal-bench` 之前已经实际跑过，但还没有完全打通，当前主要卡在高并发下的控制面稳定性问题。

## 分支说明

- `main`：当前可复现协作分支。
- `upstream-main`：从训练机导入的上游基线分支。


```bash
git clone git@github.com:to1a/rllm-private.git
cd rllm-private
git branch --show-current
```

## 最小启动

如果机器已经具备可用的 Python/CUDA/vLLM/flash-attn 环境，可以先直接按下面这组命令起 OpenThoughts 9B 训练：

```bash
git clone git@github.com:to1a/rllm-private.git
cd rllm-private

pip uninstall verl -y 2>/dev/null
pip install -e third_party/verl/

# 如果当前 shell 已经在可用环境里，可以不设 VENV_PATH。
# 如果依赖不在当前环境里，再按需指定，例如：
# export VENV_PATH=/path/to/your/venv

export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG=/data/k8s_access/kubeconfig
export TB_KUBECTL_BIN=/data/k8s_access/kubectl.real
export TB_TASKS_ROOT=/data/openthoughts-extracted-tasks
export MODEL_PATH=/data/models/Qwen3___5-9B

bash scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh
```

这条命令默认依赖共享 `/data` 上已经准备好的 3 类资源：

- `OpenThoughts` 任务数据：`/data/openthoughts-extracted-tasks`
- `Qwen3.5-9B` 模型：`/data/models/Qwen3___5-9B`
- K8s 连接材料：`/data/k8s_access/kubeconfig` 和 `/data/k8s_access/kubectl.real`

## 代码入口

- 主说明文档：`docs/openthoughts_terminal_bench.md`
- 当前 OpenThoughts 启动脚本：`scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`
- 仓内训练入口：`scripts/openthoughts_terminal_bench/train_terminal_bench_direct_rllm.py`
- 仓内数据注册入口：`scripts/openthoughts_terminal_bench/register_openthoughts_dataset.py`
- 仓内环境实现：`scripts/openthoughts_terminal_bench/terminal_bench_direct_env.py`
- 仓内固定 `verl`：`third_party/verl/`

## 当前主训练口径

- 模型：`Qwen3.5-9B`
- 当前主数据集：`openthoughts_nl2bash`
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

## 数据路径

数据集和任务目录按机器本地路径处理。

- 如果机器有共享 `/data` 挂载，可以把数据放在 `/data`。
- 通过 `TB_TASKS_ROOT` 等环境变量指向实际路径即可。

## K8s 连接材料

为了方便多人直接接手，当前两个 GPU 集群的共享 `/data` 都放了一份同路径的最小 K8s 连接材料：

| 机器 | kubeconfig | kubectl |
|---|---|---|
| 共享 `/data` 路径 | `/data/k8s_access/kubeconfig` | `/data/k8s_access/kubectl.real` |

最小环境变量示例：

```bash
export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG=/data/k8s_access/kubeconfig
export TB_KUBECTL_BIN=/data/k8s_access/kubectl.real
export TB_KUBE_NAMESPACE=terminal-bench
export TB_KUBE_READY_TIMEOUT=1800
```

注意：`kubeconfig` 带集群访问权限，按凭证文件对待，不要随意外传。

## 关于 `verl`

这里默认要求使用仓内的 `third_party/verl/`，不要再把系统里某个已安装但来源不明的 `verl` 当成主版本。

外部 Python / CUDA / flash-attn / vLLM 环境可以继续沿用机器现有环境；但只要涉及 `verl` 行为对齐，统一以仓内这份为准。

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

## 代码入口

- 主说明文档：`docs/openthoughts_terminal_bench.md`
- 当前 OpenThoughts 启动脚本：`scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`
- 仓内固定 `verl`：`third_party/verl/`

## 当前主训练口径

- 模型：`Qwen3.5-9B`
- 当前主数据集：`openthoughts_nl2bash`
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

为了方便多人直接接手，当前两台 GPU 机的 `/data` 都放了一份最小 K8s 连接材料：

| 机器 | kubeconfig | kubectl |
|---|---|---|
| `47.101.174.157:31175` | `/data/k8s_access/31175/kubeconfig` | `/data/k8s_access/31175/kubectl.real` |
| `47.101.174.157:31369` | `/data/k8s_access/31369/kubeconfig` | `/data/k8s_access/31369/kubectl.real` |

最小环境变量示例：

```bash
export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG=/data/k8s_access/31175/kubeconfig
export TB_KUBECTL_BIN=/data/k8s_access/31175/kubectl.real
export TB_KUBE_NAMESPACE=terminal-bench
export TB_KUBE_READY_TIMEOUT=1800
```

注意：`kubeconfig` 带集群访问权限，按凭证文件对待，不要随意外传。

## 关于 `verl`

这里默认要求使用仓内的 `third_party/verl/`，不要再把系统里某个已安装但来源不明的 `verl` 当成主版本。

外部 Python / CUDA / flash-attn / vLLM 环境可以继续沿用机器现有环境；但只要涉及 `verl` 行为对齐，统一以仓内这份为准。

# OpenThoughts / Terminal-Bench 实验详细说明

本文档是 [README.md](../README.md) 的补充，详细描述两个数据集方向的实验状态和技术细节。

---

## 1. 两个数据集方向

### 1.1 OpenThoughts nl2bash（当前主线）

这是当前主训练和主复现路径，`main` 分支默认就是围绕这条链路整理的。

- 数据集：`openthoughts_nl2bash`
- 主启动脚本：`scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`
- 仓内数据注册入口：`scripts/openthoughts_terminal_bench/register_openthoughts_dataset.py`
- 仓内训练入口：`scripts/openthoughts_terminal_bench/train_terminal_bench_direct_rllm.py`
- 仓内环境实现：`scripts/openthoughts_terminal_bench/terminal_bench_direct_env.py`
- 任务根目录：`/data/openthoughts-extracted-tasks/`（728 个 task，train 655 / test 73）
- 默认模型路径：`/data/models/Qwen3___5-9B`

### 1.2 Terminal-Bench（已验证，待稳定）

这条链路更早实际跑过，不是纸面预留，但目前还没有完全稳定。

当前主要瓶颈不是模型本身，而是高并发下的控制面稳定性：

- `TB_KUBE_CONTROL_MAX_PARALLEL` 提升后出现控制面抖动
- 高并发时出现过 `kubectl exit 2`
- 出现过 `newosproc` / `Resource temporarily unavailable` 等进程/线程创建压力问题

当前态度：

- 代码路径完整保留
- 已验证可以部分运行
- 主复现目标仍然优先 OpenThoughts nl2bash

---

## 2. 当前主训练口径

| 参数 | 值 |
|---|---|
| 模型 | Qwen3.5-9B |
| 模型路径 | `/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-9B` |
| 算法 | GRPO |
| 数据集 | openthoughts_nl2bash |
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

---

## 3. 仓内 verl 修补说明

仓库在 `third_party/verl/` 固定了一份带本地修补的 verl，所有人必须统一使用。

关键修补：

| 文件 | 修补内容 |
|---|---|
| `verl/workers/actor/dp_actor.py` | Ulysses SP 多模态误判修复 |
| `verl/utils/torch_functional.py` | Flash-Attention CE 开关逻辑 |

安装方式：

```bash
pip uninstall verl -y 2>/dev/null
pip install -e third_party/verl/
```

外部依赖环境（Python、CUDA、torch、flash-attn、vLLM）可以灵活使用机器已有的版本，但涉及 verl 逻辑一致性必须对齐到仓内代码。

---

## 4. 数据路径详解

### 4.1 共享目录 `/data`

GPU 训练机上 `/data` 是共享挂载，团队所有成员可直接访问：

```
/data/
├── models/                          # 预训练模型
│   ├── Qwen3___5-35B-A3B/           # 35B MoE（Terminal-Bench 用）
│   └── ...
├── rllm_ckpts/                      # 训练 checkpoint
├── rllm_logs/                       # 训练日志
├── rllm_tensorboard/                # TensorBoard 日志
└── rllm-work/                       # 其他共享工作目录
```

### 4.2 任务数据（训练机本地）

| 数据集 | 路径 | 说明 |
|---|---|---|
| OpenThoughts nl2bash | `/data/openthoughts-extracted-tasks/` | 当前主训练默认路径，728 个 task 目录 |
| Terminal Bench | `/data/terminal-bench-2/` | 当前共享路径，已同步到两个 GPU 集群 |
| OpenThoughts 9B 模型 | `/data/models/Qwen3___5-9B/` | 当前主训练默认模型路径 |

### 4.3 环境变量

通过环境变量定位数据路径，不要硬编码：

```bash
export TB_TASKS_ROOT=<任务根目录>
export TB_RLLM_DATASET_NAME=<注册的数据集名>
export TB_SOURCE_DATASET_NAME=<数据源名>
export TB_DEFAULT_DOCKER_IMAGE=<sandbox 镜像>
export TB_MAX_STEPS=48
export TB_EXECUTION_BACKEND=k8s
export TB_KUBE_READY_TIMEOUT=1800
export TB_KUBE_CONTROL_MAX_PARALLEL=256
```

### 4.4 环境安装里的 K8s 连接材料

为了让新机器更快接手，当前两个 GPU 集群的共享 `/data` 下都放了一份同路径的最小 K8s 连接材料：

| 机器 | kubeconfig | kubectl |
|---|---|---|
| 共享 `/data` 路径 | `/data/k8s_access/kubeconfig` | `/data/k8s_access/kubectl.real` |

推荐最小安装步骤：

```bash
pip uninstall verl -y 2>/dev/null
pip install -e third_party/verl/

export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG=/data/k8s_access/kubeconfig
export TB_KUBECTL_BIN=/data/k8s_access/kubectl.real
export TB_KUBE_NAMESPACE=terminal-bench
export TB_KUBE_READY_TIMEOUT=1800
export TB_KUBE_CONTROL_MAX_PARALLEL=256
```

注意：

- `kubeconfig` 是权限凭证，按敏感文件处理。
- `kubectl.real` 是当前训练机验证过可用的版本；如果机器自带 `kubectl` 已经一致，也可以继续使用系统版本。
- 能访问 `terminal-bench` namespace 的含义，是这套 `kubectl + kubeconfig` 组合能正常 `get/create/delete/exec` sandbox pod。

---

## 5. 训练架构

### 5.1 架构概述

```
GPU 训练机 (8× L20X)
├── Actor / Ref / Rollout (verl + vLLM async)
├── GRPO 训练循环
└── 通过 kubectl 控制 K8s sandbox
        │
        ▼
K8s 集群 (~42 nodes)
├── namespace: terminal-bench
└── sandbox pod 执行 bash 命令并返回结果
```

### 5.2 训练流程

1. `register_openthoughts_dataset.py` 将任务目录注册到 `DatasetRegistry`
2. `scripts/openthoughts_terminal_bench/train_terminal_bench_direct_rllm.py` 从 registry 加载 train/test 数据
3. 每条样本不是纯文本，而是 env 样本：
   ```python
   {"task_name": "...", "task_root": "...", "dataset_name": "...", "data_source": "terminal_bench_direct"}
   ```
4. `AgentTrainer` + `AgentWorkflowEngine` 驱动 rollout → env 交互 → reward → 参数更新

### 5.3 Terminal-Bench 已跑通基线（参考）

| 参数 | 值 |
|---|---|
| 模型 | Qwen3.5-35B-A3B (`/data/models/Qwen3___5-35B-A3B`) |
| data.train_batch_size | 28 |
| data.max_response_length | 32768 |
| rollout.tensor_model_parallel_size | 8 |
| rollout.expert_parallel_size | 8 |
| rollout.n | 1 |
| rllm.agent.max_steps | 48 |
| K8s pin 节点 | 10.0.5.76 |

典型一轮结果（core87 fullval）：
- pass@k = 0.345
- reward_1 = 30, reward_0 = 57
- TRUNCATION = 32

---

## 6. K8s 集群详情

| 项目 | 值 |
|---|---|
| API Server | `https://118.196.87.92:6443` |
| kubeconfig（GPU 机） | `/root/.kube/config` |
| kubectl | `/usr/local/bin/kubectl.real` |
| namespace | `terminal-bench` |
| Ready 节点 | ~42 个 |

常用命令：

```bash
/usr/local/bin/kubectl.real get nodes
/usr/local/bin/kubectl.real get pods -n terminal-bench -o wide
```

已知问题：
- `10.0.5.87` 因 CNI 问题不在健康 fan-out 集合中
- official full-suite 当前默认 pin 到 `10.0.5.76`（仅 Terminal-Bench）

---

## 7. TensorBoard 与日志

| 项目 | 路径 |
|---|---|
| TensorBoard event dir | `/root/work/tensorboard_log/rllm-openthoughts-nl2bash/<run_name>` |
| 共享 TensorBoard | `/data/rllm_tensorboard/` |
| 训练机运行日志 | `/root/work/launch_*.log` |

---

## 8. 新增训练线指南

### 8.1 用 OpenThoughts 做 SFT

推荐新开一条独立训练线，不要直接改 TB direct：

1. 参考 `examples/archive/sft_tinker/prepare_norobots_dataset.py` 编写数据准备脚本
2. 注册到 `DatasetRegistry.register_dataset("openthoughts_xxx", ..., "train/test")`
3. 仿照 `examples/archive/sft_tinker/train_norobots_tinker.py` 编写训练脚本
4. 用 `AgentSFTTrainer` 执行

### 8.2 新增环境交互类数据集

不能直接把纯文本数据塞给 `scripts/openthoughts_terminal_bench/train_terminal_bench_direct_rllm.py`，正确做法：

1. 编写对应的 `register_xxx_dataset.py`，生成 env 样本格式
2. 复用或扩展 `rllm/environments/` 中的环境实现
3. 通过 `AgentTrainer` 驱动训练

### 8.3 数据准备脚本参考

仓库 `examples/` 下有丰富的参考：

```
examples/deepcoder/prepare_deepcoder_data.py
examples/simple_math/prepare_math_dataset.py
examples/frozenlake/prepare_frozenlake_data.py
examples/countdown/prepare_countdown_data.py
```

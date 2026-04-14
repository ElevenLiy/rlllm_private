# Harbor Dataset 训练指南

## 数据集信息

- **来源**: SETA Harbor-Dataset (https://github.com/camel-ai/seta-env/tree/main/Harbor-Dataset)
- **任务总数**: 1376 个
- **训练集**: 1238 个任务 (90%)
- **测试集**: 138 个任务 (10%)
- **数据路径**: `/data/harbor-dataset`
- **任务格式**: 与 OpenThoughts 完全兼容

## 任务结构

每个任务目录包含：
```
task_0/
├── instruction.md          # 任务描述
├── task.toml              # 环境配置（Docker、资源、超时等）
├── tests/
│   ├── test.sh           # 验证脚本
│   └── test_outputs.py   # 测试输出解析
├── environment/          # 环境文件
└── solution/            # 参考解决方案
```

## 快速启动

### 1. 配置环境变量

```bash
cd /root/liyu/rllm-private
source setup_env.sh

# 覆盖为 Harbor 配置
export TB_TASKS_ROOT=/data/harbor-dataset
export TB_RLLM_DATASET_NAME=harbor_dataset
export TB_SOURCE_DATASET_NAME=harbor-dataset
```

### 2. 启动训练

```bash
bash scripts/openthoughts_terminal_bench/run_harbor_9b_noeval_resp24k_total32k_sp2.sh
```

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen3.5-9B | 9B 参数量 |
| 算法 | GRPO | Group Relative Policy Optimization |
| 数据集 | Harbor-Dataset | 1238 train + 138 test |
| Batch Size | 32 | 每批次任务数 |
| Rollouts | 8 | 每任务采样数 |
| 并行 Agent | 256 | K8s 并行容器数 |
| Max Prompt | 8192 | 最大提示长度 |
| Max Response | 24576 | 最大响应长度 |
| Total Length | 32768 | 总序列长度 |
| Sequence Parallel | 2 | Ulysses SP 大小 |
| GPUs | 8 | 每节点 GPU 数 |
| Total Steps | 200 | 训练步数 |
| Test Freq | 10 | 每 10 步测试一次 |

## 文件清单

### 新创建的文件

1. **数据注册脚本**
   - 路径: `scripts/openthoughts_terminal_bench/register_harbor_dataset.py`
   - 功能: 自动发现并注册 Harbor 任务

2. **启动脚本**
   - 路径: `scripts/openthoughts_terminal_bench/run_harbor_9b_noeval_resp24k_total32k_sp2.sh`
   - 功能: 配置环境变量并启动训练

3. **数据集目录**
   - 路径: `/data/harbor-dataset/`
   - 内容: 1376 个 task_* 目录

### 复用的文件

- `scripts/openthoughts_terminal_bench/train_terminal_bench_direct_rllm.py` - 训练入口
- `scripts/openthoughts_terminal_bench/terminal_bench_direct_env.py` - 环境实现
- 所有 rllm 核心组件（Agent、Trainer、Workflow 等）

## 验证链路

运行以下命令验证配置：

```bash
# 1. 验证数据集注册
export TB_TASKS_ROOT=/data/harbor-dataset
export TB_RLLM_DATASET_NAME=harbor_dataset
python scripts/openthoughts_terminal_bench/register_harbor_dataset.py

# 预期输出：
# Registered harbor_dataset: 1238 train tasks (x1=1238), 138 test tasks (x1=138)

# 2. 验证训练脚本
export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG=/data/k8s_access/kubeconfig
export TB_KUBECTL_BIN=/data/k8s_access/kubectl.real
export MODEL_PATH=/data/models/Qwen3___5-9B
python scripts/openthoughts_terminal_bench/train_terminal_bench_direct_rllm.py --help

# 预期输出：Hydra 配置信息
```

## 与 OpenThoughts 的差异

| 项目 | OpenThoughts | Harbor |
|------|--------------|--------|
| 任务数量 | 728 (655 train + 73 test) | 1376 (1238 train + 138 test) |
| 数据路径 | `/data/openthoughts-extracted-tasks` | `/data/harbor-dataset` |
| 数据集名 | `openthoughts_nl2bash` | `harbor_dataset` |
| Docker 镜像 | `openthoughts/nl2bash-base:20260402` | `ubuntu:22.04` (默认) |
| 任务类型 | nl2bash 专注 | 通用软件工程任务 |

## TensorBoard

- **日志路径**: `/data/rllm_tensorboard/rllm-harbor-dataset/harbor-9b-k8s-b32-n8-resp24k-total32k-sp2-noeval-tb-t10`
- **启动命令**: 
  ```bash
  tensorboard --logdir /data/rllm_tensorboard/rllm-harbor-dataset
  ```

## 示例任务

**任务 0** (`/data/harbor-dataset/task_0/instruction.md`):
```
You need to create a reproducible package inventory from the current Ubuntu system 
and prepare it for bulk installation on another system. Generate a complete list of 
all currently installed packages (excluding deinstalled ones), save it to a text file 
called `packages.txt` in the home directory, and create a shell script called 
`install_packages.sh` that can be used to reinstall those same packages on a fresh 
Ubuntu system. The script should handle package installation errors gracefully and 
provide a summary of successful and failed installations.
```

## 故障排查

### 问题：数据集未找到
```bash
# 检查数据目录
ls -la /data/harbor-dataset/ | head -20

# 重新注册
python scripts/openthoughts_terminal_bench/register_harbor_dataset.py
```

### 问题：K8s 连接失败
```bash
# 检查 K8s 连通性
export KUBECONFIG=/data/k8s_access/kubeconfig
/data/k8s_access/kubectl.real cluster-info
```

### 问题：Docker 镜像不存在
Harbor 任务的 `task.toml` 中可能指定了特定的 Docker 镜像。如果镜像不存在，可以：
1. 修改 `task.toml` 中的 `environment.docker_image`
2. 或者在 K8s 集群中预先拉取所需镜像

## 下一步

1. **启动训练**: 运行上述启动命令
2. **监控进度**: 查看 TensorBoard 或控制台输出
3. **调整参数**: 根据需要修改启动脚本中的 Hydra 参数
4. **分析结果**: 训练完成后分析模型在 Harbor 任务上的表现

---

**配置完成时间**: 2026-04-13  
**数据集**: SETA Harbor-Dataset (1376 tasks)  
**模型**: Qwen3.5-9B  
**状态**: ✅ 就绪

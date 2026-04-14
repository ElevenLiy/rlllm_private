# rllm-private 环境配置指南

## 环境配置状态

✅ **所有依赖已安装完成**  
✅ **链路检查全部通过**  
✅ **可以开始训练**

---

## 快速开始

### 1. 配置环境变量

```bash
cd /root/liyu/rllm-private
source setup_env.sh
```

### 2. 启动训练

```bash
bash scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh
```

---

## 已安装的组件

### Python 环境
- **Python 版本**: 3.12.7
- **核心包**: rllm 0.3.0rc0, verl 0.6.1, torch 2.10.0, vllm 0.18.0, transformers 5.3.0.dev0

### 依赖资源
- ✅ Wheelhouse: `/data/wheelhouse/openthoughts_terminal_bench_py312`
- ✅ 模型文件: `/data/models/Qwen3___5-9B` (Qwen3.5-9B)
- ✅ 任务数据: `/data/openthoughts-extracted-tasks` (655 train + 73 test)
- ✅ K8s 配置: `/data/k8s_access/kubeconfig`

### Kubernetes 集群
- ✅ 集群连通性: 正常
- ✅ Control Plane: https://118.196.87.92:6443

---

## 训练配置参数

当前默认训练配置（来自 `run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen3.5-9B | 9B 参数量 |
| 算法 | GRPO | Group Relative Policy Optimization |
| 数据集 | openthoughts_nl2bash | 655 train + 73 test |
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

---

## 链路验证

运行完整链路检查：

```bash
bash /tmp/rllm_env_check.sh
```

预期输出：
```
✅ 1. Python 环境: Python 3.12.7
✅ 2. 核心包导入测试: 正常
✅ 3. 依赖资源检查: 全部存在
✅ 4. Kubernetes 集群连通性: 连通
✅ 5. 数据集注册: 成功
✅ 6. 训练脚本可执行性: 可执行
```

---

## 已知问题

### 1. NumPy 版本警告
- **现象**: pandas 提示需要更新 numexpr 和 bottleneck
- **影响**: 仅警告，不影响功能
- **状态**: 可忽略

### 2. 依赖版本冲突
- **现象**: streamlit, opencv-python-headless 等包版本冲突
- **影响**: 不影响训练流程
- **状态**: 可忽略（这些包不在训练关键路径上）

---

## 目录结构

```
/root/liyu/rllm-private/
├── setup_env.sh                    # 环境变量配置脚本
├── SETUP_GUIDE.md                  # 本文档
├── scripts/
│   └── openthoughts_terminal_bench/
│       ├── run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh  # 启动脚本
│       ├── train_terminal_bench_direct_rllm.py                         # 训练入口
│       ├── register_openthoughts_dataset.py                            # 数据注册
│       └── terminal_bench_direct_env.py                                # 环境实现
├── rllm/                           # rllm 核心包
├── third_party/verl/               # verl 训练框架
└── docs/openthoughts_terminal_bench.md  # 详细文档
```

---

## 日志和输出

### TensorBoard
- **默认路径**: `/data/rllm_tensorboard/rllm-openthoughts-nl2bash/<run_name>`
- **启动命令**: `tensorboard --logdir /data/rllm_tensorboard/rllm-openthoughts-nl2bash`

### 训练日志
- **控制台输出**: 实时显示训练进度
- **Ray 日志**: 分布式训练日志

---

## 故障排查

### 问题：K8s 连接失败
```bash
# 检查 kubeconfig
export KUBECONFIG=/data/k8s_access/kubeconfig
/data/k8s_access/kubectl.real cluster-info
```

### 问题：模型加载失败
```bash
# 检查模型文件
ls -lh /data/models/Qwen3___5-9B/
```

### 问题：数据集未注册
```bash
# 重新注册数据集
export TB_TASKS_ROOT=/data/openthoughts-extracted-tasks
python scripts/openthoughts_terminal_bench/register_openthoughts_dataset.py
```

---

## 下一步

1. **启动训练**: 运行上述启动命令
2. **监控进度**: 查看 TensorBoard 或控制台输出
3. **调整参数**: 根据需要修改启动脚本中的 Hydra 参数

---

**配置完成时间**: 2026-04-13  
**配置人员**: Claude Code  
**环境状态**: ✅ 就绪

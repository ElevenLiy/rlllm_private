# Harbor Dataset 训练指南

## 快速启动

### 1. 启动训练（推荐方式）

```bash
cd /root/liyu/rllm-private/scripts/openthoughts_terminal_bench
bash start_training.sh
```

这个脚本会：
- 检查是否有正在运行的训练进程
- 显示配置信息和输出位置
- 在后台启动训练
- 创建带时间戳的日志文件

### 2. 直接运行训练脚本

```bash
cd /root/liyu/rllm-private/scripts/openthoughts_terminal_bench
bash run_harbor_9b_4gpu.sh
```

## 训练配置

### 硬件配置
- **GPU**: 4 张 (GPU 0,1,2,3)
- **模型**: Qwen3.5-9B
- **显存利用率**: 50%

### 训练参数
- **Batch size**: 32 tasks/batch
- **Rollouts**: 8 per task
- **Total steps**: 200
- **Max steps per episode**: 48
- **Learning rate**: 1e-6
- **Algorithm**: GRPO (Group Relative Policy Optimization)

### Token 配置
- **Max prompt length**: 8192
- **Max response length**: 24576
- **Total max length**: 32768

## 输出文件位置

### 日志文件
```
/root/liyu/rllm-private/harbor_train_YYYYMMDD_HHMMSS.log
```
每次运行都会创建新的带时间戳的日志文件

### Checkpoint 文件
```
/root/liyu/rllm-private/checkpoints/rllm-harbor-dataset/harbor-9b-k8s-b32-n4-resp24k-total32k-sp2-noeval-tb-t10/
```
包含：
- 模型权重 checkpoint
- `chat_completions/` 目录：每个 step 的对话历史

### TensorBoard 日志
```
/data/rllm_tensorboard/rllm-harbor-dataset/harbor-9b-k8s-b32-n4-resp24k-total32k-sp2-noeval-tb-t10/
```

## 监控训练

### 查看实时日志
```bash
# 找到最新的日志文件
ls -lt /root/liyu/rllm-private/harbor_train_*.log | head -1

# 实时查看
tail -f /root/liyu/rllm-private/harbor_train_YYYYMMDD_HHMMSS.log
```

### 查看 GPU 使用情况
```bash
watch -n 1 nvidia-smi
```

### 查看训练进程
```bash
ps aux | grep train_terminal_bench_direct_rllm.py
```

### 查看 TensorBoard
```bash
tensorboard --logdir=/data/rllm_tensorboard/rllm-harbor-dataset/harbor-9b-k8s-b32-n4-resp24k-total32k-sp2-noeval-tb-t10 --port=6006
```

## 停止训练

### 方法 1: 使用 PID
```bash
# 找到进程 PID
ps aux | grep train_terminal_bench_direct_rllm.py | grep -v grep

# 停止进程
kill <PID>
```

### 方法 2: 使用 pkill
```bash
pkill -f train_terminal_bench_direct_rllm.py
```

## 分析失败样本

训练过程中会产生失败样本，可以使用分析工具进行分析。

### 1. 运行失败样本分析
```bash
cd /root/liyu/rllm-private/failed_samples_analysis

# 更新日志文件路径（如果需要）
# 编辑 analyze_failed_trajectories.py，修改 LOG_FILE 变量

# 运行分析
python3 analyze_failed_trajectories.py
```

### 2. 查看分析报告
```bash
cd /root/liyu/rllm-private/failed_samples_analysis

# 查看可读报告
cat failure_report.txt

# 查看统计摘要
cat failure_summary.json | jq

# 查看失败样本列表
cat all_failed_trajectories.json | jq '.[0:10]'
```

### 3. 提取具体 trajectory 的对话
```bash
cd /root/liyu/rllm-private/failed_samples_analysis

# 提取 trajectory 3 的对话（文本格式）
python3 extract_trajectory.py 3 txt

# 提取 trajectory 118 的对话（JSON 格式）
python3 extract_trajectory.py 118 json

# 查看提取的对话
cat trajectories/trajectory_3.txt
```

## 重要说明

### Checkpoint 覆盖策略
- **Chat completions**: 新训练会覆盖旧的 `chat_completions/` 目录
- **日志文件**: 每次运行创建新的带时间戳的日志文件，不会覆盖
- **模型 checkpoint**: 按 step 保存，不会覆盖

### 环境要求
- **Conda 环境**: tb
- **Python 版本**: 3.12
- **CUDA 版本**: 12.9
- **必需的库**: flashinfer, vLLM, PyTorch, transformers

### 已解决的问题
- ✅ GLIBCXX_3.4.32 版本依赖问题
- ✅ flashinfer JIT 编译问题
- ✅ libcudart 链接问题
- ✅ 线程资源耗尽问题

## 故障排查

### 问题 1: 训练无法启动
```bash
# 检查 conda 环境
conda activate tb
python3 -c "import torch, vllm, flashinfer; print('OK')"

# 检查 GPU 可用性
nvidia-smi
```

### 问题 2: vLLM EngineCore 崩溃
```bash
# 检查日志中的错误信息
grep -i "error\|exception\|died" harbor_train_*.log

# 检查 flashinfer 缓存
ls -la /root/.cache/flashinfer/
```

### 问题 3: 显存不足
```bash
# 降低 gpu_memory_utilization
# 编辑 run_harbor_9b_4gpu.sh，修改：
# actor_rollout_ref.rollout.gpu_memory_utilization=0.40
```

## 文件结构

```
/root/liyu/rllm-private/
├── harbor_train_YYYYMMDD_HHMMSS.log          # 训练日志（带时间戳）
├── scripts/openthoughts_terminal_bench/
│   ├── start_training.sh                      # 启动脚本（推荐使用）
│   ├── run_harbor_9b_4gpu.sh                  # 4 GPU 训练脚本
│   ├── run_harbor_9b_noeval_resp24k_total32k_sp2.sh  # 8 GPU 训练脚本（旧）
│   └── train_terminal_bench_direct_rllm.py    # 训练主程序
├── checkpoints/rllm-harbor-dataset/
│   └── harbor-9b-k8s-b32-n4-resp24k-total32k-sp2-noeval-tb-t10/
│       ├── chat_completions/                  # 对话历史
│       │   ├── 1.jsonl                        # Step 1 (128 trajectories)
│       │   ├── 2.jsonl                        # Step 2 (128 trajectories)
│       │   └── ...
│       └── checkpoint_*.pt                    # 模型权重
└── failed_samples_analysis/
    ├── README.md                              # 分析工具说明
    ├── analyze_failed_trajectories.py         # 失败样本分析脚本
    ├── extract_trajectory.py                  # Trajectory 提取脚本
    ├── failure_summary.json                   # 统计摘要
    ├── failure_report.txt                     # 可读报告
    ├── all_failed_trajectories.json           # 所有失败样本
    ├── truncation_failures.json               # TRUNCATION 失败
    ├── env_done_failed_failures.json          # ENV_DONE 失败
    └── trajectories/                          # 提取的对话
        ├── trajectory_3.txt
        └── ...
```

## 联系与支持

如有问题，请检查：
1. 日志文件中的错误信息
2. GPU 显存使用情况
3. Conda 环境是否正确激活
4. 必需的环境变量是否设置正确

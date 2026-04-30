# Harbor 全量评估 — 重建指南

本文档记录了 Harbor 数据集（1376 tasks × 8 rollouts）全量评估的完整搭建过程、代码改动、以及遇到的所有报错和解决方案。如果服务器崩溃需要重建，按本文档操作即可。

---

## 目录

1. [环境概览](#1-环境概览)
2. [Conda 环境搭建](#2-conda-环境搭建)
3. [代码改动清单](#3-代码改动清单)
4. [报错与解决方案](#4-报错与解决方案)
5. [运行评估](#5-运行评估)
6. [输出文件说明](#6-输出文件说明)
7. [评估后筛选](#7-评估后筛选)

---

## 1. 环境概览

| 项目 | 值 |
|---|---|
| 服务器 | 8× L20X GPU（使用 GPU 4-7） |
| 项目路径 | `/data/users/liyu/rlllm_private` |
| 数据集路径 | `/data/users/liyu/seta-env/Harbor-Dataset`（1376 个数字目录） |
| 模型路径 | `/data/models/Qwen3___5-9B` |
| Conda 环境 | `tb`（Python 3.12.13） |
| K8s 配置 | `/data/k8s_access/kubeconfig`、`/data/k8s_access/kubectl.real` |
| 执行后端 | K8s sandbox（每个 trajectory 跑在独立 pod 中） |

### 关键包版本

| 包 | 版本 |
|---|---|
| Python | 3.12.13 |
| torch | 2.10.0 |
| vllm | 0.18.0 |
| transformers | 5.3.0.dev0 |
| ray | 2.54.0 |
| flashinfer-python | 0.6.6 |
| cuda-nvcc | 12.9（通过 conda 安装） |
| tensorboard | 2.20.0 |

---

## 2. Conda 环境搭建

```bash
# 创建环境（必须 Python 3.12，wheelhouse 中的包是 cp312）
conda create -n tb python=3.12 -y
conda activate tb

# 从 wheelhouse 安装（跳过依赖冲突）
pip install --no-index --no-deps --find-links /data/users/liyu/rlllm_private/wheelhouse /data/users/liyu/rlllm_private/wheelhouse/*.whl

# 补装 wheelhouse 遗漏的包（需要网络）
pip install codetiming torchdata peft tensorboard

# 安装 cuda-nvcc（flashinfer JIT 编译需要）
conda install -c nvidia cuda-nvcc -y

# 安装项目本身
cd /data/users/liyu/rlllm_private
pip install -e .
pip install -e third_party/verl

# 创建 libcuda.so 符号链接（flashinfer 链接阶段需要）
mkdir -p /data/users/liyu/miniconda3/envs/tb/lib64/stubs
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so /data/users/liyu/miniconda3/envs/tb/lib64/stubs/libcuda.so
```

### K8s 权限

```bash
# 如果 /data/k8s_access/kubeconfig 权限不足（属于 cgf 用户）：
# 以 root 身份执行
chmod 644 /data/k8s_access/kubeconfig
```

---

## 3. 代码改动清单

### 3.1 容错：`rllm/engine/agent_execution_engine.py`

**问题**：单个 trajectory 失败后异常向上传播 → async generator 崩溃 → 主线程 queue.get() 死锁。

**改动 A — `launch_one_trajectory_task()`**：
- 将 `except` 中的 `raise e` 改为构造 dummy trajectory 返回
- 失败时打印 traceback 和红色警告，然后调用 `_make_dummy_trajectory()`

**改动 B — 新增 `_make_dummy_trajectory()` 方法**：
```python
def _make_dummy_trajectory(self, env_idx: int, mode: str):
    """Return a zero-reward placeholder when a trajectory fails unrecoverably."""
    if mode == "Token":
        bos = self.tokenizer.bos_token_id or 0
        eos = self.tokenizer.eos_token_id or 0
        return {
            "prompt_tokens": torch.tensor([bos], dtype=torch.long),
            "response_tokens": torch.tensor([eos], dtype=torch.long),
            "response_masks": torch.tensor([0], dtype=torch.long),
            "trajectory_reward": 0.0,
            "idx": env_idx,
            "chat_completions": [],
            "metrics": {
                "steps": 0, "reward_time": 0.0, "env_time": 0.0,
                "llm_time": 0.0, "total_time": 0.0, "token_mismatch": 0.0,
                "failed": True,
            },
        }
    # 也支持 Step / Text / Conversation 模式
```

**改动 C — `as_completed` 循环安全兜底**：
- 添加 try/except，即使有遗漏异常也返回 dummy 而非 raise
- 保证 generator 总是 yield 恰好 `len(self.envs)` 个结果

### 3.2 断点续跑：`rllm/trainer/verl/agent_ppo_trainer.py`

**改动 A — `_validate_agent()` 开头添加 resume 逻辑**：
- 扫描 `eval_results/eval_results_batch{N}.json`
- 加载已完成 batch 的 rewards/uids/task_names，跳过重跑

**改动 B — 每 batch 完成后保存**：
- 保存 `eval_results_batch{batch_idx}.json`（含 rewards、task_names、uids）

**改动 C — 全部完成后汇总**：
- 保存 `eval_results.json`，含 per-task 分组 reward + 统计摘要
- 包括 `per_task_rewards`（每个 task 的 all_rewards、success_rate、best_score）

**注意**：文件头部需要 `import glob`

### 3.3 评估数据集注册：`register_harbor_eval_dataset.py`（新建）

- 扫描 Harbor-Dataset 下所有数字目录
- 全部 1376 个 task 注册为 "test" split
- 额外注册 32 条 dummy "train" split（满足 verl 框架非空 train dataloader 断言）

### 3.4 评估启动脚本：`run_harbor_eval_all.sh`（新建）

关键参数：
- `CUDA_VISIBLE_DEVICES=4,5,6,7`（使用空闲 GPU）
- `trainer.n_gpus_per_node=4`
- `trainer.val_only=true` + `trainer.val_before_train=true`
- `trainer.logger=[console]`（不用 tensorboard，避免权限问题）
- `n_parallel_agents=64`（128/256 会 OOM）
- `val_kwargs.n=8`（每 task 8 rollouts）
- `data.val_batch_size=128`（每 batch 128 tasks → 1024 trajectories）
- `MASTER_PORT=29501`（不同于训练的 29500，避免冲突）
- `TB_DEFAULT_DOCKER_IMAGE=ubuntu:24.04`（Harbor 用 ubuntu:24.04）
- `gpu_memory_utilization=0.50`
- `enforce_eager=True`
- K8s 后端配置 + LD_LIBRARY_PATH 指向 conda env lib

### 3.5 任务筛选：`filter_tasks_by_success_rate.py`（新建）

- 读取 `eval_results.json`
- 按 task 分组计算 success rate
- 筛选 25%-75% 区间的 task（适合 RL 训练）
- 输出 JSON + 统计摘要

---

## 4. 报错与解决方案

### 4.1 Python 版本不匹配

**报错**：`pip install` wheelhouse 中的 `.whl` 文件失败，提示 cp312 不兼容。

**原因**：wheelhouse 中的包是为 Python 3.12 编译的，但默认环境是 Python 3.10。

**解决**：
```bash
conda create -n tb python=3.12 -y
```

### 4.2 pip 依赖冲突

**报错**：`--no-index` 安装失败，缺少 blinker、numpy 版本冲突。

**解决**：用 `--no-deps` 跳过依赖检查，缺失的包从 PyPI 补装。

### 4.3 K8s kubeconfig 权限

**报错**：`Permission denied: /data/k8s_access/kubeconfig`

**原因**：文件属于 `cgf:cgf`，权限 660，liyu 无法读取。

**解决**：`chmod 644 /data/k8s_access/kubeconfig`（以 root 执行）

### 4.4 vLLM RoPE TypeError

**报错**：
```
TypeError: unsupported operand type(s) for -=: 'set' and 'list'
```
在 `_check_received_keys` 中触发。

**原因**：vLLM 的 `qwen3_5.py` 第 71 行把 `ignore_keys_at_rope_validation` 设为 **list**，但 `_check_received_keys` 做 `received_keys -= ignore_keys` 需要 **set**。

**解决**：修改 vLLM 源码：
```
文件：/data/users/liyu/miniconda3/envs/tb/lib/python3.12/site-packages/vllm/transformers_utils/configs/qwen3_5.py
第 71 行

# 修改前（list）：
kwargs["ignore_keys_at_rope_validation"] = ["mrope_section", "mrope_interleaved"]

# 修改后（set）：
kwargs["ignore_keys_at_rope_validation"] = {"mrope_section", "mrope_interleaved"}
```

### 4.5 tensorboard 未安装

**报错**：`ModuleNotFoundError: No module named 'tensorboard'`

**解决**：
```bash
pip install tensorboard
```

### 4.6 tensorboard 写入权限

**报错**：
```
PermissionError: [Errno 13] Permission denied: '/data/rllm_tensorboard/...'
```

**原因**：`/data/rllm_tensorboard/` 目录属于 `cgf` 用户，liyu 无写权限。

**解决**：在 `run_harbor_eval_all.sh` 中将 logger 改为只用 console：
```bash
# 修改前：
'trainer.logger=[console,tensorboard]'

# 修改后：
'trainer.logger=[console]'
```

### 4.7 flashinfer nvcc 未找到（EngineDeadError）

**报错**：
```
/bin/sh: 1: /data/users/liyu/miniconda3/envs/tb/bin/nvcc: not found
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

**原因**：flashinfer 0.6.6 需要 JIT 编译 GDN prefill kernel（Qwen3.5 的线性注意力层），需要 nvcc。

**解决**：
```bash
conda install -n tb -c nvidia cuda-nvcc -y
```

### 4.8 flashinfer 链接失败 — libcuda.so 未找到

**报错**：
```
cannot find -lcuda: No such file or directory
collect2: error: ld returned 1 exit status
```

**原因**：nvcc 编译 .o 文件成功，但链接 .so 时找不到 `libcuda.so`。conda 的链接器在 `lib64/stubs` 下搜索，而该目录不存在 `libcuda.so`。

**解决**：
```bash
mkdir -p /data/users/liyu/miniconda3/envs/tb/lib64/stubs
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so /data/users/liyu/miniconda3/envs/tb/lib64/stubs/libcuda.so
```

然后需要清除失败的缓存并手动触发编译：
```bash
# 清除失败的缓存
rm -rf /home/liyu/.cache/flashinfer/0.6.6/90a/cached_ops/gdn_prefill_sm90

# 重启评估后，flashinfer 会自动用 nvcc 编译并缓存 .so
# 或者手动编译（如果 .o 文件已存在）：
cd /home/liyu/.cache/flashinfer/0.6.6/90a/cached_ops/gdn_prefill_sm90
PATH="/data/users/liyu/miniconda3/envs/tb/bin:$PATH" ninja
```

编译成功后会生成：
```
/home/liyu/.cache/flashinfer/0.6.6/90a/cached_ops/gdn_prefill_sm90/gdn_prefill_sm90.so (约 4MB)
```

这是一次性操作，后续启动会直接加载缓存的 .so。

### 4.9 flashinfer GDN warmup 警告（非致命）

**现象**：
```
WARNING: GDN prefill kernel warmup (T=16/32/64) failed for layer ...
First inference may OOM due to autotuner.
```

**说明**：这些是 FSDP worker 上的 warmup 失败警告，**不影响运行**。vLLM EngineCore 会在推理时按需编译/加载 kernel。只要 `gdn_prefill_sm90.so` 存在且链接成功，EngineCore 就不会崩溃。

---

## 5. 运行评估

```bash
conda activate tb
cd /data/users/liyu/rlllm_private

# 前台运行
bash scripts/openthoughts_terminal_bench/run_harbor_eval_all.sh

# 后台运行（推荐）
nohup bash -c 'source /data/users/liyu/miniconda3/envs/tb/etc/profile.d/conda.sh && \
  conda activate tb && \
  bash scripts/openthoughts_terminal_bench/run_harbor_eval_all.sh' \
  > harbor_eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 评估数学

| 项目 | 值 |
|---|---|
| 总 tasks | 1376 |
| 每 task rollouts | 8 |
| 总 trajectories | 11,008 |
| Batch size | 128 tasks × 8 rollouts = 1024 trajectories |
| Batch 数 | 11（10×128 + 1×96） |
| 并发 agents | 64 |
| 每 agent 最大步数 | 48 |

### 监控命令

```bash
# 查看完成数
grep -c "completed due to" harbor_eval_*.log

# 查看成功率
echo "Success: $(grep -c 'Reward is 1.0' harbor_eval_*.log)"
echo "Fail:    $(grep -c 'Reward is 0.0' harbor_eval_*.log)"

# 查看当前 batch 进度
grep "Number of Trajectories" harbor_eval_*.log | tail -3

# 检查错误
grep -c "EngineDeadError\|Error executing\|FATAL\|OOM" harbor_eval_*.log

# 实时监控
tail -f harbor_eval_*.log | grep "completed due to\|batch_idx\|eval_results"

# 查看 GPU 使用
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 查看 K8s pod 活动
ps -eo args | grep kubectl | grep -v grep | wc -l
```

### 断点续跑

如果评估中途崩溃（OOM、网络异常等），直接重新运行同样的脚本即可。已完成的 batch 会被自动跳过（通过检测 `eval_results_batch{N}.json` 文件）。

---

## 6. 输出文件说明

```
checkpoints/rllm-harbor-dataset/harbor-eval-all-9b-base/
├── chat_completions/
│   └── 0.jsonl                    # 所有 trajectory 的完整 agent 对话记录（持续追加）
├── eval_results/
│   ├── eval_results_batch0.json   # Batch 0 的 rewards/uids/task_names
│   ├── eval_results_batch1.json   # Batch 1
│   ├── ...                        # Batch 2-10
│   └── eval_results.json          # 全部完成后的汇总文件（含 per-task 统计）
```

### eval_results.json 结构

```json
{
  "total_tasks": 1376,
  "total_trajectories": 11008,
  "avg_success_rate": 0.xxx,
  "tasks_zero_success": N,
  "tasks_full_success": N,
  "tasks_25_75": N,
  "metrics": { ... },
  "per_task_rewards": {
    "uid_xxx": {
      "best_score": 1.0,
      "all_rewards": [0, 1, 0, 1, ...],
      "success_rate": 0.5
    },
    ...
  },
  "task_names": [...]
}
```

---

## 7. 评估后筛选

全量评估完成后，使用筛选脚本找出适合 RL 训练的 tasks：

```bash
python scripts/openthoughts_terminal_bench/filter_tasks_by_success_rate.py \
  checkpoints/rllm-harbor-dataset/harbor-eval-all-9b-base/eval_results/eval_results.json \
  --min-rate 0.25 \
  --max-rate 0.75 \
  --output harbor_filtered_tasks.json
```

输出示例：
```
Total tasks evaluated: 1376
  0% success (too hard):     XXX
  100% success (too easy):   XXX
  [25%-75%] (good for RL):   XXX
  Avg success rate:          XX.X%
Written to harbor_filtered_tasks.json
```

---

## 快速重建 Checklist

如果服务器崩溃需要从头重建，按以下顺序操作：

1. [ ] 创建 conda env `tb`（Python 3.12）
2. [ ] 从 wheelhouse 安装 + 补装缺失包（codetiming, torchdata, peft, tensorboard）
3. [ ] `conda install -c nvidia cuda-nvcc -y`
4. [ ] `pip install -e .` 和 `pip install -e third_party/verl`
5. [ ] 创建 libcuda.so 符号链接
6. [ ] 修复 vLLM qwen3_5.py 第 71 行（list → set）
7. [ ] 确认 K8s kubeconfig 权限（644）
8. [ ] 清除旧的 flashinfer 缓存（如有）：`rm -rf ~/.cache/flashinfer/`
9. [ ] 运行评估脚本（首次启动会自动编译 flashinfer kernel，约 5 分钟）
10. [ ] 监控日志确认无 EngineDeadError

# Harbor Dataset 训练失败样本分析工具

这个目录包含用于分析 Harbor Dataset 训练过程中失败样本的工具和结果。

## 工具说明

### 1. analyze_failed_trajectories.py

从训练日志中提取和分类失败的 trajectory。

**用法:**
```bash
python3 analyze_failed_trajectories.py
```

**功能:**
- 解析训练日志文件 (`harbor_train.log`)
- 提取所有 trajectory 的完成状态和奖励
- 将失败样本分类为:
  - **TRUNCATION**: 超出最大步数限制（模型无法在规定步数内完成任务）
  - **ENV_DONE_FAILED**: 正常完成但答案错误（模型完成了执行但给出了错误答案）
- 生成统计报告和详细失败列表

**输出文件:**
- `failure_summary.json`: 统计摘要
- `failure_report.txt`: 可读的分析报告
- `all_failed_trajectories.json`: 所有失败样本的列表
- `truncation_failures.json`: TRUNCATION 类型失败列表
- `env_done_failed_failures.json`: ENV_DONE_FAILED 类型失败列表

### 2. extract_trajectory.py

根据 trajectory_id 提取原始对话记录。

**用法:**
```bash
python3 extract_trajectory.py <trajectory_id> [output_format]
```

**参数:**
- `trajectory_id`: 要提取的 trajectory ID（整数）
- `output_format`: 输出格式，可选 `json` 或 `txt`（默认 `txt`）

**示例:**
```bash
# 提取 trajectory 3 并保存为文本格式
python3 extract_trajectory.py 3 txt

# 提取 trajectory 118 并保存为 JSON 格式
python3 extract_trajectory.py 118 json
```

**功能:**
- 根据 trajectory_id 自动计算对应的 checkpoint 文件位置
- 提取完整的对话历史（包括 system、user、assistant、tool 消息）
- 提取任务 ID 和任务描述
- 支持两种输出格式：
  - `json`: 结构化 JSON 格式，便于程序处理
  - `txt`: 可读文本格式，便于人工查看

**输出位置:**
- 提取的 trajectory 保存在 `trajectories/` 子目录下
- 文件命名格式: `trajectory_<id>.json` 或 `trajectory_<id>.txt`

## Trajectory ID 和文件的对应关系

训练过程中，每个 step 处理 128 个 trajectory：
- Step 1: trajectory 0-127
- Step 2: trajectory 128-255
- Step 3: trajectory 256-383
- ...

每个 trajectory 对应 checkpoint 文件中的一行：
- 文件路径: `checkpoints/.../chat_completions/<step>.jsonl`
- 每行是一个完整的对话历史（JSON 数组）

## 分析工作流程

1. **运行失败样本分析:**
   ```bash
   python3 analyze_failed_trajectories.py
   ```

2. **查看分析报告:**
   ```bash
   cat failure_report.txt
   ```

3. **查看具体失败样本列表:**
   ```bash
   # 查看所有失败样本
   cat all_failed_trajectories.json | jq '.[0:10]'
   
   # 查看 TRUNCATION 失败
   cat truncation_failures.json | jq '.[0:10]'
   
   # 查看 ENV_DONE 失败
   cat env_done_failed_failures.json | jq '.[0:10]'
   ```

4. **提取具体 trajectory 的对话历史:**
   ```bash
   # 从失败列表中选择一个 trajectory_id
   python3 extract_trajectory.py 3 txt
   
   # 查看提取的对话
   cat trajectories/trajectory_3.txt
   ```

5. **分析失败原因:**
   - 对于 TRUNCATION 失败：检查模型是否陷入循环、策略是否低效
   - 对于 ENV_DONE_FAILED 失败：检查模型的理解能力、测试用例理解是否正确

## 目录结构

```
failed_samples_analysis/
├── README.md                          # 本文件
├── analyze_failed_trajectories.py     # 失败样本分析脚本
├── extract_trajectory.py              # Trajectory 提取脚本
├── failure_summary.json               # 统计摘要
├── failure_report.txt                 # 可读报告
├── all_failed_trajectories.json       # 所有失败样本
├── truncation_failures.json           # TRUNCATION 失败
├── env_done_failed_failures.json      # ENV_DONE 失败
└── trajectories/                      # 提取的 trajectory 对话
    ├── trajectory_3.txt
    ├── trajectory_118.txt
    └── ...
```

## 注意事项

1. 脚本默认读取 `/root/liyu/rllm-private/harbor_train.log` 作为训练日志
2. Checkpoint 目录默认为 `/root/liyu/rllm-private/checkpoints/rllm-harbor-dataset/harbor-9b-k8s-b32-n8-resp24k-total32k-sp2-noeval-tb-t10/chat_completions`
3. 如果需要修改路径，请编辑脚本开头的配置变量
4. 提取的文本文件会自动截断过长的内容（>10000 字符）以避免文件过大

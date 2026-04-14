#!/bin/bash
# ============================================================
# Harbor Dataset Training Launcher (4 GPU version)
# ============================================================

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${REPO_ROOT}/harbor_train_${TIMESTAMP}.log"

# 训练脚本路径
TRAIN_SCRIPT="${SCRIPT_DIR}/run_harbor_9b_4gpu.sh"

# Checkpoint 目录
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints/rllm-harbor-dataset/harbor-9b-k8s-b32-n4-resp24k-total32k-sp2-noeval-tb-t10"

echo "============================================================"
echo "Harbor Dataset Training - 4 GPU Configuration"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - GPUs: 0,1,2,3 (4 GPUs)"
echo "  - Model: Qwen3.5-9B"
echo "  - Batch size: 32 tasks/batch"
echo "  - Rollouts: 8 per task"
echo "  - Total steps: 200"
echo "  - Max steps per episode: 48"
echo ""
echo "Output locations:"
echo "  - Log file: ${LOG_FILE}"
echo "  - Checkpoints: ${CHECKPOINT_DIR}"
echo "  - Chat completions: ${CHECKPOINT_DIR}/chat_completions/"
echo "  - TensorBoard: /data/rllm_tensorboard/rllm-harbor-dataset/harbor-9b-k8s-b32-n4-resp24k-total32k-sp2-noeval-tb-t10"
echo ""
echo "Analysis tools:"
echo "  - Analyze failures: cd ${REPO_ROOT}/failed_samples_analysis && python3 analyze_failed_trajectories.py"
echo "  - Extract trajectory: cd ${REPO_ROOT}/failed_samples_analysis && python3 extract_trajectory.py <id> txt"
echo ""
echo "============================================================"
echo ""

# 检查是否有正在运行的训练进程
if pgrep -f "train_terminal_bench_direct_rllm.py" > /dev/null; then
    echo "ERROR: Training process is already running!"
    echo ""
    echo "Running processes:"
    ps aux | grep "train_terminal_bench_direct_rllm.py" | grep -v grep
    echo ""
    echo "Please stop the existing process first:"
    echo "  pkill -f train_terminal_bench_direct_rllm.py"
    exit 1
fi

# 确认启动
echo "Press Enter to start training, or Ctrl+C to cancel..."
read -r

echo ""
echo "Starting training in background..."
echo "Log file: ${LOG_FILE}"
echo ""

# 后台运行训练脚本
nohup bash "${TRAIN_SCRIPT}" > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

echo "Training started with PID: ${TRAIN_PID}"
echo ""
echo "Monitor training with:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Stop training:"
echo "  kill ${TRAIN_PID}"
echo "  # or"
echo "  pkill -f train_terminal_bench_direct_rllm.py"
echo ""
echo "============================================================"

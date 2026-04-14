#!/bin/bash
# ============================================================
# 快速启动 Harbor Dataset 训练 (4 GPU)
# ============================================================

echo "启动 Harbor Dataset 训练..."
echo ""
echo "配置: 4 GPU (0,1,2,3), Qwen3.5-9B, 200 steps"
echo ""

cd /root/liyu/rllm-private/scripts/openthoughts_terminal_bench
bash start_training.sh

#!/bin/bash
# rllm-private 环境配置脚本
# 使用方法: source setup_env.sh

# 基础路径配置
export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG=/data/k8s_access/kubeconfig
export TB_KUBECTL_BIN=/data/k8s_access/kubectl.real
export TB_TASKS_ROOT=/data/openthoughts-extracted-tasks
export MODEL_PATH=/data/models/Qwen3___5-9B

# K8s 运行配置
export TB_RUN_OWNER="${TB_RUN_OWNER:-${USER:-root}}"
export TB_KUBE_READY_TIMEOUT=1800
export TB_KUBE_CONTROL_MAX_PARALLEL=256

# 动态生成命名空间（每次 source 都会更新）
export TB_KUBE_NAMESPACE="terminal-bench-${TB_RUN_OWNER}-$(date +%m%d%H%M%S)"

# 数据集配置
export TB_RLLM_DATASET_NAME=openthoughts_nl2bash
export TB_SOURCE_DATASET_NAME=openthoughts-extracted-tasks
export TB_DEFAULT_DOCKER_IMAGE=openthoughts/nl2bash-base:20260402
export TB_MAX_STEPS=48
export TB_REPEAT_TRAIN=1
export TB_REPEAT_TEST=1

# PyTorch 配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# NCCL 配置
export NCCL_NVLS_ENABLE=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0

# TensorBoard 配置
export TENSORBOARD_BASE_DIR="${TENSORBOARD_BASE_DIR:-/data/rllm_tensorboard/rllm-openthoughts-nl2bash}"

echo "✅ rllm-private 环境变量已配置"
echo "   - TB_EXECUTION_BACKEND: $TB_EXECUTION_BACKEND"
echo "   - TB_KUBECONFIG: $TB_KUBECONFIG"
echo "   - TB_TASKS_ROOT: $TB_TASKS_ROOT"
echo "   - MODEL_PATH: $MODEL_PATH"
echo "   - TB_KUBE_NAMESPACE: $TB_KUBE_NAMESPACE"
echo ""
echo "启动训练："
echo "   bash scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh"

#!/bin/bash
set -euo pipefail

# ============================================================
# Harbor Filtered RL Training — Qwen3.5-9B (4 GPU, GPU 4-7)
# 763 tasks (12.5%-87.5% success rate), 8 rollouts/task
# 200 training steps, checkpoint every 50, validate every 10
# ============================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

python_env_ok() {
  python3 -c 'import hydra, rllm' >/dev/null 2>&1
}

if python_env_ok; then
  echo "Using current Python environment" >&2
else
  echo "Current Python environment is missing required packages." >&2
  echo "Activate the tb conda environment first: conda activate tb" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# Limit thread counts to prevent resource exhaustion
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export POLARS_MAX_THREADS=4
export RAYON_NUM_THREADS=4

# Distributed environment
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

# Use GPUs 4-7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# --- Harbor filtered dataset config ---
export TB_TASKS_ROOT="${TB_TASKS_ROOT:-/data/users/liyu/seta-env/Harbor-Dataset}"
export TB_FILTERED_TASKS_JSON="${TB_FILTERED_TASKS_JSON:-${SCRIPT_DIR}/harbor_filtered_tasks_12_85.json}"
export TB_RLLM_DATASET_NAME=harbor_filtered_12_85
export TB_SOURCE_DATASET_NAME=harbor-dataset
export TB_DATA_SOURCE=terminal_bench_direct
export TB_DEFAULT_DOCKER_IMAGE=ubuntu:24.04
export TB_MAX_STEPS=48
export TB_REPEAT_TRAIN=1
export TB_REPEAT_TEST=1

# --- K8s sandbox config ---
export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG="${TB_KUBECONFIG:-/data/k8s_access/kubeconfig}"
export TB_KUBECTL_BIN="${TB_KUBECTL_BIN:-/data/k8s_access/kubectl.real}"
export TB_KUBE_READY_TIMEOUT=1800
export TB_KUBE_CONTROL_MAX_PARALLEL="${TB_KUBE_CONTROL_MAX_PARALLEL:-256}"
export TB_RUN_OWNER="$(printf %s "${TB_RUN_OWNER:-${USER:-root}}" | tr "[:upper:]" "[:lower:]" | tr -cs "a-z0-9-" "-" | sed "s/^-*//; s/-*$//")"
TB_HOST_TAG="$(hostname | tr "[:upper:]" "[:lower:]" | tr -cs "a-z0-9" "-" | sed "s/^-*//; s/-*$//" | cut -c1-8)"
TB_TIME_TAG="$(date +%m%d%H%M%S)"
export TB_RUN_ID="${TB_RUN_ID:-train-${TB_TIME_TAG}-${TB_HOST_TAG}}"
export TB_KUBE_NAMESPACE="${TB_KUBE_NAMESPACE:-terminal-bench-${TB_RUN_OWNER:-root}-${TB_RUN_ID}}"

export NCCL_NVLS_ENABLE=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0

# Force tb environment's libstdc++ and CUDA
TB_ENV_LIB="/data/users/liyu/miniconda3/envs/tb/lib"
export LD_LIBRARY_PATH="${TB_ENV_LIB}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${TB_ENV_LIB}:${LIBRARY_PATH:-}"
export CUDA_HOME="/data/users/liyu/miniconda3/envs/tb"
export PATH="${CUDA_HOME}/bin:${PATH}"

export RUN_NAME=harbor-9b-filtered-12-85
export TENSORBOARD_BASE_DIR="${TENSORBOARD_BASE_DIR:-/data/rllm_tensorboard/rllm-harbor-dataset}"
export TENSORBOARD_DIR="${TENSORBOARD_BASE_DIR}/${RUN_NAME}"

cd "${REPO_ROOT}"

# Register filtered Harbor tasks for training
python3 "${SCRIPT_DIR}/register_harbor_filtered_dataset.py"

MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3___5-9B}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${REPO_ROOT}/harbor_train_filtered_${TIMESTAMP}.log"

echo "Starting Harbor filtered training (763 tasks, 200 steps)..."
echo "Log file: ${LOG_FILE}"
echo "Checkpoint dir: ${REPO_ROOT}/checkpoints/rllm-harbor-dataset/${RUN_NAME}"
echo "TensorBoard dir: ${TENSORBOARD_DIR}"
echo "  View with: tensorboard --logdir ${TENSORBOARD_DIR} --port 6006 --bind_all"

exec < /dev/null

exec python3 "${SCRIPT_DIR}/train_terminal_bench_direct_rllm.py" \
  algorithm.adv_estimator=grpo \
  data.train_batch_size=16 \
  data.val_batch_size=128 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=8192 \
  data.max_response_length=24576 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.model.lora_rank=0 \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  algorithm.kl_ctrl.kl_coef=0.02 \
  rllm.mask_truncated_samples=False \
  trainer.critic_warmup=0 \
  'trainer.logger=[console,tensorboard]' \
  trainer.project_name=rllm-harbor-dataset \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.log_episodes=true \
  trainer.val_before_train=false \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.default_hdfs_dir=null \
  trainer.total_epochs=999 \
  trainer.total_training_steps=200 \
  +rllm.agent.engine_args.n_parallel_agents=256 \
  rllm.agent.max_steps=48 \
  rllm.stepwise_advantage.enable=False

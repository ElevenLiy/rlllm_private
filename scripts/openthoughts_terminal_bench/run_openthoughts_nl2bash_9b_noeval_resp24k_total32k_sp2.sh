#!/bin/bash
set -euo pipefail

# ============================================================
# OpenThoughts nl2bash RL Training — Qwen3.5-9B (dense)
# 32 tasks/batch, 8 rollouts/task, 4 grad updates/step
# No validation before the first train step
# ============================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
VENV_PATH="${VENV_PATH:-/root/work/.venv-rllm}"
source "${VENV_PATH}/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# --- OpenThoughts dataset config ---
export TB_TASKS_ROOT="${TB_TASKS_ROOT:-/data/openthoughts-extracted-tasks}"
export TB_RLLM_DATASET_NAME=openthoughts_nl2bash
export TB_SOURCE_DATASET_NAME=openthoughts-extracted-tasks
export TB_DEFAULT_DOCKER_IMAGE=openthoughts/nl2bash-base:20260402
export TB_MAX_STEPS=48
export TB_REPEAT_TRAIN=1
export TB_REPEAT_TEST=1

# --- K8s sandbox config ---
export TB_EXECUTION_BACKEND=k8s
export TB_KUBECONFIG="${TB_KUBECONFIG:-/data/k8s_access/kubeconfig}"
export TB_KUBECTL_BIN="${TB_KUBECTL_BIN:-/data/k8s_access/kubectl.real}"
export TB_KUBE_READY_TIMEOUT=1800
export TB_KUBE_CONTROL_MAX_PARALLEL="${TB_KUBE_CONTROL_MAX_PARALLEL:-256}"

export NCCL_NVLS_ENABLE=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export RUN_NAME=ot-nl2bash-9b-k8s-b32-n8-resp24k-total32k-sp2-noeval-tb-t10
export TENSORBOARD_BASE_DIR="${TENSORBOARD_BASE_DIR:-/data/rllm_tensorboard/rllm-openthoughts-nl2bash}"
export TENSORBOARD_DIR="${TENSORBOARD_BASE_DIR}/${RUN_NAME}"

cd "${REPO_ROOT}"

# Register the OpenThoughts dataset
python3 "${SCRIPT_DIR}/register_openthoughts_dataset.py"

MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3___5-9B}"

exec python3 "${SCRIPT_DIR}/train_terminal_bench_direct_rllm.py" \
  algorithm.adv_estimator=grpo \
  data.train_batch_size=32 \
  data.val_batch_size=128 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=8192 \
  data.max_response_length=24576 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \

  actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.model.lora_rank=0 \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
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
  algorithm.kl_ctrl.kl_coef=0 \
  rllm.mask_truncated_samples=False \
  trainer.critic_warmup=0 \
  'trainer.logger=[console,tensorboard]' \
  trainer.project_name=rllm-openthoughts-nl2bash \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.log_episodes=true \
  trainer.val_before_train=false \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=10 \
  trainer.default_hdfs_dir=null \
  trainer.total_epochs=999 \
  trainer.total_training_steps=200 \
  +rllm.agent.engine_args.n_parallel_agents=256 \
  rllm.agent.max_steps=48 \
  rllm.stepwise_advantage.enable=False

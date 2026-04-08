# OpenThoughts Terminal-Bench Setup

This document describes the current reproducible OpenThoughts nl2bash + terminal-bench training setup carried by the private `main` branch.

## What To Use

- Branch: `main`
- Launch script: `scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`
- Reference `verl` source: `third_party/verl/`

## Current Training Shape

- Model: `Qwen3.5-9B`
- Dataset: `openthoughts_nl2bash`
- Train tasks: `655`
- Test tasks: `73`
- `data.train_batch_size=32`
- `actor_rollout_ref.rollout.n=8`
- `n_parallel_agents=256`
- `data.max_prompt_length=8192`
- `data.max_response_length=24576`
- Total length: `32768`
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size=2`
- `actor_rollout_ref.model.use_remove_padding=True`
- `attn_implementation=sdpa`
- `trainer.test_freq=10`
- `trainer.logger=[console,tensorboard]`
- `TB_KUBE_CONTROL_MAX_PARALLEL=256`

## Dataset Paths

Dataset and task roots are machine-local.

- If a machine has a shared `/data` mount, you can use it.
- If it does not, place datasets anywhere convenient on that machine.
- Point env vars such as `TB_TASKS_ROOT` to the correct local path.

`/data` is optional, not required.

## Reference VERL

The repository now includes a vendored reference copy of the patched `verl` source under `third_party/verl/`.

Use this copy as the source of truth for reproduction and collaboration. Machines with a compatible prebuilt environment can continue to use their own installed dependencies, but when behavior diverges, prefer the vendored `third_party/verl/` tree.

The key local fixes included in this copy are:

- the Ulysses SP multimodal mis-detection fix in `verl/workers/actor/dp_actor.py`
- the optional Flash-Attention CE toggle in `verl/utils/torch_functional.py`

## Current Server Artifacts

- TensorBoard event dir:
  - `/root/work/tensorboard_log/rllm-openthoughts-nl2bash/ot-nl2bash-9b-k8s-b32-n8-resp24k-total32k-sp2-noeval-tb-t10`
- TensorBoard server log:
  - `/root/work/tensorboard_ot_nl2bash.log`
- Current run log:
  - `/root/work/launch_sp2_tb_t10_20260408_025601.log`
- Full git history bundle:
  - `/root/work/artifacts/rllm_history_with_openthoughts_snapshot_20260408.bundle`

## Notes

- `main` is the intended collaboration and reproduction branch.
- `upstream-main` is kept only as the imported baseline reference.

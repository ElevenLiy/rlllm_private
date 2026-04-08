# rllm-private

This repository is the private collaboration mirror for the current OpenThoughts nl2bash + terminal-bench training work.

## Use This Branch

- `main` is the current reproducible collaboration branch.
- `upstream-main` preserves the imported upstream baseline from the training server.
- The main experiment notes live in `docs/openthoughts_terminal_bench.md`.
- The current launch script lives in `scripts/openthoughts_terminal_bench/`.
- The reference patched `verl` source lives in `third_party/verl/`.

Clone and start from `main`:

```bash
git clone git@github.com:to1a/rllm-private.git
cd rllm-private
git branch --show-current
```

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

- If a machine has a shared `/data` mount, you can put shared datasets there.
- If it does not, keep datasets anywhere convenient on that machine.
- Point env vars such as `TB_TASKS_ROOT` to the actual dataset path for the current environment.

`/data` is optional, not required.

## Start Here

- Docs: `docs/openthoughts_terminal_bench.md`
- Script: `scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`
- Vendored `verl`: `third_party/verl/`

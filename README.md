# rllm-private

This repository is the private collaboration mirror for the current OpenThoughts nl2bash + terminal-bench training work.

## Use This Branch

- `main` is the current reproducible collaboration branch.
- `upstream-main` preserves the imported upstream baseline from the training server.
- `snapshots/2026-04-08/` is a docs-and-patches directory inside this repo, not a separate branch.

If you want the working setup, clone `main`:

```bash
git clone git@github.com:to1a/rllm-private.git
cd rllm-private
git branch --show-current
```

## Where To Start

- Main collaboration notes: [`snapshots/2026-04-08/README.md`](./snapshots/2026-04-08/README.md)
- Current launch script: [`snapshots/2026-04-08/scripts/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`](./snapshots/2026-04-08/scripts/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh)
- Runtime patch notes: [`snapshots/2026-04-08/patches/`](./snapshots/2026-04-08/patches)

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

## Notes

- The current OpenThoughts collaboration snapshot lives in `snapshots/2026-04-08/`.
- The full local `rllm` history plus the collaboration snapshot was also preserved on the training server as:
  - `/root/work/artifacts/rllm_history_with_openthoughts_snapshot_20260408.bundle`

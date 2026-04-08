# RLLM OpenThoughts Snapshot

This snapshot captures the current private working state for the OpenThoughts nl2bash 9B training setup together with the collaboration notes needed for teammates to continue work quickly.

## Repo Layout

- Private mirror repo: `git@github.com:to1a/rllm-private.git`
- Private repo default branch: `main`
- Imported upstream baseline branch: `upstream-main`
- Collaboration branch on the training server: `codex/openthoughts-collab-20260408`
- Current collaboration commit in the private repo: `d7d5794a54ec642fef29021e8688c07b6fe1c8ec`
- Upstream base commit the snapshot was built from: `2f317d07ff96acf9e364ffa175ea5283874e56d3`

## Clone and Start

```bash
git clone git@github.com:to1a/rllm-private.git
cd rllm-private
```

For most collaborators, `main` is the right starting point because it already includes the current code snapshot plus the experiment notes under `snapshots/2026-04-08`.

## Included Snapshot Material

- `patches/rllm_worktree.patch`
  - Full diff of the tracked `rllm` worktree against the upstream base commit.
- `per_file_patches/`
  - Same changes split into review-friendly per-file patches.
- `patches/verl_dp_actor_mmfix.patch`
  - Installed-VERL hotfix for the Ulysses SP multimodal mis-detection issue.
- `patches/verl_torch_functional_flash_ce_toggle.patch`
  - Optional installed-VERL patch adding `VERL_FORCE_DISABLE_FLASH_ATTN_CE`.
- `scripts/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`
  - Current launch script snapshot.

## Current Training Shape

- Model: `Qwen3.5-9B`
- Dataset name: `openthoughts_nl2bash`
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

## Dataset Path Guidance

Dataset and task roots are intentionally treated as machine-local.

- If your machine has a shared `/data` mount, it is reasonable to place shared datasets there.
- If it does not, keep the dataset anywhere convenient for that machine.
- In either case, point the launch script or environment variables such as `TB_TASKS_ROOT` at the actual dataset location for that environment.

The snapshot documents the experiment configuration, but it does not require `/data` specifically.

## Current TensorBoard and Logs

- TensorBoard event dir on the training server:
  - `/root/work/tensorboard_log/rllm-openthoughts-nl2bash/ot-nl2bash-9b-k8s-b32-n8-resp24k-total32k-sp2-noeval-tb-t10`
- TensorBoard server log on the training server:
  - `/root/work/tensorboard_ot_nl2bash.log`
- Current run log on the training server:
  - `/root/work/launch_sp2_tb_t10_20260408_025601.log`
- Saved remote snapshot tarball:
  - `/root/work/artifacts/openthoughts_sp2_tb_t10_snapshot_20260408_040254.tar.gz`
- Full git history bundle from the training server:
  - `/root/work/artifacts/rllm_history_with_openthoughts_snapshot_20260408.bundle`

## Recommended Rebuild Flow

1. Start from the private mirror `main`, or compare against `upstream-main` if you want the imported baseline.
2. Install the matching `verl` environment.
3. Apply `patches/verl_dp_actor_mmfix.patch` into the installed `verl` package.
4. Optionally apply `patches/verl_torch_functional_flash_ce_toggle.patch` if you want the debug switch.
5. Adjust dataset paths for your own machine.
6. Use `scripts/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh` as the launch entry.

## Notes

- The snapshot intentionally stores both a full tracked diff and smaller per-file diffs for review.
- The training server worktree still contains unrelated historical backup files outside this snapshot.
- The live training run may advance after this README was generated.

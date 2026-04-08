# RLLM OpenThoughts Snapshot

This snapshot packages the current private working state for collaboration around the OpenThoughts nl2bash 9B training setup and its live TensorBoard-backed training run.

## Base

- Upstream repo: `https://github.com/rllm-org/rllm.git`
- Remote worktree path: `/root/work/rllm`
- Base commit: `2f317d07ff96acf9e364ffa175ea5283874e56d3`
- Current branch on server: `main`
- Collaboration branch on server: `codex/openthoughts-collab-20260408`
- Collaboration commit on server: `0e434ecc7efc1b4936f441ac8673d3431a9312fe`

## Included Files

- `rllm_worktree.patch`
  - Binary git patch for the current dirty `/root/work/rllm` worktree versus the base commit above.
- `per_file_patches/`
  - Per-file diffs split from the same worktree patch.
  - These are easier to review, discuss, and re-apply selectively during collaboration.
- `verl_dp_actor_mmfix.patch`
  - Installed-VERL hotfix patch for `dp_actor.py`.
  - This is the Ulysses SP multimodal mis-detection fix.
- `verl_torch_functional_flash_ce_toggle.patch`
  - Optional installed-VERL patch adding `VERL_FORCE_DISABLE_FLASH_ATTN_CE`.
  - The current active run does not rely on this toggle.
- `run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`
  - Current launch script.
  - Important settings: `rollout.n=8`, `n_parallel_agents=256`, `response_length=24576`, `ulysses_sequence_parallel_size=2`, `test_freq=10`, `trainer.logger=[console,tensorboard]`.

## Current TensorBoard / Logs

- TensorBoard event dir on remote:
  - `/root/work/tensorboard_log/rllm-openthoughts-nl2bash/ot-nl2bash-9b-k8s-b32-n8-resp24k-total32k-sp2-noeval-tb-t10`
- TensorBoard server log on remote:
  - `/root/work/tensorboard_ot_nl2bash.log`
- Current run log on remote:
  - `/root/work/launch_sp2_tb_t10_20260408_025601.log`
- Saved remote snapshot tarball:
  - `/root/work/artifacts/openthoughts_sp2_tb_t10_snapshot_20260408_040254.tar.gz`

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

## Recommended Rebuild Flow

1. Clone upstream `rllm` at commit `2f317d07ff96acf9e364ffa175ea5283874e56d3`.
2. Apply `rllm_worktree.patch`.
3. Install the matching `verl` environment.
4. Apply `verl_dp_actor_mmfix.patch` into the installed `verl` package.
5. Optionally apply `verl_torch_functional_flash_ce_toggle.patch` if you want the debug switch.
6. Use `run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh` as the launch entry.

## Collaboration Guidance

- Treat `rllm_worktree.patch` as the full snapshot.
- Use `per_file_patches/` for review and cherry-picking smaller changes.
- Keep the installed-VERL patches separate from the main repo changes; they patch the runtime environment, not the upstream `rllm` tree.
- If you need to compare with the original baseline, always anchor against base commit `2f317d07ff96acf9e364ffa175ea5283874e56d3`.

## Notes

- This snapshot intentionally stores patches plus scripts instead of a full vendored code dump.
- The current server worktree also contains unrelated historical dirty files and backups; those are not included here on purpose.
- The current active training run on the server may move forward after this snapshot was taken.
- A full git bundle preserving the complete local `rllm` history plus the collaboration commit was also saved on the server:
  - `/root/work/artifacts/rllm_history_with_openthoughts_snapshot_20260408.bundle`

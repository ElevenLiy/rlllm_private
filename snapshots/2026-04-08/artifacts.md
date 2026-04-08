# Remote Artifacts

The live run artifacts are intentionally kept on the remote GPU box instead of being committed as binary blobs.

## TensorBoard / Logs

- TensorBoard event dir:
  - `/root/work/tensorboard_log/rllm-openthoughts-nl2bash/ot-nl2bash-9b-k8s-b32-n8-resp24k-total32k-sp2-noeval-tb-t10`
- TensorBoard server log:
  - `/root/work/tensorboard_ot_nl2bash.log`
- Current training log:
  - `/root/work/launch_sp2_tb_t10_20260408_025601.log`

## Saved Snapshot

- Snapshot tarball:
  - `/root/work/artifacts/openthoughts_sp2_tb_t10_snapshot_20260408_040254.tar.gz`
- Full git history bundle:
  - `/root/work/artifacts/rllm_history_with_openthoughts_snapshot_20260408.bundle`

The TensorBoard tarball contains the current training log, TensorBoard log, TensorBoard event directory, and the active launch script.

The git bundle preserves the full local `rllm` repository history plus the collaboration snapshot commit `0e434ecc7efc1b4936f441ac8673d3431a9312fe`.

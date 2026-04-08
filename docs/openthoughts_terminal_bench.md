# OpenThoughts / terminal-bench 说明

这份文档描述当前私有 `main` 分支对应的实验背景，以及两个数据集方向目前分别处于什么状态。

## 当前涉及的两个数据集方向

### 1. OpenThoughts nl2bash

这是当前主训练和主复现路径。

- 当前主数据集：`openthoughts_nl2bash`
- 当前主启动脚本：`scripts/openthoughts_terminal_bench/run_openthoughts_nl2bash_9b_noeval_resp24k_total32k_sp2.sh`
- 当前 `main` 分支默认就是围绕这条链路整理的

### 2. terminal-bench

这条链路我们更早就实际跑过，不是纸面预留；但是目前还没有完全稳定。

当前主要问题不是模型本身，而是高并发下的控制面稳定性，例如：

- `TB_KUBE_CONTROL_MAX_PARALLEL` 提升后会出现控制面抖动
- 高并发时出现过 `kubectl exit 2`
- 出现过 `newosproc` / `Resource temporarily unavailable` 一类本机进程或线程创建压力问题

所以现在仓库里对 terminal-bench 的态度是：

- 代码路径仍然保留
- 已经验证过能部分运行
- 但当前主复现目标仍然优先是 OpenThoughts nl2bash 这条训练链

## 当前主训练口径

- 模型：`Qwen3.5-9B`
- 数据集：`openthoughts_nl2bash`
- train tasks：`655`
- test tasks：`73`
- `data.train_batch_size=32`
- `actor_rollout_ref.rollout.n=8`
- `n_parallel_agents=256`
- `data.max_prompt_length=8192`
- `data.max_response_length=24576`
- 总长度：`32768`
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size=2`
- `actor_rollout_ref.model.use_remove_padding=True`
- `attn_implementation=sdpa`
- `trainer.test_freq=10`
- `trainer.logger=[console,tensorboard]`
- `TB_KUBE_CONTROL_MAX_PARALLEL=256`

## 必须使用仓内 `verl`

仓库已经把当前带修补的 `verl` 固定在：

- `third_party/verl/`

这里的要求是：

- 统一以仓里的 `third_party/verl/` 为准
- 不再把机器里原先装着的某个 `verl` 当成主版本

外部依赖环境可以灵活一些：

- 机器自带的 Python 环境可以继续用
- 机器自带的 CUDA / torch / flash-attn / vLLM 环境也可以继续用

但只要涉及 `verl` 逻辑一致性，必须对齐到仓内这份代码。

当前这份 `verl` 里最关键的本地修补包括：

- `verl/workers/actor/dp_actor.py` 里的 Ulysses SP 多模态误判修复
- `verl/utils/torch_functional.py` 里的 Flash-Attention CE 开关逻辑

## 数据路径

数据和任务目录按机器本地路径处理。

- 如果某台机器有共享 `/data` 挂载，可以把数据放在那里
- 如果没有，也完全可以放在别的本地路径
- 通过 `TB_TASKS_ROOT` 等环境变量指向实际路径即可

`/data` 是可选项，不是硬依赖。

## 当前训练机现场信息

- TensorBoard event dir:
  - `/root/work/tensorboard_log/rllm-openthoughts-nl2bash/ot-nl2bash-9b-k8s-b32-n8-resp24k-total32k-sp2-noeval-tb-t10`
- TensorBoard server log:
  - `/root/work/tensorboard_ot_nl2bash.log`
- Current run log:
  - `/root/work/launch_sp2_tb_t10_20260408_025601.log`
- Full git history bundle:
  - `/root/work/artifacts/rllm_history_with_openthoughts_snapshot_20260408.bundle`

## 结论

- `main` 是当前唯一默认协作入口
- 当前主复现目标是 OpenThoughts nl2bash
- terminal-bench 不是未开始，而是已经跑过，但高并发稳定性还没完全收敛

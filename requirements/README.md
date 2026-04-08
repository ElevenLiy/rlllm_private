# Requirements Notes

- `openthoughts_terminal_bench.lock.txt`：当前 OpenThoughts / terminal-bench 训练链路使用的锁版本依赖。
- `openthoughts_terminal_bench.packable.lock.txt`：用于 wheelhouse / 离线分发的可打包版本。
- 两者都故意忽略了当前机器上的外部 editable 包 `arcagi3`（`/data/skillscaling/arc-agi-3-benchmarking`），因为它不属于这条训练链路。
- `rllm` 和 `verl` 不从 PyPI 安装，统一使用仓内 editable 安装：
  - `pip install --no-build-isolation --no-deps -e third_party/verl`
  - `pip install --no-build-isolation --no-deps -e .`

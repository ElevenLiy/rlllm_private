# Requirements Notes

- `openthoughts_terminal_bench.lock.txt`：当前 OpenThoughts / terminal-bench 训练链路唯一的锁版本依赖文件。
- 这份锁文件同时用于：
  - 在线安装
  - wheelhouse / 离线分发
- 当前机器上的外部 editable 包 `arcagi3`（`/data/skillscaling/arc-agi-3-benchmarking`）已经故意从锁文件中移除，因为它不属于这条训练链路。
- `rllm` 和 `verl` 不从 PyPI 安装，统一使用仓内 editable 安装：
  - `pip install --no-build-isolation --no-deps -e third_party/verl`
  - `pip install --no-build-isolation --no-deps -e .`

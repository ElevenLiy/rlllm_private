# Requirements Notes

当前只保留一套安装方式：

```bash
pip install --no-index \
  --find-links /data/wheelhouse/openthoughts_terminal_bench_py312 \
  -r requirements/openthoughts_terminal_bench.lock.txt
pip install --no-build-isolation --no-deps -e third_party/verl
pip install --no-build-isolation --no-deps -e .
```

说明：

- `requirements/openthoughts_terminal_bench.lock.txt` 是当前 OpenThoughts / terminal-bench 训练链路的唯一锁版本文件。
- `rllm` 和 `verl` 不从 PyPI 安装，统一使用仓内 editable 安装。
- `verl` 固定使用仓内 `third_party/verl`。

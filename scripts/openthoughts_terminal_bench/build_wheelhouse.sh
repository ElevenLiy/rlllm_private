#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCKFILE="${LOCKFILE:-${ROOT_DIR}/requirements/openthoughts_terminal_bench.lock.txt}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-/data/wheelhouse/openthoughts_terminal_bench_py312}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${WHEELHOUSE_DIR}.tar.gz}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-${PYTHON_BIN} -m pip}"

mkdir -p "${WHEELHOUSE_DIR}"

# Build/download all third-party wheels pinned in the shared lock file without re-resolving dependencies.
${PYTHON_BIN} -m pip wheel --no-deps -r "${LOCKFILE}" -w "${WHEELHOUSE_DIR}"

# Local repo code should still come from the checked-out workspace.
# We only package third-party dependencies here.

tar -C "$(dirname "${WHEELHOUSE_DIR}")" -czf "${ARCHIVE_PATH}" "$(basename "${WHEELHOUSE_DIR}")"

echo "wheelhouse ready: ${WHEELHOUSE_DIR}"
echo "archive ready: ${ARCHIVE_PATH}"

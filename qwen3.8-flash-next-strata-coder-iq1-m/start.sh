#!/bin/bash
# 启动 Strata SYCL 引擎 (Qwen3.8-Flash-Next GSQ-RCO Coder IQ1_M, 单张 A770, OpenAI 兼容)
# 依赖: Strata/build-sycl/strata (编译: ./build.sh)
# 用法: ./start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_DIR=~/workspace/ai/Strata
CONFIG="$SCRIPT_DIR/strata-coder-iq1-m.json"
PORT=8080

ENGINE="$STRATA_DIR/build-sycl/sycl/strata"
PYTHON="$STRATA_DIR/.venv/bin/python"

err() { echo "错误: $*" >&2; exit 1; }

[[ -x "$ENGINE" ]] || err "找不到 $ENGINE (先编译: ./build.sh)"
[[ -f "$CONFIG" ]] || err "找不到 $CONFIG"

# oneAPI 运行时库 + SYCL 环境变量
. "$SCRIPT_DIR/sycl_env.sh" || exit 1

exec "$PYTHON" "$STRATA_DIR/sycl/serve/server_intel.py" \
    --engine strata --config "$CONFIG" --port "$PORT"

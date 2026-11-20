#!/bin/bash
# Coder390 Qwen3.8 27B (Q4LynnStyle, 内置 Q4 MTP) 基准测试 — 双 Arc A770, SYCL layer / tensor 两种切分
# 用法: ./coder-bench.sh [all|layer|tensor]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"
# shellcheck source=sycl_env.sh
source "$SCRIPT_DIR/sycl_env.sh"

CODER_MODEL=~/workspace/ai/models/Merkyor/Qwen3.8-27B-Coder390-EfficientThink-GGUF/Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-MTP.gguf
SYCL_BENCH=$SYCL_BUILD/bin/llama-bench

[[ -f "$CODER_MODEL" ]] || err "找不到模型 $CODER_MODEL"

BENCH_ARGS=(
    -m "$CODER_MODEL"
    -ngl "$NGL"
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
    --flash-attn "$FA"
    --batch-size "$BS"
    --ubatch-size "$UBS"
    -p 2048 -n 128 -r 3
)

run_sycl() {
    local split=$1
    [[ -x "$SYCL_BENCH" ]] || { echo "错误: 找不到 $SYCL_BENCH (先编译: ./build.sh sycl)" >&2; return 1; }
    sycl_env "$SYCL_BENCH"
    echo "sycl-$split"
    "$SYCL_BENCH" --split-mode "$split" "${BENCH_ARGS[@]}"
}

case "${1:-all}" in
    all)
        run_sycl layer || true; sleep 8
        run_sycl tensor
        ;;
    layer)           run_sycl layer ;;
    tensor)          run_sycl tensor ;;
    -h|--help|help)  echo "用法: ./coder-bench.sh [all|layer|tensor]"; exit 0 ;;
    *)               usage "用法: ./coder-bench.sh [all|layer|tensor]"; err "未知参数 '$1'" ;;
esac

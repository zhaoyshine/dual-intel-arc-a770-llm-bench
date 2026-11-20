#!/bin/bash
# IQ4_XS MMVQ 补丁基准测试 — 双 Arc A770, master vs 补丁构建 (build-sycl-opt)
# 测试命令对齐 ggml-org/llama.cpp#31251 review (arthw):
#   llama-bench -m <gguf> -fa 1 -p 2048 -n 128 -r 3 -ctk q8_0 -ctv f16
# 用法: ./bench.sh [all|8b|27b|sycl|sycl-tensor]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"
# shellcheck source=../config.sh
source "$SCRIPT_DIR/../config.sh"
# shellcheck source=../sycl_env.sh
source "$SCRIPT_DIR/../sycl_env.sh"

MASTER_BENCH=$SYCL_BUILD/bin/llama-bench
OPT_BENCH=$LLAMA_DIR/build-sycl-opt/bin/llama-bench

MODEL_8B=~/workspace/ai/models/unsloth/Qwen3-8B-GGUF/Qwen3-8B-IQ4_XS.gguf
MODEL_27B=$MODEL

bench_args() {
    BENCH_ARGS=(
        -m "$1" -fa "$2"
        -p 2048 -n 128 -r 3
        -ctk q8_0 -ctv f16
    )
}

# 8B 单卡: 全量 IQ4_XS 量化, 直接验证补丁改动的 MMVQ 路径
run_8b() {
    local bin=$1 label=$2 fa
    [[ -f "$MODEL_8B" ]] || { echo "错误: 找不到 8B 模型 $MODEL_8B" >&2; return 1; }
    [[ -x "$bin" ]] || { echo "错误: 找不到 $bin" >&2; return 1; }
    sycl_env "$bin"
    for fa in 0 1; do
        echo "$label 8b fa=$fa"
        bench_args "$MODEL_8B" "$fa"
        "$bin" --split-mode none -dev SYCL0 "${BENCH_ARGS[@]}"
    done
}

run_27b() {
    local bin=$1 label=$2 split=$3 fa fa_list="0 1"
    [[ -f "$MODEL_27B" ]] || { echo "错误: 找不到模型 $MODEL_27B" >&2; return 1; }
    [[ -x "$bin" ]] || { echo "错误: 找不到 $bin" >&2; return 1; }
    sycl_env "$bin"
    # tensor split 强制要求 flash_attn, 只跑 fa=1
    [[ "$split" == tensor ]] && fa_list="1"
    for fa in $fa_list; do
        echo "$label 27b $split fa=$fa"
        bench_args "$MODEL_27B" "$fa"
        "$bin" --split-mode "$split" "${BENCH_ARGS[@]}"
    done
}

bench_all_27b() {
    run_27b "$MASTER_BENCH" "master (build-sycl)" layer || true; sleep 8
    run_27b "$OPT_BENCH"    "补丁 (build-sycl-opt)"    layer || true; sleep 8
    run_27b "$MASTER_BENCH" "master (build-sycl)" tensor || true; sleep 8
    run_27b "$OPT_BENCH"    "补丁 (build-sycl-opt)"    tensor
}

case "${1:-all}" in
    all)
        run_8b "$MASTER_BENCH" "master (build-sycl)" || true; sleep 8
        run_8b "$OPT_BENCH"    "补丁 (build-sycl-opt)" || true; sleep 8
        bench_all_27b
        ;;
    8b)
        run_8b "$MASTER_BENCH" "master (build-sycl)" || true; sleep 8
        run_8b "$OPT_BENCH"    "补丁 (build-sycl-opt)"
        ;;
    27b)           bench_all_27b ;;
    sycl)          run_27b "$OPT_BENCH" "补丁 (build-sycl-opt)" layer ;;
    sycl-tensor)   run_27b "$OPT_BENCH" "补丁 (build-sycl-opt)" tensor ;;
    -h|--help|help) echo "用法: ./bench.sh [all|8b|27b|sycl|sycl-tensor]"; exit 0 ;;
    *)              usage "用法: ./bench.sh [all|8b|27b|sycl|sycl-tensor]"; err "未知参数 '$1'" ;;
esac

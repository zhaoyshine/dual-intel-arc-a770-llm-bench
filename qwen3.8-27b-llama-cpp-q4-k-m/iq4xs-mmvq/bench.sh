#!/bin/bash
# IQ4_XS MMVQ 补丁基准测试 — 双 Arc A770, master vs IQ4_XS 优化构建 (build-sycl-opt)
# 测试命令对齐 ggml-org/llama.cpp#31251 review (arthw):
#   llama-bench -m <gguf> -fa 1 -p 2048 -n 128 -r 3 -ctk q8_0 -ctv f16
# 用法: ./bench.sh [all|8b|27b|sycl|sycl-tensor]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../config.sh
source "$SCRIPT_DIR/../config.sh"
# shellcheck source=../sycl_env.sh
source "$SCRIPT_DIR/../sycl_env.sh"

MASTER_BENCH_BIN=~/workspace/ai/llama.cpp/build-sycl/bin/llama-bench
OPT_BENCH_BIN=~/workspace/ai/llama.cpp/build-sycl-opt/bin/llama-bench

MODEL_8B=~/workspace/ai/models/unsloth/Qwen3-8B-GGUF/Qwen3-8B-IQ4_XS.gguf
MODEL_27B=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf

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
        echo "== 8B IQ4_XS 单卡 fa=$fa ($label) =="
        bench_args "$MODEL_8B" "$fa"
        "$bin" --split-mode none -dev SYCL0 "${BENCH_ARGS[@]}"
    done
}

run_27b() {
    local bin=$1 label=$2 split=$3
    [[ -f "$MODEL_27B" ]] || { echo "错误: 找不到模型 $MODEL_27B" >&2; return 1; }
    [[ -x "$bin" ]] || { echo "错误: 找不到 $bin" >&2; return 1; }
    sycl_env "$bin"
    # tensor split 强制要求 flash_attn, 只跑 fa=1
    local fa_list="0 1"
    [[ "$split" == tensor ]] && fa_list="1"
    local fa
    for fa in $fa_list; do
        echo "== 27B Q4_K_M $split split fa=$fa ($label) =="
        bench_args "$MODEL_27B" "$fa"
        "$bin" --split-mode "$split" "${BENCH_ARGS[@]}"
    done
}

bench_all_27b() {
    run_27b "$MASTER_BENCH_BIN" "master" layer  || true; sleep 8
    run_27b "$OPT_BENCH_BIN"    "补丁"   layer  || true; sleep 8
    run_27b "$MASTER_BENCH_BIN" "master" tensor || true; sleep 8
    run_27b "$OPT_BENCH_BIN"    "补丁"   tensor
}

usage() {
    cat <<'EOF'
用法: ./bench.sh [all|8b|27b|sycl|sycl-tensor]

  all          8B 单卡 + 27B master/补丁 × layer/tensor × fa 0/1 (默认)
  8b           Qwen3-8B-IQ4_XS 单卡: master 与补丁构建对比
  27b          27B Q4_K_M: master/补丁 × layer/tensor × fa 0/1
  sycl         Q4_K_M layer split (补丁)
  sycl-tensor  Q4_K_M tensor split (补丁)
EOF
}

case "${1:-all}" in
    all)
        run_8b "$MASTER_BENCH_BIN" "master (build-sycl)"    || true; sleep 8
        run_8b "$OPT_BENCH_BIN"    "补丁 (build-sycl-opt)" || true; sleep 8
        bench_all_27b
        ;;
    8b)
        run_8b "$MASTER_BENCH_BIN" "master (build-sycl)"    || true; sleep 8
        run_8b "$OPT_BENCH_BIN"    "补丁 (build-sycl-opt)" || true
        ;;
    27b)             bench_all_27b ;;
    sycl)            run_27b "$OPT_BENCH_BIN" "补丁" layer ;;
    sycl-tensor)     run_27b "$OPT_BENCH_BIN" "补丁" tensor ;;
    -h|--help|help)  usage ;;
    *)       usage >&2; err "未知参数 '$1'" ;;
esac

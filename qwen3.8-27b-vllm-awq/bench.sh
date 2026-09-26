#!/bin/bash

set -euo pipefail

VENV=~/workspace/ai/vllm/.venv
MODEL=~/workspace/ai/models/cyankiwi/Qwen3.8-27B-AWQ-INT4

PP_TOKENS=1024
TG_TOKENS=32
MAX_LEN=4096

UTIL_TP=0.95
# pp 权重+激活峰值高, 更低时报 No available memory for the cache blocks
UTIL_PP=0.95
NUM_ITERS=3        # vllm bench latency 默认 30, tg32 单次迭代十几秒
WARMUP_ITERS=1
REST=8

err() { echo "错误: $*" >&2; exit 1; }

[[ -x "$VENV/bin/vllm" ]] || err "找不到 $VENV/bin/vllm (先编译: ./build.sh)"
[[ -f "$MODEL/config.json" ]] || err "找不到模型 $MODEL"

export CUDA_VISIBLE_DEVICES=       # 屏蔽 GT 1030, 否则 CUDA 探针与 XPU 同时激活
export ZES_ENABLE_SYSMAN=1         # 否则读不到显存余量
export VLLM_XPU_USE_SAMPLER_KERNEL=0    # 融合采样内核要 fp64, A770 没有
# XPU 只支持 spawn 多进程方式
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# 两张卡在同一台机器, oneCCL 走共享内存传输
export FI_PROVIDER=shm
export CCL_ATL_SHM=1
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_WORKER_COUNT=2
export VLLM_LOGGING_LEVEL=ERROR
export CCL_LOG_LEVEL=error
export PYTHONWARNINGS=ignore

RUN=0
TOTAL=4
ENGINE=()
LABEL=

set_config() {
    local util=$UTIL_TP
    case "$1" in
        tp) LABEL="张量并行 TP2"
                ENGINE=(--tensor-parallel-size 2) ;;
        pp) LABEL="流水线并行 PP2"; util=$UTIL_PP
                ENGINE=(--pipeline-parallel-size 2 --tensor-parallel-size 1) ;;
        *)  usage >&2; err "未知配置 '$1'" ;;
    esac
    ENGINE+=(--model "$MODEL" --max-model-len "$MAX_LEN" --dtype float16
             --attention-backend TRITON_ATTN
             --limit-mm-per-prompt '{"image":0,"video":0}'
             --gpu-memory-utilization "$util" --enforce-eager)
}

# 否则 Ctrl-C 只停 bench.sh, spawn 出的 worker 变孤儿继续占 20G 显存
BENCH_PID=""
cleanup() {
    if [[ -n "$BENCH_PID" ]]; then
        kill -TERM -- "-$BENCH_PID" 2>/dev/null || true
        sleep 3
        kill -KILL -- "-$BENCH_PID" 2>/dev/null || true
        BENCH_PID=""
    fi
}
trap cleanup EXIT INT TERM

run_case() {
    RUN=$((RUN + 1))
    echo "== [$RUN/$TOTAL] $LABEL | $1 (输入 $2 / 输出 $3 / batch 1, $NUM_ITERS 次迭代) =="
    setsid "$VENV/bin/vllm" bench latency "${ENGINE[@]}" \
            --input-len "$2" --output-len "$3" --batch-size 1 \
            --num-iters "$NUM_ITERS" --num-iters-warmup "$WARMUP_ITERS" &
    BENCH_PID=$!
    wait "$BENCH_PID" || true
    BENCH_PID=""
    sleep "$REST"
}

# pp 启动时 FLA GDN 内核有一批配置在 DG2 上编译失败, autotune 跳过要几分钟, 别当成卡死
run_config() {
    set_config "$1"
    run_case "pp$PP_TOKENS" "$PP_TOKENS" 1
    run_case "tg$TG_TOKENS" 1 "$TG_TOKENS"
}

ALL_CONFIGS=(tp pp)

show_devices() {
    [[ -x "$VENV/bin/python" ]] || err "找不到 $VENV/bin/python (先编译: ./build.sh)"
    "$VENV/bin/python" - <<'EOF'
import torch

for i in range(torch.xpu.device_count()):
    print(f"XPU{i}: {torch.xpu.get_device_properties(i).name}")
EOF
}

usage() {
    cat <<'EOF'
用法: ./bench.sh [配置]

  all      跑两个配置 (默认, 4 次测试, 结果见 README.md)
  tp       张量并行 TP2 (prefill 快)
  pp       流水线并行 PP2 (对应 llama.cpp 的 layer split, decode 快)
  devices  列出 vLLM 可见的 XPU 设备
EOF
}

case "${1:-all}" in
    all)            TOTAL=$(( ${#ALL_CONFIGS[@]} * 2 ))
                    for c in "${ALL_CONFIGS[@]}"; do run_config "$c"; done ;;
    tp|pp)          TOTAL=2; run_config "$1" ;;
    devices)        show_devices ;;
    -h|--help|help) usage ;;
    *)              usage >&2; err "未知参数 '$1'" ;;
esac

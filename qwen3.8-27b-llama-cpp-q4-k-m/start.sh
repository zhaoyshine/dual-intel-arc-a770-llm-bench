#!/bin/bash
# 启动 Qwen3.8 27B (Q4_K_M) — 2x Arc A770, SYCL 后端 (llama-server, OpenAI 兼容)
# 依赖: ~/workspace/ai/llama.cpp/build-sycl (编译: ./build.sh sycl)
# 思考档位 low 写死; MTP 投机解码常开

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"
# shellcheck source=sycl_env.sh
source "$SCRIPT_DIR/sycl_env.sh"

SYCL_SERVER=$SYCL_BUILD/bin/llama-server

[[ $# -eq 0 ]] || err "不接受参数 (思考档位已写死 low)"
[[ -x "$SYCL_SERVER" ]] || err "找不到 $SYCL_SERVER (先编译: ./build.sh sycl)"
[[ -f "$MODEL" ]] || err "找不到模型 $MODEL"
[[ -f "$DRAFT_MODEL" ]] || err "找不到 MTP draft 模型 $DRAFT_MODEL"
[[ -f "$MMPROJ" ]] || err "找不到视觉投影 $MMPROJ"

sycl_env "$SYCL_SERVER"

ARGS=(
    -m "$MODEL"
    --mmproj "$MMPROJ"
    --no-mmproj-offload
    --alias qwen3.8-27b
    --split-mode tensor
    -ngl "$NGL"
    -c "$CTX"
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
    --flash-attn "$FA"
    # 采样为官方思考模式推荐值; 非思考模式应改 temp 0.7, top-p 0.80, presence 1.5
    "${SAMPLING_ARGS[@]}"
    --n-predict 32768
    --reasoning-budget "$REASONING_BUDGET"
    --reasoning-effort low
    --batch-size "$BS"
    --ubatch-size "$UBS"
    --parallel 1
    -md "$DRAFT_MODEL"
    --spec-type draft-mtp
    --spec-draft-n-max "$DRAFT_N_MAX"
    --spec-draft-p-min 0.6
    --cache-type-k-draft "$KV_TYPE"
    --cache-type-v-draft "$KV_TYPE"
    -ngld "$NGL"
    -devd "$MTP_DEVICE"
    -ts "$MAIN_TS"
)

exec "$SYCL_SERVER" "${ARGS[@]}" --host 127.0.0.1 --port 8080

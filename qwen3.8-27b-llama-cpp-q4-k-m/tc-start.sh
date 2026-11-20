#!/bin/bash
# 启动 ThinkingCap Qwen3.8 27B (Q4_K_M) — 2x Arc A770, SYCL 后端 (llama-server, OpenAI 兼容)
# MTP 头内置于 GGUF, 无需独立 draft 模型; 视觉走独立 mmproj
# 思考档位 medium 写死

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"
# shellcheck source=sycl_env.sh
source "$SCRIPT_DIR/sycl_env.sh"

TC_MODEL=~/workspace/ai/models/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF/ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf
TC_MMPROJ=~/workspace/ai/models/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF/mmproj-ThinkingCap-Qwen3.8-27B-f16.gguf
SYCL_SERVER=$SYCL_BUILD/bin/llama-server

[[ $# -eq 0 ]] || err "不接受参数 (思考档位已写死 medium)"
[[ -x "$SYCL_SERVER" ]] || err "找不到 $SYCL_SERVER (先编译: ./build.sh sycl)"
[[ -f "$TC_MODEL" ]] || err "找不到模型 $TC_MODEL"
[[ -f "$TC_MMPROJ" ]] || err "找不到视觉投影 $TC_MMPROJ"

sycl_env "$SYCL_SERVER"

ARGS=(
    -m "$TC_MODEL"
    --mmproj "$TC_MMPROJ"
    --no-mmproj-offload
    --alias thinkingcap-27b
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
    --reasoning-effort medium
    --batch-size "$BS"
    --ubatch-size "$UBS"
    --parallel 1
    --spec-type draft-mtp
    --spec-draft-n-max "$DRAFT_N_MAX"
    --spec-draft-p-min 0.6
    --cache-type-k-draft "$KV_TYPE"
    --cache-type-v-draft "$KV_TYPE"
    -ngld "$NGL"
)

exec "$SYCL_SERVER" "${ARGS[@]}" --host 127.0.0.1 --port 8080

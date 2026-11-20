#!/bin/bash
# 启动 Coder390 Qwen3.8 27B (Q4LynnStyle, 内置 Q4 MTP) — 2x Arc A770, SYCL 后端 (llama-server, OpenAI 兼容)
# MTP 头内置于 GGUF, 无需独立 draft 模型; 视觉走独立 mmproj
# 思考档位 high 写死

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"
# shellcheck source=sycl_env.sh
source "$SCRIPT_DIR/sycl_env.sh"

CODER_MODEL=~/workspace/ai/models/Merkyor/Qwen3.8-27B-Coder390-EfficientThink-GGUF/Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-MTP.gguf
CODER_MMPROJ=~/workspace/ai/models/Merkyor/Qwen3.8-27B-Coder390-EfficientThink-GGUF/mmproj-Qwen3.8-27B-Q8_0.gguf
SYCL_SERVER=$SYCL_BUILD/bin/llama-server

[[ $# -eq 0 ]] || err "不接受参数 (思考档位已写死 high)"
[[ -x "$SYCL_SERVER" ]] || err "找不到 $SYCL_SERVER (先编译: ./build.sh sycl)"
[[ -f "$CODER_MODEL" ]] || err "找不到模型 $CODER_MODEL"
[[ -f "$CODER_MMPROJ" ]] || err "找不到视觉投影 $CODER_MMPROJ"

sycl_env "$SYCL_SERVER"

ARGS=(
    -m "$CODER_MODEL"
    --mmproj "$CODER_MMPROJ"
    --no-mmproj-offload
    --alias qwen3.8-27b-coder390
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

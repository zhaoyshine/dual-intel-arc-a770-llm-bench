#!/bin/bash
# 启动 Qwen3.8 27B (Q4_K_M) — 2x Arc A770, SYCL 后端 + IQ4_XS 优化 (llama-server, OpenAI 兼容)
# 依赖: ~/workspace/ai/llama.cpp/build-sycl-opt (编译: ./build.sh)
# 用法: ./start.sh [low|medium|high]   思考档位, 默认 low; MTP 投机解码常开

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../config.sh
source "$SCRIPT_DIR/../config.sh"
# shellcheck source=../sycl_env.sh
source "$SCRIPT_DIR/../sycl_env.sh"

SYCL_SERVER_BIN=~/workspace/ai/llama.cpp/build-sycl-opt/bin/llama-server

usage() {
    cat <<'EOF'
用法: ./start.sh [low|medium|high]   思考档位, 默认 low; MTP 投机解码常开

  low     简短思考 (默认)
  medium  中等思考
  high    最深思考 (模板内部映射 xhigh)
EOF
}

EFFORT="${1:-low}"
case "$EFFORT" in
    low|medium|high) readonly EFFORT ;;
    -h|--help|help) usage; exit 0 ;;
    *)              usage >&2; err "未知档位 '$EFFORT' (可选: low medium high)" ;;
esac

[[ -x "$SYCL_SERVER_BIN" ]] || err "找不到 $SYCL_SERVER_BIN (先编译: ./build.sh)"
[[ -f "$MODEL" ]] || err "找不到模型 $MODEL"
[[ -f "$DRAFT_MODEL" ]] || err "找不到 MTP draft 模型 $DRAFT_MODEL"

sycl_env "$SYCL_SERVER_BIN"

ARGS=(
    -m "$MODEL"
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
    --reasoning-effort "$EFFORT"
    --batch-size "$BS"
    --ubatch-size "$UBS"
    -md "$DRAFT_MODEL"
    --spec-type draft-mtp
    --spec-draft-n-max "$DRAFT_N_MAX"
    --cache-type-k-draft "$KV_TYPE"
    --cache-type-v-draft "$KV_TYPE"
    --parallel 1
    -ngld "$NGL"
    -devd "$MTP_DEVICE"
    -ts "$MAIN_TS"
)

echo "构建: build-sycl-opt | 档位: $EFFORT | MTP 投机解码 draft-n-max $DRAFT_N_MAX | ctx $CTX | MTP 卡 $MTP_DEVICE | 主模型 -ts $MAIN_TS"
echo "监听: http://127.0.0.1:8080 (OpenAI 兼容)"

exec "$SYCL_SERVER_BIN" "${ARGS[@]}" --host 127.0.0.1 --port 8080

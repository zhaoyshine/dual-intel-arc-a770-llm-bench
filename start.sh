#!/bin/bash
# 启动 Qwen3.8 27B (Q4_K_M) — 2x Arc A770, SYCL 后端 (llama-server, OpenAI 兼容)
# 依赖: ~/workspace/ai/llama.cpp/build-sycl (编译: ./build_sycl.sh)
# 用法: ./start.sh [mtp|base]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=sycl_env.sh
source "$SCRIPT_DIR/sycl_env.sh"

MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
DRAFT_MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
SYCL_BIN=~/workspace/ai/llama.cpp/build-sycl/bin/llama-server
HOST=127.0.0.1          # 局域网访问改 0.0.0.0
PORT=8080
NGL=999
KV_TYPE=f16
DRAFT_N_MAX=3
MTP_CTX=90000
BASE_CTX=122880

usage() {
    cat <<'EOF'
用法: ./start.sh [mtp|base]

  mtp   默认. 开 MTP 投机解码 (主模型 + draft 双 KV), ctx 90k, decode 快约 48%
  base  关投机 (单 KV), ctx 120k, 上下文更长, decode 慢约 1/3
EOF
}

MODE="${1:-mtp}"
case "$MODE" in
    mtp|base)       readonly MODE ;;
    -h|--help|help) usage; exit 0 ;;
    *)              usage >&2; err "未知参数 '$MODE'" ;;
esac

if [[ "$MODE" == mtp ]]; then
    CTX=$MTP_CTX
else
    CTX=$BASE_CTX
fi

[[ -x "$SYCL_BIN" ]] || err "找不到 $SYCL_BIN (先编译: ./build_sycl.sh)"
[[ -f "$MODEL" ]] || err "找不到模型 $MODEL"
if [[ "$MODE" == mtp ]]; then
    [[ -f "$DRAFT_MODEL" ]] || err "找不到 MTP draft 模型 $DRAFT_MODEL (或用 base 模式启动)"
fi

sycl_env "$SYCL_BIN"

ARGS=(
    -m "$MODEL"
    --split-mode layer
    -ngl "$NGL"
    -c "$CTX"
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
    --flash-attn on
    # 采样为官方「思考模式」推荐值; 非思考模式改 temp 0.7, top-p 0.80, presence 1.5
    --temp 1.0
    --top-k 20
    --top-p 0.95
    --min-p 0.0
    --presence-penalty 0.0
    --repeat-penalty 1.0
    --n-predict 32768
    --reasoning-budget 4096
    --ubatch-size 1024
)

if [[ "$MODE" == mtp ]]; then
    ARGS+=(
        -md "$DRAFT_MODEL"
        --spec-type draft-mtp
        --spec-draft-n-max "$DRAFT_N_MAX"
        -ngld 999
    )
    echo "模式: mtp (MTP 投机解码, draft-n-max $DRAFT_N_MAX) | ctx $CTX"
else
    echo "模式: base (无投机, 单 KV) | ctx $CTX"
fi
echo "监听: http://$HOST:$PORT (OpenAI 兼容)"

exec "$SYCL_BIN" "${ARGS[@]}" --host "$HOST" --port "$PORT"

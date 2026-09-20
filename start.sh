#!/bin/bash
# 启动 Qwen3.8 27B (Q5_K_M) — 2x Arc A770, SYCL 后端 (llama-server, OpenAI 兼容)
# 依赖: ~/workspace/ai/llama.cpp/build-sycl (编译: ./build_sycl.sh)
# 用法: ./start.sh [mtp|base]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=sycl_env.sh
source "$SCRIPT_DIR/sycl_env.sh"

MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
DRAFT_MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
SYCL_BIN=~/workspace/ai/llama.cpp/build-sycl/bin/llama-server
HOST=127.0.0.1
PORT=8080
# 最优参数
NGL=999
KV_TYPE=q8_0
FA=on
DRAFT_N_MAX=2
MTP_DEVICE=SYCL0
MAIN_TS=0.47,0.53
MTP_CTX=200000
BASE_CTX=200000
BS=1920
UBS=640
REASONING_EFFORT=medium

usage() {
    cat <<'EOF'
用法: ./start.sh [mtp|base]

  mtp   默认 MTP 投机解码 建议开启
  base  关闭 MTP 投机解码
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
    --flash-attn "$FA"
    # 采样为官方思考模式推荐值; 非思考模式应改 temp 0.7, top-p 0.80, presence 1.5
    --temp 1.0
    --top-k 20
    --top-p 0.95
    --min-p 0.0
    --presence-penalty 0.0
    --repeat-penalty 1.0
    --n-predict 32768
    --reasoning-budget 4096
    --reasoning-effort "$REASONING_EFFORT"
    --batch-size "$BS"
    --ubatch-size "$UBS"
)

if [[ "$MODE" == mtp ]]; then
    ARGS+=(
        -md "$DRAFT_MODEL"
        --spec-type draft-mtp
        --spec-draft-n-max "$DRAFT_N_MAX"
        --cache-type-k-draft "$KV_TYPE"
        --cache-type-v-draft "$KV_TYPE"
        -ngld 999
        -devd "$MTP_DEVICE"
        -ts "$MAIN_TS"
    )
    echo "模式: mtp (MTP 投机解码, draft-n-max $DRAFT_N_MAX) | ctx $CTX | MTP 卡 $MTP_DEVICE | 主模型 -ts $MAIN_TS"
else
    echo "模式: base (无投机, 单 KV) | ctx $CTX"
fi
echo "监听: http://$HOST:$PORT (OpenAI 兼容)"

exec "$SYCL_BIN" "${ARGS[@]}" --host "$HOST" --port "$PORT"

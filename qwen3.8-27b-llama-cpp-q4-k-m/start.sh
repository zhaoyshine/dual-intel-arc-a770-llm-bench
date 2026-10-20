#!/bin/bash
# 启动 Qwen3.8 27B (Q4_K_M) — 2x Arc A770, SYCL 后端 (llama-server, OpenAI 兼容)
# 依赖: ~/workspace/ai/llama.cpp/build-sycl (编译: ./build_sycl.sh)
# 用法: ./start.sh [low|medium|high]   思考档位, 默认 low; MTP 投机解码常开

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=sycl_env.sh
source "$SCRIPT_DIR/sycl_env.sh"

MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
DRAFT_MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
MODEL_ALIAS=qwen3.8-27b
SYCL_BIN=~/workspace/ai/llama.cpp/build-sycl/bin/llama-server
HOST=127.0.0.1
PORT=8080
# 最优参数
KV_TYPE=q8_0
FA=on
DRAFT_N_MAX=3
CTX=200000
BS=2048
UBS=1024

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

[[ -x "$SYCL_BIN" ]] || err "找不到 $SYCL_BIN (先编译: ./build_sycl.sh)"
[[ -f "$MODEL" ]] || err "找不到模型 $MODEL"
[[ -f "$DRAFT_MODEL" ]] || err "找不到 MTP draft 模型 $DRAFT_MODEL"

sycl_env "$SYCL_BIN"

ARGS=(
    -m "$MODEL"
    --alias "$MODEL_ALIAS"
    --split-mode tensor
    -ngl 999
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
    --reasoning-effort "$EFFORT"
    --batch-size "$BS"
    --ubatch-size "$UBS"
    -md "$DRAFT_MODEL"
    --spec-type draft-mtp
    --spec-draft-n-max "$DRAFT_N_MAX"
    --cache-type-k-draft "$KV_TYPE"
    --cache-type-v-draft "$KV_TYPE"
    --parallel 1
    -ngld 999
    -devd SYCL0
    -ts 0.48,0.52
)

echo "档位: $EFFORT | MTP 投机解码 draft-n-max $DRAFT_N_MAX | ctx $CTX | MTP 卡 SYCL0 | 主模型 -ts 0.48,0.52"
echo "监听: http://$HOST:$PORT (OpenAI 兼容)"

exec "$SYCL_BIN" "${ARGS[@]}" --host "$HOST" --port "$PORT"

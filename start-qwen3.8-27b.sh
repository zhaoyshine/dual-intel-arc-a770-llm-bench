#!/usr/bin/env bash
# 启动 Qwen3.8 27B (Q4_K_M) — 2x Arc A770, SYCL 后端 (llama-server, OpenAI 兼容)
# 依赖: ~/workspace/ai/llama.cpp/build-sycl (编译: ./rebuild-sycl.sh)
#
# 用法:
#   ./start-qwen3.8-27b.sh            # 默认 mtp: MTP 投机解码, ctx 90k (decode 快约 48%)
#   ./start-qwen3.8-27b.sh nomtp      # 关投机: 单 KV, ctx 120k (上下文更长)
#   ./start-qwen3.8-27b.sh --help

set -euo pipefail

usage() {
    cat <<'EOF'
用法: ./start-qwen3.8-27b.sh [mtp|nomtp]

  mtp    默认. 开 MTP 投机解码 (主模型 + draft 双 KV), ctx 90k, decode 快约 48%
  nomtp  关投机 (单 KV), ctx 120k, 上下文更长, decode 慢约 1/3
EOF
}

MODE="${1:-mtp}"
case "$MODE" in
    mtp|--mtp)              MODE="mtp" ;;
    nomtp|no-mtp|--no-mtp)  MODE="nomtp" ;;
    -h|--help|help)         usage; exit 0 ;;
    *) echo "错误: 未知参数 '$MODE'"; usage; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=sycl-env.sh
source "$SCRIPT_DIR/sycl-env.sh"

MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
SYCL_BIN="${SYCL_BIN:-$HOME/workspace/ai/llama.cpp/build-sycl/bin/llama-server}"

HOST="127.0.0.1"          # 监听地址; 局域网访问改 0.0.0.0
PORT="8080"               # OpenAI 兼容 API 端口
NGL=999                   # 全部层 offload 到 GPU
KV_TYPE="f16"             # KV 缓存不量化, 全精度 f16 (精度最高, 占显存多)

# ---- 上下文长度 (模型原生 262144) ------------------------------------------
# mtp: draft + 主模型双 KV, 90k 下默认 4 槽不超内存 (102400 会 OUT_OF_HOST_MEMORY)
# nomtp: 单 KV, 120k (f16 KV 约 13GB, 两卡 32GB 够放)
MTP_CTX=90000
NO_MTP_CTX=122880

# ---- MTP 投机解码 (参数与 bench-qwen3.8-27b.sh run_sycl_mtp 一致) ------------
MTP_MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
MTP_N_MAX=3               # draft 长度; 实测 3 最优 (见 README bench 结论)

if [ "$MODE" = "mtp" ]; then
    CTX="$MTP_CTX"
else
    CTX="$NO_MTP_CTX"
fi

# ---- 前置检查 -------------------------------------------------------------
[ -x "$SYCL_BIN" ] || { echo "错误: 找不到 $SYCL_BIN (先编译: ./rebuild-sycl.sh)"; exit 1; }
[ -f "$MODEL" ] || { echo "错误: 找不到模型 $MODEL"; exit 1; }
if [ "$MODE" = "mtp" ] && [ ! -f "$MTP_MODEL" ]; then
    echo "错误: 找不到 MTP draft 模型 $MTP_MODEL (或用 nomtp 模式启动)"
    exit 1
fi

sycl_env "$SYCL_BIN"

# ---- 启动 (llama-server, OpenAI 兼容) ------------------------------------
ARGS=(
    -m "$MODEL"
    --split-mode layer
    -ngl "$NGL"
    -c "$CTX"
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
    --flash-attn on
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

if [ "$MODE" = "mtp" ]; then
    ARGS+=(
        -md "$MTP_MODEL"
        --spec-type draft-mtp
        --spec-draft-n-max "$MTP_N_MAX"
        -ngld 999
    )
    echo "模式: mtp (MTP 投机解码, draft-n-max $MTP_N_MAX) | ctx $CTX"
else
    echo "模式: nomtp (无投机, 单 KV) | ctx $CTX"
fi
echo "监听: http://$HOST:$PORT (OpenAI 兼容)"

exec "$SYCL_BIN" "${ARGS[@]}" --host "$HOST" --port "$PORT"

# 参数说明:
#   --split-mode layer  逐层切分; 双 A770 同型号均分, 避免卡间行切分带宽瓶颈
#   -m/-ngl/-c/cache-type/flash-attn 基础配置与 bench 脚本一致
#   --temp/top-k/top-p/min-p  采样参数取自 Qwen3.8 27B 官方最佳实践「思考模式」;
#                       非思考模式改: temp 0.7, top_p 0.80, top_k 20, min_p 0.0, presence 1.5
#   --n-predict 32768 总输出上限; --reasoning-budget 4096 思考段上限
#   -md/--spec-type draft-mtp/--spec-draft-n-max 3: MTP 投机解码, 只加速 decode;
#   draft+主模型双 KV, ctx 降至 90k (见 bench run_sycl_mtp 同参数)

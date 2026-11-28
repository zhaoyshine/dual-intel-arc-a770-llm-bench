#!/bin/bash
# 启动 c0ven 服务 (Qwen3.8-Flash-Next GSQ-RCO IQ3_S, OpenAI + Anthropic 兼容)
# 依赖: ./build.sh 编出的 build/core/libengine.so
# 用法: ./start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"

err() { echo "错误: $*" >&2; exit 1; }

[[ -f "$BUILD_DIR/core/libengine.so" ]] || err "找不到 $BUILD_DIR/core/libengine.so (先编译: ./build.sh)"
[[ -f "$MODEL_FILE" ]] || err "找不到模型 $MODEL_FILE"
[[ -x "$VENV/bin/python" ]] || err "找不到 $VENV/bin/python (先编译: ./build.sh)"

mkdir -p "$ENGINE_CACHE_DIR"

# ADR-0002 的环境: 设备 0 固定为 PCI 地址低的卡; sysman 读每卡剩余显存
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZES_ENABLE_SYSMAN=1
export ENGINE_LIBRARY="$BUILD_DIR/core/libengine.so"
export ENGINE_CACHE_DIR
export PYTHONPATH="$C0VEN_DIR"
export ENGINE_NO_P2P=1

cd "$C0VEN_DIR"
exec "$VENV/bin/python" -m frontend \
    --model "$MODEL_FILE" \
    --slots "$SLOTS" \
    --context "$CONTEXT" \
    --host "$HOST" \
    --port "$PORT" \
    "${ENGINE_ARGS[@]}"

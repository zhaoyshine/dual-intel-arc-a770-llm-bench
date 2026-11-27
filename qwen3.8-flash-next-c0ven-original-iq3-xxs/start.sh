#!/bin/bash
# 启动官方 alchemist-support 分支的服务 (IQ3_S, 参数即 README Quick start 的 docker 命令; 镜像里 --host 0.0.0.0
# 由 -p 127.0.0.1:8080 收口, 本机直跑用 127.0.0.1). 环境变量照抄 Dockerfile 的 ENV 行.
# A770 的 Xe-HPG 路径由 core 的 is_hpg() 自动启用, 不需要任何覆盖; vision 有 HPG 版, 不用 --no-vision
# --log-level DEBUG: frontend/engine/uvicorn 三个 logger 一起调到最细
# --metrics-log 20: frontend.metrics 每 20 秒一行 /metrics 计数器与派生比率的 JSON, 外加每个请求结束时的记录
# 用法: ./start.sh  (指标日志想留档: ./start.sh 2>&1 | tee metrics.log)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

C0VEN_DIR="${C0VEN_DIR:-$SCRIPT_DIR/c0ven}"
BUILD_DIR="$SCRIPT_DIR/build"
VENV="$C0VEN_DIR/.venv"
MODEL_FILE="${MODEL_FILE:-$HOME/workspace/ai/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
ENGINE_CACHE_DIR="${ENGINE_CACHE_DIR:-$SCRIPT_DIR/cache}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SLOTS="${SLOTS:-2}"
CONTEXT="${CONTEXT:-262144}"

err() { echo "错误: $*" >&2; exit 1; }

[[ -f "$BUILD_DIR/core/libengine.so" ]] || err "找不到 $BUILD_DIR/core/libengine.so (先编译: ./build.sh)"
[[ -f "$MODEL_FILE" ]] || err "找不到模型 $MODEL_FILE"
[[ -x "$VENV/bin/python" ]] || err "找不到 $VENV/bin/python (先编译: ./build.sh)"

mkdir -p "$ENGINE_CACHE_DIR"

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZES_ENABLE_SYSMAN=1
export ENGINE_LIBRARY="$BUILD_DIR/core/libengine.so"
export ENGINE_CACHE_DIR
export PYTHONDONTWRITEBYTECODE=1
export ENGINE_NO_P2P=1

cd "$C0VEN_DIR"
exec "$VENV/bin/python" -m frontend \
    --model "$MODEL_FILE" \
    --slots "$SLOTS" \
    --context "$CONTEXT" \
    --host "$HOST" \
    --port "$PORT" \
    --log-level DEBUG \
    --metrics-log 20

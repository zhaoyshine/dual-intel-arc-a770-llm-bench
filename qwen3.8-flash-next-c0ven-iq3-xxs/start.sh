#!/bin/bash
# 启动 c0ven 服务 (Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, OpenAI + Anthropic 兼容), 参数同 README 的双 B60 启动命令
# 依赖: ./build.sh 编出的 build/core/libengine.so
# 用法: ./start.sh [额外参数...]

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
[[ -n "$ZE_AFFINITY_MASK" ]] && export ZE_AFFINITY_MASK

# Every line the service and the engine print goes to one file, flushed line by line, so a hard reset leaves the last
# one on disk. SERVICE_LOG overrides it; test.sh points it at the run's own log.
export PYTHONUNBUFFERED=1
SERVICE_LOG="${SERVICE_LOG:-$SCRIPT_DIR/logs/service-live.log}"
mkdir -p "$(dirname "$SERVICE_LOG")"
# Line by line with fsync: a hard reset leaves every line the service printed on disk, the last one included.
LOGGER="$SCRIPT_DIR/logs/log-sync.py"
cat > "$LOGGER" <<'LOGEOF'
import os, sys
f = os.fdopen(os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_SYNC), "w", buffering=1)
for line in sys.stdin:
    f.write(line)
    f.flush()
    os.fsync(f.fileno())
LOGEOF
exec > >(exec python3 -u "$LOGGER" "$SERVICE_LOG") 2>&1

export ENGINE_LIBRARY="$BUILD_DIR/core/libengine.so"
export ENGINE_CACHE_DIR
export PYTHONPATH="$C0VEN_DIR"

cd "$C0VEN_DIR"
exec "$VENV/bin/python" -m frontend \
    --model "$MODEL_FILE" \
    --slots "$SLOTS" \
    --context "$CONTEXT" \
    --host "$HOST" \
    --port "$PORT" \
    "${ENGINE_ARGS[@]}" \
    "$@"

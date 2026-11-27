#!/bin/bash
# 按官方 DEVELOPING.md 的非镜像构建方式编译 alchemist-support 分支 (作者官方支持的 A770):
# 所有 cmake 选项用默认值 (AOT 默认含 acm-g10/11/12), ggml 用 ADR-0012 钉住的 llama.cpp commit,
# 通过作者留的 ENGINE_GGML_DIR 指到本目录解包的树 (Dockerfile 里同样的 fetch 命令)
# 用法: ./build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

C0VEN_DIR="${C0VEN_DIR:-$SCRIPT_DIR/c0ven}"
# ADR-0012 固定的 ggml commit, 与 Dockerfile 里的 /opt/llama.cpp 一致
LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
LLAMA_URL=https://github.com/ggml-org/llama.cpp
GGML_DIR="$SCRIPT_DIR/opt/llama.cpp"
BUILD_DIR="$SCRIPT_DIR/build"
VENV="$C0VEN_DIR/.venv"
MODEL_FILE="${MODEL_FILE:-$HOME/workspace/ai/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"

err() { echo "错误: $*" >&2; exit 1; }

[[ -f "$C0VEN_DIR/CMakeLists.txt" ]] || err "找不到 c0ven 源码 $C0VEN_DIR"
[[ "$(git -C "$C0VEN_DIR" branch --show-current)" == alchemist-support ]] ||
    err "$C0VEN_DIR 不在 alchemist-support 分支"
pkg-config --exists level-zero || err "缺 level-zero 开发包 (Dockerfile 钉 libze-dev 1.32.0)"
command -v ocloc >/dev/null || err "缺 ocloc (Dockerfile 钉 intel-ocloc 26.40)"
command -v cmake >/dev/null || err "缺 cmake"
command -v ninja >/dev/null || err "缺 ninja"
uv="$(command -v uv || true)"
[[ -n "$uv" && "$("$uv" --version)" == "uv 0.12.22"* ]] || err "需要 uv 0.12.22 (ADR-0002 钉的版本)"
py="$(command -v python3.14 || true)"
[[ -n "$py" ]] || err "缺 python3.14 (.python-version 钉 3.14.4, requires-python <3.15)"
[[ -f "$MODEL_FILE" ]] || err "找不到模型 $MODEL_FILE"

# ggml 取自 llama.cpp 的固定 commit, 解包到 opt/llama.cpp (与镜像的 /opt/llama.cpp 对齐)
if [[ "$(cat "$GGML_DIR/.c0ven-commit" 2>/dev/null)" != "$LLAMA_COMMIT" ]]; then
    echo "== 拉取 llama.cpp@${LLAMA_COMMIT:0:8} 到 $GGML_DIR =="
    rm -rf "$GGML_DIR"
    git init -q "$GGML_DIR"
    git -C "$GGML_DIR" fetch -q --depth 1 "$LLAMA_URL" "$LLAMA_COMMIT"
    git -C "$GGML_DIR" checkout -q FETCH_HEAD
    echo "$LLAMA_COMMIT" > "$GGML_DIR/.c0ven-commit"
fi

# 与镜像同样的 uv 环境 (ENV 行); .venv 建在工程目录, 即镜像的 /opt/c0ven/.venv
echo "== uv sync -> $VENV =="
UV_PYTHON="$py" UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    "$uv" sync --project "$C0VEN_DIR" --locked --no-dev

# 不传任何设备/等级选项: AOT 默认 bmg-g21;bmg-g31;acm-g10;acm-g11;acm-g12 (ADR-0082),
# CPU 等级默认 9 档, libengine.so 运行时按本机 CPU 特性选档
echo "== configure $BUILD_DIR =="
cmake -S "$C0VEN_DIR" -B "$BUILD_DIR" -G Ninja -DENGINE_GGML_DIR="$GGML_DIR"

echo "== build c0ven =="
cmake --build "$BUILD_DIR"

echo "完成: $BUILD_DIR/core/libengine.so"
echo "启动: ./start.sh"

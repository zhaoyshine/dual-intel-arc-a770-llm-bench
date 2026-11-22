#!/bin/bash
# 在本机编译 c0ven (C++ 核心 + ocloc 预编译 kernel + Python 前端), 复刻 Dockerfile 的 serve 阶段
# 用法: ./build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"

err() { echo "错误: $*" >&2; exit 1; }

case "${1:-}" in
    "")        ;;
    -h|--help) echo "用法: ./build.sh"; exit 0 ;;
    *)         err "用法: ./build.sh" ;;
esac

[[ -f "$C0VEN_DIR/CMakeLists.txt" ]] || err "找不到 c0ven 源码 $C0VEN_DIR"
[[ -f /usr/include/level_zero/ze_api.h ]] ||
    err "缺 ze_api.h, 安装: sudo dnf install level-zero-devel"
command -v ocloc >/dev/null || err "缺 ocloc, 安装: sudo dnf install intel-ocloc"
[[ -f "$MODEL_FILE" ]] || err "找不到模型 $MODEL_FILE"

# ggml 取自 llama.cpp 的固定 commit, 解包到 opt/llama.cpp (与官方镜像的 /opt/llama.cpp 对齐)
ggml_dir="$LLAMA_CPP_DIR"
if [[ "$(cat "$ggml_dir/.c0ven-commit" 2>/dev/null)" != "$LLAMA_COMMIT" ]]; then
    git -C "$LLAMA_REPO" cat-file -e "$LLAMA_COMMIT^{commit}" 2>/dev/null ||
        err "$LLAMA_REPO 里没有 commit $LLAMA_COMMIT (先 git fetch)"
    echo "== 解包 llama.cpp@${LLAMA_COMMIT:0:8} 到 $ggml_dir =="
    rm -rf "$ggml_dir"
    mkdir -p "$ggml_dir"
    git -C "$LLAMA_REPO" archive "$LLAMA_COMMIT" | tar -x -C "$ggml_dir"
    echo "$LLAMA_COMMIT" > "$ggml_dir/.c0ven-commit"
fi

# pyproject 要求 uv 版本精确匹配, 系统 uv 版本不同时装一份到本目录
uv="$(command -v uv || true)"
if [[ -z "$uv" || "$("$uv" --version)" != "uv 0.12.22"* ]]; then
    if [[ ! -x "$UV_DIR/bin/uv" ]]; then
        echo "== 安装 uv 0.12.22 到 $UV_DIR =="
        python3 -m venv "$UV_DIR"
        "$UV_DIR/bin/pip" install -q uv==0.12.22
    fi
    uv="$UV_DIR/bin/uv"
fi

# .python-version 钉 3.14.4, 本机只有 3.14.8, 直接指定解释器, 不让 uv 下载 Python
py="$(command -v python3.14 || true)"
[[ -n "$py" ]] || err "缺 python3.14"
echo "== uv sync -> $VENV =="
UV_PROJECT_ENVIRONMENT="$VENV" UV_PYTHON="$py" UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    "$uv" sync --project "$C0VEN_DIR" --locked --no-dev

# 只编译本机 CPU 支持的最高指令集等级, 全等级会重复编译 9 份 ggml
flags=" $(grep -m1 '^flags' /proc/cpuinfo) "
has() { [[ "$flags" == *" $1 "* ]]; }
if [[ -n "${ENGINE_CPU_LEVELS:-}" ]]; then
    levels="$ENGINE_CPU_LEVELS"
elif has avx512f && has avx512_vbmi && has avx512_vnni && has avx512_bf16; then
    levels=zen4
elif has avx512f && has avx512_vnni; then
    levels=cascadelake
elif has avx512f; then
    levels=skylakex
elif has avx_vnni; then
    levels=alderlake
elif has avx2 && has bmi2; then
    levels=haswell
elif has avx; then
    levels=sandybridge
else
    levels=x64
fi
echo "== CPU 等级: $levels =="

echo "== configure $BUILD_DIR =="
cmake -S "$C0VEN_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DENGINE_GGML_DIR="$ggml_dir" \
    -DENGINE_CPU_LEVELS="$levels" \
    -DENGINE_AOT_DEVICES="$AOT_DEVICES" \
    -DENGINE_DG2_DEVICES="$DG2_DEVICES" \
    -DENGINE_KERNEL_SPIRV_DEVICE="$SPIRV_DEVICE"

echo "== build c0ven =="
cmake --build "$BUILD_DIR"

echo "完成: $BUILD_DIR/core/libengine.so"
echo "启动: ./start.sh"

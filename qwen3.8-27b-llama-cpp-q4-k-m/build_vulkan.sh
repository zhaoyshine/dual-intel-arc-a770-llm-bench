#!/bin/bash
# 编译 llama.cpp Vulkan 后端 (双 Arc A770 对照基准用)
# 依赖: vulkan-headers glslc spirv-headers-devel vulkan-loader-devel
# 用法: ./build_vulkan.sh

set -euo pipefail

LLAMA_DIR=~/workspace/ai/llama.cpp
BUILD_DIR=build-vulkan
NJOBS=24
TARGETS=(llama-bench)

err() { echo "错误: $*" >&2; exit 1; }

[[ -d "$LLAMA_DIR" ]] || err "找不到 llama.cpp 源码 $LLAMA_DIR"
# 缺 glslc 时 cmake 只警告并跳过着色器编译, 编出的二进制无 GPU 内核
command -v glslc >/dev/null 2>&1 ||
    err "缺 glslc, 安装: sudo dnf install glslc vulkan-headers spirv-headers-devel vulkan-loader-devel"

cd "$LLAMA_DIR"

git checkout -- .
git pull

echo "== configure $BUILD_DIR =="
cmake -B "$BUILD_DIR" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release

echo "== build: ${TARGETS[*]} (-j $NJOBS) =="
cmake --build "$BUILD_DIR" --config Release -j "$NJOBS" --target "${TARGETS[@]}"

echo "完成: $LLAMA_DIR/$BUILD_DIR/bin/"

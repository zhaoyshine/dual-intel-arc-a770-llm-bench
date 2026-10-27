#!/bin/bash
# 导出 llama.cpp HEAD + 打本目录补丁, 构建到 ~/workspace/ai/llama.cpp/build-sycl-opt
# 上游工作树不动 (git archive 到 ~/workspace/ai/llama.cpp-opt-src 后打补丁)
# 用法: ./build.sh [--clean]    --clean 清掉源码副本与构建目录, 全量重建

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR=~/workspace/ai/llama.cpp
SRC_DIR=~/workspace/ai/llama.cpp-opt-src
BUILD_DIR="$LLAMA_DIR/build-sycl-opt"
PATCHES=("$DIR"/patches/*.patch)
TARGETS=(llama-bench llama-server llama-cli)

err() { echo "错误: $*" >&2; exit 1; }

[[ -d "$LLAMA_DIR" ]] || err "找不到 llama.cpp 源码 $LLAMA_DIR"
[[ -f "${PATCHES[0]}" ]] || err "找不到补丁 ${PATCHES[0]}"

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf "$SRC_DIR" "$BUILD_DIR"
fi

# llama.cpp HEAD 或补丁内容变化时重新导出, 否则沿用已有副本
stamp=$({ git -C "$LLAMA_DIR" rev-parse HEAD; cat "${PATCHES[@]}"; } | md5sum | cut -d' ' -f1)
if [[ ! -f "$SRC_DIR/.patch-stamp" || "$(cat "$SRC_DIR/.patch-stamp")" != "$stamp" ]]; then
    echo "== 导出源码 + 打补丁 -> $SRC_DIR =="
    rm -rf "$SRC_DIR"
    mkdir -p "$SRC_DIR"
    git -C "$LLAMA_DIR" archive HEAD | tar -x -C "$SRC_DIR"
    for p in "${PATCHES[@]}"; do
        echo "-- $(basename "$p")"
        patch -d "$SRC_DIR" -p1 --no-backup-if-mismatch < "$p"
    done
    # 导出树自己的 git 仓库: cmake 会在这里跑 git describe, 没有仓库就报 "不是 Git 仓库" 且 build 号变成 unknown
    git -C "$SRC_DIR" init -q
    git -C "$SRC_DIR" add -A
    git -C "$SRC_DIR" -c user.name=build -c user.email=build@local \
        commit -qm "llama.cpp $(git -C "$LLAMA_DIR" rev-parse --short=8 HEAD) + IQ4_XS MMVQ" || true
    echo "$stamp" > "$SRC_DIR/.patch-stamp"
fi

# 缺 ze_api.h 编出的二进制启动即 GGML_ABORT
[[ -f /usr/include/level_zero/ze_api.h ]] ||
    err "缺 ze_api.h, 安装: sudo dnf install oneapi-level-zero-devel (Ubuntu: libze-dev)"

# setvars 的探测命令会失败, source 期间需临时关闭 -e/-u
if ! command -v icpx >/dev/null 2>&1; then
    setvars=/opt/intel/oneapi/setvars.sh
    [[ -f "$setvars" ]] || err "找不到 $setvars (需先装 intel-oneapi-toolkit)"
    echo "== source $setvars =="
    set +eu
    # shellcheck disable=SC1090
    source "$setvars"
    set -euo pipefail
fi

echo "== configure $BUILD_DIR =="
# 与 ../build.sh 的 sycl 配置保持一致, 只多一个补丁源码目录, 保证结果可对比
cmake -B "$BUILD_DIR" -S "$SRC_DIR" \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_F16=ON \
    -DGGML_SYCL_DEVICE_ARCH=acm_g10 \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx

echo "== build: ${TARGETS[*]} (-j 12) =="
cmake --build "$BUILD_DIR" --config Release -j 12 --target "${TARGETS[@]}"

echo "完成: $BUILD_DIR/bin/"

#!/bin/bash
# 编译 llama.cpp SYCL 后端 (双 Arc A770 基准/服务用)
# 依赖: oneAPI toolkit + oneapi-level-zero-devel
# 用法: ./build_sycl.sh

set -euo pipefail

LLAMA_DIR=~/workspace/ai/llama.cpp
BUILD_DIR=build-sycl
NJOBS=24
TARGETS=(llama-bench llama-server llama-cli)

err() { echo "错误: $*" >&2; exit 1; }

[[ -d "$LLAMA_DIR" ]] || err "找不到 llama.cpp 源码 $LLAMA_DIR"
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

cd "$LLAMA_DIR"

git checkout -- .
git pull

# cmake --build 的 auto-reconfigure 不重跑 find_path, 装新依赖头后须显式 configure
echo "== configure $BUILD_DIR =="
cmake -B "$BUILD_DIR" -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_SYCL_F16=ON \
    -DGGML_SYCL_DEVICE_ARCH=acm_g10

echo "== build: ${TARGETS[*]} (-j $NJOBS) =="
cmake --build "$BUILD_DIR" --config Release -j "$NJOBS" --target "${TARGETS[@]}"

echo "完成: $LLAMA_DIR/$BUILD_DIR/bin/"

#!/bin/bash
# 编译 llama.cpp (双 Arc A770 基准/服务用)
# 用法: ./build.sh sycl|vulkan

set -euo pipefail

LLAMA_DIR=~/workspace/ai/llama.cpp
BACKEND="${1:-}"

err() { echo "错误: $*" >&2; exit 1; }

case "$BACKEND" in
    sycl)
        BUILD_DIR=build-sycl
        TARGETS=(llama-bench llama-server llama-cli)
        CMAKE_ARGS=(
            -DGGML_SYCL=ON
            -DGGML_SYCL_F16=ON
            -DGGML_SYCL_DEVICE_ARCH=acm_g10
            -DCMAKE_C_COMPILER=icx
            -DCMAKE_CXX_COMPILER=icpx
        )

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
        ;;
    vulkan)
        BUILD_DIR=build-vulkan
        TARGETS=(llama-bench)
        CMAKE_ARGS=(-DGGML_VULKAN=ON)

        # 缺 glslc 时 cmake 只警告并跳过着色器编译, 编出的二进制无 GPU 内核
        command -v glslc >/dev/null 2>&1 ||
            err "缺 glslc, 安装: sudo dnf install glslc vulkan-headers spirv-headers-devel vulkan-loader-devel"
        ;;
    -h|--help)
        echo "用法: ./build.sh sycl|vulkan"
        exit 0
        ;;
    *)
        err "用法: ./build.sh sycl|vulkan"
        ;;
esac

[[ -d "$LLAMA_DIR" ]] || err "找不到 llama.cpp 源码 $LLAMA_DIR"

cd "$LLAMA_DIR"

# cmake --build 的 auto-reconfigure 不重跑 find_path, 装新依赖头后须显式 configure
echo "== configure $BUILD_DIR =="
cmake -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"

echo "== build: ${TARGETS[*]} (-j 12) =="
cmake --build "$BUILD_DIR" --config Release -j 12 --target "${TARGETS[@]}"

echo "完成: $LLAMA_DIR/$BUILD_DIR/bin/"

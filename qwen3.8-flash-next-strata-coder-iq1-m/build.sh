#!/bin/bash
# 编译 Strata SYCL 引擎 (双 Arc A770 本地跑)
# A770 是 Alchemist (DG2): 不设 STRATA_SYCL_AOT, GPU 代码为 SPIR-V, 首次启动 JIT 约 1 分钟
# 用法: ./build.sh

set -euo pipefail

STRATA_DIR=~/workspace/ai/Strata
BUILD_DIR="$STRATA_DIR/build-sycl"

err() { echo "错误: $*" >&2; exit 1; }

case "${1:-}" in
    "")        ;;
    -h|--help) echo "用法: ./build.sh"; exit 0 ;;
    *)         err "用法: ./build.sh" ;;
esac

[[ -d "$STRATA_DIR/sycl" ]] || err "找不到 Strata 源码 $STRATA_DIR"
[[ -f /usr/include/level_zero/ze_api.h ]] ||
    err "缺 ze_api.h, 安装: sudo dnf install oneapi-level-zero-devel"

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

# 旧构建缓存指向 sycl/ 子项目, 与根目录配置不兼容
cache="$BUILD_DIR/CMakeCache.txt"
if [[ -f "$cache" ]] && ! grep -qx "CMAKE_HOME_DIRECTORY:INTERNAL=$STRATA_DIR" "$cache"; then
    echo "== 源码目录变了, 清空 $BUILD_DIR =="
    rm -rf "$BUILD_DIR"
fi

# 不传 -DSTRATA_SYCL_AOT: JIT 构建默认带 -ze-opt-greater-than-4GB-buffer-required,
# 缺了它在 A770 上 4 GiB 之后的专家权重会读成 NaN
echo "== configure $BUILD_DIR =="
cmake -S "$STRATA_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DSTRATA_ENABLE_SYCL=ON

echo "== build strata =="
cmake --build "$BUILD_DIR" --target strata

echo "完成: $BUILD_DIR/sycl/strata"

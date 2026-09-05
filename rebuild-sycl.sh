#!/usr/bin/env bash
# 重新编译 llama.cpp SYCL 后端 (双 Arc A770 基准/服务用)
# 依赖: oneAPI toolkit + oneapi-level-zero-devel (ze_api.h 规范头)
#
# 背景: llama.cpp master a32af33 (#27968) 起内存查询默认走 Level Zero sysman,
#   缺规范头时 GGML_SYCL_SUPPORT_LEVEL_ZERO_API 未定义 → llama-bench 启动即 GGML_ABORT。
#   装 oneapi-level-zero-devel 后宏生效。
#
# 用法: ./rebuild-sycl.sh
#   env: LLAMA_DIR (默认 ~/workspace/ai/llama.cpp), BUILD_DIR (默认 build-sycl), NJOBS (默认 24)

set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/workspace/ai/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-build-sycl}"
NJOBS="${NJOBS:-24}"
TARGETS=(llama-bench llama-server)

[ -d "$LLAMA_DIR" ] || { echo "错误: 找不到 llama.cpp 源码 $LLAMA_DIR"; exit 1; }

# ---- oneAPI 环境 (icx/icpx 不在 PATH 时 source setvars) --------------------
if ! command -v icpx >/dev/null 2>&1; then
    SETVARS=/opt/intel/oneapi/setvars.sh
    [ -f "$SETVARS" ] || { echo "错误: 找不到 $SETVARS (需先装 intel-oneapi-toolkit)"; exit 1; }
    echo "== sourcing $SETVARS =="
    set +e +u                    # setvars.sh 引用未绑定变量, nounset 下会炸
    # shellcheck disable=SC1090
    source "$SETVARS"            # 也不能用 -e: setvars 某些探测命令"失败"是正常的
    set -euo pipefail
fi

# ---- 前置检查: Level Zero 规范头 (缺了编出的二进制启动即崩) ---------------
if ! rpm -q oneapi-level-zero-devel >/dev/null 2>&1; then
    if [ ! -f /usr/include/level_zero/ze_api.h ]; then
        echo "警告: 没找到 oneapi-level-zero-devel / ze_api.h" >&2
        echo "  llama.cpp 新内存查询代码需要 Level Zero 规范头, 否则 llama-bench 启动即崩" >&2
        echo "  安装: sudo dnf install oneapi-level-zero-devel   (Ubuntu 是 libze-dev)" >&2
        exit 1
    fi
fi

cd "$LLAMA_DIR"

echo "== configure $BUILD_DIR =="
cmake -B "$BUILD_DIR" -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_SYCL_F16=ON
# 注: 装完新依赖头必须重新 configure, cmake --build 的 auto-reconfigure 不重跑 find_path

echo "== build: ${TARGETS[*]} (-j $NJOBS) =="
cmake --build "$BUILD_DIR" --config Release -j "$NJOBS" --target "${TARGETS[@]}"

echo
for t in "${TARGETS[@]}"; do
    echo "完成: $LLAMA_DIR/$BUILD_DIR/bin/$t"
done
echo "快速自检 (应列出两个 A770, 不再 abort):"
echo "  ONEAPI_DEVICE_SELECTOR=level_zero:0;level_zero:1 $LLAMA_DIR/$BUILD_DIR/bin/llama-bench --list-devices"

#!/usr/bin/env bash
# oneAPI SYCL 运行环境（被 bench/start 脚本 source）
# 设备选择用 env (ONEAPI_DEVICE_SELECTOR) 不用 --device; 双 A770 → level_zero:0;1
# 库路径手动指定, 不 source setvars.sh:
#   libsycl/libsvml/libirng/libimf/libintlc/libiomp5 → compiler/latest/lib
#   libdnnl (GGML_SYCL_DNN=ON 默认)                 → dnnl/latest/lib
#   libumf (oneAPI 2026 unified runtime 依赖)       → umf/latest/lib
#   libhwloc.so.15 (level_zero adapter 依赖)        → oneapi/<ver>/lib, 动态查找
#     缺依赖时 adapter 加载失败 → "No device of requested type available"

sycl_env() {
    local bin="$1"
    export ONEAPI_DEVICE_SELECTOR="${SYCL_DEVICES:-level_zero:0;level_zero:1}"
    export ZES_ENABLE_SYSMAN=1    # layer split 查显存余量
    local hwloc_lib="$(find /opt/intel/oneapi -maxdepth 3 -name 'libhwloc.so.15' 2>/dev/null | head -1)"
    local hwloc_dir=""
    [ -n "$hwloc_lib" ] && hwloc_dir="$(dirname "$hwloc_lib")"   # 找不到时跳过, 不把 "." 加进库路径
    for d in /opt/intel/oneapi/compiler/latest/lib /opt/intel/oneapi/dnnl/latest/lib /opt/intel/oneapi/umf/latest/lib $hwloc_dir; do
        [ -d "$d" ] || { echo "错误: 找不到 oneAPI 库目录 $d" >&2; exit 1; }
        export LD_LIBRARY_PATH="${bin%/*}:$d:${LD_LIBRARY_PATH:-}"
    done
}

#!/bin/bash
# SYCL 运行环境 (oneAPI 库路径 + 设备选择), 被 bench.sh / start.sh source
# 手动拼 LD_LIBRARY_PATH 而不 source setvars.sh: 只需 compiler/dnnl/umf 三个 lib 目录 + 动态查到的 hwloc
# 缺依赖时 level_zero adapter 加载失败, 报 "No device of requested type available"

err() { echo "错误: $*" >&2; exit 1; }

sycl_env() {
    local dir libs=${1%/*} hwloc_lib
    for dir in /opt/intel/oneapi/{compiler,dnnl,umf}/latest/lib; do
        [[ -d "$dir" ]] || err "找不到 oneAPI 库目录 $dir"
        libs+=":$dir"
    done
    hwloc_lib=$(find /opt/intel/oneapi -maxdepth 3 -name libhwloc.so.15 -print -quit 2>/dev/null) || true
    [[ -n "$hwloc_lib" ]] || err "找不到 libhwloc.so.15"
    libs+=":$(dirname "$hwloc_lib")"

    export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"
    export ZES_ENABLE_SYSMAN=1    # 查显存余量
    export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

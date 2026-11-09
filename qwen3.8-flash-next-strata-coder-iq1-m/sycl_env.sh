#!/bin/bash
# 给当前 shell 注入 oneAPI 运行时库路径和 SYCL 环境变量
# 缺了 level_zero adapter 加载失败; 用法: . ./sycl_env.sh

libs=""
for dir in /opt/intel/oneapi/{compiler,mkl,dnnl,umf}/latest/lib; do
    [[ -d "$dir" ]] || { echo "错误: 找不到 oneAPI 库目录 $dir" >&2; return 1; }
    libs+=":$dir"
done
hwloc_lib=$(find /opt/intel/oneapi -maxdepth 3 -name 'libhwloc.so.15' -print -quit 2>/dev/null) || true
[[ -n "$hwloc_lib" ]] || { echo "错误: 找不到 libhwloc.so.15" >&2; return 1; }
libs+=":$(dirname "$hwloc_lib")"
export LD_LIBRARY_PATH="${libs#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export SYCL_UR_USE_LEVEL_ZERO_V2=1
export ZES_ENABLE_SYSMAN=1

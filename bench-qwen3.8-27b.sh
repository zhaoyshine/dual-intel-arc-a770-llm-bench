#!/usr/bin/env bash
# Qwen3.8 27B (Q4_K_M) 基准测试 — 双 Arc A770 四种配置
# 测试结果见同目录 README.md
#
# 用法:
#   ./bench-qwen3.8-27b.sh               # 默认串行跑全部 4 种
#   ./bench-qwen3.8-27b.sh vulkan-official   # 只跑指定一种
#   ./bench-qwen3.8-27b.sh vulkan        #   自编译 Vulkan (build-vulkan)
#   ./bench-qwen3.8-27b.sh sycl          #   SYCL layer split
#   ./bench-qwen3.8-27b.sh sycl-tensor   #   SYCL tensor split
#
# 四种配置:
#   vulkan-official  官方预编译单文件版 (b10217), 设备 --device Vulkan1,Vulkan2
#   vulkan           同源码自编译 (build-vulkan), 与 SYCL 公平对比
#   sycl             同源码自编译 (build-sycl, FP16), ONEAPI_DEVICE_SELECTOR 选双卡
#   sycl-tensor      同上, 但 --split-mode tensor

set -euo pipefail

MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf

# ---- 运行配置 (与 start-qwen3.8-27b.sh 一致) --------------------------------
NGL=999                   # 全部层 offload
KV_TYPE="f16"             # KV 全精度

# ---- bench 参数 -----------------------------------------------------------
PP=512                    # prompt tokens
TG=128                    # 生成 tokens
REPEATS=3                 # 重复取平均

# ---- 二进制路径 (可用 env 覆盖) -------------------------------------------
VULKAN_OFFICIAL_BIN="${VULKAN_OFFICIAL_BIN:-$HOME/.local/bin/llama}"
VULKAN_BIN="${VULKAN_BIN:-$HOME/workspace/ai/llama.cpp/build-vulkan/bin/llama-bench}"
SYCL_BIN="${SYCL_BIN:-$HOME/workspace/ai/llama.cpp/build-sycl/bin/llama-bench}"

# 公共参数; split 模式由各 runner 显式传入 (tensor 与 layer 不混用)
BENCH_ARGS=(
    -m "$MODEL"
    -ngl "$NGL"
    --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE"
    --flash-attn on
    -p "$PP" -n "$TG" -r "$REPEATS"
)

[ -f "$MODEL" ] || { echo "错误: 找不到模型 $MODEL"; exit 1; }

# ---- SYCL 运行环境 --------------------------------------------------------
# 设备选择用 env (ONEAPI_DEVICE_SELECTOR) 不用 --device; 双 A770 → level_zero:0;1
# 库路径手动指定, 不 source setvars.sh:
#   libsycl/libsvml/libirng/libimf/libintlc/libiomp5 → compiler/latest/lib
#   libdnnl (GGML_SYCL_DNN=ON 默认)                → dnnl/latest/lib
#   libumf (oneAPI 2026 unified runtime 依赖)      → umf/latest/lib
#   libhwloc.so.15 (level_zero adapter 依赖)       → oneapi/<ver>/lib, 动态查找
#     缺依赖时 adapter 加载失败 → "No device of requested type available"
sycl_env() {
    export ONEAPI_DEVICE_SELECTOR="${SYCL_DEVICES:-level_zero:0;level_zero:1}"
    export ZES_ENABLE_SYSMAN=1    # layer split 查显存余量
    local hwloc_dir="$(dirname "$(find /opt/intel/oneapi -maxdepth 3 -name 'libhwloc.so.15' 2>/dev/null | head -1)")"
    for d in /opt/intel/oneapi/compiler/latest/lib /opt/intel/oneapi/dnnl/latest/lib /opt/intel/oneapi/umf/latest/lib "$hwloc_dir"; do
        [ -d "$d" ] || { echo "错误: 找不到 oneAPI 库目录 $d" >&2; exit 1; }
        export LD_LIBRARY_PATH="${SYCL_BIN%/*}:$d:${LD_LIBRARY_PATH:-}"
    done
}

# ---- 四种配置 --------------------------------------------------------------
run_vulkan_official() {
    [ -x "$VULKAN_OFFICIAL_BIN" ] || { echo "错误: 找不到 $VULKAN_OFFICIAL_BIN"; return 1; }
    echo "== [1/4] Vulkan 官方预编译 $(basename "$VULKAN_OFFICIAL_BIN") =="
    "$VULKAN_OFFICIAL_BIN" bench --device "Vulkan1,Vulkan2" --split-mode layer "${BENCH_ARGS[@]}"
}

run_vulkan() {
    [ -x "$VULKAN_BIN" ] || { echo "错误: 找不到 $VULKAN_BIN (需先编译 build-vulkan)"; return 1; }
    echo "== [2/4] Vulkan 自编译 (build-vulkan) =="
    "$VULKAN_BIN" --device "Vulkan1,Vulkan2" --split-mode layer "${BENCH_ARGS[@]}"
}

run_sycl() {
    [ -x "$SYCL_BIN" ] || { echo "错误: 找不到 $SYCL_BIN (需先编译 build-sycl)"; return 1; }
    sycl_env
    echo "== [3/4] SYCL layer split (build-sycl, $ONEAPI_DEVICE_SELECTOR) =="
    "$SYCL_BIN" --split-mode layer "${BENCH_ARGS[@]}"
}

run_sycl_tensor() {
    [ -x "$SYCL_BIN" ] || { echo "错误: 找不到 $SYCL_BIN (需先编译 build-sycl)"; return 1; }
    sycl_env
    echo "== [4/4] SYCL tensor split (build-sycl, $ONEAPI_DEVICE_SELECTOR) =="
    "$SYCL_BIN" --split-mode tensor "${BENCH_ARGS[@]}"
}

# ---- 分发 --------------------------------------------------------------
case "${1:-all}" in
    all)
        run_vulkan_official || true
        sleep 8
        run_vulkan || true
        sleep 8
        run_sycl || true
        sleep 8
        run_sycl_tensor || true
        ;;
    vulkan-official) run_vulkan_official ;;
    vulkan)          run_vulkan ;;
    sycl)            run_sycl ;;
    sycl-tensor)     run_sycl_tensor ;;
    *) echo "错误: 未知配置 '$1', 支持: vulkan-official|vulkan|sycl|sycl-tensor (默认 all)"; exit 1 ;;
esac

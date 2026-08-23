#!/usr/bin/env bash
# 启动 Qwen3.8 27B (Q4_K_M) 于 2x Arc A770 — SYCL 后端 (build-sycl 自编译版)
# 依赖: ~/workspace/ai/llama.cpp/build-sycl (编译方法见 README.md)
# 用法: ./start-qwen3.8-27b.sh   (需先 chmod +x)

set -euo pipefail

MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
SYCL_BIN="${SYCL_BIN:-$HOME/workspace/ai/llama.cpp/build-sycl/bin/llama-server}"

HOST="127.0.0.1"          # 监听地址; 局域网访问改 0.0.0.0
PORT="8080"               # OpenAI 兼容 API 端口
N_CTX=131072              # 上下文长度。模型原生 262144; 128k 下 f16 KV 约 13GB, 两卡 32GB 够放
NGL=999                   # 全部 65 层 offload 到 GPU
KV_TYPE="f16"             # KV 缓存不量化, 全精度 f16 (精度最高, 占显存多)

# ---- 前置检查 -------------------------------------------------------------
[ -x "$SYCL_BIN" ] || {
    echo "错误: 找不到 $SYCL_BIN"
    echo "需先编译: cmake -B build-sycl -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON"
    exit 1
}
[ -f "$MODEL" ] || { echo "错误: 找不到模型 $MODEL"; exit 1; }

export ONEAPI_DEVICE_SELECTOR="${SYCL_DEVICES:-level_zero:0;level_zero:1}"
export ZES_ENABLE_SYSMAN=1    # layer split 查显存余量
# 缺 umf/latest/lib (libumf.so.1) / hwloc 时 level_zero adapter 加载失败 → 0 devices
hwloc_dir="$(dirname "$(find /opt/intel/oneapi -maxdepth 3 -name 'libhwloc.so.15' 2>/dev/null | head -1)")"
for d in /opt/intel/oneapi/compiler/latest/lib /opt/intel/oneapi/dnnl/latest/lib /opt/intel/oneapi/umf/latest/lib "$hwloc_dir"; do
    [ -d "$d" ] || { echo "错误: 找不到 oneAPI 库目录 $d"; exit 1; }
    export LD_LIBRARY_PATH="${SYCL_BIN%/*}:$d:${LD_LIBRARY_PATH:-}"
done

# ---- 启动 (llama-server, OpenAI 兼容) ------------------------------------
exec "$SYCL_BIN" \
    -m "$MODEL" \
    --split-mode layer \
    -ngl "$NGL" \
    -c "$N_CTX" \
    --cache-type-k "$KV_TYPE" \
    --cache-type-v "$KV_TYPE" \
    --flash-attn on \
    --temp 1.0 \
    --top-k 20 \
    --top-p 0.95 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --repeat-penalty 1.0 \
    --n-predict 32768 \
    --reasoning-budget 4096 \
    --host "$HOST" \
    --port "$PORT"

# 参数说明:
#   --split-mode layer  逐层切分; 双 A770 同型号均分, 避免卡间行切分带宽瓶颈
#   -m/-ngl/-c/cache-type/flash-attn 基础配置与 bench 脚本一致
#   --temp/top-k/top-p/min-p  采样参数取自 Qwen3.8 27B 官方最佳实践「思考模式」;
#                       非思考模式改: temp 0.7, top_p 0.80, top_k 20, min_p 0.0, presence 1.5
#   --n-predict 32768 总输出上限; --reasoning-budget 4096 思考段上限

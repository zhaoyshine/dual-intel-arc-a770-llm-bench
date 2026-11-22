#!/bin/bash
# 本目录所有脚本共用配置; 由 build.sh/start.sh 在设置 SCRIPT_DIR 后 source

C0VEN_DIR="${C0VEN_DIR:-$HOME/workspace/ai/c0ven}"
LLAMA_REPO="${LLAMA_REPO:-$HOME/workspace/ai/llama.cpp}"
# ADR-0012 固定的 ggml commit, c0ven 的 CPU 专家路径和 gguf-py 都依赖它
LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d

MODEL_FILE="${MODEL_FILE:-$HOME/workspace/ai/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"

# 与官方镜像同样的路径布局, 只是放在本目录下
LLAMA_CPP_DIR="$SCRIPT_DIR/opt/llama.cpp"
BUILD_DIR="$SCRIPT_DIR/build"
VENV="$SCRIPT_DIR/.venv"
UV_DIR="$SCRIPT_DIR/.uv"
ENGINE_CACHE_DIR="${ENGINE_CACHE_DIR:-$SCRIPT_DIR/cache}"

HOST=127.0.0.1
PORT=8080
SLOTS=1
CONTEXT=32768
# 双卡 A770 (15.1 GiB/卡) 时按这段打包; 单卡调试设 ZE_AFFINITY_MASK=0
ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-}"

# A770: acm-g10 的 8-lane 变体, SPIR-V 兜底也用 acm 的可移植版本; bmg 仍按原样编译
AOT_DEVICES="${AOT_DEVICES:-acm-g10}"
DG2_DEVICES="${DG2_DEVICES:-acm-g10}"
SPIRV_DEVICE="${SPIRV_DEVICE:-acm-g10}"

# A770 的显存比 B60 少 7.6 GiB/卡, 专家更多留在 pinned RAM; 本机 31 GiB 内存下 IQ3_XXS @32K
# 的规划 host 24.7 GiB, 只留 ~1 GiB 余量, 因此关掉 checkpoint, 并跳过图片编码 (vision 模块不编 A770 版本)
ENGINE_ARGS=(
    --engine-arg=--checkpoints=0
    --engine-arg=--no-vision
)

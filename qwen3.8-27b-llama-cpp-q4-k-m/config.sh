#!/bin/bash
# shellcheck disable=SC2034
# 各脚本共用: 模型路径 + llama.cpp 构建目录 + 推理参数

LLAMA_DIR=~/workspace/ai/llama.cpp
SYCL_BUILD=$LLAMA_DIR/build-sycl
VULKAN_BUILD=$LLAMA_DIR/build-vulkan

MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
DRAFT_MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
MMPROJ=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf

# 通用推理
NGL=999
CTX=190000
KV_TYPE=q8_0
FA=on
BS=2048
UBS=1024
DRAFT_N_MAX=4
MTP_DEVICE=SYCL0
MAIN_TS=0.48,0.52
REASONING_BUDGET=4096

# llama-cli / llama-server 共用的官方思考模式采样值
SAMPLING_ARGS=(
    --temp 1.0
    --top-k 20
    --top-p 0.95
    --min-p 0.0
    --presence-penalty 0.0
    --repeat-penalty 1.0
)

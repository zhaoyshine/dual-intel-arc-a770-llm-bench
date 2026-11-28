#!/bin/bash
# 本目录所有脚本共用配置; 由 build.sh/start.sh 在设置 SCRIPT_DIR 后 source

# c0ven 和 llama.cpp 都在本目录 opt/ 下, 路径写死
C0VEN_DIR="$SCRIPT_DIR/opt/c0ven"
# 路径布局与官方镜像的 /opt/llama.cpp 对齐
LLAMA_CPP_DIR="$SCRIPT_DIR/opt/llama.cpp"
# ADR-0012 固定的 ggml commit, c0ven 的 CPU 专家路径和 gguf-py 都依赖它
LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d

MODEL_FILE="$HOME/workspace/ai/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"

BUILD_DIR="$SCRIPT_DIR/build"
VENV="$SCRIPT_DIR/.venv"
UV_DIR="$SCRIPT_DIR/.uv"
ENGINE_CACHE_DIR="$SCRIPT_DIR/cache"

HOST=127.0.0.1
PORT=8080
SLOTS=1
CONTEXT=200000
# 双卡 A770 (15.1 GiB/卡) 时按这段打包; 单卡调试在环境里设 ZE_AFFINITY_MASK=0

# 上游默认编 bmg-g21;bmg-g31;acm-g10;acm-g11;acm-g12; 本机只有 A770, 只编 acm-g10 这一档
# (单家族 build 时 kernels.cmake 自己给出 acm 的 HPG SPIR-V 目录, 另一家族留空)
AOT_DEVICES="acm-g10"

# 双卡 A770 各 15.1 GiB 显存, 装不下 IQ3_S 的专家, 多出的部分留在 pinned RAM; 本机内存 46.9 GiB,
# host 规划要在本机实测; 模型目录里没有 mmproj-*.gguf, vision 不加载
# 前端自己只给 engine 一个临时目录, 所以 tensor 表要靠 --engine-arg 指到 cache 里, 重启不再重读 GGUF 头部
ENGINE_ARGS=(
    --engine-arg=--checkpoints=32
    --engine-arg=--no-vision
    --engine-arg=--table-cache
    --engine-arg="$ENGINE_CACHE_DIR/tables"
)

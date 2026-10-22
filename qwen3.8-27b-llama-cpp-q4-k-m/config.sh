#!/bin/bash
# shellcheck disable=SC2034
# bench.sh / start.sh 共用配置

MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
DRAFT_MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf

# 通用推理
NGL=999
CTX=200000
KV_TYPE=q8_0
FA=on
BS=2048
UBS=1024
DRAFT_N_MAX=3
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


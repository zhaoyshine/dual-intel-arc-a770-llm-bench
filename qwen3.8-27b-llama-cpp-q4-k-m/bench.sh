#!/bin/bash
# Qwen3.8 27B (Q4_K_M) 基准测试 — 双 Arc A770, 五种配置
# 用法: ./bench.sh [all|vulkan-official|vulkan|sycl|sycl-tensor|sycl-mtp|devices]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=sycl_env.sh
source "$SCRIPT_DIR/sycl_env.sh"

MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
VULKAN_OFFICIAL_BIN=~/.local/bin/llama
VULKAN_BIN=~/workspace/ai/llama.cpp/build-vulkan/bin/llama-bench
SYCL_BENCH_BIN=~/workspace/ai/llama.cpp/build-sycl/bin/llama-bench
SYCL_CLI_BIN=~/workspace/ai/llama.cpp/build-sycl/bin/llama-cli
DRAFT_MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf

NGL=999
KV_TYPE=q8_0
FA=on
PP=5120
TG=256
REPEATS=3
DRAFT_N_MAX=3
MTP_DEVICE=SYCL0
MAIN_TS=0.47,0.53
MTP_CTX=200000
BS=2048
UBS=1024
REASONING_EFFORT=medium

# 别用重复短句当提示词——输出与输入同分布, 无法代表真实场景
read -r -d '' PROMPT <<'EOF' || true
请从第一性原理出发，分析下面这个工程问题，给出完整的推理过程和最终结论。
问题：假设你为一个在轨深空望远镜设计纠错系统，其星敏感器与主成像系统共享一套 PCIe 4.0 x16 链路，该链路已被硬件拆分卡分成两个 x8 通道（类似消费级双显卡工作站）。成像数据以 4K@60fps、每像素 14bit 无压缩流式写入两块独立显存，要求零丢帧。
约束如下：
1. 两条 x8 通道总带宽独立，各自约 8GB/s 可用，突发时可达 16GB/s 但持续时间不能超过 100us；
2. 星敏感器每 10ms 需要读取 2MB 姿态参考帧，延迟超过 2ms 会触发姿态失稳；
3. 系统内存仅 15GB，其中 12GB 已被不可回收的推理工作负载占用；
4. 主控 CPU 只有 12 核，必须保留 4 核给姿态控制环，不能借用。
请回答：第一，用排队论模型估算最小缓冲深度并解释为什么不能简单地把两路数据合并调度；第二，写出一个基于双缓冲生产者-消费者模型的伪代码实现，明确说明每个锁、每块内存池的大小与生命周期；第三，指出如果显存总线宽度减半，你的模型哪个参数最先失效，为什么；第四，给出你最不确定的三条假设，并分别说明如何用实验验证。回答请用中文，推理过程分步展开。
EOF

[[ -f "$MODEL" ]] || err "找不到模型 $MODEL"

BENCH_ARGS=(
    -m "$MODEL"
    -ngl "$NGL"
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
    --flash-attn "$FA"
    --batch-size "$BS"
    --ubatch-size "$UBS"
    -p "$PP" -n "$TG" -r "$REPEATS"
)

run_vulkan_official() {
    [[ -x "$VULKAN_OFFICIAL_BIN" ]] || { echo "错误: 找不到 $VULKAN_OFFICIAL_BIN" >&2; return 1; }
    echo "== [1/5] Vulkan 官方预编译 =="
    "$VULKAN_OFFICIAL_BIN" bench --device Vulkan1,Vulkan2 --split-mode layer "${BENCH_ARGS[@]}"
}

run_vulkan() {
    [[ -x "$VULKAN_BIN" ]] || { echo "错误: 找不到 $VULKAN_BIN (需先编译 build-vulkan)" >&2; return 1; }
    echo "== [2/5] Vulkan 自编译 (build-vulkan) =="
    "$VULKAN_BIN" --device Vulkan1,Vulkan2 --split-mode layer "${BENCH_ARGS[@]}"
}

run_sycl() {
    [[ -x "$SYCL_BENCH_BIN" ]] || { echo "错误: 找不到 $SYCL_BENCH_BIN (需先编译 build-sycl)" >&2; return 1; }
    sycl_env "$SYCL_BENCH_BIN"
    echo "== [3/5] SYCL layer split (build-sycl) =="
    "$SYCL_BENCH_BIN" --split-mode layer "${BENCH_ARGS[@]}"
}

run_sycl_tensor() {
    [[ -x "$SYCL_BENCH_BIN" ]] || { echo "错误: 找不到 $SYCL_BENCH_BIN (需先编译 build-sycl)" >&2; return 1; }
    sycl_env "$SYCL_BENCH_BIN"
    echo "== [4/5] SYCL tensor split (build-sycl) =="
    "$SYCL_BENCH_BIN" --split-mode tensor "${BENCH_ARGS[@]}"
}

# llama-bench 不支持 draft 参数, 投机只能走 llama-cli
run_sycl_mtp() {
    [[ -x "$SYCL_CLI_BIN" ]] || { echo "错误: 找不到 $SYCL_CLI_BIN (先编译: ./build_sycl.sh)" >&2; return 1; }
    [[ -f "$DRAFT_MODEL" ]] || { echo "错误: 找不到 MTP draft 模型 $DRAFT_MODEL" >&2; return 1; }
    sycl_env "$SYCL_CLI_BIN"
    echo "== [5/5] SYCL + MTP 投机解码 (llama-cli, draft-n-max $DRAFT_N_MAX) =="
    echo "draft: $(basename "$DRAFT_MODEL") | prompt: ${#PROMPT} 字符"
    "$SYCL_CLI_BIN" -m "$MODEL" -md "$DRAFT_MODEL" \
        --spec-type draft-mtp \
        --spec-draft-n-max "$DRAFT_N_MAX" \
        --cache-type-k-draft "$KV_TYPE" \
        --cache-type-v-draft "$KV_TYPE" \
        --split-mode layer \
        -ngl "$NGL" -ngld 999 \
        -devd "$MTP_DEVICE" \
        -ts "$MAIN_TS" \
        --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE" \
        --flash-attn "$FA" \
        --batch-size "$BS" \
        --ubatch-size "$UBS" \
        -c "$MTP_CTX" -p "$PROMPT" -n 512 \
        --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
        --presence-penalty 0.0 --repeat-penalty 1.0 \
        --reasoning-budget 4096 \
        --reasoning-effort "$REASONING_EFFORT" \
        >/dev/null
}

show_devices() {
    [[ -x "$SYCL_BENCH_BIN" ]] || err "找不到 $SYCL_BENCH_BIN (需先编译 build-sycl)"
    sycl_env "$SYCL_BENCH_BIN"
    "$SYCL_BENCH_BIN" --list-devices
}

usage() {
    cat <<'EOF'
用法: ./bench.sh [all|vulkan-official|vulkan|sycl|sycl-tensor|sycl-mtp|devices]

  all              串行跑全部五种配置 (默认, 结果与结论见 README.md)
  vulkan-official  官方预编译 llama (~/.local/bin/llama)
  vulkan           自编译 Vulkan (build-vulkan)
  sycl             SYCL layer split (build-sycl)
  sycl-tensor      SYCL tensor split (build-sycl)
  sycl-mtp         SYCL + MTP 投机解码 (llama-cli + draft 模型)
  devices          列出 llama.cpp 可见设备
EOF
}

case "${1:-all}" in
    all)
        run_vulkan_official || true; sleep 8
        run_vulkan          || true; sleep 8
        run_sycl            || true; sleep 8
        run_sycl_tensor     || true; sleep 8
        run_sycl_mtp        || true
        ;;
    vulkan-official) run_vulkan_official ;;
    vulkan)          run_vulkan ;;
    sycl)            run_sycl ;;
    sycl-tensor)     run_sycl_tensor ;;
    sycl-mtp)        run_sycl_mtp ;;
    devices)         show_devices ;;
    -h|--help|help) usage ;;
    *)       usage >&2; err "未知参数 '$1'" ;;
esac

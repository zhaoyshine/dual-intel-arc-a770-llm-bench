#!/bin/bash
# Qwen3.8 27B (Q4_K_M) 基准测试 — 双 Arc A770, 五种配置
# 用法: ./bench.sh [all|vulkan|sycl|sycl-tensor|sycl-mtp|sycl-tensor-mtp|devices]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
# shellcheck source=config.sh
source "$SCRIPT_DIR/config.sh"
# shellcheck source=sycl_env.sh
source "$SCRIPT_DIR/sycl_env.sh"

VULKAN_BENCH=$VULKAN_BUILD/bin/llama-bench
SYCL_BENCH=$SYCL_BUILD/bin/llama-bench
SYCL_CLI=$SYCL_BUILD/bin/llama-cli

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
    -p 2048 -n 128 -r 3
)

run_vulkan() {
    [[ -x "$VULKAN_BENCH" ]] || { echo "错误: 找不到 $VULKAN_BENCH (先编译: ./build.sh vulkan)" >&2; return 1; }
    echo "vulkan"
    "$VULKAN_BENCH" --device Vulkan1,Vulkan2 --split-mode layer "${BENCH_ARGS[@]}"
}

run_sycl() {
    local split=$1
    [[ -x "$SYCL_BENCH" ]] || { echo "错误: 找不到 $SYCL_BENCH (先编译: ./build.sh sycl)" >&2; return 1; }
    sycl_env "$SYCL_BENCH"
    echo "sycl-$split"
    "$SYCL_BENCH" --split-mode "$split" "${BENCH_ARGS[@]}"
}

# llama-bench 不支持 draft 参数, 投机只能走 llama-cli
run_sycl_mtp() {
    local split=$1
    [[ -x "$SYCL_CLI" ]] || { echo "错误: 找不到 $SYCL_CLI (先编译: ./build.sh sycl)" >&2; return 1; }
    [[ -f "$DRAFT_MODEL" ]] || { echo "错误: 找不到 MTP draft 模型 $DRAFT_MODEL" >&2; return 1; }
    sycl_env "$SYCL_CLI"
    echo "sycl-$split-mtp"
    "$SYCL_CLI" -m "$MODEL" -md "$DRAFT_MODEL" \
        --spec-type draft-mtp \
        --spec-draft-n-max "$DRAFT_N_MAX" \
        --cache-type-k-draft "$KV_TYPE" \
        --cache-type-v-draft "$KV_TYPE" \
        --split-mode "$split" \
        -ngl "$NGL" -ngld "$NGL" \
        -devd "$MTP_DEVICE" \
        -ts "$MAIN_TS" \
        --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE" \
        --flash-attn "$FA" \
        --batch-size "$BS" \
        --ubatch-size "$UBS" \
        -c "$CTX" -p "$PROMPT" -n 512 \
        "${SAMPLING_ARGS[@]}" \
        --reasoning-budget "$REASONING_BUDGET" \
        --reasoning-effort medium \
        >/dev/null
}

show_devices() {
    [[ -x "$SYCL_BENCH" ]] || err "找不到 $SYCL_BENCH (先编译: ./build.sh sycl)"
    sycl_env "$SYCL_BENCH"
    "$SYCL_BENCH" --list-devices
}

case "${1:-all}" in
    all)
        run_vulkan || true; sleep 8
        run_sycl layer || true; sleep 8
        run_sycl tensor || true; sleep 8
        run_sycl_mtp layer || true; sleep 8
        run_sycl_mtp tensor || true
        ;;
    vulkan)          run_vulkan ;;
    sycl)            run_sycl layer ;;
    sycl-tensor)     run_sycl tensor ;;
    sycl-mtp)        run_sycl_mtp layer ;;
    sycl-tensor-mtp) run_sycl_mtp tensor ;;
    devices)         show_devices ;;
    -h|--help|help)  echo "用法: ./bench.sh [all|vulkan|sycl|sycl-tensor|sycl-mtp|sycl-tensor-mtp|devices]"; exit 0 ;;
    *)               usage "用法: ./bench.sh [all|vulkan|sycl|sycl-tensor|sycl-mtp|sycl-tensor-mtp|devices]"; err "未知参数 '$1'" ;;
esac

#!/usr/bin/env bash
# Qwen3.8 27B (Q4_K_M) 基准测试 — 双 Arc A770, 五种配置
# 结果与结论见同目录 README.md
#
# 用法:
#   ./bench-qwen3.8-27b.sh               # 串行跑全部五种
#   ./bench-qwen3.8-27b.sh <配置>        # 只跑一种 (-h/--help 查看配置列表)
#
# 配置:
#   vulkan-official  官方预编译 (b10217), --device Vulkan1,Vulkan2
#   vulkan           同源码自编译 (build-vulkan), 与 SYCL 公平对比
#   sycl             同源码自编译 (build-sycl, FP16), layer split
#   sycl-tensor      同上, --split-mode tensor
#   sycl-mtp         同上, + MTP 投机解码 (llama-cli, 只测 decode)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=sycl-env.sh
source "$SCRIPT_DIR/sycl-env.sh"

MODEL=~/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf

# ---- 运行配置 (与 start-qwen3.8-27b.sh 一致) --------------------------------
NGL=999                   # 全部层 offload
KV_TYPE="f16"             # KV 全精度

# ---- bench 参数 -----------------------------------------------------------
PP=5120                   # prompt tokens
TG=128                    # 生成 tokens
REPEATS=3                 # 重复取平均

# ---- 二进制路径 (可用 env 覆盖) -------------------------------------------
VULKAN_OFFICIAL_BIN="${VULKAN_OFFICIAL_BIN:-$HOME/.local/bin/llama}"
VULKAN_BIN="${VULKAN_BIN:-$HOME/workspace/ai/llama.cpp/build-vulkan/bin/llama-bench}"
SYCL_BIN="${SYCL_BIN:-$HOME/workspace/ai/llama.cpp/build-sycl/bin/llama-bench}"
SYCL_CLI_BIN="${SYCL_CLI_BIN:-$HOME/workspace/ai/llama.cpp/build-sycl/bin/llama-cli}"

# ---- MTP 投机解码 (run_sycl_mtp) ------------------------------------------
# 投机解码只作用于 decode, llama-bench 不支持 draft 参数, 用 llama-cli 测:
#   llama-cli 非交互单次跑完打印 eval timings; 生成文本重定向丢弃, 只看 stderr 统计
#   llama-bench 自动均值 (-r), llama-cli 无此参数, 手动循环 REPEATS 次
# 运行参数对齐 start-qwen3.8-27b.sh (采样/ctx/ubatch), 保证投机测试与 server 一致
MTP_MODEL="${MTP_MODEL:-$HOME/workspace/ai/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
MTP_N_MAX="${MTP_N_MAX:-3}"  # draft 长度; 实测 3 最优, 4 打平, 1 仅 +17%
MTP_CTX="${MTP_CTX:-102400}" # 投机=主+draft 双 KV, 15GB RAM 上限 (start 脚本保守取 90000)
# 默认提示词: 真实"高级问题" (~600 token), 让模型认真思考输出
# 别用重复短句——输出会和输入同分布, 无法代表真实场景
if [ -z "${MTP_PROMPT:-}" ]; then
    read -r -d '' MTP_PROMPT <<'MTP_EOF' || true
请从第一性原理出发，分析下面这个工程问题，给出完整的推理过程和最终结论。
问题：假设你为一个在轨深空望远镜设计纠错系统，其星敏感器与主成像系统共享一套 PCIe 4.0 x16 链路，该链路已被硬件拆分卡分成两个 x8 通道（类似消费级双显卡工作站）。成像数据以 4K@60fps、每像素 14bit 无压缩流式写入两块独立显存，要求零丢帧。
约束如下：
1. 两条 x8 通道总带宽独立，各自约 8GB/s 可用，突发时可达 16GB/s 但持续时间不能超过 100us；
2. 星敏感器每 10ms 需要读取 2MB 姿态参考帧，延迟超过 2ms 会触发姿态失稳；
3. 系统内存仅 15GB，其中 12GB 已被不可回收的推理工作负载占用；
4. 主控 CPU 只有 12 核，必须保留 4 核给姿态控制环，不能借用。
请回答：第一，用排队论模型估算最小缓冲深度并解释为什么不能简单地把两路数据合并调度；第二，写出一个基于双缓冲生产者-消费者模型的伪代码实现，明确说明每个锁、每块内存池的大小与生命周期；第三，指出如果显存总线宽度减半，你的模型哪个参数最先失效，为什么；第四，给出你最不确定的三条假设，并分别说明如何用实验验证。回答请用中文，推理过程分步展开。
MTP_EOF
fi

# 公共参数; split 模式由各 runner 显式传入 (tensor 与 layer 不混用)
BENCH_ARGS=(
    -m "$MODEL"
    -ngl "$NGL"
    --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE"
    --flash-attn on
    --ubatch-size 1024
    -p "$PP" -n "$TG" -r "$REPEATS"
)

[ -f "$MODEL" ] || { echo "错误: 找不到模型 $MODEL"; exit 1; }

# ---- 五种配置 --------------------------------------------------------------
run_vulkan_official() {
    [ -x "$VULKAN_OFFICIAL_BIN" ] || { echo "错误: 找不到 $VULKAN_OFFICIAL_BIN"; return 1; }
    echo "== [1/5] Vulkan 官方预编译 $(basename "$VULKAN_OFFICIAL_BIN") =="
    "$VULKAN_OFFICIAL_BIN" bench --device "Vulkan1,Vulkan2" --split-mode layer "${BENCH_ARGS[@]}"
}

run_vulkan() {
    [ -x "$VULKAN_BIN" ] || { echo "错误: 找不到 $VULKAN_BIN (需先编译 build-vulkan)"; return 1; }
    echo "== [2/5] Vulkan 自编译 (build-vulkan) =="
    "$VULKAN_BIN" --device "Vulkan1,Vulkan2" --split-mode layer "${BENCH_ARGS[@]}"
}

run_sycl() {
    [ -x "$SYCL_BIN" ] || { echo "错误: 找不到 $SYCL_BIN (需先编译 build-sycl)"; return 1; }
    sycl_env "$SYCL_BIN"
    echo "== [3/5] SYCL layer split (build-sycl, $ONEAPI_DEVICE_SELECTOR) =="
    "$SYCL_BIN" --split-mode layer "${BENCH_ARGS[@]}"
}

run_sycl_tensor() {
    [ -x "$SYCL_BIN" ] || { echo "错误: 找不到 $SYCL_BIN (需先编译 build-sycl)"; return 1; }
    sycl_env "$SYCL_BIN"
    echo "== [4/5] SYCL tensor split (build-sycl, $ONEAPI_DEVICE_SELECTOR) =="
    "$SYCL_BIN" --split-mode tensor "${BENCH_ARGS[@]}"
}

run_sycl_mtp() {
    [ -x "$SYCL_CLI_BIN" ] || { echo "错误: 找不到 $SYCL_CLI_BIN (需编译 llama-cli)"; return 1; }
    [ -f "$MTP_MODEL" ] || { echo "错误: 找不到 MTP draft 模型 $MTP_MODEL"; return 1; }
    sycl_env "$SYCL_CLI_BIN"
    echo "== [5/5] SYCL + MTP 投机解码 (llama-cli, draft-n-max $MTP_N_MAX, $ONEAPI_DEVICE_SELECTOR) =="
    echo "== draft: $(basename "$MTP_MODEL") | prompt: ${#MTP_PROMPT} 字符 =="
    # 采样参数与 start 脚本一致; 生成文本丢 stdout, 统计在 stderr
    # llama-cli 跑完驻留不退出, 看结果后 Ctrl-C 结束本轮
    "$SYCL_CLI_BIN" -m "$MODEL" -md "$MTP_MODEL" \
        --spec-type draft-mtp \
        --spec-draft-n-max "$MTP_N_MAX" \
        --split-mode layer \
        -ngl "$NGL" -ngld 999 \
        --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE" \
        --flash-attn on \
        --ubatch-size 1024 \
        -c "$MTP_CTX" -p "$MTP_PROMPT" -n "$TG" \
        --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
        --presence-penalty 0.0 --repeat-penalty 1.0 \
        >/dev/null
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
        sleep 8
        run_sycl_mtp || true
        ;;
    vulkan-official) run_vulkan_official ;;
    vulkan)          run_vulkan ;;
    sycl)            run_sycl ;;
    sycl-tensor)     run_sycl_tensor ;;
    sycl-mtp)        run_sycl_mtp ;;
    -h|--help)
        echo "用法: $0 [all|vulkan-official|vulkan|sycl|sycl-tensor|sycl-mtp]"
        echo "不带参数串行跑全部五种; 结果与结论见同目录 README.md"
        ;;
    *) echo "错误: 未知配置 '$1', 支持: vulkan-official|vulkan|sycl|sycl-tensor|sycl-mtp (默认 all)"; exit 1 ;;
esac

# Qwen3.8 27B 双 A770 基准测试 / Qwen3.8 27B Dual A770 Benchmark Report

家人们，用你发财的小手点点右上角的星星⭐吧 / Dear friends, please click the star ⭐ at the top right with your lucky hand!

**日期 / Date:** 2026-08-23 初测，2026-09-05 补充 ubatch / MTP 复测 / Initial run 2026-08-23, ubatch / MTP follow-ups 2026-09-05

## 目录 / Contents

1. [测试环境 / System Environment](#1-测试环境--system-environment)
2. [软件版本 / Software Versions](#2-软件版本--software-versions)
3. [测试配置 / Benchmark Configs](#3-测试配置--benchmark-configs)
4. [测试方法 / Methodology](#4-测试方法--methodology)
5. [测试结果 / Results](#5-测试结果--results)
6. [使用说明 / Usage](#6-使用说明--usage)

---

## 1. 测试环境 / System Environment

**中文：** 本测试在个人工作站上进行，双 A770 分别以 SYCL 与 Vulkan 后端跑 llama.cpp 推理基准。GT 1030 仅作显示输出，未参与推理。

**English:** This benchmark ran on a personal workstation using two Arc A770 GPUs with the SYCL and Vulkan backends of llama.cpp. A GT 1030 handles display only and did not participate in inference.

| 项目 / Item | 值 / Value |
|---|---|
| 操作系统 / OS | Fedora Linux 44 (KDE Plasma Desktop Edition) |
| 内核 / Kernel | 7.1.7-200.fc44.x86_64 |
| CPU | AMD Ryzen 9 5900X, 12C/24T |
| 内存 / RAM | 15 GiB |
| GPU 1 | Intel Arc A770 16GB (DG2) — level_zero:0 / Vulkan1 |
| GPU 2 | Intel Arc A770 16GB (DG2) — level_zero:1 / Vulkan2 |
| GPU 3 (显示 / display) | NVIDIA GeForce GT 1030 2GB — Vulkan0 |
| 拆分卡 / bifurcation card | Intel PCIe switch (Device 4fa0, 07:00.0)，将 PCIe 4.0 x16 拆分为 x8+x8 / splits PCIe 4.0 x16 into x8+x8 |
| 显卡驱动 / GPU Drivers | intel-level-zero 26.22.38646, intel-opencl 26.22.38646, vulkan-loader 1.4.341, mesa 26.1.6 |

---

## 2. 软件版本 / Software Versions

**中文：** llama.cpp 从源码编译，官方预编译单文件版（b10217）用作 Vulkan 对照。SYCL 需 Intel oneAPI 编译，Vulkan 用系统 GCC。

**English:** llama.cpp built from source; the official prebuilt single binary (b10217) served as the Vulkan baseline. SYCL required the Intel oneAPI toolkit; Vulkan used the system GCC.

| 组件 / Component | 版本 / Version | 备注 / Note |
|---|---|---|
| llama.cpp 源码 / Source | commit `8144f31` (master, 2026-08-23) | `git clone --depth 1` |
| 官方预编译 / Official Prebuilt | b10217-ddd4ec142 (`~/.local/bin/llama`) | Vulkan 对照基准 / Vulkan baseline |
| 自编译 SYCL 版 / Self-Built SYCL | 8144f31, `build-sycl/bin/` | `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON` |
| 自编译 Vulkan 版 / Self-Built Vulkan | 8144f31, `build-vulkan/bin/` | `-DGGML_VULKAN=ON` |
| Intel oneAPI | 2026.1.1 (2026.1.1.20260724) | `intel-oneapi-toolkit` via yum |
| DPC++ 编译器 / Compiler | icpx 2026.1.1 | 编译 SYCL 后端 / builds SYCL backend |
| 模型 / Model | Qwen3.8-27B-UD-Q4_K_M.gguf, 15.32 GiB, 27.32 B params | unsloth GGUF |

---

## 3. 测试配置 / Benchmark Configs

**中文：** 五种配置互为对照：同源码编译的 Vulkan 与 SYCL 对比（消除版本差异），官方预编译 Vulkan 作额外参考；SYCL 分 layer（逐层切分）与 tensor（张量并行）两种 split 模式；sycl-mtp 用官方 MTP 模块测投机解码。

**English:** Five configs for cross-comparison: self-built Vulkan vs SYCL from the same source commit (eliminating version skew), the official prebuilt Vulkan as an extra reference, SYCL in both `layer` (layer-wise split) and `tensor` (tensor parallelism) split modes, and sycl-mtp measuring speculative decoding with the official MTP module.

| 配置 / Config | 二进制 / Binary | Split 模式 / Mode | 设备选择 / Device Select |
|---|---|---|---|
| vulkan-official | `~/.local/bin/llama` (b10217) | layer | `--device Vulkan1,Vulkan2` |
| vulkan | `build-vulkan/bin/llama-bench` | layer | `--device Vulkan1,Vulkan2` |
| sycl | `build-sycl/bin/llama-bench` | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| sycl-tensor | `build-sycl/bin/llama-bench` | tensor | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| sycl-mtp | `build-sycl/bin/llama-cli` + MTP draft | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |

---

## 4. 测试方法 / Methodology

**中文：** 用 `llama-bench` 测纯计算吞吐：5120 token prompt 测 prefill（pp5120），128 token 生成测 decode（tg128），各重复 3 次取平均。所有配置统一：全部层 offload（ngl 999）、KV 缓存 f16、flash-attention 开启、`--ubatch-size 1024`。Vulkan 按卡分行输出（双卡各自独立测），SYCL 为双卡合并吞吐。

**English:** Pure compute throughput via `llama-bench`: 5120 prompt tokens for prefill (pp5120), 128 generated tokens for decode (tg128), 3 repeats averaged. Identical settings across all configs: full offload (ngl 999), f16 KV cache, flash-attention on, `--ubatch-size 1024`. Vulkan reports per-GPU rows; SYCL reports combined dual-GPU throughput.

```sh
llama-bench -m <model> -ngl 999 --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --ubatch-size 1024 -p 5120 -n 128 -r 3
```

sycl-mtp 用 llama-cli 测（llama-bench 不支持投机参数），采样参数与 start 脚本一致，见 [5.3.2](#532-mtp-投机解码--mtp-speculative-decoding)。/ sycl-mtp uses llama-cli (llama-bench has no speculative support) with sampling identical to the start script; see [5.3.2](#532-mtp-投机解码--mtp-speculative-decoding).

---

## 5. 测试结果 / Results

**中文：** SYCL layer split 全面领先：prefill 合并吞吐 428 t/s（Vulkan 双卡合计约 225 t/s，快约 1.9 倍），decode 9.76 t/s（比 Vulkan 7.8 t/s 快约 25%）。tensor split 最慢——27B 逐层张量并行时，卡间 ring all-reduce 的 PCIe 通信开销大于收益。官方预编译与自编译 Vulkan 结果几乎一致，版本差异可忽略。Vulkan 慢并非 CPU 瓶颈：ngl 999 已全 offload（本机 CPU 跑 27B 上限仅约 pp 10 / tg 2 t/s），差距源于 Mesa Vulkan 驱动对 Arc 的优化弱于 Intel 自家 SYCL/oneDNN 路径。

**English:** SYCL layer split wins across the board: combined prefill 428 t/s vs ~225 t/s for both Vulkan GPUs together (~1.9x), and decode 9.76 t/s vs 7.8 t/s (~25% faster). Tensor split is the slowest — per-layer tensor parallelism on a 27B model pays more in PCIe ring all-reduce traffic than it gains. The official prebuilt and self-built Vulkan results match closely, confirming version skew is negligible. The slower Vulkan numbers are not a CPU issue: ngl 999 keeps every layer on GPU (CPU on this machine tops out around pp 10 / tg 2 t/s for a 27B); the gap comes from the Mesa Vulkan driver's weaker ggml optimizations vs Intel's own SYCL/oneDNN path.

### 5.1 Vulkan 官方预编译 / Official Prebuilt Vulkan
| model                          |       size |     params | backend    | ngl |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           pp512 |        109.59 ± 0.15 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           tg128 |          7.83 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           pp512 |        114.83 ± 0.23 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           tg128 |          7.79 ± 0.00 |

### 5.2 Vulkan 自编译 / Self-Built Vulkan
| model                          |       size |     params | backend    | ngl |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           pp512 |        109.52 ± 0.19 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           tg128 |          7.83 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           pp512 |        114.93 ± 0.16 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           tg128 |          7.79 ± 0.00 |

### 5.3 SYCL 逐层切分 / SYCL layer split
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |   1 |           pp512 |        428.01 ± 5.12 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |   1 |           tg128 |          9.76 ± 0.01 |

#### 5.3.1 ubatch 对 prefill 吞吐的影响 / Effect of ubatch on prefill throughput (2026-09-05)

**中文：** 复测时把 prompt 加长到 5120 token（pp5120）。调大 `--ubatch-size` 显著提升 prefill：默认值 → 432.7 t/s，1024 → 570.2 t/s，2048 → 637.4 t/s（较默认快约 47%）。注意 n_ubatch 只影响 prefill，且此表与 5.3 原表（pp512）不可直接对比。

**English:** Re-measured with the prompt length raised to 5120 tokens (pp5120). Raising `--ubatch-size` boosts prefill sharply: default → 432.7 t/s, 1024 → 570.2 t/s, 2048 → 637.4 t/s (~47% faster than default). Note n_ubatch only affects prefill, and these rows are not directly comparable with the pp512 rows in 5.3.

| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |  512(默认) |   1 |          pp5120 |        432.73 ± 0.18 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |     1024 |   1 |          pp5120 |        570.16 ± 0.30 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |     2048 |   1 |          pp5120 |        637.37 ± 0.21 |

#### 5.3.2 MTP 投机解码 / MTP speculative decoding (2026-09-05)

**中文：** 用官方 Qwen3.8-27B MTP 模块（`mtp-Qwen3.8-27B-Q4_0.gguf`，1.3 GiB）作 draft 模型，主+draft 同跑双 A770。llama-bench 不支持投机参数，改用 llama-cli：真实中文推理题 533 字符（~400 token）prompt、ctx 100k（投机=主+draft 双 KV，15 GB 内存下 128k 会报 OUT_OF_HOST_MEMORY）、采样参数与 start 脚本一致（temp 1.0/top-k 20/top-p 0.95）、每档 2–3 次取均值。无投机 llama-cli 对照为 9.6 t/s，与 llama-bench tg 的 9.x 一致（两条路径可比）。

**English:** Used the official Qwen3.8-27B MTP module (`mtp-Qwen3.8-27B-Q4_0.gguf`, 1.3 GiB) as a draft model, main+draft on both A770s. llama-bench has no speculative support, so llama-cli was used: real Chinese reasoning prompt (533 chars, ~400 tokens), ctx 100k (dual-KV for main+draft hits OUT_OF_HOST_MEMORY at 128k on the 15 GB host), sampling identical to the start script (temp 1.0/top-k 20/top-p 0.95), 2–3 runs averaged. The no-draft llama-cli baseline is 9.6 t/s, matching llama-bench's 9.x (both paths agree).

| draft 配置 / config | test | t/s |
|---|---|---|
| 无投机 / no speculation | tg128 | 9.6 |
| MTP `--spec-draft-n-max 1` | tg128 | 11.3 |
| MTP `--spec-draft-n-max 2` | tg128 | 12.8 |
| MTP `--spec-draft-n-max 3` | tg128 | 14.2 |
| MTP `--spec-draft-n-max 4` | tg128 | 14.0 |

> **注 / Note:** 选型 `--spec-draft-n-max 3`（官方推荐，与 4 打平），**比无投机快约 48%**；长度 1 仅 +17%（draft 推理开销摊不薄），2 为 +33%。draft 模型已全 offload（`-ngld 999`）。测试入口：`./bench-qwen3.8-27b.sh sycl-mtp`。/ Chose `--spec-draft-n-max 3` (official recommendation, ties with 4), **~48% faster than no speculation**; length 1 only +17% (draft overhead not amortized), 2 gives +33%. The draft model is fully offloaded (`-ngld 999`). Entry: `./bench-qwen3.8-27b.sh sycl-mtp`.

### 5.4 SYCL 张量并行 / SYCL tensor split
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 | tensor |   1 |           pp512 |        325.07 ± 3.01 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 | tensor |   1 |           tg128 |          3.99 ± 0.01 |

> **注 / Note:** Vulkan 行值为每张卡处理自己分片的速度，非"只用了单卡"：prefill 阶段两卡并行，合并吞吐约为两行之和（~225 t/s）；decode 阶段两卡流水线作业、同一时刻仅一卡计算，故每行值即整体值。SYCL 为双卡合并吞吐。Each Vulkan row shows one GPU processing its half of the layers — both GPUs are active. During prefill they run in parallel (combined ≈ sum of the two rows, ~225 t/s); during decode they form a pipeline with one GPU computing at a time, so each row equals the overall rate. SYCL reports combined throughput.

---

## 6. 使用说明 / Usage

### 基准测试 / Benchmark

```sh
./bench-qwen3.8-27b.sh                    # 串行跑全部五种 / run all five
./bench-qwen3.8-27b.sh sycl               # 单种 / one: vulkan-official|vulkan|sycl|sycl-tensor|sycl-mtp
```

### 服务启动 / Start Server (SYCL)

```sh
./start-qwen3.8-27b.sh                    # 默认 mtp: MTP 投机, ctx 90k; llama-server, OpenAI 兼容 API, 127.0.0.1:8080
./start-qwen3.8-27b.sh nomtp              # 关投机: 单 KV, ctx 120k (上下文更长, decode 慢约 1/3)
curl http://127.0.0.1:8080/v1/chat/completions
```

两种模式 / Two modes:

| 模式 / Mode | ctx | KV | decode 速度 / speed |
|---|---|---|---|
| `mtp`（默认 / default） | 90000 | 主 + draft 双份 / main + draft | 约 +48%（见 [5.3.2](#532-mtp-投机解码--mtp-speculative-decoding)） |
| `nomtp` | 122880 | 单份 / single | 基线 / baseline (9.6 t/s) |

> MTP 模式 draft 与主模型各占一份 KV，128k 在 15 GB 主机内存下会 `OUT_OF_HOST_MEMORY`，故 90k；关掉投机后单 KV 可放到 120k。/ In MTP mode the draft and main model each hold a KV cache, and 128k hits `OUT_OF_HOST_MEMORY` on the 15 GB host, hence 90k; with speculation off a single KV fits 120k.

### 编译 / Build

```sh
# SYCL: 推荐用脚本 (含 Level Zero 规范头检查) / recommended: script (with Level Zero header check)
./rebuild-sycl.sh                         # 依赖 oneAPI / requires oneAPI: sudo yum install intel-oneapi-toolkit
# 或手动 / or manually:
cmake -B build-sycl -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
cmake --build build-sycl --config Release -j 24 --target llama-bench llama-server

# Vulkan (依赖 vulkan-headers glslc spirv-headers-devel vulkan-loader-devel)
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j 24 --target llama-bench
```

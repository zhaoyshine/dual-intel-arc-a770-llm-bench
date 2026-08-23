# Qwen3.8 27B 双 A770 基准测试记录 / Qwen3.8 27B Dual A770 Benchmark Report

**日期 / Date:** 2026-08-23

## 目录 / Contents
1. [测试环境 / System Environment](#1-测试环境--system-environment)
2. [软件版本 / Software Versions](#2-软件版本--software-versions)
3. [四种测试配置 / Four Benchmark Configs](#3-四种测试配置--four-benchmark-configs)
4. [测试方法 / Methodology](#4-测试方法--methodology)
5. [测试结果 / Results](#5-测试结果--results)
6. [使用说明 / Usage](#6-使用说明--usage)

---

## 1. 测试环境 / System Environment

**中文：** 本测试在个人工作站上进行，双 Intel Arc A770 显卡以 SYCL 与 Vulkan 后端分别跑 llama.cpp 推理基准。系统另有 GT 1030 仅作显示输出，未参与推理。

**English:** This benchmark ran on a personal workstation using two Intel Arc A770 GPUs with both SYCL and Vulkan backends of llama.cpp. A GT 1030 handles display only and did not participate in inference.

| 项目 / Item | 值 / Value |
|---|---|
| 操作系统 / OS | Fedora Linux 44 (KDE Plasma Desktop Edition) |
| 内核 / Kernel | 7.1.7-200.fc44.x86_64 |
| CPU | AMD Ryzen 9 5900X, 12C/24T |
| 内存 / RAM | 15 GiB |
| GPU 1 | Intel Arc A770 16GB (DG2) — level_zero:0 / Vulkan1 |
| GPU 2 | Intel Arc A770 16GB (DG2) — level_zero:1 / Vulkan2 |
| GPU 3 (显示) | NVIDIA GeForce GT 1030 2GB — Vulkan0 |
| 拆分卡 / bifurcation card | Intel PCIe switch (Device 4fa0, 07:00.0)，将 PCIe 4.0 x16 拆分为 x8+x8 |
| 显卡驱动 / GPU Drivers | intel-level-zero 26.22.38646, intel-opencl 26.22.38646, vulkan-loader 1.4.341, mesa 26.1.6 |

---

## 2. 软件版本 / Software Versions

**中文：** llama.cpp 从源码编译，官方预编译单文件版（b10217）用作 Vulkan 对照。SYCL 需 Intel oneAPI 编译，Vulkan 用系统 GCC。

**English:** llama.cpp built from source; the official prebuilt single binary (b10217) served as the Vulkan baseline. SYCL required the Intel oneAPI toolkit; Vulkan used the system GCC.

| 组件 / Component | 版本 / Version | 备注 / Note |
|---|---|---|
| llama.cpp 源码 / Source | commit `8144f31` (master, 2026-08-23) | `git clone --depth 1` |
| 官方预编译 / Official Prebuilt | b10217-ddd4ec142 (`~/.local/bin/llama`) | Vulkan 对照基准 |
| 自编译 SYCL 版 / Self-Built SYCL | 8144f31, `build-sycl/bin/` | `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON` |
| 自编译 Vulkan 版 / Self-Built Vulkan | 8144f31, `build-vulkan/bin/` | `-DGGML_VULKAN=ON` |
| Intel oneAPI | 2026.1.1 (2026.1.1.20260724) | `intel-oneapi-toolkit` via yum |
| DPC++ 编译器 / Compiler | icpx 2026.1.1 | 编译 SYCL 后端 |
| 模型 / Model | Qwen3.8-27B-UD-Q4_K_M.gguf, 15.32 GiB, 27.32 B params | unsloth GGUF |

---

## 3. 四种测试配置 / Four Benchmark Configs

**中文：** 四种配置互为对照：同源码编译的 Vulkan 与 SYCL 对比（消除版本差异），官方预编译 Vulkan 作额外参考；SYCL 再分 layer（逐层切分）与 tensor（张量并行）两种 split 模式。

**English:** Four configs for cross-comparison: self-built Vulkan vs SYCL from the same source commit (eliminating version skew), the official prebuilt Vulkan as an extra reference, and SYCL in both `layer` (layer-wise split) and `tensor` (tensor parallelism) split modes.

| 配置 / Config | 二进制 / Binary | Split 模式 / Mode | 设备选择 / Device Select |
|---|---|---|---|
| 1. vulkan-official | `~/.local/bin/llama` (b10217) | layer | `--device Vulkan1,Vulkan2` |
| 2. vulkan | `build-vulkan/bin/llama-bench` | layer | `--device Vulkan1,Vulkan2` |
| 3. sycl | `build-sycl/bin/llama-bench` | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| 4. sycl-tensor | `build-sycl/bin/llama-bench` | tensor | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |

---

## 4. 测试方法 / Methodology

**中文：** 用 `llama-bench` 测纯计算吞吐：prompt 512 token 测 prefill（pp512），生成 128 token 测 decode（tg128），各重复 3 次取平均。所有配置统一：全部层 offload（ngl 999）、KV 缓存 f16、flash-attention 开启。Vulkan 版按卡分行输出（双卡各自独立测），SYCL 为双卡合并吞吐。

**English:** Pure compute throughput via `llama-bench`: 512 prompt tokens for prefill (pp512), 128 generated tokens for decode (tg128), 3 repeats averaged. Identical settings across all configs: full offload (ngl 999), f16 KV cache, flash-attention on. Vulkan reports per-GPU rows; SYCL reports combined dual-GPU throughput.

```sh
llama-bench -m <model> -ngl 999 --cache-type-k f16 --cache-type-v f16 --flash-attn on -p 512 -n 128 -r 3
```

---

## 5. 测试结果 / Results

**中文：** SYCL 逐层切分（layer split）全面领先：prefill 合并吞吐 427 t/s（Vulkan 双卡合计约 225 t/s，SYCL 快约 1.9 倍），decode 9.73 t/s（比 Vulkan 7.8 t/s 快约 25%）。SYCL 张量并行（tensor split）最慢——27B 模型逐层做张量并行时，卡间 ring all-reduce 的 PCIe 通信开销大于收益。官方预编译与自编译 Vulkan 结果几乎一致，版本差异可忽略。Vulkan 慢并非 CPU 瓶颈：ngl 999 已将全部层 offload 至 GPU，CPU 跑 27B 上限仅约 pp 10 / tg 2 t/s；差距源于 Mesa Vulkan 驱动对 Arc 的优化弱于 Intel 自家 SYCL/oneDNN 路径。

**English:** SYCL layer split wins across the board: combined prefill 427 t/s vs ~225 t/s for both Vulkan GPUs together (~1.9x), and decode 9.73 t/s vs 7.8 t/s (~25% faster). SYCL tensor split is the slowest — per-layer tensor parallelism on a 27B model pays more in PCIe ring all-reduce traffic than it gains. The official prebuilt and self-built Vulkan results match closely, confirming version skew is negligible. The slower Vulkan numbers are not a CPU issue: ngl 999 keeps every layer on GPU (CPU on this machine tops out around pp 10 / tg 2 t/s for a 27B); the gap comes from the Mesa Vulkan driver's weaker ggml optimizations vs Intel's own SYCL/oneDNN path.

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

### 5.4 SYCL 张量并行 / SYCL tensor split
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 | tensor |   1 |           pp512 |        325.07 ± 3.01 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 | tensor |   1 |           tg128 |          3.99 ± 0.01 |

> **注 / Note:** Vulkan 行值为每张卡处理自己分片的速度，非"只用了单卡"。prefill 阶段两卡并行，合并吞吐约为两行之和（~225 t/s）；decode 阶段两卡流水线作业、同一时刻仅一卡计算，故每行值即整体值。SYCL 为双卡合并吞吐。Each Vulkan row shows one GPU processing its half of the layers — both GPUs are active. During prefill they run in parallel (combined ≈ sum of the two rows, ~225 t/s); during decode they form a pipeline with one GPU computing at a time, so each row equals the overall rate. SYCL reports combined throughput.

---

## 6. 使用说明 / Usage

### 基准测试 / Benchmark

```sh
./bench-qwen3.8-27b.sh                    # 跑全部 4 种配置 / run all four
./bench-qwen3.8-27b.sh vulkan-official    # 只跑一种 / run one: vulkan-official|vulkan|sycl|sycl-tensor
```

### 服务启动 / Start Server (SYCL)

```sh
./start-qwen3.8-27b.sh                    # 启动 llama-server：OpenAI 兼容 API，监听 127.0.0.1:8080
curl http://127.0.0.1:8080/v1/chat/completions
```

### 编译记录 / Build Notes

```sh
# SYCL (需要 oneAPI: sudo yum install intel-oneapi-toolkit)
cmake -B build-sycl -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
cmake --build build-sycl --config Release -j 24 --target llama-bench llama-server

# Vulkan (需要 vulkan-headers glslc spirv-headers-devel vulkan-loader-devel)
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j 24 --target llama-bench
```

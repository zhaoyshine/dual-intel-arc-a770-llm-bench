# Qwen3.8 27B 双 A770 基准测试

家人们，用你发财的小手点一点右上角的星星⭐吧

**日期:** 2026-08-23 初测，2026-09-05 补充 ubatch / MTP 复测，2026-09-17 补充 xe 驱动复测

## 目录

1. [测试环境](#1-测试环境)
2. [软件版本](#2-软件版本)
3. [测试配置](#3-测试配置)
4. [测试方法](#4-测试方法)
5. [测试结果](#5-测试结果)
6. [使用说明](#6-使用说明)

---

## 1. 测试环境

| 项目 | 值 |
|---|---|
| 操作系统 | Fedora Linux 44 (KDE Plasma Desktop Edition) |
| 内核 | 7.1.13-200.fc44.x86_64（初测为 7.1.7） |
| CPU | AMD Ryzen 9 5900X, 12C/24T |
| 内存 | 15 GiB |
| GPU 1 | Intel Arc A770 16GB (DG2) — level_zero:0 / Vulkan1 |
| GPU 2 | Intel Arc A770 16GB (DG2) — level_zero:1 / Vulkan2 |
| GPU 3 (显示) | NVIDIA GeForce GT 1030 2GB — Vulkan0 |
| 拆分卡 | Intel PCIe switch (Device 4fa0, 07:00.0)，将 PCIe 4.0 x16 拆分为 x8+x8 |
| 显卡驱动 | intel-level-zero 26.22.38646, intel-opencl 26.22.38646, vulkan-loader 1.4.341, mesa 26.1.6 |

---

## 2. 软件版本

| 组件 | 版本 | 备注 |
|---|---|---|
| llama.cpp 源码 | commit `8144f31` (master, 2026-08-23) | `git clone --depth 1` |
| 官方预编译 | b10217-ddd4ec142 (`~/.local/bin/llama`) | Vulkan 对照基准 |
| 自编译 SYCL 版 | 8144f31, `build-sycl/bin/` | `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON` |
| 自编译 Vulkan 版 | 8144f31, `build-vulkan/bin/` | `-DGGML_VULKAN=ON`，系统 GCC |
| Intel oneAPI | 2026.1.1 (2026.1.1.20260724) | `intel-oneapi-toolkit` via yum |
| DPC++ 编译器 | icpx 2026.1.1 | 编译 SYCL 后端 |
| 模型 | Qwen3.8-27B-UD-Q4_K_M.gguf, 15.32 GiB, 27.32 B params | unsloth GGUF |

---

## 3. 测试配置

SYCL 与 Vulkan 用同一份源码编译，对比可排除版本差异；官方预编译 Vulkan 作额外参考。

| 配置 | 二进制 | Split 模式 | 设备选择 |
|---|---|---|---|
| vulkan-official | `~/.local/bin/llama` (b10217) | layer | `--device Vulkan1,Vulkan2` |
| vulkan | `build-vulkan/bin/llama-bench` | layer | `--device Vulkan1,Vulkan2` |
| sycl | `build-sycl/bin/llama-bench` | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| sycl-tensor | `build-sycl/bin/llama-bench` | tensor | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| sycl-mtp | `build-sycl/bin/llama-cli` + MTP draft | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |

---

## 4. 测试方法

各配置参数一致，每档重复 3 次取平均；Vulkan 按卡分行输出（双卡各自独立测量），SYCL 为双卡合并吞吐。

```sh
llama-bench -m <model> -ngl 999 --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --ubatch-size 1024 -p 5120 -n 128 -r 3
```

sycl-mtp 用 llama-cli 测，采样参数与 start 脚本一致，见 [5.3.2](#532-mtp-投机解码-2026-09-05)。

---

## 5. 测试结果

SYCL layer split 全面领先：prefill 合并吞吐 428 t/s（Vulkan 双卡合计约 225 t/s，快约 1.9 倍），decode 9.76 t/s（比 Vulkan 7.8 t/s 快约 25%）。tensor split 最慢——27B 逐层张量并行时，卡间 ring all-reduce 的 PCIe 通信开销大于收益。官方预编译与自编译 Vulkan 的结果几乎一致，版本差异可忽略。Vulkan 慢并非 CPU 瓶颈：ngl 999 已全 offload（本机 CPU 跑 27B 上限仅约 pp 10 / tg 2 t/s），差距源于 Mesa Vulkan 驱动对 Arc 的优化弱于 Intel 自家 SYCL/oneDNN 路径。

> **注:** Vulkan 行值为每张卡处理自己分片的速度，非“只用了单卡”：prefill 阶段两卡并行，合并吞吐约为两行之和（~225 t/s）；decode 阶段两卡流水线作业、同一时刻仅一卡计算，故每行值即整体值。SYCL 为双卡合并吞吐。

### 5.1 Vulkan 官方预编译
| model                          |       size |     params | backend    | ngl |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           pp512 |        109.59 ± 0.15 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           tg128 |          7.83 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           pp512 |        114.83 ± 0.23 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           tg128 |          7.79 ± 0.00 |

### 5.2 Vulkan 自编译
| model                          |       size |     params | backend    | ngl |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           pp512 |        109.52 ± 0.19 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           tg128 |          7.83 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           pp512 |        114.93 ± 0.16 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           tg128 |          7.79 ± 0.00 |

### 5.3 SYCL 逐层切分
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |   1 |           pp512 |        428.01 ± 5.12 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |   1 |           tg128 |          9.76 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL (xe)  | 999 |   1 |           tg128 |         13.90 ± 0.06 |

> **注:** `SYCL (xe)` 行为 2026-09-17 xe 驱动复测，参数与二进制同 5.3 的 i915 行，说明见 [5.3.1](#531-ubatch-对-prefill-吞吐的影响-2026-09-05)。

#### 5.3.1 ubatch 对 prefill 吞吐的影响 (2026-09-05)

复测时把 prompt 加长到 5120 token（pp5120），调大 `--ubatch-size` 显著提升 prefill。注意 n_ubatch 只影响 prefill，且此表与 5.3 原表（pp512）不可直接对比。

| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |  512(默认) |   1 |          pp5120 |        432.73 ± 0.18 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |     1024 |   1 |          pp5120 |        570.16 ± 0.30 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |     2048 |   1 |          pp5120 |        637.37 ± 0.21 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL (xe)  | 999 |     1024 |   1 |          pp5120 |        538.93 ± 0.08 |

> **注:** `SYCL (xe)` 行 prefill 比 i915 的 570.16 低约 5%。

#### 5.3.2 MTP 投机解码 (2026-09-05)

用官方 Qwen3.8-27B MTP 模块（`mtp-Qwen3.8-27B-Q4_0.gguf`，1.3 GiB）作 draft 模型，主+draft 同跑双 A770。llama-bench 不支持投机参数，改用 llama-cli：真实中文推理题作为 prompt（533 字符，~400 token）、ctx 100k（投机时主+draft 各占一份 KV，15 GB 内存下 128k 会报 OUT_OF_HOST_MEMORY）、采样参数与 start 脚本一致、每档 2–3 次取均值。无投机 llama-cli 对照为 9.6 t/s，与 llama-bench tg 的 9.x 一致（两条路径可比）。

| draft 配置 | test | t/s |
|---|---|---|
| 无投机 | tg128 | 9.6 |
| MTP `--spec-draft-n-max 1` | tg128 | 11.3 |
| MTP `--spec-draft-n-max 2` | tg128 | 12.8 |
| MTP `--spec-draft-n-max 3` | tg128 | 14.2 |
| MTP `--spec-draft-n-max 4` | tg128 | 14.0 |
| MTP `--spec-draft-n-max 3` (xe) | tg128 | 19.6 |

> **注:** 选型 `--spec-draft-n-max 3`（官方推荐，与 4 打平），长度 1 的增益被 draft 推理开销摊薄。draft 模型已全 offload（`-ngld 999`）。测试入口: `make bench sycl-mtp`
>
> `(xe)` 行为 2026-09-17 复测，`--spec-draft-n-max 3` 连跑 4 次均值 18.7 / 23.1 / 15.8 / 20.6 t/s，离散度大。

### 5.4 SYCL 张量并行
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 | tensor |   1 |           pp512 |        325.07 ± 3.01 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 | tensor |   1 |           tg128 |          3.99 ± 0.01 |

---

## 6. 使用说明

```sh
make                     # 列出全部命令
make build               # 编译 SYCL 后端
make bench               # 串行跑全部五种
make bench sycl          # 单种: vulkan-official|vulkan|sycl|sycl-tensor|sycl-mtp
make start               # 默认 mtp: MTP 投机, ctx 90k; llama-server, OpenAI 兼容 API, 127.0.0.1:8080
make start base          # 关投机: 单 KV, ctx 120k (上下文更长, decode 慢约 1/3)
make devices             # 列出 llama.cpp 可见设备
curl http://127.0.0.1:8080/v1/chat/completions
```

两种启动模式:

| 模式 | ctx | KV | decode 速度 |
|---|---|---|---|
| `mtp`（默认） | 90000 | 主 + draft 双份 | 约 +48%（见 [5.3.2](#532-mtp-投机解码-2026-09-05)） |
| `base` | 122880 | 单份 | 基线 (9.6 t/s) |

> MTP 模式 draft 与主模型各占一份 KV，128k 在 15 GB 主机内存下会 `OUT_OF_HOST_MEMORY`，故 90k；关掉投机后单 KV 可放到 120k。

### GPU 驱动切换 i915 / xe

A770 由 i915 驱动（内核默认），也可切到新的 xe，切换只改引导参数，重启生效：

```sh
make driver-status       # 当前驱动 + 引导参数
make driver-xe           # 切 xe (sudo grubby + dracut -f)
make driver-i915         # 切回 i915
```

### 编译

```sh
make build                # SYCL: 依赖 oneAPI, 含 Level Zero 头文件检查
                          # sudo dnf install intel-oneapi-toolkit oneapi-level-zero-devel

# Vulkan (依赖 vulkan-headers glslc spirv-headers-devel vulkan-loader-devel)
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j 24 --target llama-bench
```

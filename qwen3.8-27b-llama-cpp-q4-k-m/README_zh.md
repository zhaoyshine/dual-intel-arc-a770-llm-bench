# Qwen3.8 27B 双 A770 基准测试

[English](README.md)

家人们，用你发财的小手点一点右上角的星星⭐吧

如无意外，本项目的 start 配置是运行速度、输出质量与 context 长度的最佳平衡；若后续无大变更，则不再继续测试。

## 目录

1. [测试环境](#1-测试环境)
2. [软件版本](#2-软件版本)
3. [测试配置](#3-测试配置)
4. [测试方法](#4-测试方法)
5. [测试结果](#5-测试结果)
6. [使用说明](#6-使用说明)
7. [历史测试结果](#7-历史测试结果)

---

## 1. 测试环境

| 项目 | 值 |
|---|---|
| 操作系统 | Fedora Linux 44 (KDE Plasma Desktop Edition) |
| 内核 | 7.2.11-200.fc44.x86_64 |
| CPU | AMD Ryzen 9 5900X, 12C/24T |
| 内存 | 15 GiB |
| GPU 1 | Intel Arc A770 16GB (DG2) — level_zero:0 / Vulkan1 |
| GPU 2 | Intel Arc A770 16GB (DG2) — level_zero:1 / Vulkan2 |
| GPU 3 (显示) | NVIDIA GeForce GT 1030 2GB — Vulkan0 |
| 拆分卡 | Intel PCIe switch (4fa0)，两张 A770 链路 PCIe 3.0 x8 |
| 显卡驱动 | intel-level-zero 26.35.39758, intel-opencl 26.35.39758, vulkan-loader 1.4.341, mesa 26.2.4 |
| 内核驱动 | xe (`xe.force_probe=56a0`)，两张 A770 |

---

## 2. 软件版本

| 组件 | 版本 | 备注 |
|---|---|---|
| llama.cpp 源码 | commit `5f5acfe2`（上游 master, 2026-10-22） | |
| 自编译 SYCL 版 | `5f5acfe2`, build 1227, `build-sycl/bin/` | `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON` |
| 自编译 Vulkan 版 | `5f5acfe2`, build 1227, `build-vulkan/bin/` | `-DGGML_VULKAN=ON`，系统 GCC |
| Intel oneAPI | 2026.1.1 (2026.1.1.20260724) | `intel-oneapi-toolkit` via yum |
| DPC++ 编译器 | icpx 2026.1.1 | 编译 SYCL 后端 |
| 模型 | Qwen3.8-27B-UD-Q4_K_M.gguf, 15.32 GiB, 27.32 B params | unsloth GGUF |

---

## 3. 测试配置

| 配置 | 二进制 (build) | Split 模式 | 设备选择 |
|---|---|---|---|
| vulkan | `build-vulkan/bin/llama-bench` (5f5acfe2) | layer | `--device Vulkan1,Vulkan2` |
| sycl | `build-sycl/bin/llama-bench` (5f5acfe2) | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;level_zero:1` |
| sycl-tensor | `build-sycl/bin/llama-bench` (5f5acfe2) | tensor | `ONEAPI_DEVICE_SELECTOR=level_zero:0;level_zero:1` |
| sycl-mtp | `build-sycl/bin/llama-cli` (5f5acfe2) + MTP draft | layer | draft 固定 `SYCL0`，主模型 `-ts 0.48,0.52` |
| sycl-tensor-mtp | `build-sycl/bin/llama-cli` (5f5acfe2) + MTP draft | tensor | draft 固定 `SYCL0`，主模型 `-ts 0.48,0.52` |

SYCL 与 Vulkan 均编自同一 commit（`5f5acfe2`）。

---

## 4. 测试方法

各配置参数一致，每档重复 3 次取平均；Vulkan 按卡分行输出（双卡各自独立测量），SYCL 为双卡合并吞吐。

```sh
llama-bench -m <model> -ngl 999 --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --batch-size 2048 --ubatch-size 1024 -p 2048 -n 128 -r 3
```

llama-bench 不支持投机参数，sycl-mtp 与 sycl-tensor-mtp 改用 llama-cli 测，采样参数与 start 脚本一致，见 [5.4](#54-sycl--mtp-投机解码)。

---

## 5. 测试结果

### 5.1 Vulkan 自编译（按卡分行）
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |  fa | dev     |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | --: | ------- | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 | pp2048 | 180.08 ± 20.57 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 |  tg128 |   7.93 ± 0.29 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 | pp2048 | 173.75 ± 26.41 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 |  tg128 |   7.97 ± 0.41 |

### 5.2 SYCL 逐层切分（双卡合并）
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |  fa | dev  |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | --: | ---- | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 |   1 | 双卡 | pp2048 | 608.06 ± 1.03 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 |   1 | 双卡 |  tg128 |  15.68 ± 0.01 |

### 5.3 SYCL 张量并行（双卡合并，链路 PCIe 3.0 x8）
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |     sm |  fa | dev  |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | -----: | --: | ---- | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 | tensor |   1 | 双卡 | pp2048 | 585.48 ± 1.53 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 | tensor |   1 | 双卡 |  tg128 |  19.28 ± 0.12 |

### 5.4 SYCL + MTP 投机解码

draft 模型：官方 Qwen3.8-27B MTP 模块（`mtp-Qwen3.8-27B-Q4_0.gguf`，1.3 GiB），固定 GPU 0；主模型按 0.48/0.52 切分；`--spec-draft-n-max 3`。prompt：中文推理题（533 字符，约 400 token）；ctx 200k；`--reasoning-effort medium`；每种 split 各跑一次。

| split | prompt t/s | generation t/s |
| --- | --: | --: |
| layer | 87.0 | 26.6 |
| tensor | 173.7 | 32.4 |

测试入口: `make bench sycl-mtp` / `make bench sycl-tensor-mtp`

---

## 6. 使用说明

```sh
make                     # 列出全部命令
make build sycl          # 编译 SYCL 后端
make build vulkan        # 编译 Vulkan 对照
make bench               # 串行运行全部五种配置
make bench sycl          # 指定配置: vulkan|sycl|sycl-tensor|sycl-mtp|sycl-tensor-mtp
make start               # 档位 low (默认); MTP 投机解码; llama-server, OpenAI 兼容 API, 127.0.0.1:8080
make start high          # 思考档位: low (默认) | medium | high
make devices             # 列出 llama.cpp 可见设备
curl http://127.0.0.1:8080/v1/chat/completions
```

三种思考档位; MTP 投机解码常开:

| 档位 | `--reasoning-effort` | 思考 |
|---|---|---|
| `low`（默认） | `low` | 简短 |
| `medium` | `medium` | 均衡 |
| `high` | `high` → `xhigh` | 最深 |

> MTP 将 draft 模型固定在 GPU 0，主模型按 0.48/0.52 切分；KV 类型 q8_0，batch 2048，ubatch 1024。

### GPU 驱动切换 i915 / xe

A770 内核默认绑定 i915 驱动，本机跑在 xe 上；切换仅修改引导参数，重启生效：

```sh
make driver-status       # 当前驱动 + 引导参数
make driver-xe           # 切 xe (sudo grubby + dracut -f，重启生效)
make driver-i915         # 切回 i915 (重启生效)
```

### 编译

```sh
make build sycl           # 编译 SYCL 后端
make build vulkan         # 编译 Vulkan 对照
                          # SYCL 依赖: oneAPI 及 Level Zero 头文件
                          #   sudo dnf install intel-oneapi-toolkit oneapi-level-zero-devel
                          # Vulkan 依赖: vulkan-headers glslc spirv-headers-devel vulkan-loader-devel
```

---

## 7. 历史测试结果

历史各轮的数字，留档。

| 日期 | 变化 | 结果 |
|---|---|---|
| 2026-08-23 | 初测：i915, f16 KV, pp512 | SYCL layer pp512 428.01 / tg128 9.76；Vulkan pp512 109.59 与 114.83 / tg128 7.83 与 7.79；tensor split pp512 325.07 / tg128 3.99；官方预编译与自编译一致 |
| 2026-09-05 | prompt 加长至 pp5120，扫 `--ubatch-size` | ubatch 512 / 1024 / 2048 分别为 432.73 / 570.16 / 637.37 t/s |
| 2026-09-05 | 扫 MTP `--spec-draft-n-max` | 无投机 9.6 t/s；draft 长度 1 / 2 / 3 / 4 分别为 11.3 / 12.8 / 14.2 / 14.0 |
| 2026-09-17 | 内核驱动 i915 切 xe | SYCL layer tg128 9.76 → 13.90；MTP draft 长度 3 由 14.2 → 19.6（四次连跑 18.7 / 23.1 / 15.8 / 20.6） |
| 2026-10-19 | 重编 `1ae3092`（b1148，master + 3 本地 SYCL commit） | 官方预编译 Vulkan pp5120 73.87 与 76.79 / tg256 8.77 与 8.75；自编译 Vulkan pp5120 109.13 与 112.91 / tg128 11.06 与 10.89；SYCL layer pp5120 543.09 / tg256 14.44；tensor pp5120 589.45 / tg256 16.16；MTP layer 160.9 / 25.4，tensor 159.6 / 30.4 |
| 2026-10-22 | 重编 `5f5acfe2`（b1227，上游 master）；bench 改为 pp2048/tg128，移除官方预编译 | Vulkan pp2048 180.08 与 173.75 / tg128 7.93 与 7.97；SYCL layer pp2048 608.06 / tg128 15.68；tensor pp2048 585.48 / tg128 19.28；MTP layer 87.0 / 26.6，tensor 173.7 / 32.4 |

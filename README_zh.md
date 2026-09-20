# Qwen3.8 27B 双 A770 基准测试

[English](README.md)

家人们，用你发财的小手点一点右上角的星星⭐吧

如无意外，本项目的配置是运行速度、输出质量与 context 长度的最佳平衡；若后续无大变更，则不再继续测试。

- 运行速度：SYCL layer split + MTP 投机解码，prefill 575.77 t/s，decode 约 25 t/s
- 输出质量：q8_0 KV cache，相比 f16 KV 的质量损失可忽略
- context 长度：180k token，够一到两个小任务跑完整上下文

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
| 内核 | 7.1.13-200.fc44.x86_64 |
| CPU | AMD Ryzen 9 5900X, 12C/24T |
| 内存 | 15 GiB |
| GPU 1 | Intel Arc A770 16GB (DG2) — level_zero:0 / Vulkan1 |
| GPU 2 | Intel Arc A770 16GB (DG2) — level_zero:1 / Vulkan2 |
| GPU 3 (显示) | NVIDIA GeForce GT 1030 2GB — Vulkan0 |
| 拆分卡 | Intel PCIe switch (Device 4fa0, 07:00.0)，将 PCIe 4.0 x16 拆分为 x8+x8 |
| 显卡驱动 | intel-level-zero 26.22.38646, intel-opencl 26.22.38646, vulkan-loader 1.4.341, mesa 26.1.6 |
| 内核驱动 | xe (`xe.force_probe=56a0`)，两张 A770 |

---

## 2. 软件版本

| 组件 | 版本 | 备注 |
|---|---|---|
| llama.cpp 源码 | commit `b23701f` (master, 2026-09-20) | `git clone --depth 1` |
| 官方预编译 | build 11046 (`60081bb2b`)，`~/.local/bin/llama` | Vulkan 对照基准 |
| 自编译 SYCL 版 | `b23701f`, build 457, `build-sycl/bin/` | `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON` |
| 自编译 Vulkan 版 | `b23701f`, build 457, `build-vulkan/bin/` | `-DGGML_VULKAN=ON`，系统 GCC |
| Intel oneAPI | 2026.1.1 (2026.1.1.20260724) | `intel-oneapi-toolkit` via yum |
| DPC++ 编译器 | icpx 2026.1.1 | 编译 SYCL 后端 |
| 模型 | Qwen3.8-27B-UD-Q4_K_M.gguf, 15.32 GiB, 27.32 B params | unsloth GGUF |

---

## 3. 测试配置

| 配置 | 二进制 (build) | Split 模式 | 设备选择 |
|---|---|---|---|
| vulkan-official | `~/.local/bin/llama` (11046) | layer | `--device Vulkan1,Vulkan2` |
| vulkan | `build-vulkan/bin/llama-bench` (b23701f) | layer | `--device Vulkan1,Vulkan2` |
| sycl | `build-sycl/bin/llama-bench` (b23701f) | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| sycl-tensor | `build-sycl/bin/llama-bench` (b23701f) | tensor | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| sycl-mtp | `build-sycl/bin/llama-cli` (b23701f) + MTP draft | layer | draft 固定 `SYCL0`，主模型 `-ts 0.47,0.53` |

SYCL 与 Vulkan 均编自同一 commit（`b23701f`）。

---

## 4. 测试方法

各配置参数一致，每档重复 3 次取平均；Vulkan 按卡分行输出（双卡各自独立测量），SYCL 为双卡合并吞吐。

```sh
llama-bench -m <model> -ngl 999 --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --batch-size 2048 --ubatch-size 1024 -p 5120 -n 128 -r 3
```

llama-bench 不支持投机参数，sycl-mtp 改用 llama-cli 测，采样参数与 start 脚本一致，见 [5.5](#55-sycl--mtp-投机解码)。

---

## 5. 测试结果

### 5.1 Vulkan 官方预编译（按卡分行）
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |  fa | dev     |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | --: | ------- | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 | pp5120 | 109.19 ± 0.24 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 |  tg128 |  11.05 ± 0.00 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 | pp5120 | 112.86 ± 1.26 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 |  tg128 |  10.88 ± 0.06 |

### 5.2 Vulkan 自编译（按卡分行）
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |  fa | dev     |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | --: | ------- | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 | pp5120 | 109.13 ± 0.25 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 |  tg128 |  11.06 ± 0.00 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 | pp5120 | 112.91 ± 1.27 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 |  tg128 |  10.89 ± 0.06 |

### 5.3 SYCL 逐层切分（双卡合并）
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |  fa | dev  |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | --: | ---- | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 |   1 | 双卡 | pp5120 | 575.77 ± 0.13 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 |   1 | 双卡 |  tg128 |  14.21 ± 0.00 |

### 5.4 SYCL 张量并行（双卡合并）
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |     sm |  fa | dev  |   test |           t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | -----: | --: | ---- | -----: | ------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 | tensor |   1 | 双卡 | pp5120 | 48.34 ± 0.01 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 | tensor |   1 | 双卡 |  tg128 | 10.86 ± 0.00 |

### 5.5 SYCL + MTP 投机解码

draft 模型：官方 Qwen3.8-27B MTP 模块（`mtp-Qwen3.8-27B-Q4_0.gguf`，1.3 GiB），固定 GPU 0；主模型按 0.47/0.53 切分；`--spec-draft-n-max 3`。prompt：中文推理题（533 字符，约 400 token）；ctx 180k；`--reasoning-effort medium`；连跑五次。

| 轮次 | prompt t/s | generation t/s |
| --- | --: | --: |
| 1 | 207.0 | 25.0 |
| 2 | 201.9 | 23.7 |
| 3 | 171.8 | 23.5 |
| 4 | 206.2 | 27.0 |
| 5 | 206.9 | 26.2 |

测试入口: `make bench sycl-mtp`

---

## 6. 使用说明

```sh
make                     # 列出全部命令
make build               # 编译两个后端 (SYCL + Vulkan)
make bench               # 串行运行全部五种配置
make bench sycl          # 指定配置: vulkan-official|vulkan|sycl|sycl-tensor|sycl-mtp
make start               # 默认 mtp: MTP 投机解码, ctx 180k; llama-server, OpenAI 兼容 API, 127.0.0.1:8080
make start base          # 关闭投机: 单份 KV, ctx 180k
make devices             # 列出 llama.cpp 可见设备
curl http://127.0.0.1:8080/v1/chat/completions
```

两种启动模式:

| 模式 | ctx | KV | decode 速度 |
|---|---|---|---|
| `mtp`（默认） | 180000 | 主 + draft 双份 | 约 1.8 倍于 `base`（见 [5.5](#55-sycl--mtp-投机解码)） |
| `base` | 180000 | 单份 | SYCL layer split 基线 (14.21 t/s) |

> MTP 模式将 draft 模型固定在 GPU 0，主模型按 0.47/0.53 切分；KV 类型 q8_0，batch 2048，ubatch 1024。

### GPU 驱动切换 i915 / xe

A770 内核默认绑定 i915 驱动，本机跑在 xe 上；切换仅修改引导参数，重启生效：

```sh
make driver-status       # 当前驱动 + 引导参数
make driver-xe           # 切 xe (sudo grubby + dracut -f，重启生效)
make driver-i915         # 切回 i915 (重启生效)
```

### 编译

```sh
make build                # 编译两个后端
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

# Qwen3.8 27B 双 A770 vLLM XPU 基准测试

[English](README.md)

家人们，用你发财的小手点一点右上角的星星⭐吧

vLLM 的 XPU 内核只覆盖 Xe2（B580/B60、Lunar Lake）和 Xe3（Panther Lake），A770 是 Xe-HPG（DG2），直接跑不起来：`vllm-xpu-kernels` 的 SYCL chunk GDN 内核在 DG2 上会触发 IGC 的 DPAS 代码生成崩溃，本仓库的插件把 Qwen GDN 层改走 FLA 的 Triton 实现。

- 速度：prefill 114.43 t/s（pp1024，TP2）；decode 11.29 t/s（tg32，PP2）
- 采样：top-k/top-p 正常可用，融合的 XPU 采样内核关闭（`VLLM_XPU_USE_SAMPLER_KERNEL=0`）
- 不支持：图像/视频（视觉塔）、FlashAttention 后端

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
| GPU 1 | Intel Arc A770 16GB (DG2) — level_zero:0 |
| GPU 2 | Intel Arc A770 16GB (DG2) — level_zero:1 |
| GPU 3（显示） | NVIDIA GeForce GT 1030 2GB |
| 显卡驱动 | intel-level-zero 26.22.38646，intel-opencl 26.22.38646，intel-igc 2.36.3 |
| 内核驱动 | xe（`xe.force_probe=56a0`），两张 A770 均走此驱动 |

---

## 2. 软件版本

| 组件 | 版本 | 备注 |
|---|---|---|
| vLLM | 0.30.1rc1.dev150+g25b0add7b | 源码编译，`VLLM_TARGET_DEVICE=xpu`，editable 装在 `~/workspace/ai/vllm` |
| PyTorch | 2.14.0+xpu | 来自 `download.pytorch.org/whl/xpu` |
| triton | 3.8.0+xpu | shim 包，实际驱动是 `triton_xpu` |
| vllm-xpu-kernels | 0.1.15.4 | 预编译 SYCL 内核（Xe2/Xe3） |
| oneAPI DLE | intel-sycl-rt 2026.1.1，dnnl/umf 来自 oneAPI 2026.1 | oneCCL 2022.1.2 |
| 模型 | cyankiwi/Qwen3.8-27B-AWQ-INT4 | compressed-tensors W4A16，group size 32，约 20 GB |

---

## 3. 测试配置

两个配置，对应两种并行方式。每次运行各自加载一份约 20 GB 权重，独占两张卡。

| 配置 | 并行方式 | 引擎参数 |
|---|---|---|
| `tp` | 张量并行 2 | `--tensor-parallel-size 2` |
| `pp` | 流水线并行 2 | `--pipeline-parallel-size 2 --tensor-parallel-size 1` |

两者共用：

```sh
--model cyankiwi/Qwen3.8-27B-AWQ-INT4 --max-model-len 4096 --dtype float16 \
  --attention-backend TRITON_ATTN --limit-mm-per-prompt '{"image":0,"video":0}' \
  --gpu-memory-utilization 0.95 --enforce-eager
```

`--max-model-len 4096` 够跑下面两个用例；`--enforce-eager` 让内核 autotune 时间可控。`--gpu-memory-utilization 0.95` 的前提是两张卡没被别的进程占用。

---

## 4. 测试方法

数据全部来自 `vllm bench latency`，由 `make bench` 驱动；每个配置两个用例，batch 1，预热 1 次后跑 3 次迭代，报告平均延迟：

```sh
vllm bench latency --input-len 1024 --output-len 1 --batch-size 1 \
  --num-iters 3 --num-iters-warmup 1     # pp1024; tg32 为 --input-len 1 --output-len 32
```

每个用例独占一个进程组，用例之间休息 8 s。两档上下文都很短，不能和隔壁 llama.cpp 的长上下文数据（pp5120 / tg256）直接对比。

---

## 5. 测试结果

| 配置 | pp1024（输入 1024 / 输出 1） | tg32（输入 1 / 输出 32） |
|---|---|---|
| TP2 | 8.948 s — 114.43 t/s | 3.326 s — 9.62 t/s |
| PP2 | 14.052 s — 72.87 t/s | 2.833 s — 11.29 t/s |

---

## 6. 使用说明

```sh
make              # 列出全部命令
make build        # 建 venv、装 XPU 依赖、编译 vLLM、装两个插件
make bench        # 跑两个配置，共 4 次测试
make bench tp     # 指定配置：tp|pp
make devices      # 列出 vLLM 可见的 XPU 设备
```

两个插件文件夹是本方案针对 A770 的全部改动，其余都是原生 vLLM XPU。

---

## 7. 历史测试结果

历史各轮的数字，留档。

| 日期 | 变化 | 结果 |
|---|---|---|
| 2026-09-26 | 基线 | TP2 pp1024 114.43 / tg32 9.62；PP2 pp1024 72.87 / tg32 11.29 |
| 2026-09-26 | KV cache `fp8` | TP2 pp1024 110.70 / tg32 9.21；PP2 pp1024 71.54 / tg32 11.21 |
| 2026-09-26 | KV cache `int8_per_token_head` | TP2 pp1024 106.48 / tg32 8.95；PP2 pp1024 68.69 / tg32 11.24 |
| 2026-09-26 | 权重连续布局 | TP2 pp1024 115.16 / tg32 9.61；PP2 pp1024 72.83 / tg32 11.29 |
| 2026-09-26 | 关 triton tensor descriptor | TP2 pp1024 113.53 / tg32 9.59；PP2 pp1024 72.09 / tg32 11.26 |
| 2026-09-26 | 关 PP 微批 | PP2 pp1024 72.64 / tg32 11.04 |
| 2026-09-26 | block-size 128 | TP2 pp1024 115.21 / tg32 9.44；PP2 pp1024 72.85 / tg32 11.28 |
| 2026-09-26 | 首轮：服务端单请求，`--max-model-len 32768`（已废弃） | 300 token 补全（含 prefill）9.6 t/s；KV cache 56,275 token；启动到就绪约 3.5 分钟 |

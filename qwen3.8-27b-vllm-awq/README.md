# Qwen3.8 27B Dual A770 vLLM XPU Benchmark Report

[中文版 (Simplified Chinese)](README_zh.md)

Friends, please tap the star ⭐ in the top-right corner!

vLLM's XPU kernels target Xe2 (B580/B60, Lunar Lake) and Xe3 (Panther Lake), but the A770 is Xe-HPG (DG2), so it will not run as-is: the SYCL chunk GDN kernel in `vllm-xpu-kernels` crashes IGC's DPAS codegen on DG2, and one out-of-tree plugin routes the Qwen GDN layer to the FLA Triton implementation instead.

- Speed: prefill 114.43 t/s (pp1024, TP2); decode 11.29 t/s (tg32, PP2)
- Sampler: top-k/top-p work normally; the fused XPU sampler kernel is off (`VLLM_XPU_USE_SAMPLER_KERNEL=0`)
- Not supported: images/video (vision tower), the FlashAttention backend

## Contents

1. [System Environment](#1-system-environment)
2. [Software Versions](#2-software-versions)
3. [Benchmark Configs](#3-benchmark-configs)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Usage](#6-usage)
7. [Historical Results](#7-historical-results)

---

## 1. System Environment

| Item | Value |
|---|---|
| OS | Fedora Linux 44 (KDE Plasma Desktop Edition) |
| Kernel | 7.1.13-200.fc44.x86_64 |
| CPU | AMD Ryzen 9 5900X, 12C/24T |
| RAM | 15 GiB |
| GPU 1 | Intel Arc A770 16GB (DG2) — level_zero:0 |
| GPU 2 | Intel Arc A770 16GB (DG2) — level_zero:1 |
| GPU 3 (display) | NVIDIA GeForce GT 1030 2GB |
| GPU drivers | intel-level-zero 26.22.38646, intel-opencl 26.22.38646, intel-igc 2.36.3 |
| Kernel driver | xe (`xe.force_probe=56a0`) on both A770s |

---

## 2. Software Versions

| Component | Version | Note |
|---|---|---|
| vLLM | 0.30.1rc1.dev150+g25b0add7b | source build, `VLLM_TARGET_DEVICE=xpu`, editable install at `~/workspace/ai/vllm` |
| PyTorch | 2.14.0+xpu | from `download.pytorch.org/whl/xpu` |
| triton | 3.8.0+xpu | shim; the real driver is `triton_xpu` |
| vllm-xpu-kernels | 0.1.15.4 | prebuilt SYCL kernels (Xe2/Xe3) |
| oneAPI DLE | intel-sycl-rt 2026.1.1, dnnl/umf from oneAPI 2026.1 | oneCCL 2022.1.2 |
| Model | cyankiwi/Qwen3.8-27B-AWQ-INT4 | compressed-tensors W4A16, group size 32, ~20 GB |

---

## 3. Benchmark Configs

Two configs, one per parallel strategy. Every run loads its own ~20 GB copy of the weights and occupies both cards.

| Config | Parallelism | Engine flags |
|---|---|---|
| `tp` | tensor parallel 2 | `--tensor-parallel-size 2` |
| `pp` | pipeline parallel 2 | `--pipeline-parallel-size 2 --tensor-parallel-size 1` |

Shared by both:

```sh
--model cyankiwi/Qwen3.8-27B-AWQ-INT4 --max-model-len 4096 --dtype float16 \
  --attention-backend TRITON_ATTN --limit-mm-per-prompt '{"image":0,"video":0}' \
  --gpu-memory-utilization 0.95 --enforce-eager
```

`--max-model-len 4096` covers the two bench cases; `--enforce-eager` keeps kernel autotune time in check; `--gpu-memory-utilization 0.95` assumes both cards are otherwise idle.

---

## 4. Methodology

All numbers come from `vllm bench latency`, driven by `make bench`; each config gets two cases, batch size 1, 1 warmup followed by 3 iterations, reporting the average latency:

```sh
vllm bench latency --input-len 1024 --output-len 1 --batch-size 1 \
  --num-iters 3 --num-iters-warmup 1     # pp1024; tg32 is --input-len 1 --output-len 32
```

Each case runs in its own process group, with 8 s of rest between cases. Both contexts are short, so these figures are not directly comparable with the longer-context llama.cpp runs in the sibling folder (pp5120 / tg256).

---

## 5. Results

| Config | pp1024 (1024 in / 1 out) | tg32 (1 in / 32 out) |
|---|---|---|
| TP2 | 8.948 s — 114.43 t/s | 3.326 s — 9.62 t/s |
| PP2 | 14.052 s — 72.87 t/s | 2.833 s — 11.29 t/s |

---

## 6. Usage

```sh
make              # list all commands
make build        # create venv, install XPU deps, build vLLM, install both plugins
make bench        # both configs, 4 runs
make bench tp     # a single config: tp|pp
make devices      # list the XPU devices vLLM sees
```

The two plugin folders are the A770-specific parts of this setup; everything else is stock vLLM XPU.

---

## 7. Historical Results

Numbers from the earlier rounds, kept for the record.

| Date | What changed | Effect |
|---|---|---|
| 2026-09-26 | baseline | TP2 pp1024 114.43 / tg32 9.62; PP2 pp1024 72.87 / tg32 11.29 |
| 2026-09-26 | KV cache `fp8` | TP2 pp1024 110.70 / tg32 9.21; PP2 pp1024 71.54 / tg32 11.21 |
| 2026-09-26 | KV cache `int8_per_token_head` | TP2 pp1024 106.48 / tg32 8.95; PP2 pp1024 68.69 / tg32 11.24 |
| 2026-09-26 | contiguous weights | TP2 pp1024 115.16 / tg32 9.61; PP2 pp1024 72.83 / tg32 11.29 |
| 2026-09-26 | triton tensor descriptor off | TP2 pp1024 113.53 / tg32 9.59; PP2 pp1024 72.09 / tg32 11.26 |
| 2026-09-26 | PP microbatch off | PP2 pp1024 72.64 / tg32 11.04 |
| 2026-09-26 | block-size 128 | TP2 pp1024 115.21 / tg32 9.44; PP2 pp1024 72.85 / tg32 11.28 |
| 2026-09-26 | first round: server, single request, `--max-model-len 32768` (dropped) | 300-token completion incl. prefill 9.6 t/s; KV cache 56,275 tokens; startup to ready ~3.5 min |

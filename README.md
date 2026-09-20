# Qwen3.8 27B Dual A770 Benchmark Report

[中文版 (Simplified Chinese)](README_zh.md)

Friends, please tap the star ⭐ in the top-right corner!

Barring surprises, the configuration in this repo is the best balance of runtime speed, output quality, and context length; absent major changes, no further benchmarking is planned.

- Speed: SYCL layer split prefill 490.21 t/s (pp5120), decode 14.19 t/s (tg256); MTP speculative decoding at 200k context measured 204.0 t/s prompt / 23.2 t/s generation
- Quality: q8_0 KV cache, negligible quality loss versus f16 KV
- Context: 200k tokens, enough for one or two small tasks end to end

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
| GPU 1 | Intel Arc A770 16GB (DG2) — level_zero:0 / Vulkan1 |
| GPU 2 | Intel Arc A770 16GB (DG2) — level_zero:1 / Vulkan2 |
| GPU 3 (display) | NVIDIA GeForce GT 1030 2GB — Vulkan0 |
| bifurcation card | Intel PCIe switch (Device 4fa0, 07:00.0) splits PCIe 4.0 x16 into x8+x8 |
| GPU Drivers | intel-level-zero 26.22.38646, intel-opencl 26.22.38646, vulkan-loader 1.4.341, mesa 26.1.6 |
| Kernel driver | xe (`xe.force_probe=56a0`), both A770s |

---

## 2. Software Versions

| Component | Version | Note |
|---|---|---|
| Source | commit `b23701f` (master, 2026-09-20) | `git clone --depth 1` |
| Official Prebuilt | build 11046 (`60081bb2b`), `~/.local/bin/llama` | Vulkan baseline |
| Self-Built SYCL | `b23701f`, build 457, `build-sycl/bin/` | `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON` |
| Self-Built Vulkan | `b23701f`, build 457, `build-vulkan/bin/` | `-DGGML_VULKAN=ON`, system GCC |
| Intel oneAPI | 2026.1.1 (2026.1.1.20260724) | `intel-oneapi-toolkit` via yum |
| Compiler | icpx 2026.1.1 | builds SYCL backend |
| Model | Qwen3.8-27B-UD-Q4_K_M.gguf, 15.32 GiB, 27.32 B params | unsloth GGUF |

---

## 3. Benchmark Configs

| Config | Binary (build) | Split Mode | Device Select |
|---|---|---|---|
| vulkan-official | `~/.local/bin/llama` (11046) | layer | `--device Vulkan1,Vulkan2` |
| vulkan | `build-vulkan/bin/llama-bench` (b23701f) | layer | `--device Vulkan1,Vulkan2` |
| sycl | `build-sycl/bin/llama-bench` (b23701f) | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| sycl-tensor | `build-sycl/bin/llama-bench` (b23701f) | tensor | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| sycl-mtp | `build-sycl/bin/llama-cli` (b23701f) + MTP draft | layer | draft on `SYCL0`, main model `-ts 0.47,0.53` |

SYCL and Vulkan are built from the same commit (`b23701f`).

---

## 4. Methodology

All configs share identical settings; each row is the average of 3 repeats. Vulkan reports one row per GPU; SYCL reports combined dual-GPU throughput.

```sh
llama-bench -m <model> -ngl 999 --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --batch-size 2048 --ubatch-size 1024 -p 5120 -n 128 -r 3
```

llama-bench has no speculative-decoding flags, so sycl-mtp is measured with llama-cli instead, with sampling parameters identical to the start script; see [5.5](#55-sycl--mtp-speculative-decoding).

---

## 5. Results

### 5.1 Official Prebuilt Vulkan, per-GPU shard
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |  fa | dev     |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | --: | ------- | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 | pp5120 | 109.19 ± 0.24 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 |  tg128 |  11.05 ± 0.00 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 | pp5120 | 112.86 ± 1.26 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 |  tg128 |  10.88 ± 0.06 |

### 5.2 Self-Built Vulkan, per-GPU shard
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |  fa | dev     |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | --: | ------- | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 | pp5120 | 109.13 ± 0.25 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 |  tg128 |  11.06 ± 0.00 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 | pp5120 | 112.91 ± 1.27 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 |  tg128 |  10.89 ± 0.06 |

### 5.3 SYCL layer split, dual-GPU combined
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |  fa |    dev |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | --: | -----: | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 |   1 | both     | pp5120 | 575.77 ± 0.13 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 |   1 | both     |  tg128 |  14.21 ± 0.00 |

### 5.4 SYCL tensor split, dual-GPU combined
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |     sm |  fa |    dev |   test |           t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | -----: | --: | -----: | -----: | ------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 | tensor |   1 | both     | pp5120 | 48.34 ± 0.01 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 | tensor |   1 | both     |  tg128 | 10.86 ± 0.00 |

### 5.5 SYCL + MTP speculative decoding

Draft model: official Qwen3.8-27B MTP module (`mtp-Qwen3.8-27B-Q4_0.gguf`, 1.3 GiB), pinned to GPU 0; main model split 0.47/0.53; `--spec-draft-n-max 3`. Prompt: Chinese reasoning question (533 chars, ~400 tokens); ctx 180k; `--reasoning-effort medium`; five runs.

| run | prompt t/s | generation t/s |
| --- | --: | --: |
| 1 | 207.0 | 25.0 |
| 2 | 201.9 | 23.7 |
| 3 | 171.8 | 23.5 |
| 4 | 206.2 | 27.0 |
| 5 | 206.9 | 26.2 |

Entry point: `make bench sycl-mtp`

---

## 6. Usage

```sh
make                     # list all commands
make build               # build both backends (SYCL + Vulkan)
make bench               # run all five configs sequentially
make bench sycl          # a single config: vulkan-official|vulkan|sycl|sycl-tensor|sycl-mtp
make start               # default mtp: MTP speculative decoding, ctx 180k; llama-server, OpenAI-compatible API, 127.0.0.1:8080
make start base          # speculation off: single KV cache, ctx 180k
make devices             # list devices visible to llama.cpp
curl http://127.0.0.1:8080/v1/chat/completions
```

Two start modes:

| Mode | ctx | KV | decode speed |
|---|---|---|---|
| `mtp` (default) | 180000 | main + draft | ~1.8x `base` (see [5.5](#55-sycl--mtp-speculative-decoding)) |
| `base` | 180000 | single | SYCL layer split baseline (14.21 t/s) |

> MTP mode pins the draft model to GPU 0 and splits the main model 0.47/0.53; the KV type is q8_0, batch 2048, ubatch 1024.

### Switching the GPU kernel driver i915 / xe

The A770 is bound to i915 by default in the kernel and runs on xe here; the switch only modifies boot parameters and takes effect after a reboot:

```sh
make driver-status       # show current driver + boot parameters
make driver-xe           # switch to xe (sudo grubby + dracut -f, reboot)
make driver-i915         # switch back to i915 (reboot)
```

### Build

```sh
make build                # builds both backends
                          # SYCL deps: oneAPI and the Level Zero headers
                          #   sudo dnf install intel-oneapi-toolkit oneapi-level-zero-devel
                          # Vulkan deps: vulkan-headers glslc spirv-headers-devel vulkan-loader-devel
```

---

## 7. Historical Results

Numbers from the earlier rounds, kept for the record.

| Date | What changed | Effect |
|---|---|---|
| 2026-08-23 | initial run: i915, f16 KV, pp512 | SYCL layer split pp512 428.01 / tg128 9.76; Vulkan pp512 109.59 & 114.83 / tg128 7.83 & 7.79; tensor split pp512 325.07 / tg128 3.99; official prebuilt matched the self-build |
| 2026-09-05 | prompt extended to pp5120, `--ubatch-size` swept | 432.73 / 570.16 / 637.37 t/s at ubatch 512 / 1024 / 2048 |
| 2026-09-05 | MTP `--spec-draft-n-max` swept | 9.6 t/s without speculation; 11.3 / 12.8 / 14.2 / 14.0 at draft length 1 / 2 / 3 / 4 |
| 2026-09-17 | kernel driver i915 → xe | SYCL layer tg128 9.76 → 13.90; MTP draft length 3 14.2 → 19.6 (18.7 / 23.1 / 15.8 / 20.6 over four runs) |

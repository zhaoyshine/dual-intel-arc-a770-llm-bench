# Qwen3.8 27B Dual A770 Benchmark Report

[中文版 (Simplified Chinese)](README_zh.md)

Friends, please tap the star ⭐ in the top-right corner!

Barring surprises, the configuration in this repo is the best balance of runtime speed, output quality, and context length; absent major changes, no further benchmarking is planned.

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
| Kernel | 7.2.11-200.fc44.x86_64 |
| CPU | AMD Ryzen 9 5900X, 12C/24T |
| RAM | 15 GiB |
| GPU 1 | Intel Arc A770 16GB (DG2) — level_zero:0 / Vulkan1 |
| GPU 2 | Intel Arc A770 16GB (DG2) — level_zero:1 / Vulkan2 |
| GPU 3 (display) | NVIDIA GeForce GT 1030 2GB — Vulkan0 |
| bifurcation card | Intel PCIe switch (4fa0), both A770s linked at PCIe 3.0 x8 |
| GPU Drivers | intel-level-zero 26.35.39758, intel-opencl 26.35.39758, vulkan-loader 1.4.341, mesa 26.2.4 |
| Kernel driver | xe (`xe.force_probe=56a0`), both A770s |

---

## 2. Software Versions

| Component | Version | Note |
|---|---|---|
| Source | commit `5f5acfe2` (upstream master, 2026-10-22) | |
| Self-Built SYCL | `5f5acfe2`, build 1227, `build-sycl/bin/` | `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON` |
| Self-Built Vulkan | `5f5acfe2`, build 1227, `build-vulkan/bin/` | `-DGGML_VULKAN=ON`, system GCC |
| Intel oneAPI | 2026.1.1 (2026.1.1.20260724) | `intel-oneapi-toolkit` via yum |
| Compiler | icpx 2026.1.1 | builds SYCL backend |
| Model | Qwen3.8-27B-UD-Q4_K_M.gguf, 15.32 GiB, 27.32 B params | unsloth GGUF |

---

## 3. Benchmark Configs

| Config | Binary (build) | Split Mode | Device Select |
|---|---|---|---|
| vulkan | `build-vulkan/bin/llama-bench` (5f5acfe2) | layer | `--device Vulkan1,Vulkan2` |
| sycl | `build-sycl/bin/llama-bench` (5f5acfe2) | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;level_zero:1` |
| sycl-tensor | `build-sycl/bin/llama-bench` (5f5acfe2) | tensor | `ONEAPI_DEVICE_SELECTOR=level_zero:0;level_zero:1` |
| sycl-mtp | `build-sycl/bin/llama-cli` (5f5acfe2) + MTP draft | layer | draft on `SYCL0`, main model `-ts 0.48,0.52` |
| sycl-tensor-mtp | `build-sycl/bin/llama-cli` (5f5acfe2) + MTP draft | tensor | draft on `SYCL0`, main model `-ts 0.48,0.52` |

SYCL and Vulkan are built from the same commit (`5f5acfe2`).

---

## 4. Methodology

All configs share identical settings; each row is the average of 3 repeats. Vulkan reports one row per GPU; SYCL reports combined dual-GPU throughput.

```sh
llama-bench -m <model> -ngl 999 --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --batch-size 2048 --ubatch-size 1024 -p 2048 -n 128 -r 3
```

llama-bench has no speculative-decoding flags, so sycl-mtp and sycl-tensor-mtp are measured with llama-cli instead, with sampling parameters identical to the start script; see [5.4](#54-sycl--mtp-speculative-decoding).

---

## 5. Results

### 5.1 Self-Built Vulkan, per-GPU shard
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |  fa | dev     |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | --: | ------- | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 | pp2048 | 180.08 ± 20.57 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan1 |  tg128 |   7.93 ± 0.29 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 | pp2048 | 173.75 ± 26.41 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | Vulkan  | 999 |     1024 |   q8_0 |   q8_0 |   1 | Vulkan2 |  tg128 |   7.97 ± 0.41 |

### 5.2 SYCL layer split, dual-GPU combined
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |  fa |    dev |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | --: | -----: | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 |   1 | both     | pp2048 | 608.06 ± 1.03 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 |   1 | both     |  tg128 |  15.68 ± 0.01 |

### 5.3 SYCL tensor split, dual-GPU combined (PCIe 3.0 x8 link)
| model                    |       size |     params | backend | ngl | n_ubatch | type_k | type_v |     sm |  fa |    dev |   test |            t/s |
| ------------------------ | ---------: | ---------: | ------- | --: | -------: | -----: | -----: | -----: | --: | -----: | -----: | -------------: |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 | tensor |   1 | both     | pp2048 | 585.48 ± 1.53 |
| qwen35 27B Q4_K - Medium |  15.32 GiB |    27.32 B | SYCL    | 999 |     1024 |   q8_0 |   q8_0 | tensor |   1 | both     |  tg128 |  19.28 ± 0.12 |

### 5.4 SYCL + MTP speculative decoding

Draft model: official Qwen3.8-27B MTP module (`mtp-Qwen3.8-27B-Q4_0.gguf`, 1.3 GiB), pinned to GPU 0; main model split 0.48/0.52; `--spec-draft-n-max 3`. Prompt: Chinese reasoning question (533 chars, ~400 tokens); ctx 200k; `--reasoning-effort medium`; one run per split mode.

| split | prompt t/s | generation t/s |
| --- | --: | --: |
| layer | 87.0 | 26.6 |
| tensor | 173.7 | 32.4 |

Entry point: `make bench sycl-mtp` / `make bench sycl-tensor-mtp`

---

## 6. Usage

```sh
make                     # list all commands
make build sycl          # build SYCL backend
make build vulkan        # build Vulkan baseline
make bench               # run all five configs sequentially
make bench sycl          # a single config: vulkan|sycl|sycl-tensor|sycl-mtp|sycl-tensor-mtp
make start               # effort low (default); MTP speculative decoding; llama-server, OpenAI-compatible API, 127.0.0.1:8080
make start high          # effort levels: low (default) | medium | high
make devices             # list devices visible to llama.cpp
curl http://127.0.0.1:8080/v1/chat/completions
```

Three reasoning-effort levels; MTP speculative decoding is always on:

| Level | `--reasoning-effort` | thinking |
|---|---|---|
| `low` (default) | `low` | brief |
| `medium` | `medium` | balanced |
| `high` | `high` → `xhigh` | deepest |

> MTP pins the draft model to GPU 0 and splits the main model 0.48/0.52; KV type q8_0, batch 2048, ubatch 1024.

### Switching the GPU kernel driver i915 / xe

The A770 is bound to i915 by default in the kernel and runs on xe here; the switch only modifies boot parameters and takes effect after a reboot:

```sh
make driver-status       # show current driver + boot parameters
make driver-xe           # switch to xe (sudo grubby + dracut -f, reboot)
make driver-i915         # switch back to i915 (reboot)
```

### Build

```sh
make build sycl           # build SYCL backend
make build vulkan         # build Vulkan baseline
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
| 2026-10-19 | recompiled on `1ae3092` (b1148, master + 3 local SYCL commits) | official prebuilt Vulkan pp5120 73.87 & 76.79 / tg256 8.77 & 8.75; self-built Vulkan pp5120 109.13 & 112.91 / tg128 11.06 & 10.89; SYCL layer pp5120 543.09 / tg256 14.44; tensor pp5120 589.45 / tg256 16.16; MTP layer 160.9 / 25.4, tensor 159.6 / 30.4 |
| 2026-10-22 | recompiled on `5f5acfe2` (b1227, upstream master); bench moved to pp2048/tg128, official prebuilt dropped from bench | Vulkan pp2048 180.08 & 173.75 / tg128 7.93 & 7.97; SYCL layer pp2048 608.06 / tg128 15.68; tensor pp2048 585.48 / tg128 19.28; MTP layer 87.0 / 26.6, tensor 173.7 / 32.4 |

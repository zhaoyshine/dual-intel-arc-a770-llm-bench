# Qwen3.8 27B Dual A770 Benchmark Report

[中文版 (Simplified Chinese)](README_zh.md)

Friends, please tap the star ⭐ in the top-right corner!

**Date:** Initial run 2026-08-23; ubatch / MTP re-runs added 2026-09-05; xe driver re-run added 2026-09-17

## Contents

1. [System Environment](#1-system-environment)
2. [Software Versions](#2-software-versions)
3. [Benchmark Configs](#3-benchmark-configs)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Usage](#6-usage)

---

## 1. System Environment

| Item | Value |
|---|---|
| OS | Fedora Linux 44 (KDE Plasma Desktop Edition) |
| Kernel | 7.1.13-200.fc44.x86_64 (7.1.7 at initial run) |
| CPU | AMD Ryzen 9 5900X, 12C/24T |
| RAM | 15 GiB |
| GPU 1 | Intel Arc A770 16GB (DG2) — level_zero:0 / Vulkan1 |
| GPU 2 | Intel Arc A770 16GB (DG2) — level_zero:1 / Vulkan2 |
| GPU 3 (display) | NVIDIA GeForce GT 1030 2GB — Vulkan0 |
| bifurcation card | Intel PCIe switch (Device 4fa0, 07:00.0) splits PCIe 4.0 x16 into x8+x8 |
| GPU Drivers | intel-level-zero 26.22.38646, intel-opencl 26.22.38646, vulkan-loader 1.4.341, mesa 26.1.6 |

---

## 2. Software Versions

| Component | Version | Note |
|---|---|---|
| Source | commit `8144f31` (master, 2026-08-23) | `git clone --depth 1` |
| Official Prebuilt | b10217-ddd4ec142 (`~/.local/bin/llama`) | Vulkan baseline |
| Self-Built SYCL | 8144f31, `build-sycl/bin/` | `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON` |
| Self-Built Vulkan | 8144f31, `build-vulkan/bin/` | `-DGGML_VULKAN=ON`, system GCC |
| Intel oneAPI | 2026.1.1 (2026.1.1.20260724) | `intel-oneapi-toolkit` via yum |
| Compiler | icpx 2026.1.1 | builds SYCL backend |
| Model | Qwen3.8-27B-UD-Q4_K_M.gguf, 15.32 GiB, 27.32 B params | unsloth GGUF |

---

## 3. Benchmark Configs

The self-built SYCL and Vulkan builds come from the same source commit, so the comparison is free of version skew; the official prebuilt Vulkan serves as an extra reference.

| Config | Binary | Split Mode | Device Select |
|---|---|---|---|
| vulkan-official | `~/.local/bin/llama` (b10217) | layer | `--device Vulkan1,Vulkan2` |
| vulkan | `build-vulkan/bin/llama-bench` | layer | `--device Vulkan1,Vulkan2` |
| sycl | `build-sycl/bin/llama-bench` | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| sycl-tensor | `build-sycl/bin/llama-bench` | tensor | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |
| sycl-mtp | `build-sycl/bin/llama-cli` + MTP draft | layer | `ONEAPI_DEVICE_SELECTOR=level_zero:0;1` |

---

## 4. Methodology

All configs share identical settings; each row is the average of 3 repeats. Vulkan reports one row per GPU; SYCL reports combined dual-GPU throughput.

```sh
llama-bench -m <model> -ngl 999 --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --ubatch-size 1024 -p 5120 -n 128 -r 3
```

sycl-mtp is measured with llama-cli, with sampling parameters identical to the start script; see [5.3.2](#532-mtp-speculative-decoding-2026-09-05).

---

## 5. Results

SYCL layer split wins across the board: combined prefill 428 t/s vs ~225 t/s for both Vulkan GPUs together (~1.9x), and decode 9.76 t/s vs 7.8 t/s (~25% faster). Tensor split is the slowest — per-layer tensor parallelism on a 27B model costs more in inter-GPU PCIe ring all-reduce than it gains. The official prebuilt and self-built Vulkan results match closely, so version skew is negligible. The slower Vulkan is not a CPU bottleneck: ngl 999 offloads every layer to the GPUs (this machine's CPU tops out around pp 10 / tg 2 t/s on a 27B); the gap comes from the Mesa Vulkan driver being less optimized for Arc than Intel's own SYCL/oneDNN path.

> **Note:** Each Vulkan row is the speed of one GPU processing its own shard, not "only one GPU was used" — both GPUs are active. During prefill the two run in parallel, and the combined throughput is about the sum of the two rows (~225 t/s); during decode they form a pipeline with only one GPU computing at a time, so each row is the overall rate. SYCL reports combined dual-GPU throughput.

### 5.1 Official Prebuilt Vulkan
| model                          |       size |     params | backend    | ngl |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           pp512 |        109.59 ± 0.15 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           tg128 |          7.83 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           pp512 |        114.83 ± 0.23 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           tg128 |          7.79 ± 0.00 |

### 5.2 Self-Built Vulkan
| model                          |       size |     params | backend    | ngl |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           pp512 |        109.52 ± 0.19 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan1      |           tg128 |          7.83 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           pp512 |        114.93 ± 0.16 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | Vulkan     | 999 |   1 | Vulkan2      |           tg128 |          7.79 ± 0.00 |

### 5.3 SYCL layer split
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |   1 |           pp512 |        428.01 ± 5.12 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |   1 |           tg128 |          9.76 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL (xe)  | 999 |   1 |           tg128 |         13.90 ± 0.06 |

> **Note:** The `SYCL (xe)` row is the 2026-09-17 xe driver re-run, with the same params and binary as the i915 rows; see 5.3.1 for details.

#### 5.3.1 Effect of ubatch on prefill throughput (2026-09-05)

Re-measured with a longer 5120-token prompt (pp5120); raising `--ubatch-size` boosts prefill sharply. Note n_ubatch only affects prefill, and these rows are not directly comparable with the pp512 rows in 5.3.

| model                          |       size |     params | backend    | ngl | n_ubatch |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |  512 (default) |   1 |          pp5120 |        432.73 ± 0.18 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |     1024 |   1 |          pp5120 |        570.16 ± 0.30 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 |     2048 |   1 |          pp5120 |        637.37 ± 0.21 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL (xe)  | 999 |     1024 |   1 |          pp5120 |        538.93 ± 0.08 |

> **Note:** The `SYCL (xe)` row's prefill is ~5% below the i915 570.16.

#### 5.3.2 MTP speculative decoding (2026-09-05)

The official Qwen3.8-27B MTP module (`mtp-Qwen3.8-27B-Q4_0.gguf`, 1.3 GiB) was used as the draft model, with main + draft running together on the dual A770s. llama-bench does not support speculative decoding parameters, so llama-cli was used instead: a real Chinese reasoning prompt (533 chars, ~400 tokens), ctx 100k (speculation keeps a KV cache for both main and draft; 128k hits OUT_OF_HOST_MEMORY on a 15 GB host), sampling identical to the start script, 2–3 runs averaged per setting. The no-MTP llama-cli baseline is 9.6 t/s, consistent with llama-bench's 9.x (the two paths are comparable).

| config | test | t/s |
|---|---|---|
| no MTP | tg128 | 9.6 |
| MTP `--spec-draft-n-max 1` | tg128 | 11.3 |
| MTP `--spec-draft-n-max 2` | tg128 | 12.8 |
| MTP `--spec-draft-n-max 3` | tg128 | 14.2 |
| MTP `--spec-draft-n-max 4` | tg128 | 14.0 |
| MTP `--spec-draft-n-max 3` (xe) | tg128 | 19.6 |

> **Note:** `--spec-draft-n-max 3` was selected (officially recommended, tied with 4); the gain at length 1 is diluted by draft inference overhead. The draft model is fully offloaded (`-ngld 999`). Entry point: `make bench sycl-mtp`
>
> The `(xe)` row is the 2026-09-17 re-run: four consecutive runs of `--spec-draft-n-max 3` gave 18.7 / 23.1 / 15.8 / 20.6 t/s, with a wide spread.

### 5.4 SYCL tensor split
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 | tensor |   1 |           pp512 |        325.07 ± 3.01 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | SYCL       | 999 | tensor |   1 |           tg128 |          3.99 ± 0.01 |

---

## 6. Usage

```sh
make                     # list all commands
make build               # build the SYCL backend
make bench               # run all five configs sequentially
make bench sycl          # a single config: vulkan-official|vulkan|sycl|sycl-tensor|sycl-mtp
make start               # default mtp: MTP speculation, ctx 90k; llama-server, OpenAI-compatible API, 127.0.0.1:8080
make start base          # speculation off: single KV, ctx 120k (longer context, decode ~1/3 slower)
make devices             # list devices visible to llama.cpp
curl http://127.0.0.1:8080/v1/chat/completions
```

Two start modes:

| Mode | ctx | KV | decode speed |
|---|---|---|---|
| `mtp` (default) | 90000 | main + draft | ~+48% (see [5.3.2](#532-mtp-speculative-decoding-2026-09-05)) |
| `base` | 122880 | single | baseline (9.6 t/s) |

> In MTP mode the draft and main models each hold a KV cache, so 128k hits `OUT_OF_HOST_MEMORY` on a 15 GB host — hence 90k. With speculation off a single KV cache fits 120k.

### Switching the GPU kernel driver i915 / xe

The A770 runs on i915 (kernel default) and can be switched to xe; switching only changes boot parameters and takes effect after a reboot:

```sh
make driver-status       # show current driver + boot parameters
make driver-xe           # switch to xe (sudo grubby + dracut -f, reboot)
make driver-i915         # switch back to i915 (reboot)
```

### Build

```sh
make build                # SYCL: requires oneAPI, checks the Level Zero headers
                          # sudo dnf install intel-oneapi-toolkit oneapi-level-zero-devel

# Vulkan (requires vulkan-headers glslc spirv-headers-devel vulkan-loader-devel)
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j 24 --target llama-bench
```

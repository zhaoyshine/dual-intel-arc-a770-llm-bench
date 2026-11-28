# Qwen3.8 Flash-Next — c0ven on two Arc A770s

[中文](README_zh.md)

c0ven publishes only Docker images. These scripts repeat its build steps on this Fedora host: sources from
`opt/c0ven` (that checkout is yours to manage), ggml from `opt/llama.cpp` — `build.sh` refuses it unless
`opt/llama.cpp/.c0ven-commit` is the pinned `LLAMA_COMMIT`, llama.cpp@3cf03257 as the image has it — and Python
packages with uv 0.12.22 in `.venv`.

## Use

```bash
./build.sh    # into build/, AOT kernels for acm-g10
./start.sh    # serve on 127.0.0.1:8080
./test.sh     # PP, TG and the decode counters
```

The system needs `oneapi-level-zero-devel`, `intel-ocloc` (26.35 here; c0ven pins 26.40) and python3.14.

## What Xe-HPG needs

`opt/c0ven` carries the Alchemist support (ADR-0082): `ENGINE_AOT_DEVICES` names `acm-g10`, `acm-g11` and `acm-g12`
beside the Battlemage chips, `gemm_hpg.cl`, `prompt_attn_hpg.cl` and `vision_attn_hpg.cl` hold the 8-lane XMX forms,
and `ze::Card::is_hpg()` switches the runtime paths — host-event decode list splits instead of EU polling of host
flags, cache-line flushes across written staging ranges, and CPU-mapped 1D prompt staging instead of 2D arena copies.
Cross-card exchanges fall back to pinned host memory when peer access is absent (ADR-0080).

`start.sh` sets `ENGINE_NO_P2P=1` so the core skips its peer probe: `zeDeviceCanAccessPeer` is true on this Zen 3
host, and the probe's copy-engine BAR-to-BAR transfer is the thing that resets the machine.

`config.sh` passes `--no-vision`: the image encoder has an Alchemist build, and this model directory holds no
`mmproj-*.gguf`.

## Results

IQ3_S at 200K context with `--checkpoints=32` and `--table-cache`, medians and means of five `test.sh` runs:
**PP 395.6 tok/s** on a 15,828-token prompt (mean; first token after 39.88-40.31 s, the five runs spread only
392.6-396.9 tok/s, median 395.9) and **TG 23.0 tok/s** from a short prompt at 1,024 tokens (21.4-25.1 tok/s, median
22.5). Decode counters over those rounds: 1,913 rounds at 3.24 rows and 82.5 ms each, 2.32 tokens produced per
round; 4,282 drafts, 2,532 accepted (59.1%), every draft from DFlash — lookup proposed none; 2,585 draft passes
taking 25.9 s and 608 swap rounds taking 10.1 s.

# Qwen3.8 Flash-Next — c0ven on two Arc A770s

[中文](README_zh.md)

c0ven publishes only Docker images. These scripts repeat its build steps on this Fedora host: sources from
`~/workspace/ai/c0ven` (branch `dg2-a770`), ggml at `opt/llama.cpp` pinned to llama.cpp@3cf03257 as the image has
it, and Python packages with uv 0.12.22 in `.venv`.

## Use

```bash
./build.sh    # into build/, AOT kernels for acm-g10
./start.sh    # serve on 127.0.0.1:8080
./test.sh     # PP, TG and a correctness check; logs under logs/run-<time>/
```

The system needs `level-zero-devel`, `intel-ocloc` (26.35 here; c0ven pins 26.40) and python3.14.

## What Alchemist needed

c0ven targets Battlemage. For Alchemist this branch adds:

- `kernels/gemm_big.cl` and `kernels/pack.cl`: the prompt path's dense products on the XMX units, operands packed in
  the DPAS order, 128 x 128 work-group tiles with a double-buffered SLM stage. The gate takes any N and rows (the
  epilogue masks the tails); a start-up self-check compares the big tile against `gemm_f16` bit for bit and leaves
  the plain kernels on a difference.
- The two cards' exchanges go through host memory, device -> host -> device by the copy engine. A copy into the other
  card's device memory resets this machine (the Zen 3 data fabric's sync flood); Battlemage keeps its kernel stores.
- `config.sh` passes `--checkpoints=0 --no-vision`, which fits IQ3_XXS and its DFlash draft layer into 24 GiB of host
  memory.

## Results

Two A770s at 2,400 MHz, IQ3_XXS, 32K context, DFlash drafting with the engine's defaults (`draft_max` 7,
`draft_min_p` 0.5): **PP 268.6 tok/s** (1,374-token prompt) and **TG 25.0 tok/s** (512 tokens), medians of three
runs from `test.sh`.

# Qwen3.8 Flash-Next — 双 Arc A770 上的 c0ven

[English](README.md)

c0ven 官方只发 Docker 镜像，这里把它的构建步骤搬到本机 Fedora：源码在本目录 `opt/c0ven`（这份 checkout 的版本由你
自己管理），ggml 用 `opt/llama.cpp`，`build.sh` 校验 `opt/llama.cpp/.c0ven-commit` 等于钉死的 `LLAMA_COMMIT`
（llama.cpp@3cf03257，与官方镜像一致）；Python 依赖用 uv 0.12.22 装进本目录 `.venv`。

## 使用

```bash
./build.sh    # 编译，产物在 build/，AOT kernel 目标 acm-g10
./start.sh    # 启动服务 (127.0.0.1:8080)
./test.sh     # 测 PP、TG，并打解码 counters
```

系统依赖：`oneapi-level-zero-devel`、`intel-ocloc`（本机 26.35，c0ven 钉 26.40）、python3.14。

## Xe-HPG 需要的是什么

opt/c0ven 自带 Alchemist 支持（ADR-0082）：`ENGINE_AOT_DEVICES` 默认含 `acm-g10`、`acm-g11`、`acm-g12`，
`gemm_hpg.cl`、`prompt_attn_hpg.cl`、`vision_attn_hpg.cl` 是 8-lane XMX 的版本，`ze::Card::is_hpg()` 切换运行时路径
——decode 命令表按 host event 切分（不再让 EU 轮询 host flag）、写过的 staging 区间做 cache line flush、prompt
staging 走 CPU 映射的 1D 拷贝（不再用 2D arena 拷贝）。跨卡 peer access 不可用时退回 pinned host 内存（ADR-0080）。

`start.sh` 设 `ENGINE_NO_P2P=1`，core 因此跳过 peer 探测：本机 Zen 桥的 `zeDeviceCanAccessPeer` 为真，探测里的
copy-engine BAR-to-BAR 传输会让整机复位。

`config.sh` 传 `--no-vision`：vision 模块有 Alchemist 版本，模型目录里没有 `mmproj-*.gguf`。

## 结果

IQ3_S、200K context、`--checkpoints=32`、`--table-cache`，`test.sh` 五次取平均：**PP 395.6 tok/s**（15828-token 提示，
首 token 39.88-40.31s，五次 392.6-396.9 tok/s，中位数 395.9）、**TG 23.0 tok/s**（短提示、1024 token，五次
21.4-25.1 tok/s，中位数 22.5）。解码 counters：1913 round，每 round 3.24 行、82.5 ms、产出 2.32 token；drafts 4282、
accepted 2532（59.1%），lookup 一次没生成，draft 全来自 DFlash；draft pass 2585 次 25.9s，swap 608 轮 10.1s。

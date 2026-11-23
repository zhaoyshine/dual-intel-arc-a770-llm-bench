# Qwen3.8 Flash-Next — 双 Arc A770 上的 c0ven

[English](README.md)

c0ven 官方只发 Docker 镜像，这里把它的构建步骤搬到本机 Fedora：源码取自 `~/workspace/ai/c0ven`（分支
`dg2-a770`），ggml 固定在 `opt/llama.cpp`（llama.cpp@3cf03257，与官方镜像一致），Python 依赖用 uv 0.12.22
装进本目录 `.venv`。

## 使用

```bash
./build.sh    # 编译，产物在 build/，AOT kernel 目标 acm-g10
./start.sh    # 启动服务 (127.0.0.1:8080)
./test.sh     # 测 PP、TG 和正确性
```

系统依赖：`level-zero-devel`、`intel-ocloc`（本机 26.35，c0ven 钉 26.40）、python3.14。

## 为 Alchemist 做的改动

c0ven 原本只针对 Battlemage，本分支补上 A770（DG2）所需：

- `kernels/gemm_big.cl`、`kernels/pack.cl`：prompt 路径的 dense 矩阵乘走 XMX，操作数按 DPAS 布局打包，
  work-group tile 128×128，SLM 双缓冲。N 和 rows 任意（epilogue 掩掉尾部）；加载时自检拿 big tile 与
  `gemm_f16` 逐位对拍，不一致就退回普通 kernel。
- 两张卡之间的 partial 交换全部经 host 内存，由 copy engine 做 device→host→device。直接写对卡显存会让本机
  整机复位（CPU data fabric sync flood）；Battlemage 仍保留 kernel 直写。
- `config.sh` 传 `--checkpoints=0 --no-vision`，把 IQ3_XXS 和 DFlash 草稿层塞进 24 GiB 主机内存。

## 结果

双 A770、2400 MHz、IQ3_XXS、32K context、DFlash 用引擎默认（`draft_max` 7、`draft_min_p` 0.5）：
**PP 268.6 tok/s**（1374-token 提示）、**TG 25.0 tok/s**（512 token），`test.sh` 三次取中位数。

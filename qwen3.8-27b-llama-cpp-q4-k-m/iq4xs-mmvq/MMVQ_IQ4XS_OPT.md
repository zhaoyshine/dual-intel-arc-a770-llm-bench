# IQ4_XS MMVQ 优化

`patches/0001-sycl-iq4-xs-reorder-mmvq.patch` 对 llama.cpp `7ecb57f7` 的改动。目标：提高单 token decode 时 IQ4_XS GEMV 的权重读带宽。

模型 `Qwen3.8-27B-UD-Q4_K_M.gguf` 里 IQ4_XS 有 117 个张量、4.433 GiB，占 15.334 GiB 的 29%，主要是 ffn_gate / ffn_up / ffn_down。

## 改了什么

瓶颈在 `vec_dot_iq4_xs_q8_1` 的查表：每个权重字要 8 次标量索引字节加载，Intel GPU 没有 byte-permute 指令可以绕。

- **双码查表**：`kvalues_iq4nl_pair[256]` 两个 4-bit 码查一次，返回两个值打包成 16 bit。
- **SoA 重排**：`reorder_qw_iq4_xs` 把 block 拆成「全部 qs 连续在前 + 每块 8 字节 meta（d/scales_h/scales_l）」，按权重 block 解码一次供所有目标列复用。走上游的 `reorder_qw_planar`，分 32 MiB 块搬，不整张量拷一份。
- **接入上游框架**：`quants.hpp` 的 `block_q_t<IQ4_XS>` 给两个偏移，`vecdotq.hpp` 的 `reorder_vec_dot_q_sycl<IQ4_XS>` 给 `load/load_activations/apply/dot`，并声明 `shared_weights`/`shared_activations`。ncols 1–8、融合 GLU、MoE 三条路径全部由 `mul_mat_vec_q_reorder*` 模板生成，本补丁不含 kernel 代码。
- **融合 GLU**：`ggml_sycl_mul_mat_glu_fused` 的 `reorder_pair` 白名单加入 IQ4_XS，ffn_gate/up 在 reorder 布局下走融合 kernel，不再退回两次 MMVQ。
- **配套反量化**：`dequantize_block_iq4_xs_reorder` + `convert.cpp` 的 `is_reordered_src0` 分支，保证回退路径读的是重排后的布局。

## 实测

单卡 `17408×5120` IQ4_XS GEMV 带宽（`iq4_lut_probe.cpp` 隔离内核，`mmvq_probe.cpp` 端到端）：

| 布局 × 查表 | GB/s |
|---|---|
| 交错 136B + 标量查表 | 261 |
| SoA + 标量查表 | 271 |
| SoA + 双码查表 | 380 |
| SoA 不查表（内存上限） | 382 |

补丁前后，按目标列数（K=5120，预热 1 s）：

| ncols | 改前 | 0001 |
|---|---|---|
| 1 | 218.2 | 311.7 |
| 2 | 201.0 | 284.2 |
| 3 | 192.2 | 262.7 |
| 4 | 174.2 | 248.8 |
| 6 | 150.1 | 184.5 |
| 8 | 129.6 | 158.0 |

端到端 A/B（2026-10-27，同一 HEAD `7ecb57f7`，`-p 2048 -n 128 -r 3`，q8_0 KV，fa on，ub 1024）：

| 配置 | build-sycl | build-sycl-opt (0001) | 变化 |
|---|---|---|---|
| layer pp2048 | 610.82 ± 0.20 | 610.11 ± 0.24 | −0.1% |
| layer tg128 | 15.80 ± 0.01 | 16.79 ± 0.01 | +6.3% |
| tensor pp2048 | 631.63 ± 0.83 | 635.20 ± 0.22 | +0.6% |
| tensor tg128 | 19.31 ± 0.25 | 20.05 ± 0.20 | +3.8% |

正确性（2026-10-27）：

- `llama-perplexity -c 512 -b 4 -ub 4`（强制多列 MMVQ，80 KB 文本取 3 chunk）：build-sycl 5.7063 ± 0.47113，build-sycl-opt 5.7140 ± 0.47187，差 0.14%。
- 同提示词 `--temp 0 -n 48` 贪心输出逐字节相同。
- `mmvq_probe` 内置 f32 主机参考（自建 kernel 版本测得，ncols 1–8 × 5 种形状，相对误差 1.0–1.5%）。

## 复现

```bash
make build          # 导出 llama.cpp HEAD + 打补丁 → ~/workspace/ai/llama.cpp/build-sycl-opt
make bench sycl-tensor
```

对照组是外层 `../bench.sh sycl`、`../bench.sh sycl-tensor`（build-sycl，同一 HEAD，参数一致）。

`make build` 在 llama.cpp HEAD 或补丁内容变化时重新导出源码；`--clean` 全量重建。

注意：`libggml.so` 按 soname 依赖 `libggml-sycl.so.0`，探针用 rpath 指向 opt 目录时会复用已加载的库，两个构建测出一样的数；GPU 空闲降频到 600 MHz，预热必须按时间（1 s）。

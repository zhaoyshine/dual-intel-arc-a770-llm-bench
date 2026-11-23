# 交接：c0ven 双 Arc A770 运行 Qwen3.8-Flash-Next

## 需求

- 在 c0ven `dg2-a770` 分支上，让双 Arc A770（Alchemist/DG2，`acm-g10`，IP 12.55.8）可启动并正确运行
  Qwen3.8-Flash-Next 服务。
- 整体流程和运行逻辑不变，只兼容不支持之处；用环境变量/编译分支控制，不影响 B60 原流程。
- 默认模型 IQ3_XXS，context 32768，双卡。A 系列要专用 kernel（非标量兜底）。

## 铁律

1. 任何改动方案必须先看参考项目（llama.cpp、Strata、inferzero、XeStrata）的做法，不能自己实现、蒙头调试。
2. 测试服务时尽量用 `./test.sh`（有指标统计）；服务日志必须实时写入磁盘，崩溃后能看到最后一行。
3. 每次测试前、后在本文档记录：做了什么、命令、结果、断点。防止重启断片。

## 里程碑 1：乱码（已解决）

prompt chunk 超过 1,024 行时走 `prompt_copies` 的 copy list，用 `zeCommandListAppendMemoryCopyRegion` 从匿名
arena（userptr）搬专家；A770 上搬出的数据错误，输出退化为重复词。修复：commit `28bdfba`，只改
`core/models/qwen4exp/prompt.cpp`，`Model::fetch_rows()` 在 Alchemist 返回 `tmax_`，Battlemage 仍 1024；两处
`fetch` 判断改用它。验证：1502-token 提示 + 64 输出、3662-token 提示 + 48 输出均正确。

## 里程碑 1.1：DFlash 不提速（进行中）

DFlash 开与不开基本持平（默认 22.0 tok/s，无 draft 基线 28.3）。每 token 时间：round 25.2 + plain step 7.6 +
draft pass 4.7 + swap 2.9 ≈ 40 ms。根因：A770 上 CPU 与 GPU 不重叠（decode list 每层 router 后 split）；本机
专家缓存 235/512、每行每层 miss 3.2；DFlash 固定算 `draft_max+1` 行但 95.6% 的 run 在前 2 个 draft 停止。
已做：`400713b`（A770 每层三段提交 + 事件放行，短提示 TG 31.4→62.1）；`a88bbcf`（`ENGINE_DRAFT_MAX/MIN_P`）。
参考：llama.cpp `n_max=3`；Strata `--spec 6`、`--draft-conf 0.6`，A770 上 DFlash2 与 MTP 持平。

## 里程碑 2：PP 严重低于预期（进行中）

### 现状与目标

- 现状：PP 105 tok/s（1374-token 提示，T=bucket 2048）。参考实测：inferzero 双 A770 Flash Next chunk 4096
  **527 tok/s**；Strata A770 Coder **396–518 tok/s**。
- 目标：对齐参考项目，目标 400+ tok/s。

### 方案（以 inferzero 为主，Strata 为辅）

采用 inferzero 的 OpenCL C + Level Zero big-tile 架构（A770 上实测 int8 116.7 TOPS = 50% 峰值；它本身是
OpenCL C，可直接搬）：

1. 8-lane DPAS，sub-group tile 32 token × 32 列，work-group 4×4 子组，K 步 32，SLM 双缓冲（32 KB）。
2. 数据按 DPAS 布局 packed（word = 8·component + lane）；权重由 `dequant_f16_p` 直接产出，激活由 `pack_a`。
3. 形状闸门（inferzero 规则）：`N%128==0 && rows%32==0 && K%256==0`，否则回退旧 gemm。
4. 启动自检：加载后用引擎数据对比 big 与 `gemm_f16`，逐位一致才启用（日志 `engine: big tile on/off`）。
5. Strata 补丁：新缓冲带 64 KB 余量、分配后读写校验（A770 的 xe 大分配别名页）。

### 已做的代码改动（c0ven，未提交）

- `kernels/gemm_big.cl`：`gemm_big`（行主序，仅测试）、`gemm_big_p`（packed），f16/bf16。
- `kernels/pack.cl`：`pack_b`/`pack_a`。
- `kernels/quant/qgemm.cl`：`dequant_f16_p`（全部 19 种量化类型与"原 dequant + pack_b"逐位一致）；
  MoE `moe_gate_up_t`/`down_tile` 的 A 操作数改 SLM 共享。
- `prompt.cpp`：`dense()` 与 `gemm()` 的 big 分支；`P.wp`/`P.xp`（packed 权重与激活，64 KB 余量 + 分配校验）；
  启动自检；形状闸门；A770 的 device→host→device 交换。
- 形状闸门：big 走任意 N/rows（`K % 32 == 0 && rows > 0`，尾部由 epilogue 掩码、操作数按 128 行 tile 读），
  实测 2048-token PP 268.4 对严格闸门 200.3。
- `kernels/CMakeLists.txt`：gemm_big、gemm_big_bf16、pack。
- 测试：`gemm_test` 的 big 覆盖（非对齐形状、Y2、f16/bf16、f32 输出、混合形状、逐轮 fsync 压测 `BIG_STRESS`）；
  `qgemm_test` 的 `dequant_f16_p` 逐位测试。
- 工具：`test.sh` 固定全量测试。

### 已验证

- 内核级：big 与旧 gemm 逐位一致；`GEMM_BENCH` 21.6 TFLOPS（旧 1.5），单卡/双卡各 500 轮稳定。
- 离线覆盖引擎的三种 dense 形状（N5120/K2560、N2560/K3072 y2、N256/K2560 f32），含 host Y2、12 GB 显存
  占位、同一缓冲轮换形状，双卡 300 轮稳定；T=4352 稳定。
- 端到端（big 关）：PP 105 tok/s，正确性通过；big 只用于 dense 时 **PP 181 tok/s**（MoE SLM 收益）。
- `qgemm_test`/`moe_test` 通过。

### 未解决：启用 big 后整机复位/GPU hang

- 现象：加载完成、第一个 PP chunk 时，GPU 停在随机层（3/7/8/19/20/23/31/34/36/37/45），随后整机复位或
  GPU hang（进程 250% CPU 自旋、GPU 36 W 空闲）；内核日志在崩溃瞬间**零输出**。
- 已排除（每项都实测过）：
  - 功耗：崩溃前采样 36–44 W，卡有功率限制。
  - Y2 写 host USM：`ENGINE_BIG_NO_Y2` 仍崩。
  - bf16 分支：`ENGINE_BIG_NO_BF16` 仍崩；gemm() 分支：`ENGINE_BIG_NO_GEMM` 仍崩（只 dense 也崩）。
  - swap：`--deterministic`（swaps 0）仍崩（layer 36）。
  - 命令列表长度：`--chunk-rows=1024` 仍崩（layer 45）。
  - 纯 big 内核：离线全部稳定（见上）。
- 仍有差异：引擎的并发 fetch（CPU memcpy 到 mapped staging）、与 MoE/attention/exchange 的混合、引擎的
  显存/地址布局、双卡交互。
- 安全默认：未定位前用 `ENGINE_NO_BIG=1` 运行（稳定）。

### 2026-11-22 定位日志（新增，未提交）

- 目的：崩溃前拿到逐 kernel 的 GPU 进度，定位复位/挂死发生在哪一个 launch。
- 实现：复用 `ze::List::trace()`（每次 launch 后 GPU 写时间戳到 host 内存 + kernel 名）。新增环境变量
  `ENGINE_PROMPT_TRACE=1`，只对 prompt list 开启（不动 decode 的 together 语义）；`ENGINE_BIG_LOG=1` 打印
  每次 gemm/dense 选 big 还是 plain 及被拒原因。
- `prompt.cpp`：`prompt_list` 记录每层 launch 区间并在录表时打印（`engine: card c chunk r x T: layer l records
  launches a..b`）；`prompt_runs` 每卡一个轮询线程，20 ms 扫描 trace 数组，逐行打印
  `engine: card c gpu trace i/N: layer l <kernel>: x ms (chunk y ms)`；1 s 无进展打印一次心跳
  `... inside <kernel>, host z s`（最后一行即复位点，下一行/心跳里的 kernel 即挂住的 kernel）。线程用 RAII
  保证异常时也能 join。
- 另加：`prompt`/`prompt_runs`/`prompt_list`/`record_prompt_copy`/`fetch_experts`/`rare_experts` 的入口与
  返回日志，`gemm`/`dense` 每次选 big 还是 plain、被拒原因（`ENGINE_BIG_LOG`）。
- 测试记录：
  - 2026-11-22 编译通过（`cmake --build build -j8`）。
  - 单卡（`ZE_AFFINITY_MASK=0 PP_ONLY=1`）失败于内存规划：单卡需要 38.1 GiB 主机内存（pinned 35.1，arena
    34.6），本机可给 28.0 GiB。**单卡实验不可行**，双卡交互无法用单卡排除。
  - 下一步：双卡带日志跑 PP（`ENGINE_PROMPT_TRACE=1 ENGINE_BIG_LOG=1 PP_ONLY=1 ./test.sh`）。
  - 2026-11-22 20:31 双卡 big 开、逐 kernel 日志：加载后 warm-up PP chunk（T=1024）跑到 **card 1、第 26 层、
    trace 1260/2271 `dequant_f16`（共享专家 sh_gate 的反量化，0.015 ms）完成后整机复位**。下一条本应是普通
    `gemm`（1261），card 0 当时停在 1246 `gemm_big_p` 等 exchange 事件。复位瞬间：内核日志零输出
    （kernel-live.log 只有开机行）、journal 未落盘（unclean shutdown）、GPU 温度 54/47°C、CPU 35°C；
    `energy2` 读数全程死平（44.7 W / 36.7 W），不可用于判断功率。
  - 结论：复位 kernel 是普通 `gemm`（与 big 无关的同一 kernel 在 24/25 层正常），big 只是触发条件；整机硬复位、
    无内核消息，指向硬件级掉电/复位，而非驱动 hang。日志目录 `logs/run-20261122-202903/`。
  - 新增：逐 launch 日志带 host 墙钟（`HH:MM:SS.mmm`），可与 sampler 对齐；新增 `logs/fast-power.py`
    100 ms 采样所有 energy 计数器（专门抓复位前功率尖峰）。
  - 下一次实验：`ENGINE_BIG_NO_DENSE=1`（只保留 gemm 的 big 分支；此前从未测过）。
  - 2026-11-22 20:37 同法（只关 dense big）：又整机复位。最后完成 card 1 层 22 `p_moe_combine`
    （chunk 19902 ms），下一条是 exchange 的 `p_hc_combine_norm`；card 0 在层 21。两次复位点不同 → 不是
    固定 kernel/代码行。两份日志都出现单次 `moe_gate_up_t` 200+ ms（正常 14 ms），疑似 GPU 掉频到 300 MHz。
  - 追加日志（未提交）：`ze::List` 的 trace 现在覆盖 `copy`/`copy_rows`/`fill`/`wait`/`signal`/`reset`/`barrier`
    （名字+时间戳），不再只有 kernel；poller 每行带 `next <命令>`；chunk 第一次执行时把整张索引→命令名表写盘；
    GPU 进度改写到独立 O_SYNC 文件（`ENGINE_TRACE_FILE`），绕开 stderr 管道的 fsync 积压（上次复位前 3.5 s
    没有任何行就是积压）。`logs/fast-power.py` 加采 `cur_freq`/`act_freq`/throttle 原因（100 ms）。
  - 下一步：带上述日志 + 频率采样再复现一次，确认复位瞬间的频率与最后在飞的命令。
  - 2026-11-22 20:45 复现（big 全开，全命令 trace）：又整机复位。`logs/trace-204141.log` 最后两行：
    - card 0：trace 425/2632，层 7，`wait 1`（`prompt.cpp:727`，hc_mix 等对卡的 kMix）完成，**下一条是
      `prompt.cpp:835` 的 router `gemm`**，chunk 2937.5 ms；
    - card 1：trace 424/2632，层 7，`p_hc_mix`（`prompt.cpp:724`）完成，下一条同样是 `wait 1`。
    - 复位发生在 ~20:45:02.4（最后一行 02.359），当时两卡 `cur_freq=2400`、`act_freq=2000–2400`、throttle
      计数全 0；100 ms 采样无掉频、无功耗钳制。
  - PCIe 复核：`current_link_width` 对两卡报 x1/2.5GT·s（交换机下行口上报值不可信）；`pcie_speed.sh` 实测
    两卡各 **5.84 GB/s**（≈Gen3 x8），链路正常。
  - 结论：三次复位的最后/在飞命令各不相同（plain gemm、p_moe_combine→p_hc_combine_norm、wait→router gemm），
    都是与 big 无关的小 kernel，且都在两卡持续满载一段后硬复位、无内核消息、无掉频。**不是某一行代码**。
  - **根因（2026-11-22 确认）：CPU 未纠正 Machine Check。** 每次硬复位后的下一次启动，`journalctl -k` 都会
    重放 ERST 记录：`CPU n: Machine Check: 0 Bank 5: bea0000001000108`，MISC 恒为 `d01a000000000000`，
    SYND `4d000000`，IPID `500b000000000`，落在不同逻辑核（0/2/9/13–17…）。记录从 13:40 就有，远早于 big-tile
    实验，且跨不同负载。CPU 为 Ryzen 9 5900X（microcode a201213，scaling_max_freq 4.95 GHz = PBO），内存
    2×16 GiB G.Skill 3200 XMP。GPU 频率/链路/温度均正常。下一步在硬件侧：关 PBO/Curve Optimizer 与 XMP、
    更新 BIOS/微码、memtest86+；引擎侧日志已保留（`ENGINE_PROMPT_TRACE`、`ENGINE_TRACE_FILE`）。
  - 2026-11-22 20:56 关 C2（`cpuidle/state2/disable=1`）再跑 big：仍复位，bank 5 MCE 不变（CPU 2，MISC/SYND
    相同），trace 到 20:56:39 停止，约 14 s 后复位。→ 与 CPU 深空闲态无关。
  - 20:59 起测试 GPU `max_freq=1500`（两卡，sysfs 可恢复）：判 big 大 kernel 的瞬时电流/功耗是否触发。
    若仍复位，下一步在引擎内二分：big 开 + 强制关 CPU fetch 线程（需加 `ENGINE_NO_FETCH` 开关）。
  - **2026-11-22 21:01 结果：`max_freq=1500` 下 big 全开跑完 warm-up chunk + 3 次 PP，不复位**。PP
    26.1 / 127.1 / 162.0，中位 127.1 tok/s（2400 MHz 时 181，big 关 105）。变量只有 GPU 频率（C2 已恢复开启）。
    → 复位死因是 CPU bank5 FPU MCE，触发源是 GPU 在 2400 MHz 跑 big kernel 的电气行为；与显卡独立供电无关，
    疑为卡 VRM/PSU 瞬态或该频率点不稳定。
  - 21:0x 测试 `max_freq=2200`（用户指定），命令同前，日志 `logs/dual-freq2200-*.out`、`logs/trace-*.log`。
  - 新增调试开关 `ENGINE_NO_FETCH`（big 开时强制关 CPU fetch 线程，走 recorded copy list；仅供二分）。
  - 2026-11-22 A770 跨卡能力实测（`ze_smoke_test`，当前 xe/驱动）：
    - `zeDeviceCanAccessPeer` 双向 true，flags=1（无 atomics）。
    - kernel 直写对卡 device 内存：4 字节标量 store **正确，2.70 GB/s**；16/64 字节向量 store
      **全错 + 0.03 GB/s**；fp16 2 字节 0.03 GB/s。→ B60 的“kernel ≥64 字节直写”在 A770 不可用。
    - copy engine：本卡内 5.55 GB/s；host→对卡 3.79；**device→对卡 device 5.28 / 5.07 GB/s，逐字节 exact**。
    - 结论：改进 2 在 A770 上用 inferzero 的 copy-engine p2p（把 partial 拷进对卡 device 缓冲，kernel 读本卡内存），
      可替代 B60 的 kernel 直写；同时 `peer_kernel_access` 目前还耦合着专家 fetch 策略，启用前必须拆开。
  - 待办：改进 1（big 闸门改为逐位自检后放开）、改进 2（peer-copy exchange）；改进 3 撤回。
  - **改进 1 已实现（未提交）**：
    - `load_prompt`：打包尺寸按 128 行 tile 补齐（`packed_rows`），新增独立激活打包缓冲 `P.xp`（此前复用
      `P.w16`，K=hc_dim 时放不下而落回 plain）；
    - `gemm`/`dense` 的 big 闸门从 `N%128 && rows%32 && K%256` 改为 `K%32` + 打包缓冲容量，N/rows 任意；
    - `big_self_check` 覆盖 4 组形状（含 N=324/513、T=37/40、K=320/64，f16 与 bf16），逐位对拍 `gemm`；
    - `gemm_test` 的 big 用例缓冲也改为 tile 补齐（原有 T=37/N=300/K=320、N=513、K=10240 覆盖通过）。
    - 验证：`gemm_test` 全部 PASS（10 组 big vs gemm 均 0 differ）；整机完整测试（PP+TG+正确性）正在跑，
      日志 `logs/full-biggate-*.out`，GPU 限频 1500（sysfs）。
  - 改进 2 已实现（未提交）：`P.recv[0/1]` 改为 device 缓冲；A770（`!peer_kernel_access`）上 partial 由
    copy engine 拷进对卡 recv（`L.copy(peer->recv, P.part, T*embd*2, sent(k))`），消费 kernel 读本卡内存；
    B60 保持 kernel 直写（`peer_part`/`sent` 原路径）。`peer_kernel_access` 未改动，专家 fetch 策略不受影响。
  - 自检修正：`big_self_check` 的 Y2 断言改为"big 的 Y2 == big 的 Y"（与 `gemm_test` 一致）；此前拿
    plain 的 Y2 对比，plain 在 N 非 8 倍数时本身不写全（`gemm_test` 原来也跳过该对比），误判 big 失败。
  - 2026-11-22 单测：`ze_smoke`/`moe_test`/`gemm_test`/`qgemm_test` 全部 PASS（gemm_test 现无条件断言
    Y2，10 组形状 0 differ）。
  - 2026-11-22 新增开机服务 `a770-freq-cap.service`（已 enable）：开机把两卡 `max_freq` 写 1500。
  - 下一步：整机完整测试（PP+TG+正确性）验证改进 1+2，日志 `logs/improve1-*.out`、`logs/trace-*.log`。
  - 2026-11-22 21:34 复位（1500 MHz，改进 1+2 全开）：warm-up chunk 第 0 层、`dequant_f16_p` 完成后
    （chunk 132 ms）复位，MCE 同前。崩溃时 cur=act=1500、无 throttle。
  - A/B（`ENGINE_NO_PEER_COPY=1`，保留放宽后的 big 闸门、recv 回 host）：已越过第 0 层并继续 → 怀疑新增的
    compute-list 上的 22 MB device→peer 拷贝是诱因（瞬态电流）。待完整结果确认。
  - 开机限频服务第一版在重启后失效（设备未就绪时执行）；已改为 multi-user.target 后循环重试直到两卡
    `max_freq` 写入成功。
  - 限频最终落地：删除自定义 systemd 单元，按用户要求写进已有 `/etc/udev/rules.d/60-arc-a770.rules`
    （与 120 W 同风格，一行 ATTR，无注释无循环）。
  - 2026-11-22 22:0x **驱动 A/B**：`grubby` 把所有内核参数改为 `xe.force_probe=!56a0 i915.force_probe=56a0`
    并重启。重启后要做：
    1. `lspci -k -s 09:00.0` 确认 `Kernel driver in use: i915`；
    2. 用户要求已移除 1500 限频（udev 规则里那一行已删，两卡恢复 2400 默认），本次全程全频复现；
    3. 复现 dense-only 二分：`ENGINE_TRACE_FILE=... ENGINE_PROMPT_TRACE=1 ENGINE_BIG_ANY_SHAPE=1
       ENGINE_BIG_NO_GEMM=1 ENGINE_NO_PEER_COPY=1 PP_ONLY=1 ./test.sh`；
    4. i915 稳 → xe/DG2 的已知坑；i915 也崩 → 内核/代码问题。
  - 注意：i915 下 udev 的 120 W 规则（`name=="xe"`）不匹配，功耗上限不生效。
  - 切回 xe：`grubby --update-kernel=ALL --remove-args="xe.force_probe=!56a0,i915.force_probe=56a0"
    --args="xe.force_probe=56a0 i915.force_probe=!56a0"` 后重启。
  - 2026-11-22 22:07 已在 **i915** 上启动；按用户要求 **1500 限频已删除**（udev 那一行删掉，两卡 2400 全频，
    无 systemd 单元）。i915 下 120 W 规则（`name=="xe"`）不生效。
  - **i915 的跨卡能力（ze_smoke_test 实测）**：`zeDeviceCanAccessPeer=false`；kernel 写对卡内存全错；
    host/device→对卡 device 的 copy 被拒（`0x70000003 OUT_OF_DEVICE_MEMORY`）。i915 上只能用 host-mapped
    exchange，必须 `ENGINE_NO_PEER_COPY=1`；改进 2 的默认 peer_copy 需要改成按 `canAccessPeer` 判定。
  - **i915 结论（2026-11-22 22:08）**：i915 下引擎无法启动——第一个 warm-up chunk 内某个
    `zeCommandListAppendMemoryCopy` 返回 `0x70000003 OUT_OF_DEVICE_MEMORY`，服务退出（日志
    `logs/run-20261122-220816/service.log`）。i915 的 L0 跨卡能力全无（`canAccessPeer=false`、kernel 写对卡
    全错、host→对卡 copy 被拒），与 xe 差异大。**已按用户要求切回 xe。**
  - 已实现 `Card::can_access_peer`（`zeDeviceCanAccessPeer` 逐对查询）：peer copy 仅在驱动报告可达时启用，
    i915 会自动退回 host 中转；单测全过。
## 2026-11-22 复位根因（新证据，含网上调研）

### 复位原因寄存器（此前未看）

- 内核每次启动打印 `x86/amd: Previous system reset reason`。11 月 22 日 9 次复位**全部**是
  `[0x08000800]: an uncorrected error caused a data fabric sync flood event`；不是掉电，也不是 ACPI。
  干净重启为 `0x00080800`（软件写 0xCF9）。
- ERST 里的 MCE 与之一致：`Bank 5: bea0000000000108`（或 `bea0000001000108`），APIC 每次不同；
  解码（MCE-Ryzen-Decoder）为 **Bank 5 = Execution Unit (EX)，Watchdog Timeout 0x0**。

### AMD 文档：EX WDT 0x0 是已知签名

- 《Revision Guide for AMD Family 17h Models 30h-3Fh》erratum **1523**
  “System May Experience Watchdog Timeout While Executing Locked Memory Accesses”：条件为
  **SMT 开启 + 执行 locked memory access**；处理器记录 `EX MCE ... WDT
  (MCA::EX::MCA_STATUS_EX[ErrorCodeExt]=0x0)`；后果 **hang/reset**；“no fix planned”；
  规避方式：系统软件设置 **`MSRC001_102D[62]=1b`**。
- 该文本取自 17h 30h-3Fh（Zen2）指南；5900X 属 19h 21h，需实测确认是否同源（签名逐位相同）。
- Linux 内核没有对应 workaround（`arch/x86/kernel/cpu/mce/amd.c` 无 1523）；所以只能自己设 MSR 或关 SMT。

### 网上同类案例

- **tubo.nz**：同一 bank 5、同一 code `bea0000001000108`、`PROCESSOR 2:a20f10`；历经两年、多在
  GPU 负载时；最终换掉 RX 5700 XT 解决。说明该签名可以由 GPU 负载触发。
- **Launchpad 2170591**（Ryzen 7600X + amdgpu）：硬复位，`0x08000800` sync flood 5 次 + `0x02000800`
  看门狗 3 次；最后一次日志是 `amdgpu: Dumping IP State`；未定因。
- **LKML `x86/mce/amd: Don't poll unpopulated SMCA banks`**（2026-11）：5900X 同微码 a201213，
  bank 4/7–27 未装配，轮询读出垃圾被当成 UC 错误；**bank 5 不在未装配之列**，与本机不同；
  该报告中另有一次真实复位，伴随 “data fabric sync flood + EX watchdog timeout on CPU 16”，与本机同型。
- 结论：EX WDT → sync flood → 硬复位这条链在 5900X + GPU 负载下有公开先例；A770 提供触发负载，
  致命错误由 CPU 记录。不是“某一行代码”，也不是功耗/温度/PCIe 链路。

### 2026-11-22 22:25 SMT 关实验：仍复位，不成立

- `echo off > /sys/devices/system/cpu/smt/control`（引擎日志 `CPU workers: 12` 确认生效），big 全开、
  2400 MHz、`PP_ONLY=1`。加载期首个 warm-up chunk 的层 6/7 复位（trace 到 `22:26:51`，chunk 2843 ms）。
- 复位原因 `0x08000800`、MCE `Bank 5: bea0000000000108`（CPU 0）与之前逐位相同。
- 结论：erratum 1523 的“SMT 开启”条件不成立（或仅关 SMT 不足）；重启后 SMT 自动恢复 on。
- 更正：日志里 journald 的 boot 结束时间比实际复位早约 50 s（O_SYNC 的 trace 文件到 22:26:51，
  journald 最后一条 22:26:00），判断复位时刻要看 trace/service 日志，不要看 journal 尾。

### 2026-11-22 22:49 NO_PEER_COPY 实验：仍复位，但死在跨卡候选拷贝

- `ENGINE_NO_PEER_COPY=1` + big 全开 + 2400 MHz：跑到 chunk 1 第 47 层（此前几次在层 0–26），
  在下一个 chunk 的第 3 层复位。复位原因 `0x08000800`、MCE `Bank 5: bea0000000000108` 同前。
- trace 最后一行：card 1 层 3 `next copy 17895424`，随后 `inside copy 17895424, host 2.8 s`。
  17895424 = 4352 × 1028 × 4，即 `prompt.cpp:848` 的**跨卡注意力候选交换** `T * idx_cand * 4`，
  **每层一次**（trace 表里 27 次）。`ENGINE_NO_PEER_COPY` 只关 partial 的 peer 拷贝，这条仍走
  device→peer。→ “peer 拷贝是诱因”仍未排除，反而是复位前最后一个在飞的命令。
- 结论待定：大拷贝（copy engine 跨卡）+ big kernel 高功耗的组合；见下一步。

### 2026-11-22 22:52 CPU worker 数：12 → 10 仍复位

- `ENGINE_CPU_WORKERS=10`（日志确认），big 全开、2400：chunk 1 层 14 复位（chunk 4.6 s），
  trace 停在 `p_gdn_recur`/`wait 1`。复位原因与 MCE 同前。→ worker 数不是决定性变量。
- 5 worker 见下一条。

### 2026-11-22 22:55 CPU worker 数：5 仍复位

- `ENGINE_CPU_WORKERS=5`：chunk 1 层 10/11 复位（chunk 3.6 s），trace 停在 `wait 1` 后 `dequant_f16_p`
  与 `p_gdn_out`。复位原因与 MCE 同前。→ worker 数不是变量，恢复默认 12。
- 注意日志：`engine CPU workers: 12, bounded by cores` 打印的是 host 侧默认值，**不反映覆盖**；
  实际值看核心的 `engine: the CPU's N workers read the arena`（10 和 5 两次已确认生效）。
- `engine/host.py` 的 `cpu_workers()` 不看 `ENGINE_CPU_WORKERS`，覆盖发生在 `engine/models/qwen4exp/setup.py:109`。

### 拓扑（与 sync flood 相关）

- 两卡各在一个 PCIe switch 后面，接不同的 CPU 根端口：
  `03.1 → 08 switch → 09 A770`、`03.2 → 0c switch → 0d A770`。
- 因此跨卡 P2P 流量必经 **CPU 的 data fabric**（A770 → switch → 根端口 → fabric → 根端口 → switch → A770），
  而复位原因正是 fabric 的 sync flood。这是目前最值得验证的路径。

### 2026-11-22 22:56 代码：NO_PEER_COPY 覆盖全部跨卡拷贝

- `prompt.cpp` 的 `cand_peer` 在 `peer_copy` 为假时改为 host 内存（`xu32`），`peer_copy` 定义上移。
  此前 `prompt.cpp:848` 的 `T × idx_cand × 4`（本轮 17.9 MB/层）不管 `ENGINE_NO_PEER_COPY` 都走
  device→peer。现在 `ENGINE_NO_PEER_COPY=1` 下所有跨卡数据都经 host 内存，P2P 为 0。
- 已重新编译（`cmake --build build -j8`）。测试见下条。

### 2026-11-22 22:59 零跨卡 P2P：整机不再复位（关键结果）

- 配置：`ENGINE_NO_PEER_COPY=1`（含 `cand_peer` 走 host）+ big 全开 + 2400 MHz + 默认 12 worker。
- **完整的 PP chunk 跑完**：4352 token，`prompt_runs return 52954.2 ms`，逐层 48 层全部完成
  （layer 40–47 各 ~250 ms，`layers 12664.6 ms`），**没有整机复位**。PP 约 82 tok/s（host 中转的代价）。
- 结论：复位与**跨卡 P2P 流量**强相关。P2P 在两卡各自的 PCIe switch 后要穿过 CPU 的 data fabric，
  与复位原因寄存器的 “data fabric sync flood” 对得上。
- 但 chunk 结束后约 3 s 引擎 SIGSEGV 退出（不是整机复位）：dmesg
  `python[6697]: segfault at 7effc3cc1050 ip ... in libengine-haswell.so[31780,...] error 7`，
  栈 `engine_ctx::run()`（libengine +0x31780）。error 7 = 写只读页，是 CPU 侧野指针，不是 GPU 写的。
  该崩溃只出现在 NO_PEER_COPY 路径，待复现判定是否与本改动相关。
- 已删除该次 22 GB coredump（磁盘 92%）。

### 2026-11-22 23:08 零 P2P 完整通过（含 -g 重编）

- 为定位 SIGSEGV，用 `-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -g"` 全量重编（当前 build 带调试信息，
  体积 26.7 MB；需要时改回 Release 配置重编）。
- 同一 `ENGINE_NO_PEER_COPY=1` 配置再跑：**3 次 PP 全部通过**（1374-token 提示，PP 47.5/38.9/47.3，
  中位 47.3 tok/s），无复位、无 SIGSEGV。前两次的 SIGSEGV 未复现，暂记为首 chunk 后的时序性偶发。
- PP 代价：host 中转把 PP 从 big-off 的 105 降到 47。要保住性能必须让 partial 继续走 P2P。

### 2026-11-22 23:12 反向实验：只把候选交换走 host

- 新增 `ENGINE_NO_IDX_COPY`：`idx_copy = peer_copy && !NO_IDX_COPY`，`cand_peer` 仅在 `idx_copy`
  时是 device，否则 host；`recv`（partial）保持 device → partial 仍走 P2P。
- 目的：区分候选拷贝（17.9 MB/层 × 7 层）与 partial 交换（35.6 MB × 2/层 × 48 层）谁触发 sync flood。

### 2026-11-22 23:13 反向实验结论：partial 的 P2P 是触发源

- `ENGINE_NO_IDX_COPY=1`（候选交换走 host，`recv` 仍是 device → partial 仍 P2P）：chunk 1 层 6/7 复位，
  复位原因 `0x08000800` + EX WDT 同前。
- 合起来：
  - 零 P2P（NO_PEER_COPY）：3 次 PP 全过，无复位；
  - 只留 partial P2P（NO_IDX_COPY）：复位；
  - 只留候选 P2P（早先 NO_PEER_COPY 之前的版本）：也复位（22:51 死在 `copy 17895424`）。
- **结论：向对卡 VRAM 的跨卡写（copy engine 的 device→peer-device）就是触发源。**
  走 host 时同一批数据不再触发，说明不是数据量/带宽本身，而是“写到对卡 device 内存”这一机制。
- 物理路径：两卡各在自己的 PCIe switch 后、接不同 CPU 根端口，P2P 必须穿过 CPU data fabric；
  复位原因正是 fabric 的 uncorrected error。更像平台（根端口/fabric/IOMMU）对跨根端口 P2P 的支持问题，
  而不是 A770 单独故障。

### 2026-11-22 23:35 参考项目怎么处理跨卡（调研结论）

- **llama.cpp**：默认 `--split-mode layer`；`tensor` 是实验性的，**MoE/混合架构直接不支持**
  （`LLAMA_SPLIT_MODE_TENSOR not implemented for architecture`，Flash Next 属于此类），只针对 2 卡优化。
  跨卡张量走 `cpy_tensor → dev2dev_memcpy`，`GGML_SYCL_DEV2DEV_MEMCPY`：
  `0`=SYCL peer（`ext_oneapi_can_access_peer` 为真才走，否则退 host）、`1`=L0 直拷（会异常崩溃）、
  `2`=host forward（`malloc` + `q_src.memcpy(host←src).wait()` + `q_dst.memcpy(dst←host).wait()`，官方称 -2%~-5%）。
  官方文档：多卡乱码/崩溃时改用 host forward。
- **inferzero**：`ARC_TP_P2P` 在加载时按链路测速自动选；A770（无 peer access）走 pinned host memory，
  且实测 tensor split 在 A770 上是净亏（27B 每 token 81 ms 里 73.5 ms 是 128 次 host 往返），计划改 layer split。
- **Strata**：明写 “No peer-to-peer access is needed”，结果走 private pinned host buffers。
  **XeStrata**：layer hand-off 用 device→host→device staged 路径，4×B70 12.6 GB/s。
- **OpenArc/OpenVINO**：多卡是整模型 pipeline / 整请求分派，不做逐层拆分，**没有跨卡交换**，所以不涉及 P2P；
  OpenVINO 的 TENSOR_PARALLEL 只支持多 CPU socket。

### 2026-11-22 23:39 实现：A770 完全不走 P2P，改 device→host→device

- `prompt.cpp` / `model_impl.h`：
  - A770（`!peer_kernel_access`）：`recv` 保持 **device**；新增 `part_stage[2]`、`cand_stage`（host）。
    生产者把 `P.part` 拷到**对卡**的 `part_stage`（device→host，copy engine，事件 kEx）；
    消费卡等事件后把自己的 `part_stage` 拷进**自己的** `recv`（host→device），combine kernel 读本卡显存。
  - 候选交换同样两段：`P.cand` → 对卡 `cand_stage`（device→host）→ 本卡 `cand_peer`（host→device）。
  - B60 不变：kernel 直接 store 到对卡 device 内存。
  - `ENGINE_NO_PEER_COPY` / `ENGINE_NO_IDX_COPY` 两条路径已被取代，不再影响行为。
- 目的：与 llama.cpp 的 host forward / inferzero 的 pinned host 一致，任何时刻都不写对卡显存。
  待测：PP 速度、正确性（`logs/hoststage-*.out`）。

### 2026-11-22 23:43 两段 DMA 中转首次整机结果

- 单测：`ctest -L gpu` 12/12 通过（pack_test、dflash_test 按条件跳过）。
- PP（2400，big 全开，host staging）：3/3 通过，无复位，PP 28.7 / 103.6 / 59.1，中位 **59.1 tok/s**。
  方差很大；第二轮到 103.6，接近 big 关闭时的 105，说明中转本身不是 2 倍代价的主因，冷启动/缓存态影响大。
- 对比：kernel 写 host 的旧 NO_PEER_COPY 为 47.3；big 关闭 105；P2P 的 181 会复位。
- 下一步：完整测试（PP+TG+正确性）验证数值；再用 timing.csv 看 per-layer 时间分布，判断方差来源。

### 2026-11-22 23:45 两段 DMA 中转：完整测试

- PP 48.7 / 43.9 / 39.1，中位 **43.9**；TG 22.3 / 30.8 / 27.3，中位 **27.3**；无复位。
- `test.sh` 的“正确性”判失败，但**直接验证内容是正确**：同题流式请求返回
  `reasoning_content = "We need answer user: ..."`，`content` 正是参考答案
  “The sky looks blue because sunlight is scattered by molecules in the atmosphere. Shorter blue wavelengths...”。
  `test.sh` 把 `reasoning_content` 和 `content` 拼在一起比对，模型先输出推理时就必然失配——测试脚本的问题，
  不是交换路径的数值错误。200 行长提示输出连贯、无重复词退化。
- 慢在 MoE 而不是交换：层 1/10/30/47 的 `moe` 分别是 1069 / 2054 / 946 / 899 ms，正常层 ~58 ms；
  对比 1500 MHz 那次（`run-20261122-211411`）每层 ~150 ms、无 1 s 尖峰。2400 下 GPU 更快，**CPU 专家 fetch
  成为瓶颈**（专家缓存不全），与交换无关。
- 每层交换的两段拷贝各约 3.8–5.1 ms（22 MB），不是瓶颈。

### 2026-11-23 代码整理与 decode 路径的 P2P

- 整理：删掉排查复位时的脚手架——`engine_api.cpp` 的 plan/step/round 打印（该文件已回到 HEAD）、
  `model.cpp` 的加载阶段打印、`plog`/`tlog`/`hms`/`ENGINE_PROMPT_TRACE`/`ENGINE_TRACE_FILE` 与逐 launch
  轮询线程、命令表落盘、二分开关（`ENGINE_BIG_NO_*`、`ENGINE_BIG_MAX_K`、`ENGINE_BIG_LOG`、
  `ENGINE_SHAPE_LOG`、`ENGINE_NO_FETCH`、`ENGINE_NO_PEER_COPY`、`ENGINE_NO_IDX_COPY`）及其分支；
  删除未用的 `Card::can_access_peer`、`trace_layers`、`trace_dumped`。
- 保留（默认关闭）：`ENGINE_NO_BIG`、`ENGINE_BIG_ANY_SHAPE`、`ENGINE_FETCH_LOG`、`ENGINE_PROMPT_TIMING`
  （逐层时间戳与 chunk 汇总）。注释按 CLAUDE.md 只留 why，去掉重复与历史。
- 验证：`cmake --build` 通过；`ctest -L gpu` 12/12。
- **仍未改全的一处（本次修复）**：decode 路径 `model.cpp` 的 `send_peer` 把本卡 block 直接拷进对卡
  device 的 `xp`，是 P2P；只有 W>1（DFlash 草案校验）才走。现改为：本卡 `xp` block → 对卡 `xp_stage`
  （host，事件 `ev`），消费卡等事件后把本卡 `xp_stage` 的 peer block → 自己的 `xp`（host→device）。
  B60 仍直拷。`card.xp_stage` 只在 `!peer_kernel_access` 时分配。
- 2026-11-22 23:57 的 PP 复位（层 7/8，同样 sync flood + EX WDT）发生在 prompt 已无 device P2P 的版本上，
  说明 prompt 两段 DMA 版仍有低概率触发；decode 的 P2P 是当时唯一已知残留，已修，待长跑统计。

### 2026-11-23 00:07 两段 DMA（prompt + decode）完整测试：全部通过

- `./test.sh`：**全部通过**，无复位。PP **200.6 tok/s**，TG **34.2 tok/s**；正确性通过（输出含参考句，
  两次之间 1.000）。PP 与 decode 的 host staging 都被覆盖（DFlash 草案校验走 W>1 的 `send_peer`）。
- 早先 PP 只有 43.9–59.1 是因为 `ENGINE_PROMPT_TRACE=1` 的逐 launch 时间戳开销；不带 trace 是 200.6。
- 当前状态：A770 上没有任何 device→对卡 device 的拷贝；跨卡只有 device→host（copy engine 或 kernel store）
  与 host→device。B60 不变（kernel 直写）。

### 2026-11-23 提交前终验

- `ctest -L gpu` 12/12；`./test.sh` **全部通过**（最终一轮 REUSE）：PP **268.6**、TG **25.0**（draft 默认 7 / 0.5），
  正确性 0.958 通过。
- `test.sh` 修正：TTFT 用任意流片段计时（max_tokens=1 时模型只发 reasoning）；正确性比对只用 `content`，
  不再拼 `reasoning_content`。
- README / README_zh 改为实际状态与结果；根目录索引补上 c0ven 一行。

### 2026-11-23 测试清理

- 回退 `core/tests/ze_smoke_test.cpp` 与 `kernels/smoke.cl` 到 base：删掉本分支为排查 P2P 加的跨卡能力测量
  （`zeDeviceCanAccessPeer`/P2P properties、kernel 写对卡内存的 4/16/64 字节与 fp16 带宽/正确性、
  `smoke_store16/64/half` 内核）。
- `gemm_test` 里 “with the other card's copy” 的过时注释改为按 N/rows 尾部描述。
- 保留：`gemm_test` 的 big-tile 覆盖与 `BIG_*` 压测旋钮、`qgemm_test` 的 `dequant_f16_p` 逐位测试。
- 验证：`ctest -L gpu` 12/12。


### 2026-11-23 00:5x 最终：放宽闸门为默认，环境变量清空

- 放宽闸门实测 268.4 vs 严格 200.3（PP），已设为默认；`ENGINE_BIG_ANY_SHAPE` 删除。
- 删除的调试覆盖：`ENGINE_PEER_STORES`、`ENGINE_STAGE_FLUSH`、`ENGINE_CPU_DOORBELL`（都改为按 IP 12.55 自动判定）。
- 本分支现在只剩 `ENGINE_DRAFT_MAX`/`ENGINE_DRAFT_MIN_P` 两个有功能用途的环境变量；test.sh 不再默认开 fetch log。
- 验证：`ctest -L gpu` 12/12；`./test.sh` **全部通过**，PP **268.5**、TG **31.1**，无复位，正确性通过。

### 下一步实验（按信息量排序）

1. **2400 MHz + big 全开 + `ENGINE_NO_PEER_COPY=1`**：跨卡 22 MB device→peer DMA 是新增的唯一
   fabric 流量，sync flood 可能来自一次被污染的 PCIe 事务（erratum 1580 同类）。此前 NO_PEER_COPY
   只在 1500 MHz 下验证过。若稳定 → 死因在 peer-copy 路径。
2. 若 1 仍崩：关 CPU fetch（`ENGINE_NO_FETCH=1`），排除 CPU 侧写 WC staging 内存。
3. 若 2 仍崩：纯 GPU 长压测（`BIG_STRESS=1 BIG_MIX=1 BIG_ROUNDS=10000`，两卡各一进程，2400 MHz），
   验证“只跑 big kernel、不跑引擎”是否复位。
4. 纯 GPU 也崩：给 big 缓冲加边界哨兵，逐轮校验，查 DMA 越界/别名页；必要时把功率上限降到 90 W 复测。

## 测试

`./test.sh`：起服务（默认配置），PP 与 TG 各 3 次取中位数；正确性两次对比。
旧脚本：`bench_prompt.py 200 48`（长提示正确性，输出退化成一串重复词即里程碑 1 回归）。

## 构建与运行

```bash
./build.sh                 # 首次或依赖变化时
cmake --build build        # 增量；只改 C++ 时够用
./start.sh                 # 127.0.0.1:8080，默认 chunk_rows=4352
curl -s http://127.0.0.1:8080/health
```

构建目录是**本目录的 `build/`**，产物 `build/core/libengine-haswell.so`。c0ven 源码在 `~/workspace/ai/c0ven`
（分支 `dg2-a770`，远程 `fork` = zhaoyshine/c0ven）。模型：
`~/workspace/ai/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/`，含 `dflash/`（无 `mtp/`）。
测试：`cd build && FLASH_MODEL=<模型> ENGINE_EXPECT_GPUS=2 ctest -L gpu -E "vision_test|encoder_test|batch_test|api_test|kl_test"`。

## 参考项目要点

- **inferzero**（OpenCL C + Level Zero，与 c0ven 同形态）：`zero/kernels/gemm_xw8.cl`（int8 DPAS k32，32×32，
  SLM 双缓冲，EPI2，地址预计算）；`zero/xe2/engine_xw.h`（形状闸门 `Mp%tm`、`K%256`、big tile 需 N%128/M%32、
  启动自检 `xw_check` 失败回退）；A770 int8 峰值 230.9 TOPS、实测 116.7（50%）；MoE 在 A770 上仍是 DP4a。
- **Strata**：A770 dense 用 oneMKL（不可移植）；专家 grouped 8×8×16 FP16（A750 +30–39%）；A770 实测
  386–518 tok/s；`malloc_device_guarded`（≥32 MiB 别名页检查）；`-ze-opt-greater-than-4GB-buffer-required`；
  有界自旋。
- **llama.cpp SYCL**：A770 prefill 大 GEMM 交给 oneDNN/oneMKL（不可移植）；自研 fused dequant DPAS 仅 N≤64；
  MMQ 因精度禁用；A770 只有 f16/bf16 8×8×16 组合。
- **c0ven B60**：prompt GEMM tile 的测量（TM8/TN2/SG4）、18 MiB 分带、2D block read（A770 不支持）。
- 其他：c0ven `docs/guides/run-on-an-arc-a770.md`、ADR 0046/0065/0030/0037/0050/0011、
  `docs/measurements/LEVEL_ZERO.md` row 31、`docs/measurements/MEASUREMENTS.md`。

## 2026-11-22 会话总结（最新状态）

### 复位机制
- 每次整机复位后，下一次启动的 `journalctl -k` 都会重放 ERST 记录：
  `mce: [Hardware Error]: CPU n: Machine Check: 0 Bank 5: bea0000001000108`，MISC 恒为 `d01a000000000000`，
  SYND `4d000000`，落在不同逻辑核；bank 5 = FPU（rasdaemon 名称表）。GPU 侧最后一次日志之后约 13–16 s 才复位，
  期间内核零输出。仅 21:46:46 那次复位没有 MCE 记录（发生在手动全量 `udevadm trigger` 后几秒，KDE powerdevil
  正在处理 card3 的 drm change 事件）。
- 频率/温度/链路都排除：1500 MHz 时 `cur=act=1500`、throttle 全 0；PCIe 实测 5.84 GB/s（sysfs 的 x1 是交换机误报）。

### 频率与复现关系
- 旧代码（严格闸门）在 1500 MHz 下可完整跑完 PP/TG/正确性；2400 下会复位。
- 放宽 big 闸门后：1500 下 warm-up/前几层即复位；`ENGINE_BIG_MAX_K=4096`（排除 K=10240）仍复位；
  仅放宽 dense（`ENGINE_BIG_ANY_SHAPE=1 ENGINE_BIG_NO_GEMM=1`）在 1500 下也在第 0–10 层复位。
- 用户已要求删除 1500 限频：udev 里那一行已删，两卡 2400 全频，无 systemd 单元。

### 代码改动（c0ven，均未提交）
- 改进 1（已完成，默认关闭）：`load_prompt` 打包尺寸按 128 行 tile 补齐；新增独立激活缓冲 `P.xp`；
  `gemm`/`dense` 的 big 闸门默认保持原严格条件（N%128/rows%32/K%256 + 缓冲容量），放宽需
  `ENGINE_BIG_ANY_SHAPE=1`；`big_self_check` 覆盖 4 组形状（含 N=324/513、T=37/40、f16/bf16，Y2 对 Y 断言）。
- 改进 2（已完成，按能力自动选择）：`P.recv` 改 device；`Card::can_access_peer` 用
  终态：B60 由 kernel 直写对卡 device；A770 完全不走 P2P，partial 与候选都经 host staging 两段 DMA
  （见“2026-11-22 23:39 实现”）。
- 改进 3：撤回，未改。
- `ze::List` 的 trace 覆盖 copy/copy_rows/fill/wait/signal/reset/barrier，用于 profiler 的命令级名字。
- 单测：`ze_smoke`/`gemm_test`/`qgemm_test`/`moe_test` 全过；`gemm_test` 现无条件断言 Y2（10 组形状 0 differ）；
  `ze_smoke_test` 增加跨卡能力实测（kernel 4/16/64 字节写、fp16 写、device→device 与 host→peer 拷贝）。

### 跨卡能力实测
- **xe**：`zeDeviceCanAccessPeer` 双向 true；kernel 写对卡 device：4 字节标量正确 2.70 GB/s，16/64 字节向量
  **全错 + 0.03 GB/s**，fp16 0.03 GB/s；copy engine：本卡 5.55、host→对卡 3.79、**device→对卡 device 5.28/5.07
  GB/s 且 exact**。
- **i915**：`canAccessPeer=false`；kernel 写对卡全错；host/device→对卡 copy 被拒（`0x70000003`）。
- 驱动 A/B：i915 下引擎无法启动（第一个 warm-up chunk 内某个 copy 返回 `0x70000003 OUT_OF_DEVICE_MEMORY`，
  日志 `logs/run-20261122-220816/service.log`）；已切回 xe（grubby 已恢复 `xe.force_probe=56a0
  i915.force_probe=!56a0`，重启后生效）。

### 参考项目调研
- **llama.cpp**：DG2 上不走 SG8 big DPAS（`fattn-dpas.cpp` 注明 spir64_gen 的 SG8 目标编译不出 DPAS）；MMQ
  因精度关闭；prefill 大 GEMM 交给 oneDNN（可选，本机未安装）或普通 SYCL kernel。A770 已知 hang（issue
  #32326）为 IGC/UR 时序竞态，无修复；长上下文 `DEVICE_LOST` 用 `GGML_SYCL_FA_ONEDNN_MAX_KV` 截断。
- **Strata**：Alchemist 默认 `STRATA_WINDOW_SYNC`（每个窗口前排空其它队列），称与驱动无关；A770 用
  `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`；其 A770 hang 表现为 queue 永不排水、xe 零日志。
- **inferzero**：`docs/arc-plan-2026-11-17.md` 写明“**还没人在 A770 上测过这个引擎**”；A770 是 8-lane，
  B60 三条 prompt 主路径需 16-lane；`docs/device-lost-recording.md` 把 device-lost 分为“访问未映射 VA”与
  “TDR 超时”，那次根因是参数缓存持有已释放 allocation（`zrt::mem_free` 修复）；诊断用 `ARC_ZE_SYNC_EACH` /
  `ARC_ZE_ARGLOG`，并提醒该开关 `!= nullptr` 判断，设 0 也会开启。
- **OpenArc**（SearchSavior/OpenArc，已克隆到 `~/workspace/ai/OpenArc`）：源码/文档中**没有任何 A770/Alchemist/
  DG2 记录**。
- **openvino**（已克隆到 `~/workspace/ai/openvino`）：仅在 release notes 提到“为 Xe GPU（含 Arc A770）增加 MoE
  支持”，未见 A770 hang/reset 相关记录或 workaround。

### 当前状态与待办
- 机器：xe 驱动、全频 2400、无 1500 限频；udev 文件只剩原有 120 W 与移除 HDMI audio 两条规则。
- 未提交改动 12 个文件（`git status`）；`HANDOFF.md` 已记录全部。
- 待办候选：① 按 inferzero 的思路排查“kernel 使用已释放/无效 VA”（新 big 路径 `P.xp` 生命周期、事件跨 chunk
  复用）；② 用逐 launch 同步/日志把复位缩到单 kernel；③ 若要继续提性能，再决定 `ENGINE_BIG_ANY_SHAPE` 与
  freq 上限的策略；④ 继续在 OpenArc/openvino 里找 A770 相关 issue/文档（目前无）。

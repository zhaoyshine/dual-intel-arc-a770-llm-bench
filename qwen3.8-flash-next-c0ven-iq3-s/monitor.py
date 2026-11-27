#!/usr/bin/env python3
"""c0ven 终端监控, 版式照 Strata 的 Overview: 标题行 + 横线 + 四列信息块 (大数字/副行/sparkline)
+ 最近请求表, 宽度上限 110 列。速率与比值一律取 /metrics 的 derived/计数, 不按刷新间隔做差。
用法: monitor.py [URL] [--interval 2]"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
import urllib.error
import urllib.request

from rich import box
from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.segment import Segment, Segments
from rich.table import Table
from rich.text import Text

ACCENT = "#34d399"
BLUE = "#5aa9ff"
AMBER = "#f5c56b"
RED = "#f28b82"
MAX_WIDTH = 110

STATE_STYLE = {"LIVE": f"bold {ACCENT}", "STALE": f"bold {AMBER}", "DISCONNECTED": f"bold {RED}"}

def fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=2) as resp:
        return json.loads(resp.read().decode())


def num(v: object) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def n(v: object) -> int:
    return int(num(v))


def opt(v: object) -> float | None:
    return None if v is None else num(v)


def api_values(m: dict) -> dict:
    """速率与比值全部来自 API: derived 里有的直接用; 解码总速率由计数算 (卡时, 不涉及刷新间隔)。"""
    core = m.get("core", {}) or {}
    der = m.get("derived", {}) or {}
    tot = m.get("totals", {}) or {}
    eng = m.get("engine", {}) or {}
    tokens = (n(core.get("steps")) + n(core.get("rounds"))
              + n(core.get("layer_accepted")) + n(core.get("lookup_accepted")))
    decode_ns = num(core.get("step_ns")) + num(core.get("round_ns")) + num(core.get("lookup_ns"))
    reqs = n(tot.get("requests"))
    recent = m.get("recent") or []
    last = recent[-1] if recent else {}
    pool, kv_used = n(eng.get("pages")), n(core.get("kv_pages_used"))
    return {
        "decode_tps": (tokens / decode_ns * 1e9) if decode_ns > 0 else None,
        "step_tps": opt(der.get("step_tokens_per_s")),
        "layer_tps": opt(der.get("layer_tokens_per_s")),
        "prompt_tps": opt(der.get("prompt_rows_per_s")),
        "step_ms": opt(der.get("step_ms")),
        "round_ms": opt(der.get("round_ms")),
        "draft_accept": opt(der.get("layer_accepted_per_run")),
        "ngram_hit": opt(der.get("ngram_hit_rate")),
        "miss_per_row": opt(der.get("misses_per_row")),
        "busy_share": opt(der.get("busy")),
        "swap_share": opt(der.get("swap_share")),
        "queue_s": (num(tot.get("queue_seconds")) / reqs) if reqs > 0 else None,
        "last_prompt_tps": (max(0, n(last.get("prompt_tokens")) - n(last.get("cached_tokens")))
                            / num(last.get("prompt_ms")) * 1000) if num(last.get("prompt_ms")) > 0 else None,
        "last_decode_tps": (n(last.get("completion_tokens")) / num(last.get("decode_ms")) * 1000)
        if num(last.get("decode_ms")) > 0 and n(last.get("completion_tokens")) >= 2 else None,
        "requests": reqs,
        "errors": n(tot.get("errors")),
        "prompt_tokens": n(tot.get("prompt_tokens")),
        "completion_tokens": n(tot.get("completion_tokens")),
        "cached_tokens": n(tot.get("cached_tokens")),
        "queue_seconds": num(tot.get("queue_seconds")),
        "busy": n(m.get("busy")),
        "slots": n(eng.get("slots")),
        "waiting": n(core.get("waiting")),
        "reading": n(core.get("reading")),
        "writing": n(core.get("writing")),
        "kv_used": kv_used,
        "kv_pool": pool,
        "kv_ratio": (kv_used / pool) if pool > 0 else None,
    }


class History:
    """最近 16 个样本, 只画 sparkline。"""

    KEYS = ("decode_tps", "prompt_tps", "last_decode_tps", "last_prompt_tps",
            "step_ms", "round_ms", "busy_share", "ngram_hit", "miss_per_row")
    MAX = 16

    def __init__(self) -> None:
        self.samples: list[dict] = []

    def add(self, live: dict) -> None:
        self.samples.append({key: live.get(key) for key in self.KEYS})
        del self.samples[: -self.MAX]

    def values(self, key: str) -> list:
        return [sample.get(key) for sample in self.samples]


class RecentLog:
    """API 的 recent 只返回最后 12 条; 每轮和本地已有记录对齐后追加, 不设上限。"""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def update(self, incoming: list) -> None:
        if not incoming:
            return
        k = min(len(self.items), len(incoming))
        while k > 0 and self.items[-k:] != list(incoming[:k]):
            k -= 1
        self.items.extend(incoming[k:])


def spark(values: list, width: int, color: str = ACCENT) -> Text:
    """一行 block 字符画最近 width 个样本; 无样本时为空格。"""
    vals = list(values)[-width:]
    present = [v for v in vals if v is not None]
    if not present:
        return Text(" " * width)
    low, high = min(present), max(present)
    if high - low < 1e-9:
        high = low + 1.0
    body = "".join(" " if v is None else "▁▂▃▄▅▆▇█"[min(7, max(0, int((v - low) / (high - low) * 7.999)))]
                   for v in vals)
    return Text(" " * (width - len(vals)) + body, style=color)


def span(values: list, digits: int = 1) -> str:
    """最近样本的最小-最大值, 没有两个样本时返回 --。"""
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return "--"
    lo, hi = min(valid), max(valid)
    a, b = f"{lo:,.{digits}f}", f"{hi:,.{digits}f}"
    return a if a == b else f"{a}-{b}"


def bar(frac: float | None, width: int, color: str = ACCENT) -> Text:
    if frac is None:
        return Text("░" * width, style="dim")
    filled = int(round(max(0.0, min(1.0, frac)) * width))
    return Text.assemble(Text("█" * filled, style=color), Text("░" * (width - filled), style="dim"))


def fnum(v: float | None, digits: int = 1, unit: str = "") -> str:
    return "--" if v is None else f"{v:,.{digits}f}{unit}"


def ms(v: float | None) -> str:
    return "--" if v is None else f"{v:,.1f} ms"


def pct(v: float | None) -> str:
    return "--" if v is None else f"{v * 100:.1f}%"


# 7x10 像素点阵, 每两行像素压进一行字符 (▀▄█)
_DIGITS: dict[str, tuple[str, ...]] = {
    "0": (" ####  ", "##  ## ", "##  ## ", "##  ## ", "##  ## ",
          "##  ## ", "##  ## ", "##  ## ", "##  ## ", " ####  "),
    "1": ("   ##  ", "  ###  ", " ####  ", "   ##  ", "   ##  ",
          "   ##  ", "   ##  ", "   ##  ", "   ##  ", " ######"),
    "2": (" ####  ", "##  ## ", "    ## ", "    ## ", "   ##  ",
          "  ##   ", " ##    ", "##     ", "##     ", "###### "),
    "3": (" ##### ", "    ## ", "    ## ", "   ##  ", "  ###  ",
          "    ## ", "    ## ", "    ## ", "##  ## ", " ####  "),
    "4": ("   ### ", "  #### ", " ## ## ", "##  ## ", "##  ## ",
          "###### ", "    ## ", "    ## ", "    ## ", "    ## "),
    "5": ("###### ", "##     ", "##     ", "#####  ", "    ## ",
          "    ## ", "    ## ", "    ## ", "##  ## ", " ####  "),
    "6": (" ####  ", "##  ## ", "##     ", "##     ", "#####  ",
          "##  ## ", "##  ## ", "##  ## ", "##  ## ", " ####  "),
    "7": ("###### ", "##  ## ", "    ## ", "   ##  ", "   ##  ",
          "  ##   ", "  ##   ", " ##    ", " ##    ", " ##    "),
    "8": (" ####  ", "##  ## ", "##  ## ", "##  ## ", " ####  ",
          "##  ## ", "##  ## ", "##  ## ", "##  ## ", " ####  "),
    "9": (" ####  ", "##  ## ", "##  ## ", "##  ## ", " ##### ",
          "    ## ", "    ## ", "    ## ", "##  ## ", " ####  "),
    ".": ("   ", "   ", "   ", "   ", "   ",
          "   ", "   ", "   ", " ##", " ##"),
    ",": ("   ", "   ", "   ", "   ", "   ",
          "   ", "   ", "   ", " ##", "## "),
    "-": ("       ", "       ", "       ", "       ", "       ",
          " ##### ", "       ", "       ", "       ", "       "),
}


def _halves(top: str, bottom: str) -> str:
    return "".join("█" if a == "#" and b == "#" else "▀" if a == "#" else "▄" if b == "#" else " "
                   for a, b in zip(top, bottom))


def big_rate(value: float | None, color: str = ACCENT) -> list[Text]:
    """五排数码管 (10 行像素压成 5 行); 超过 9 位退回纯文本。"""
    label = f"{value:,.1f}" if value is not None else "--"
    if value is None or len(label) > 9:
        return [Text(label + " tok/s", style=f"bold {color}")]
    return [Text(" ".join(_halves(_DIGITS[ch][2 * row], _DIGITS[ch][2 * row + 1]) for ch in label),
                 style=f"bold {color}", no_wrap=True) for row in range(5)]


def kv(label: str, value: str) -> Text:
    """副行: 标签暗色, 数值默认前景 (比标题亮)。"""
    return Text.assemble(Text(label, style="dim"), Text(value, style="default"))


def tile(title: str, big: str, sub: str | Text, series: Text, width: int | None = None,
         accent: str = ACCENT, pad: bool = False, extra: Text | None = None) -> Panel:
    body = Text(no_wrap=True, overflow="crop")
    body.append(big + "\n", style=f"bold {accent}")
    if isinstance(sub, Text):
        body.append_text(sub)
        body.append("\n")
    else:
        body.append(sub + "\n", style="default")
    if isinstance(extra, Text):
        body.append_text(extra)
        body.append("\n")
    body.append_text(series)
    if pad:
        body.append("\n")
    return Panel(body, title=Text(title, style="dim"), title_align="left", box=box.ROUNDED,
                 border_style="dim", padding=(0, 1),
                 **({"width": width} if width else {}))


def phase(live: dict, state: str) -> tuple[str, str]:
    if state != "LIVE":
        return "离线", f"bold {RED}"
    if live.get("writing"):
        return "解码", f"bold {ACCENT}"
    if live.get("reading"):
        return "读提示", f"bold {BLUE}"
    if live.get("busy"):
        return "排队", f"bold {AMBER}"
    return "空闲", "dim"


def header(url: str, state: str, model_path: str | None, eng: dict, up_s: float,
           live: dict, width: int, show_info: bool = True, extra: str = "") -> Group:
    left = Text(no_wrap=True, overflow="crop")
    left.append("C0VEN 总览", style=f"bold {ACCENT}")
    left.append("   " + url, style=ACCENT)
    left.append("   " + os.path.basename(model_path or ""), style="bold")
    label, style = phase(live, state)
    right = Text(no_wrap=True)
    right.append("● " + label, style=style)
    right.append("   ")
    right.append(f"[{state}]", style=STATE_STYLE[state])
    pad = max(2, width - cell_len(left.plain) - cell_len(right.plain))
    line = Text.assemble(left, " " * pad, right, no_wrap=True, overflow="crop")
    info = Text(f"{eng.get('version', '-')} · {n(eng.get('cards'))} 卡 · 上下文 {n(eng.get('max_context')):,} tok"
                f" · 运行 {up_s:,.0f}s · pid {n(eng.get('pid')):,}{extra}", style="dim")
    parts = [line, Text("─" * width, style="dim")]
    if show_info:
        parts.append(info)
    return Group(*parts)


def k(v: float | None, digits: int = 1) -> str:
    if v is None:
        return "--"
    return f"{v / 1e6:.{digits}f}M" if v >= 1e6 else f"{v / 1e3:.{digits}f}K" if v >= 1e3 else f"{v:,.0f}"


def speed_tile(title: str, digits: list[Text], note: Text, note2: Text, series: Text,
               width: int | None = None) -> Panel:
    """与 Strata tile 同构: 标题 (标签) / 大数字 / 两行说明 / 走势。"""
    body = Text(no_wrap=True, overflow="crop")
    for row in digits:
        body.append_text(row)
        body.append("\n")
    body.append_text(note)
    body.append("\n")
    body.append_text(note2)
    body.append("\n")
    body.append_text(series)
    return Panel(body, title=Text(title, style="dim"), title_align="left", box=box.ROUNDED,
                 border_style="dim", padding=(0, 1),
                 **({"width": width} if width else {}))


def tiles(m: dict, live: dict, hist: History, core: dict, eng: dict, width: int,
          interval: float) -> tuple:
    """两行四列等宽小格子, 上面两块速度大屏。"""
    tile_w = max(18, (width - 3) // 4)
    inner = max(6, tile_w - 4)
    block_w = 2 * tile_w + 1
    window = History.MAX * interval

    hero = Table.grid(padding=(0, 1), expand=True)
    for _ in range(2):
        hero.add_column(ratio=1)
    hero.add_row(
        speed_tile("提示速度 tok/s（上一条请求）", big_rate(live.get("last_prompt_tps"), BLUE),
                   kv("平均 ", fnum(live.get("prompt_tps"), 1) + " tok/s"),
                   Text.assemble(kv("分块 ", fnum(opt(core.get('chunk_ms')), 0, ' ms')),
                                 kv(" · 每行 ", ms(opt(core.get('row_ms'))))),
                   spark(hist.values("last_prompt_tps"), block_w - 6, BLUE)),
        speed_tile("解码速度 tok/s（上一条请求）", big_rate(live.get("last_decode_tps")),
                   kv("平均 ", fnum(live.get("decode_tps"), 1) + " tok/s"),
                   Text.assemble(kv("普通步 ", fnum(live.get('step_tps'), 0) + " tok/s"),
                                 kv(" · 草稿轮 ", fnum(live.get('layer_tps'), 0) + " tok/s")),
                   spark(hist.values("last_decode_tps"), block_w - 6)),
    )

    grid = Table.grid(padding=(0, 1), expand=True)
    for _ in range(4):
        grid.add_column(ratio=1)
    grid.add_row(
        tile("单步耗时", fnum(live.get("step_ms"), 1, " ms"),
             Text.assemble(kv("草稿轮 ", fnum(live.get('round_ms'), 0, ' ms'))),
             spark(hist.values("step_ms"), inner),
             extra=Text.assemble(kv("每轮草稿接受 ", fnum(live.get('draft_accept'), 2) + " 个"))),
        tile("查表命中", pct(live.get("ngram_hit")),
             Text.assemble(kv("命中 ", f"{k(core.get('ngram_hits'))} / {k(core.get('ngram_rows'))} 行")),
             bar(live.get("ngram_hit"), inner,
                 AMBER if (live.get("ngram_hit") or 0) < 0.5 else ACCENT),
             extra=Text.assemble(kv("查表读取 ", f"{k(core.get('ngram_reads'))} 行"))),
        tile("槽位占用", f"{live['busy']} 请求 · {live['slots']} 槽",
             Text.assemble(kv("等待 ", str(live['waiting'])), kv(" 读提示 ", str(live['reading'])),
                           kv(" 解码 ", str(live['writing']))),
             bar(live["busy"] / live["slots"] if live["slots"] else None, inner,
                 RED if live["slots"] and live["busy"] > live["slots"] else ACCENT),
             extra=Text.assemble(kv("平均排队 ", fnum(live.get('queue_s'), 2, ' s')))),
        tile("引擎线程", pct(live.get("busy_share")),
             Text.assemble(kv("换出占比 ", pct(live.get('swap_share')))),
             spark(hist.values("busy_share"), inner,
                   AMBER if (live.get("busy_share") or 0) >= 0.95 else ACCENT),
             extra=Text.assemble(kv("空闲轮 ", f"{n(core.get('idle_rounds'))} 轮"))),
    )
    grid.add_row(
        tile("KV 页", f"{live['kv_used']:,}/{live['kv_pool']:,}",
             Text.assemble(kv("ckpt ", f"{n(core.get('kv_pages_checkpoints'))} 个"),
                           kv(" · 占用 ", pct(live.get('kv_ratio')))),
             bar(live.get("kv_ratio"), inner,
                 AMBER if (live.get("kv_ratio") or 0) >= 0.9 else ACCENT),
             extra=Text.assemble(kv("槽位持有 ", f"{n(core.get('kv_pages_slots')):,} 页"))),
        tile("专家缓存", f"{n(core.get('experts_cached'))}/{n(core.get('experts_floor'))}",
             Text.assemble(kv("每行未命中 ", fnum(live.get('miss_per_row'), 1, ' 个'))),
             spark(hist.values("miss_per_row"), inner),
             extra=Text.assemble(kv("下限 ", f"{n(core.get('experts_floor'))} 个"))),
        tile("缺页", f"{n(core.get('major_faults')):,} 次",
             Text.assemble(kv("停顿 ", f"{n(core.get('stalls'))} 次"),
                           kv("·空转 ", f"{n(core.get('idle_rounds'))} 轮")),
             Text("自启动累计", style="dim"),
             extra=Text.assemble(kv("换出到主机 ", f"{n(core.get('swapped_out_pages'))} 页"))),
        tile("会话累计", f"{live['requests']} 请求",
             Text.assemble(kv("提示 ", f"{k(live.get('prompt_tokens'), 0)}"),
                           kv(" 输出 ", f"{k(live.get('completion_tokens'), 0)} tok")),
             Text.assemble(kv("累计排队 ", fnum(live.get('queue_seconds'), 0, ' s'))),
             extra=Text.assemble(kv("错误 ", str(live['errors'])),
                                 kv(" · 缓存 ", f"{k(live.get('cached_tokens'))} tok"))),
    )
    return hero, grid


def finish_style(finish: object) -> str:
    return {"stop": ACCENT, "tool_calls": BLUE, "length": AMBER}.get(str(finish), "dim")


def req_pp(r: dict) -> str:
    """与 engine 的 prompt_tokens_per_sec 一致: 缓存命中的前缀不读, 不算进提示速度。"""
    pm, pt_, cached = num(r.get("prompt_ms")), n(r.get("prompt_tokens")), n(r.get("cached_tokens"))
    return f"{max(0, pt_ - cached) / pm * 1000:,.0f}" if pm > 0 else "--"


def req_tg(r: dict) -> str:
    dm, ct = num(r.get("decode_ms")), n(r.get("completion_tokens"))
    return f"{ct / dm * 1000:,.1f}" if dm > 0 and ct >= 2 else "--"


def _f(v: object) -> str:
    """fit_* 的一部分: "-" 和空值显示为 —, 下划线换成空格。"""
    text = str(v if v is not None else "")
    return "" if text in ("", "-") else text.replace("_", " ")


def gbs(v: str) -> str:
    """45.5GB/s -> 45.5 GB/s, 单位前留空格。"""
    return v.replace("GB/s", " GB/s")


def _fit(raw: object) -> list[str]:
    """host/plan/core 三段; 数值里的 GB/s 不能被当成分隔符。"""
    parts = str(raw or "").split("/")
    out: list[str] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] == "s":
            out.append(parts[i] + "/s")
            i += 2
        else:
            out.append(parts[i])
            i += 1
    return out


def facts(m: dict, live: dict, core: dict, eng: dict, width: int) -> tuple[Panel, Panel]:
    """硬件面板 (卡/内存/CPU/拷贝) 和统计面板 (载入/窗口/draft), 都放在各自的位置。"""
    def sheet() -> tuple[Table, object]:
        t = Table.grid(padding=(0, 2))
        t.add_column(width=6)
        t.add_column()

        def row(label: str, *parts) -> None:
            t.add_row(Text(label, style="dim"), Text.assemble(*parts))
        return t, row

    hw, hw_row = sheet()
    st, st_row = sheet()
    dot = Text(" · ", style="dim")
    der = m.get("derived") or {}

    for i in (0, 1):
        raw = eng.get(f"fit_card{i}")
        if not raw:
            continue
        host, plan, _ = (_fit(raw) + ["", "", ""])[:3]
        f = host.split(",")
        name = (f[0].replace("(R)", "").replace("(TM)", "").replace("_", " ")
                .replace(" Graphics", "") if f else "")
        vram = _f(f[1]) if len(f) > 1 else ""
        cores = _f(f[3]).replace("Xe cores", "Xe 核") if len(f) > 3 else ""
        pcie = _f(f[4]) if len(f) > 4 else ""
        used = _f(plan)
        parts = [Text(name, style="default")]
        for x in (cores, vram, pcie):
            if x:
                parts += [dot, Text(x, style="default")]
        if used:
            parts += [dot, kv("占用 ", used)]
        hw_row(f"GPU {i}", *parts)

    avail, arena = _fit(eng.get("fit_mem_available")), _fit(eng.get("fit_arena"))
    pinned, swap = _fit(eng.get("fit_pinned")), _fit(eng.get("fit_swap"))

    def pick(parts: list[str]) -> str:
        return "/".join(x for x in parts if x and x != "-") or "—"

    hw_row("内存", kv("可用 ", "/".join(avail[:3]) if any(avail) else "—"),
           Text("（主机/规划/核心）", style="dim"))
    hw_row("", kv("arena ", pick(arena)), dot, kv("pinned ", pick(pinned)),
           dot, kv("交换 ", _f(swap[0].split(",")[0]) if swap and swap[0] else "—"))

    cpu = _fit(eng.get("fit_cpu"))
    cpu_host = cpu[0].split(",") if cpu and cpu[0] else []
    cpu_name = (cpu_host[0].replace("_", " ").replace(" 12-Core Processor", "")
                if cpu_host else "")
    workers = (_f(cpu[2]).replace("workers", "工作线程") if len(cpu) > 2 else "")
    read, loader = _fit(eng.get("fit_host_read")), _fit(eng.get("fit_loader"))
    threads = _f(loader[2]).replace("threads", "线程") if len(loader) > 2 else ""
    hw_row("CPU", kv("型号 ", cpu_name),
           dot, Text(", ".join(x.replace("CPUs", "CPU").replace("cores", "核")
                               for x in (_f(v) for v in cpu_host[1:3]) if x), style="default"),
           dot, Text(workers, style="default"),
           dot, kv("读 ", gbs(_f(read[2])) if len(read) > 2 else "—"),
           dot, kv("loader ", gbs(_f(loader[0])) or "—"),
           kv(" · ", threads))

    c0, c1, ngram = _fit(eng.get("fit_copy0")), _fit(eng.get("fit_copy1")), _fit(eng.get("fit_ngram"))
    hw_row("拷贝", kv("卡0 ", gbs(_f(c0[2])) if len(c0) > 2 else "—"),
           dot, kv("卡1 ", gbs(_f(c1[2])) if len(c1) > 2 else "—"),
           dot, kv("ngram ", (_f(ngram[0]).replace("us a 4KiB read", " us / 4 KiB 读")
                              if ngram and ngram[0] else "—")))

    st_row("载入", kv("耗时 ", fnum(opt(eng.get("load_s")), 1, " s")),
           dot, kv("后端 ", _f(eng.get("backend")) or "—"),
           dot, kv("计划数 ", f"{n(core.get('plans')):,}"),
           dot, kv("swap ", f"{n(core.get('swaps')):,} 页 / {n(core.get('swap_rounds')):,} 轮"))
    st_row("", kv("checkpoint ", f"存 {n(core.get('checkpoints_taken'))} / 取 {n(core.get('restores'))}"),
           dot, kv("冷启 ", str(n(core.get("cold_admits")))),
           dot, kv("欠时 ", fnum(opt(core.get("owed_ms")), 0, " ms")))
    st_row("窗口", kv("读入 ", fnum(der.get("window_rows_per_s"), 0, " 行/s")),
           dot, kv("lookup ", fnum(der.get("lookup_tokens_per_s"), 1, " tok/s")),
           dot, kv("draft pass ", fnum(der.get("draft_pass_ms"), 1, " ms")),
           kv(" (", pct(der.get("draft_pass_share")) + ")"),
           dot, kv("未草稿步 ", pct(der.get("undrafted_share"))),
           dot, kv("读提示轮 ", pct(der.get("reading_round_share"))))
    depths = " ".join(f"@{d} {pct(der.get(f'layer_accepted_at_{d}'))}"
                      for d in range(1, 8) if der.get(f"layer_accepted_at_{d}") is not None)
    stops = " ".join(f"{label} {pct(der.get(f'layer_stopped_at_{key}'))}"
                     for key, label in (("max", "到顶"), ("share", "到份额"),
                                        ("p", "低于阈值"), ("choice", "核心选"))
                     if der.get(f"layer_stopped_at_{key}") is not None)
    if depths:
        st_row("draft", kv("接受 ", depths))
    if stops:
        st_row("", kv("停止 ", stops))

    return (Panel(st, title=Text("统计", style="dim"), title_align="left", border_style="dim",
                  box=box.ROUNDED, width=width, padding=(0, 1)),
            Panel(hw, title=Text("硬件", style="dim"), title_align="left", border_style="dim",
                  box=box.ROUNDED, width=width, padding=(0, 1)))


def recent_table(recent: list, width: int) -> Panel:
    table = Table(box=None, pad_edge=False, expand=True)
    for col in ("结束", "提示 tok", "缓存 tok", "输出 tok", "耗时", "排队", "提示 tok/s", "解码 tok/s",
                "步数", "草稿", "接受"):
        table.add_column(col, justify="left" if col == "结束" else "right", style="dim" if col == "结束" else "")
    for r in recent:
        table.add_row(
            Text(str(r.get("finish", "-")), style=finish_style(r.get("finish"))),
            f"{n(r.get('prompt_tokens')):,}", f"{n(r.get('cached_tokens')):,}",
            f"{n(r.get('completion_tokens')):,}",
            f"{num(r.get('seconds')):,.3f}s", ms(opt(r.get("queue_ms"))), req_pp(r), req_tg(r),
            f"{n(r.get('steps')):,}", f"{n(r.get('drafts')):,}", f"{n(r.get('accepted')):,}",
        )
    return Panel(table, title=Text(f"最近请求 · {len(recent)}", style="dim"), title_align="left",
                 border_style="dim", box=box.ROUNDED, padding=(0, 1), width=width)


def hint(interval: float) -> Text:
    return Text(f"↑↓ PgUp/PgDn 滚动整页   g/G 到顶/到底   Ctrl+C 退出   · 每 {interval:g}s 刷新", style="dim")


def render(m: dict, live: dict, hist: History, state: str, model_path: str | None,
           url: str, interval: float, width: int) -> Group:
    core = m.get("core", {})
    eng = m.get("engine", {})
    up_s = num(core.get("up_ns")) / 1e9
    hero, grid = tiles(m, live, hist, core, eng, width, interval)
    stats, hardware = facts(m, live, core, eng, width)
    body = Table.grid()
    body.add_column(width=width)
    body.add_row(hero)
    body.add_row(Text(""))
    body.add_row(grid)
    blank = [Text("")]
    return Group(header(url, state, model_path, eng, up_s, live, width),
                 *blank, body, *blank,
                 stats, *blank,
                 recent_table(m.get("recent", []), width), *blank,
                 hardware, *blank, hint(interval))


def page(console: Console, group: Group, offset: int | None) -> tuple[Segments, int, int]:
    """整页渲染成行, 取屏幕高度的一段; offset 是距底部的行数 (0 = 跟随最新)。"""
    lines = console.render_lines(group, console.options, pad=False)
    height = max(1, console.size.height)
    max_off = max(0, len(lines) - height)
    offset = max_off if offset is None else max(0, min(offset, max_off))
    start = max(0, len(lines) - height - offset)
    segs: list[Segment] = []
    for line in lines[start:start + height]:
        segs.extend(line)
        segs.append(Segment.line())
    return Segments(segs), offset, max_off


class Keys:
    """cbreak 模式读按键, 退出时还原本来的终端设置。"""

    SEQS = ((b"\x1b[A", "up"), (b"\x1bOA", "up"), (b"\x1b[B", "down"), (b"\x1bOB", "down"),
            (b"\x1b[5~", "pgup"), (b"\x1b[6~", "pgdn"),
            (b"\x1b[H", "home"), (b"\x1bOH", "home"), (b"\x1b[F", "end"), (b"\x1bOF", "end"))

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.attrs = None

    def __enter__(self) -> "Keys":
        if self.fd is not None:
            self.attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.attrs is not None and self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.attrs)

    def read(self) -> list[str]:
        if self.fd is None:
            return []
        try:
            ready, _, _ = select.select([self.fd], [], [], 0)
        except (OSError, ValueError):
            return []
        if not ready:
            return []
        try:
            data = os.read(self.fd, 32)
        except OSError:
            return []
        out = [name for seq, name in self.SEQS if seq in data]
        if not out:
            for ch in data.decode(errors="ignore"):
                if ch in "j":
                    out.append("down")
                elif ch in "k":
                    out.append("up")
                elif ch == "g":
                    out.append("home")
                elif ch == "G":
                    out.append("end")
        return out

    @staticmethod
    def scroll(offset: int, key: str, total: int, visible: int) -> int:
        if key == "up":
            offset += 1
        elif key == "down":
            offset -= 1
        elif key == "pgup":
            offset += visible
        elif key == "pgdn":
            offset -= visible
        elif key == "home":
            offset = total
        elif key == "end":
            offset = 0
        return max(0, min(offset, total))


def main() -> None:
    p = argparse.ArgumentParser(prog="monitor", description="c0ven 终端监控, 每 2s 刷新")
    p.add_argument("url", nargs="?", default="http://127.0.0.1:8080")
    p.add_argument("--interval", type=float, default=2.0)
    args = p.parse_args()

    console = Console()
    hist = History()
    log = RecentLog()
    last_good = 0.0
    model_path: str | None = None
    live: dict = {}
    prev: dict = {}
    offset: int | None = None

    with Live(Group(header(args.url, "DISCONNECTED", None, {}, 0.0, {}, MAX_WIDTH),
                    Text("连接中…", style="dim")), console=console, screen=True,
              refresh_per_second=4) as live_view, Keys() as keys:
        while True:
            t0 = time.monotonic()
            m: dict | None = None
            try:
                m = fetch(args.url + "/metrics")
                if model_path is None:
                    try:
                        model_path = fetch(args.url + "/props").get("model_path")
                    except Exception:
                        pass
                last_good = time.monotonic()
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                pass

            if m is not None:
                live = api_values(m)
                hist.add(live)
                log.update(m.get("recent", []))
                state = "LIVE"
            else:
                state = "STALE" if last_good and (time.monotonic() - last_good) < 30 else "DISCONNECTED"
                m = prev
            m = {**m, "recent": log.items}
            prev = m
            group: Group = (render(m, live, hist, state, model_path, args.url, args.interval,
                                   min(console.size.width, MAX_WIDTH)) if m.get("core") else
                            Group(header(args.url, state, model_path, {}, 0.0, live, MAX_WIDTH),
                                  Text("没有数据, 等下一次刷新…", style="dim")))
            view, offset, max_off = page(console, group, offset)
            pressed = keys.read()
            if pressed:
                for key in pressed:
                    offset = Keys.scroll(offset, key, max_off, console.size.height)
                view, offset, max_off = page(console, group, offset)
            live_view.update(view)

            deadline = t0 + args.interval
            while (left := deadline - time.monotonic()) > 0:
                time.sleep(min(0.1, left))
                pressed = keys.read()
                if not pressed:
                    continue
                before = offset
                for key in pressed:
                    offset = Keys.scroll(offset, key, max_off, console.size.height)
                if offset != before:
                    view, offset, max_off = page(console, group, offset)
                    live_view.update(view)


if __name__ == "__main__":
    main()

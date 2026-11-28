#!/usr/bin/env python3
"""c0ven 终端监控, 版式照 Strata 的 Overview: 标题行 + 横线 + 四列信息块 (大数字/副行/sparkline)
+ 最近请求表, 宽度上限 110 列。速率与比值一律取 /metrics 的 derived/计数 (实时那栏是计数的窗口差分), 不用墙钟差。
用法: monitor.py [URL] [--interval 1]

刷新方式 (照 Strata: 轮询与绘制分开, 谁都不等谁):
- /metrics 由后台线程抓, 渲染循环不等网络 (引擎忙时抓取会超过 2s);
- 整页只在数据/终端尺寸变化时渲染一次, 按键只换切片不重新渲染;
- 自己画屏 (Painter), 每帧只重画内容变了的行, 所以没有 rich Live 那种 4 次/秒整屏重写。
两块大屏的大数字是实时速率 (Rates): 最近 3*interval 内的 core 计数器差分, 与平均同口径; 没跑的那一相就是 0。
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
import urllib.error
import urllib.request

from rich import box
from rich.cells import cell_len
from rich.console import Console, Group
from rich.panel import Panel
from rich.segment import Segment, Segments
from rich.table import Table
from rich.text import Text

ACCENT = "#34d399"
BLUE = "#5aa9ff"
AMBER = "#f5c56b"
RED = "#f28b82"
MAX_WIDTH = 110
SCROLL_STEP = 5

STATE_STYLE = {"LIVE": f"bold {ACCENT}", "STALE": f"bold {AMBER}", "DISCONNECTED": f"bold {RED}"}

class Poller:
    """后台抓 /metrics (和一次 /props), 只留最新一份; 抓取慢或超时不影响界面。"""

    def __init__(self, url: str, interval: float) -> None:
        self.url = url
        self.interval = interval
        self.lock = threading.Lock()
        self._payload: dict | None = None
        self._props: dict | None = None
        self._model_id: str | None = None
        self.last_good = 0.0
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=2) as resp:
            return json.loads(resp.read().decode())

    def _run(self) -> None:
        while True:
            try:
                payload = self._get("/metrics")
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                pass
            else:
                with self.lock:
                    if not payload.get("core") and self._payload:
                        # 引擎在跑长提示时 STATS 会超时 (实测 11s 不答), totals 还活着: 只保住上一份 core
                        payload = {**payload, "core": self._payload["core"],
                                   "derived": self._payload.get("derived") or {}}
                    self._payload = payload
                    self.last_good = time.monotonic()
                if self._props is None:
                    try:
                        self._props = self._get("/props")
                        self._model_id = ((self._get("/v1/models").get("data") or [{}])[0]).get("id")
                    except Exception:
                        pass
            if self._stop.wait(self.interval):
                return

    def current(self) -> tuple[dict | None, dict | None, str | None]:
        with self.lock:
            return self._payload, self._props, self._model_id


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
            "live_decode_tps", "live_prompt_tps",
            "step_ms", "round_ms", "busy_share", "ngram_hit", "miss_per_row")
    MAX = 16

    def __init__(self) -> None:
        self.samples: list[dict] = []

    def add(self, live: dict) -> None:
        self.samples.append({key: live.get(key) for key in self.KEYS})
        del self.samples[: -self.MAX]

    def values(self, key: str) -> list:
        return [sample.get(key) for sample in self.samples]


class Rates:
    """实时速率: 和平均同一口径 (tokens / 卡时), 只是分子分母都取最近 window_s 内的计数器差分
    (engine_api.h: 读法就是取两次快照做差)。最近两轮没动的那一相直接归 0, 免得相位结束后还挂着旧数。"""

    DECODE_NS = ("step_ns", "round_ns", "lookup_ns")
    DECODE_TOKENS = ("steps", "rounds", "layer_accepted", "lookup_accepted")
    COUNTERS = DECODE_NS + DECODE_TOKENS + ("prompt_ns", "prompt_rows")

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.samples: list[tuple[float, dict]] = []
        self.decode = 0.0
        self.prompt = 0.0

    def add(self, core: dict, now: float) -> None:
        sample = {key: num(core.get(key)) for key in self.COUNTERS}
        self.samples = [(t, s) for t, s in self.samples if now - t <= self.window_s]
        first = self.samples[0][1] if self.samples else sample
        delta = {key: sample[key] - first[key] for key in self.COUNTERS}
        if any(value < 0 for value in delta.values()):
            self.samples, first = [(now, sample)], sample       # 引擎重启, 计数器回到小值
            delta = dict.fromkeys(self.COUNTERS, 0.0)
        else:
            self.samples.append((now, sample))
        prev = self.samples[-3][1] if len(self.samples) > 2 else first
        fresh = {key: sample[key] - prev[key] for key in self.COUNTERS}
        self.decode = self._rate(delta, fresh, self.DECODE_NS, self.DECODE_TOKENS)
        self.prompt = self._rate(delta, fresh, ("prompt_ns",), ("prompt_rows",))

    @staticmethod
    def _rate(delta: dict, fresh: dict, ns_keys: tuple, tok_keys: tuple) -> float:
        ns = sum(delta[key] for key in ns_keys)
        if ns <= 0 or sum(fresh[key] for key in ns_keys) <= 0:
            return 0.0
        return sum(delta[key] for key in tok_keys) / ns * 1e9


class RecentLog:
    """API 的 recent 只返回最后 12 条; 每轮和本地已有记录对齐后追加, 最多留 30 条。"""

    MAX = 30

    def __init__(self) -> None:
        self.items: list[dict] = []

    def update(self, incoming: list) -> None:
        if not incoming:
            return
        k = min(len(self.items), len(incoming))
        while k > 0 and self.items[-k:] != list(incoming[:k]):
            k -= 1
        self.items.extend(incoming[k:])
        del self.items[: -self.MAX]


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
        return "offline", f"bold {RED}"
    if live.get("writing"):
        return "decoding", f"bold {ACCENT}"
    if live.get("reading"):
        return "prefilling", f"bold {BLUE}"
    if live.get("busy"):
        return "queued", f"bold {AMBER}"
    return "idle", "dim"


def sampling(props: dict | None) -> str:
    """模型自带的采样默认值 (/props 的 generation_settings), 缺的字段不写。"""
    gen = (props or {}).get("generation_settings") or {}
    pairs = (
        ("temp", f"{num(gen['temperature']):.2f}" if "temperature" in gen else ""),
        ("top_p", f"{num(gen['top_p']):.3f}" if "top_p" in gen else ""),
        ("top_k", f"{n(gen['top_k'])}" if "top_k" in gen else ""),
        ("max", f"{n(gen['max_tokens']):,}" if gen.get("max_tokens")
                else "unset" if "max_tokens" in gen else ""),
    )
    return " · ".join(f"{name} {value}" for name, value in pairs if value)


def model_panel(props: dict | None, name: str | None, eng: dict, up_s: float, width: int) -> Panel:
    """模型与生成配置 (/props, /v1/models), 版式同 Hardware。"""
    t = Table.grid(padding=(0, 2))
    t.add_column(width=8)
    t.add_column()

    def row(label: str, *parts) -> None:
        t.add_row(Text(label, style="dim"), Text.assemble(*parts))

    dot = Text(" · ", style="dim")
    path = (props or {}).get("model_path") or ""
    template = (props or {}).get("chat_template") or ""
    folder = os.path.basename(os.path.dirname(path)) if path else ""

    parts = [Text(name or os.path.basename(path) or "—", style="default")]
    if folder:
        parts += [dot, Text(folder, style="default")]
    if "-of-" in path:  # 分片名形如 -00001-of-00002.gguf
        index, total = path.rsplit("-of-", 1)
        try:
            parts += [dot, kv("shard ", f"{int(index.rsplit('-', 1)[-1])}/{int(total.split('.')[0])}")]
        except ValueError:
            pass
    row("Model", *parts)
    row("Path", Text(path or "—", style="dim", no_wrap=True, overflow="crop"))
    row("Engine", Text(str(eng.get("arch") or "—")), dot, Text(str(eng.get("backend") or "—")), dot,
        Text(str(eng.get("version") or "—")), dot, kv("context ", f"{n(eng.get('max_context')):,} tok"), dot,
        kv("", f"{n(eng.get('slots'))} slot"), dot, kv("", f"{n(eng.get('cards'))} cards"))
    row("Sampling", Text(sampling(props) or "—"))
    vision = n(eng.get("image_tokens_max"))
    row("MoE", kv("experts ", f"{n(eng.get('experts_planned'))} planned / {n(eng.get('experts_cached'))} cached"), dot,
        kv("pages ", f"{n(eng.get('pages')):,}"), dot,
        kv("vision ", f"up to {vision:,} tok" if vision else "off"))
    switches = [w for w in ("enable_thinking", "reasoning_effort", "preserve_thinking") if w in template]
    row("Template", kv("chars ", f"{len(template):,}") if template else Text("—"),
        *(p for w in switches for p in (dot, Text(w, style="default"))))
    row("Runtime", kv("pid ", f"{n(eng.get('pid')):,}"), dot, kv("up ", f"{up_s:,.0f}s"))
    return Panel(t, title=Text("Model", style="dim"), title_align="left", border_style="dim",
                 box=box.ROUNDED, width=width, padding=(0, 1))


def header(url: str, state: str, live: dict, width: int) -> Group:
    left = Text(no_wrap=True, overflow="crop")
    left.append("C0VEN OVERVIEW", style=f"bold {ACCENT}")
    left.append("   " + url, style=ACCENT)
    label, style = phase(live, state)
    right = Text(no_wrap=True)
    right.append("● " + label, style=style)
    right.append("   ")
    right.append(f"[{state}]", style=STATE_STYLE[state])
    pad = max(2, width - cell_len(left.plain) - cell_len(right.plain))
    line = Text.assemble(left, " " * pad, right, no_wrap=True, overflow="crop")
    return Group(line, Text("─" * width, style="dim"))


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

    hero = Table.grid(padding=(0, 1), expand=True)
    for _ in range(2):
        hero.add_column(ratio=1)
    hero.add_row(
        speed_tile("Prompt speed tok/s (live)", big_rate(live.get("live_prompt_tps"), BLUE),
                   Text.assemble(kv("last request ", fnum(live.get('last_prompt_tps'), 1)),
                                 kv(" tok/s · mean ", fnum(live.get('prompt_tps'), 1) + " tok/s")),
                   Text.assemble(kv("chunk ", fnum(opt(core.get('chunk_ms')), 0, ' ms')),
                                 kv(" · per row ", ms(opt(core.get('row_ms'))))),
                   spark(hist.values("live_prompt_tps"), block_w - 6, BLUE)),
        speed_tile("Decode speed tok/s (live)", big_rate(live.get("live_decode_tps")),
                   Text.assemble(kv("last request ", fnum(live.get('last_decode_tps'), 1)),
                                 kv(" tok/s · mean ", fnum(live.get('decode_tps'), 1) + " tok/s")),
                   Text.assemble(kv("plain steps ", fnum(live.get('step_tps'), 0) + " tok/s"),
                                 kv(" · draft rounds ", fnum(live.get('layer_tps'), 0) + " tok/s")),
                   spark(hist.values("live_decode_tps"), block_w - 6)),
    )

    grid = Table.grid(padding=(0, 1), expand=True)
    for _ in range(4):
        grid.add_column(ratio=1)
    grid.add_row(
        tile("Step time", fnum(live.get("step_ms"), 1, " ms"),
             Text.assemble(kv("draft round ", fnum(live.get('round_ms'), 0, ' ms'))),
             spark(hist.values("step_ms"), inner),
             extra=Text.assemble(kv("accepted per run ", fnum(live.get('draft_accept'), 2)))),
        tile("n-gram hits", pct(live.get("ngram_hit")),
             Text.assemble(kv("hits ", f"{k(core.get('ngram_hits'))} / {k(core.get('ngram_rows'))} rows")),
             bar(live.get("ngram_hit"), inner,
                 AMBER if (live.get("ngram_hit") or 0) < 0.5 else ACCENT),
             extra=Text.assemble(kv("SSD reads ", f"{k(core.get('ngram_reads'))} rows"))),
        tile("Slot usage", f"{live['busy']} requests · {live['slots']} slots",
             Text.assemble(kv("wait ", str(live['waiting'])), kv(" read ", str(live['reading'])),
                           kv(" write ", str(live['writing']))),
             bar(live["busy"] / live["slots"] if live["slots"] else None, inner,
                 RED if live["slots"] and live["busy"] > live["slots"] else ACCENT),
             extra=Text.assemble(kv("mean wait ", fnum(live.get('queue_s'), 2, ' s')))),
        tile("Engine busy", pct(live.get("busy_share")),
             Text.assemble(kv("swap share ", pct(live.get('swap_share')))),
             spark(hist.values("busy_share"), inner,
                   AMBER if (live.get("busy_share") or 0) >= 0.95 else ACCENT),
             extra=Text.assemble(kv("idle rounds ", f"{n(core.get('idle_rounds'))}"))),
    )
    grid.add_row(
        tile("KV pages", f"{live['kv_used']:,}/{live['kv_pool']:,}",
             Text.assemble(kv("ckpt ", f"{n(core.get('kv_pages_checkpoints'))}"),
                           kv(" · used ", pct(live.get('kv_ratio')))),
             bar(live.get("kv_ratio"), inner,
                 AMBER if (live.get("kv_ratio") or 0) >= 0.9 else ACCENT),
             extra=Text.assemble(kv("held by slots ", f"{n(core.get('kv_pages_slots')):,}"))),
        tile("Expert cache", f"{n(core.get('experts_cached'))}/{n(core.get('experts_floor'))}",
             Text.assemble(kv("misses per row ", fnum(live.get('miss_per_row'), 1))),
             spark(hist.values("miss_per_row"), inner),
             extra=Text.assemble(kv("floor ", f"{n(core.get('experts_floor'))}"))),
        tile("Major faults", f"{n(core.get('major_faults')):,}",
             Text.assemble(kv("stalls ", f"{n(core.get('stalls'))}"),
                           kv(" · idle ", f"{n(core.get('idle_rounds'))}")),
             Text("since open", style="dim"),
             extra=Text.assemble(kv("swapped out ", f"{n(core.get('swapped_out_pages'))}"))),
        tile("Session totals", f"{live['requests']} requests",
             Text.assemble(kv("prompt ", f"{k(live.get('prompt_tokens'), 0)}"),
                           kv(" output ", f"{k(live.get('completion_tokens'), 0)}")),
             Text.assemble(kv("queued ", fnum(live.get('queue_seconds'), 0, ' s'))),
             extra=Text.assemble(kv("errors ", str(live['errors'])),
                                 kv(" · cached ", f"{k(live.get('cached_tokens'))}"))),
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
        cores = _f(f[3]) if len(f) > 3 else ""
        pcie = _f(f[4]) if len(f) > 4 else ""
        used = _f(plan)
        parts = [Text(name, style="default")]
        for x in (cores, vram, pcie):
            if x:
                parts += [dot, Text(x, style="default")]
        if used:
            parts += [dot, kv("used ", used)]
        hw_row(f"GPU {i}", *parts)

    avail, arena = _fit(eng.get("fit_mem_available")), _fit(eng.get("fit_arena"))
    pinned, swap = _fit(eng.get("fit_pinned")), _fit(eng.get("fit_swap"))

    def pick(parts: list[str]) -> str:
        return "/".join(x for x in parts if x and x != "-") or "—"

    hw_row("Memory", kv("available ", "/".join(avail[:3]) if any(avail) else "—"),
           Text(" (host / plan / core)", style="dim"))
    hw_row("", kv("arena ", pick(arena)), dot, kv("pinned ", pick(pinned)),
           dot, kv("swap ", _f(swap[0].split(",")[0]) if swap and swap[0] else "—"))

    cpu = _fit(eng.get("fit_cpu"))
    cpu_host = cpu[0].split(",") if cpu and cpu[0] else []
    cpu_name = (cpu_host[0].replace("_", " ").replace(" 12-Core Processor", "")
                if cpu_host else "")
    workers = _f(cpu[2]) if len(cpu) > 2 else ""
    read, loader = _fit(eng.get("fit_host_read")), _fit(eng.get("fit_loader"))
    threads = _f(loader[2]) if len(loader) > 2 else ""
    hw_row("CPU", Text(cpu_name, style="default"),
           dot, Text(", ".join(x for x in (_f(v) for v in cpu_host[1:3]) if x), style="default"),
           dot, Text(workers, style="default"))
    hw_row("", kv("read ", gbs(_f(read[2])) if len(read) > 2 else "—"),
           dot, kv("loader ", gbs(_f(loader[0])) or "—"),
           kv(" · ", threads))

    c0, c1, ngram = _fit(eng.get("fit_copy0")), _fit(eng.get("fit_copy1")), _fit(eng.get("fit_ngram"))
    hw_row("Copies", kv("card 0 ", gbs(_f(c0[2])) if len(c0) > 2 else "—"),
           dot, kv("card 1 ", gbs(_f(c1[2])) if len(c1) > 2 else "—"),
           dot, kv("n-gram ", (_f(ngram[0]).replace("us a 4KiB read", " us per 4 KiB read")
                               if ngram and ngram[0] else "—")))

    st_row("Load", kv("time ", fnum(opt(eng.get("load_s")), 1, " s")),
           dot, kv("backend ", _f(eng.get("backend")) or "—"),
           dot, kv("plans ", f"{n(core.get('plans')):,}"),
           dot, kv("swap ", f"{n(core.get('swaps')):,} experts / {n(core.get('swap_rounds')):,} rounds"))
    st_row("", kv("checkpoints ", f"taken {n(core.get('checkpoints_taken'))} / restored {n(core.get('restores'))}"),
           dot, kv("cold admits ", str(n(core.get("cold_admits")))),
           dot, kv("decode owed ", fnum(opt(core.get("owed_ms")), 0, " ms")))
    st_row("Window", kv("read ", fnum(der.get("window_rows_per_s"), 0, " rows/s")),
           dot, kv("lookup ", fnum(der.get("lookup_tokens_per_s"), 1, " tok/s")),
           dot, kv("draft pass ", fnum(der.get("draft_pass_ms"), 1, " ms")),
           kv(" (", pct(der.get("draft_pass_share")) + ")"))
    st_row("", kv("undrafted steps ", pct(der.get("undrafted_share"))),
           dot, kv("rounds while reading ", pct(der.get("reading_round_share"))))
    depths = " ".join(f"@{d} {pct(der.get(f'layer_accepted_at_{d}'))}"
                      for d in range(1, 8) if der.get(f"layer_accepted_at_{d}") is not None)
    stops = " ".join(f"{label} {pct(der.get(f'layer_stopped_at_{key}'))}"
                     for key, label in (("max", "max"), ("share", "share"),
                                        ("p", "min_p"), ("choice", "core's choice"))
                     if der.get(f"layer_stopped_at_{key}") is not None)
    if depths:
        st_row("draft", kv("accepted ", depths))
    if stops:
        st_row("", kv("stopped ", stops))

    return (Panel(st, title=Text("Stats", style="dim"), title_align="left", border_style="dim",
                  box=box.ROUNDED, width=width, padding=(0, 1)),
            Panel(hw, title=Text("Hardware", style="dim"), title_align="left", border_style="dim",
                  box=box.ROUNDED, width=width, padding=(0, 1)))


def recent_table(recent: list, width: int) -> Panel:
    table = Table(box=None, pad_edge=False, expand=True)
    for col in ("finish", "prompt", "cached", "output", "time", "queue", "prompt tok/s",
                "decode tok/s", "steps", "drafts", "accepted"):
        table.add_column(col, justify="left" if col == "finish" else "right", style="dim" if col == "finish" else "")
    for r in recent:
        table.add_row(
            Text(str(r.get("finish", "-")), style=finish_style(r.get("finish"))),
            f"{n(r.get('prompt_tokens')):,}", f"{n(r.get('cached_tokens')):,}",
            f"{n(r.get('completion_tokens')):,}",
            f"{num(r.get('seconds')):,.3f}s", ms(opt(r.get("queue_ms"))), req_pp(r), req_tg(r),
            f"{n(r.get('steps')):,}", f"{n(r.get('drafts')):,}", f"{n(r.get('accepted')):,}",
        )
    return Panel(table, title=Text(f"Recent requests · {len(recent)}", style="dim"), title_align="left",
                 border_style="dim", box=box.ROUNDED, padding=(0, 1), width=width)


def hint(interval: float) -> Text:
    return Text(f"↑↓ {SCROLL_STEP} rows   PgUp/PgDn page   g/G top/bottom   Ctrl+C quit   · every {interval:g}s refresh",
                style="dim")


def render(m: dict, live: dict, hist: History, state: str,
           props: dict | None, model_name: str | None, url: str, interval: float, width: int) -> Group:
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
    return Group(header(url, state, live, width),
                 *blank, body, *blank,
                 stats, *blank,
                 recent_table(m.get("recent", []), width), *blank,
                 model_panel(props, model_name, eng, up_s, width), *blank,
                 hardware, *blank, hint(interval))


class Painter:
    """自己画屏: doc 是已经渲染好的整页行, 每帧只重写内容和上屏不同的行, 滚动只换切片。
    行首带绝对定位, 所以不依赖游标状态; 终端尺寸一变就整屏重画。"""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.rows: list = []
        self.size: tuple[int, int] | None = None
        self.active = False

    def __enter__(self) -> "Painter":
        if not self.console.is_terminal:
            raise SystemExit("monitor needs a terminal")
        self.active = self.console.set_alt_screen(True)
        self.console.show_cursor(False)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.active:
            self.console.show_cursor(True)
            self.console.set_alt_screen(False)

    def show(self, doc: list, offset: int | None) -> tuple[int, int]:
        """offset 是距底部的行数 (None = 停在页顶); 返回落定后的 offset 和可滚行数。"""
        size = self.console.size
        height = max(1, size.height)
        buf: list = []
        if size != self.size:
            self.rows, self.size = [None] * height, size
            buf.append(Segment("\x1b[2J"))
        max_off = max(0, len(doc) - height)
        offset = max_off if offset is None else max(0, min(offset, max_off))
        start = max(0, len(doc) - height - offset)
        for row in range(height):
            want = doc[start + row] if start + row < len(doc) else []
            if self.rows[row] == want:
                continue
            buf.append(Segment(f"\x1b[{row + 1};1H\x1b[0m\x1b[K"))
            buf.extend(want)
            self.rows[row] = want
        if buf:
            self.console.print(Segments(buf), end="", crop=False)
        return offset, max_off


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
            offset += SCROLL_STEP
        elif key == "down":
            offset -= SCROLL_STEP
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
    p = argparse.ArgumentParser(prog="monitor", description="c0ven terminal monitor, refreshed every --interval seconds")
    p.add_argument("url", nargs="?", default="http://127.0.0.1:8080")
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    console = Console()
    hist = History()
    log = RecentLog()
    rates = Rates(3 * args.interval)
    poller = Poller(args.url, args.interval)
    live: dict = {}
    props: dict | None = None
    model_name: str | None = None
    offset: int | None = None
    doc: list | None = None
    doc_raw: dict | None = None
    doc_state: str | None = None
    doc_size = (0, 0)

    with Painter(console) as paint, Keys() as keys:
        while True:
            t0 = time.monotonic()
            raw, fresh, fresh_name = poller.current()
            props = fresh or props
            model_name = fresh_name or model_name
            if raw is not None and raw != doc_raw:
                live = api_values(raw)
                rates.add(raw.get("core") or {}, time.monotonic())
                live["live_prompt_tps"] = rates.prompt
                live["live_decode_tps"] = rates.decode
                hist.add(live)
                log.update(raw.get("recent") or [])
                state = "LIVE"
            else:
                good = poller.last_good
                state = ("STALE" if good and (time.monotonic() - good) < 30 else "DISCONNECTED")
            size = console.size
            if doc is None or raw != doc_raw or state != doc_state or size != doc_size:
                width = min(size.width, MAX_WIDTH)
                group: Group = (
                    render({**raw, "recent": log.items}, live, hist, state, props, model_name,
                           args.url, args.interval, width)
                    if raw and raw.get("core") else
                    Group(header(args.url, state, live, width),
                          Text("no data yet, waiting for the next refresh…", style="dim")))
                doc = console.render_lines(group, console.options, pad=False)
                doc_raw, doc_state, doc_size = raw, state, size
            _, max_off = paint.show(doc, offset)

            deadline = t0 + args.interval
            while (left := deadline - time.monotonic()) > 0:
                time.sleep(min(0.05, left))
                if console.size != doc_size:
                    break
                for key in keys.read():
                    base = max_off if offset is None else offset
                    offset, max_off = paint.show(doc, Keys.scroll(base, key, max_off, doc_size[1]))


if __name__ == "__main__":
    main()

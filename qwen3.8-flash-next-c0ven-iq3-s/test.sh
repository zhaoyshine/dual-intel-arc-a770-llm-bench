#!/bin/bash
# 测速度：PP 与 TG 分开 bench —— PP 用约 16k-token 提示，TG 用短提示纯 decode 跑 1024 token，各 5 次取平均值
# $URL 上已经有就绪的服务就复用，只有本次启动的才在跑完后关掉。
# 计时方式同 docs/BENCHMARKS.md：PP = prompt tokens / 首 token 时间，TG = 首 token 之后 / 末 token 时间。
# 本次启动的服务输出丢弃，终端只打印测试结果。
# 用法: ./test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"

URL="http://$HOST:$PORT"
RUNS=5

err() { echo "错误: $*" >&2; exit 1; }

stop_all() {
    pkill -TERM -f "python -m frontend" 2>/dev/null || true
    for _ in $(seq 1 60); do
        pgrep -f "[p]ython -m engine" >/dev/null || break
        sleep 1
    done
    for _ in $(seq 1 60); do
        pgrep -f "[p]ython -m frontend" >/dev/null || return 0
        sleep 1
    done
    echo "警告: 进程 60s 未退出" >&2
}

started=0
if /usr/bin/curl -sf --max-time 2 "$URL/health" | grep -q ok; then
    served="$(/usr/bin/curl -sf --max-time 5 "$URL/props" |
        python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["model_path"], d["context"])')"
    [[ "$served" = "$MODEL_FILE $CONTEXT" ]] ||
        err "$URL 上的服务不是这套配置（跑的是 $served），换个 PORT 或先停掉它"
    echo "== 复用已在跑的服务 $URL =="
else
    [[ -f "$BUILD_DIR/core/libengine.so" ]] || err "找不到 $BUILD_DIR/core/libengine.so (先编译)"
    [[ -f "$MODEL_FILE" ]] || err "找不到模型 $MODEL_FILE"
    trap 'if [[ $started = 1 ]]; then stop_all; fi' EXIT

    echo "== 停止已有服务 =="
    stop_all
    sleep 1

    echo "== 启动服务 =="
    cd "$C0VEN_DIR"
    "$SCRIPT_DIR/start.sh" >/dev/null 2>&1 &
    cd "$SCRIPT_DIR"
    started=1

    echo "== 等待加载 =="
    ok=0
    for _ in $(seq 1 200); do
        if /usr/bin/curl -sf --max-time 2 "$URL/health" | grep -q ok; then
            ok=1
            break
        fi
        if ! pgrep -f "[p]ython -m frontend" >/dev/null; then
            echo "服务进程已退出" >&2
            break
        fi
        sleep 2
    done
    [[ $ok = 1 ]] || err "服务未就绪，跑 ./start.sh 看输出"
fi

python3 - "$URL" "$RUNS" <<'PY'
import json, sys, statistics, time, urllib.request

url, runs = sys.argv[1], int(sys.argv[2])

def core():
    with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
        return json.load(r)["core"]

def diff_keys(a, b):
    keys = ["rounds", "round_rows", "round_ns", "steps", "step_rows", "step_ns",
            "layer_drafts", "layer_accepted", "lookup_drafts", "lookup_accepted",
            "draft_passes", "draft_ns", "swap_rounds", "swap_ns"]
    return {k: b[k] - a[k] for k in keys}

def ask(prompt, n):
    """Streaming request: returns (first-token seconds, last-token seconds, usage, text)."""
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": n, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    parts = []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ""
                if first is None and (piece or delta.get("reasoning_content")):
                    first = time.time() - t0
                if piece:
                    parts.append(piece)
    last = time.time() - t0
    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": len(parts)}
    return first, last, usage, "".join(parts)

lines = 876
def prompt_long(run):
    body = "\n".join(f"Line {i + (run + 1) * lines:04d}: the quick brown fox jumps over the lazy dog."
                     for i in range(lines))
    return body + "\nWrite a detailed story about a fox."
print()
print("== PP ==")
pps = []
for i in range(runs):
    first, _, u, _ = ask(prompt_long(i), 1)
    pp = u["prompt_tokens"] / first if first else 0.0
    pps.append(pp)
    print(f"  run {i + 1}: {u['prompt_tokens']} token, 首 token {first:.2f}s -> PP {pp:.1f} tok/s")
pp_mean = statistics.mean(pps)

print("== TG（短提示，纯 decode）==")
prompt_tg = "Write a detailed story about a fox."
c_before = core()
tgs = []
for i in range(runs):
    first, last, u, _ = ask(prompt_tg, 1024)
    tg = (u["completion_tokens"] - 1) / (last - first) if last > first else 0.0
    tgs.append(tg)
    print(f"  run {i + 1}: {u['completion_tokens']} token, 首 token {first:.2f}s, "
          f"解码 {last - first:.2f}s -> TG {tg:.1f} tok/s")
tg_mean = statistics.mean(tgs)
print(f"平均值: PP {pp_mean:.1f} tok/s, TG {tg_mean:.1f} tok/s")
print(f"中位数: PP {statistics.median(pps):.1f} tok/s, TG {statistics.median(tgs):.1f} tok/s")
d = diff_keys(c_before, core())
r = max(1, d["rounds"])
drafts = d["layer_drafts"] + d["lookup_drafts"]
accepted = d["layer_accepted"] + d["lookup_accepted"]
print("  解码 counters："
      f"rounds {d['rounds']}"
      f"（{d['round_rows'] / r:.2f} 行/round，{d['round_ns'] / r / 1e6:.1f} ms/round），"
      f"产出 {(accepted + r) / r:.2f} token/round；"
      f"drafts {drafts}，accepted {accepted}（{accepted / drafts if drafts else 0:.1%}）；"
      f"lookup {d['lookup_accepted']}/{d['lookup_drafts']}；"
      f"draft pass {d['draft_passes']} 次 {d['draft_ns'] / 1e6:.0f} ms；"
      f"swap {d['swap_rounds']} 轮 {d['swap_ns'] / 1e6:.0f} ms")
PY
rc=$?

echo
if [[ $rc -ne 0 ]]; then
    echo "== 测速失败（退出码 $rc）=="
fi
exit $rc

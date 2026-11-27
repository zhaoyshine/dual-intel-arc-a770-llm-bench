#!/bin/bash
# 启动服务，跑两项测试后停止服务：
#   1. 速度：PP 与 TG 分开 bench —— PP 用 1024-token 提示，TG 用短提示（纯 decode），各 3 次取中位数
#   2. 正确性：10-20 token 内容输入 + 128 输出，同一输入跑两次（temperature 0），与参考答案比对
# 计时方式同 docs/BENCHMARKS.md：PP = prompt tokens / 首 token 时间，TG = 首 token 之后 / 末 token 时间。
# 服务输出丢弃，终端只打印测试结果。
# 用法: ./test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"

URL="http://$HOST:$PORT"
SIM_MIN=0.90

err() { echo "错误: $*" >&2; exit 1; }

[[ -f "$BUILD_DIR/core/libengine.so" ]] || err "找不到 $BUILD_DIR/core/libengine.so (先编译)"
[[ -f "$MODEL_FILE" ]] || err "找不到模型 $MODEL_FILE"

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

trap stop_all EXIT

echo "== 停止已有服务 =="
stop_all
sleep 1

echo "== 启动服务 =="
cd "$C0VEN_DIR"
"$SCRIPT_DIR/start.sh" >/dev/null 2>&1 &
cd "$SCRIPT_DIR"

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

python3 - "$URL" "$SIM_MIN" <<'PY'
import json, sys, difflib, statistics, time, urllib.request

url, sim_min = sys.argv[1], float(sys.argv[2])

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

# 1) 速度：约 1024 token（每行约 14 token）。PP 与 TG 各用独立请求，3 次取中位数。
lines = 73
prompt_long = "\n".join(f"Line {i:04d}: the quick brown fox jumps over the lazy dog." for i in range(lines))
prompt_long += "\nWrite a detailed story about a fox."
print()
print("== 测试 1a：PP ==")
pps = []
for i in range(3):
    first, _, u, _ = ask(prompt_long, 1)
    pp = u["prompt_tokens"] / first if first else 0.0
    pps.append(pp)
    print(f"  run {i + 1}: {u['prompt_tokens']} token, 首 token {first:.2f}s -> PP {pp:.1f} tok/s")
pp_med = statistics.median(pps)

print("== 测试 1b：TG（短提示，纯 decode）==")
prompt_tg = "Write a detailed story about a fox."
c_before = core()
tgs = []
for i in range(3):
    first, last, u, _ = ask(prompt_tg, 512)
    tg = (u["completion_tokens"] - 1) / (last - first) if last > first else 0.0
    tgs.append(tg)
    print(f"  run {i + 1}: {u['completion_tokens']} token, 首 token {first:.2f}s, "
          f"解码 {last - first:.2f}s -> TG {tg:.1f} tok/s")
tg_med = statistics.median(tgs)
print(f"中位数: PP {pp_med:.1f} tok/s, TG {tg_med:.1f} tok/s")
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

# 2) 正确性：内容 10-20 token（prompt_tokens 另含 chat template），输出 128，与参考答案比较前缀相似率
prompt_short = "Explain why the sky looks blue in a short paragraph."
ref = ("The sky looks blue because sunlight is scattered by molecules in the atmosphere. "
       "Shorter blue wavelengths scatter more s")
_, _, ua, a = ask(prompt_short, 128)
_, _, ub, b = ask(prompt_short, 128)
if ref not in a and ref not in b:
    # 128 token 可能全用在 reasoning 上、答案还没开始写；延长一次再看。
    print("  （128 token 内未见答案，延长到 256 重测）")
    _, _, ua, a = ask(prompt_short, 256)
    _, _, ub, b = ask(prompt_short, 256)
sim_ref = difflib.SequenceMatcher(None, ref, a[:len(ref)]).ratio()
sim_two = difflib.SequenceMatcher(None, a, b).ratio()
print()
print("== 测试 2：正确性 ==")
print(f"输入 {ua['prompt_tokens']} token, 输出 {ua['completion_tokens']} / {ub['completion_tokens']} token")
contains_ref = ref in a
print(f"与参考相似率 {sim_ref:.3f}（阈值 {sim_min:.2f}）；两次之间 {sim_two:.3f}；含参考: {contains_ref}")
print("参考:", repr(ref))
print("实际:", repr(a[:len(ref)]))
if ua["completion_tokens"] == 0 or ub["completion_tokens"] == 0:
    print("失败: 有输出为 0 token")
    sys.exit(1)
# 输出有时以 reasoning 开头，前缀相似率会低；参考句出现在全文任一位置也算通过。
if sim_ref < sim_min and not contains_ref:
    print("失败: 输出与参考不符")
    sys.exit(1)
print("通过")
PY
rc=$?

echo
if [[ $rc -eq 0 ]]; then
    echo "== 全部通过 =="
else
    echo "== 失败（退出码 $rc）=="
fi
exit $rc

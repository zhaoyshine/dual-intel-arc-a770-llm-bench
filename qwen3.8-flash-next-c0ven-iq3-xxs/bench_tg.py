#!/usr/bin/env python3
"""TG 速率：两点法，先 max_tokens=1 测得 prefill，再 max_tokens=N+1，差值为纯 decode。

用法: python3 bench_tg.py [输出token=128] [提示词行数=2]
对比 draft 开关时，先停服务，再分别启动:
  ./start.sh                       # DFlash 开（dflash/ 存在时自动）
  ./start.sh --engine-arg=--no-draft
同一份提示词、temperature 0 下两臂的 TG 和输出文本都要一致（batching 差异只允许 logits 噪声）。
"""
import json, sys, time, urllib.request

no = int(sys.argv[1]) if len(sys.argv) > 1 else 128
nl = int(sys.argv[2]) if len(sys.argv) > 2 else 2
prompt = "\n".join(f"Line {i:04d}: the quick brown fox jumps over the lazy dog." for i in range(nl))
prompt += "\nWrite a detailed story about a fox."

def run(n_out):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": n_out, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    text = (msg.get("content") or msg.get("reasoning_content") or "")[:120]
    return time.time() - t0, out["usage"]["completion_tokens"], text

t1, n1, _ = run(1)
t2, n2, text = run(no + 1)
print(f"tg {(n2 - n1) / (t2 - t1):.2f} tok/s over {n2 - n1} tokens "
      f"(prefill {t1:.2f}s, total {t2:.2f}s)")
print(repr(text))

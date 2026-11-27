#!/usr/bin/env python3
"""正确性与计时：发一个 N 行的可复现提示词，打印 token 数、耗时和完整输出。

用法: python3 bench_prompt.py [行数=80] [输出token=64]
输出退化成一串重复词即为乱码（里程碑 1 的回归项）。
"""
import json, sys, time, urllib.request

nl = int(sys.argv[1]) if len(sys.argv) > 1 else 80
no = int(sys.argv[2]) if len(sys.argv) > 2 else 64
prompt = "\n".join(f"Line {i:04d}: the quick brown fox jumps over the lazy dog." for i in range(nl))
prompt += "\nReply with exactly one short sentence about the fox."
body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": no, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=3600) as r:
    out = json.load(r)
dt = time.time() - t0
u = out["usage"]
ttft = None
msg = out["choices"][0]["message"]
text = msg.get("content") or msg.get("reasoning_content") or ""
print(f"prompt_tokens {u['prompt_tokens']} completion_tokens {u['completion_tokens']} elapsed {dt:.2f}s")
print(repr(text)[:1000])

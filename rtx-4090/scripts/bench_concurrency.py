#!/usr/bin/env python3
"""N concurrent streamed chat requests against the running server (stdlib only).

Start the server with MAX_BATCH_SIZE at or above --n, then:

    python3 scripts/bench_concurrency.py --n 4

Each CONC line is one round: total_tps is all streams together, first token to last token, client-side.
"""
import argparse, json, threading, time, urllib.request
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and three unit tests.",
    "Explain how DNS caching works, from the stub resolver to the authoritative server, in about 250 words.",
    "Write a systemd unit file for a Python web service with automatic restart, then explain each directive.",
    "Describe how a transistor amplifies a signal, then give a worked example with numbers.",
]
def one(base, model, prompt, max_tokens, out, i, gate):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    gate.wait()
    t0 = time.time(); ttft = None; usage = None; err = None
    try:
        req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout = 900) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                ev = json.loads(line[5:])
                if ev.get("usage"): usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if ((d.get("content") or "") + (d.get("reasoning_content") or "")) and ttft is None: ttft = time.time()
    except Exception as e:
        err = repr(e)[:200]
    t1 = time.time(); u = usage or {}
    out[i] = {"t0": t0, "ttft": ttft, "t1": t1, "n": u.get("completion_tokens"), "server_tps": u.get("completion_tokens_per_sec"), "err": err}
ap = argparse.ArgumentParser()
ap.add_argument("--base", default = "http://127.0.0.1:8000/v1"); ap.add_argument("--model", default = "Inkling-2-Mini-SAGE-REXL3-4.23bpw")
ap.add_argument("--n", type = int, default = 1); ap.add_argument("--max-tokens", type = int, default = 300)
ap.add_argument("--repeats", type = int, default = 3); ap.add_argument("--tag", default = "")
a = ap.parse_args()
for rep in range(a.repeats):
    out = [None] * a.n; gate = threading.Event()
    th = [threading.Thread(target = one, args = (a.base, a.model, PROMPTS[(i + rep) % 4], a.max_tokens, out, i, gate)) for i in range(a.n)]
    for t in th: t.start()
    gate.set()
    for t in th: t.join()
    ok = [o for o in out if o and o["n"] and o["ttft"]]
    rec = {"tag": a.tag, "n": a.n, "rep": rep, "ok": len(ok), "errs": [o["err"] for o in out if o and o["err"]]}
    if ok:
        first = min(o["ttft"] for o in ok); last = max(o["t1"] for o in ok)
        rec["total_tps"] = round(sum(o["n"] - 1 for o in ok) / (last - first), 2)          # all streams, first token to last token
        rec["per_stream_tps"] = [round((o["n"] - 1) / (o["t1"] - o["ttft"]), 2) for o in ok]
        rec["ttft_s"] = [round(o["ttft"] - o["t0"], 2) for o in ok]; rec["tokens"] = [o["n"] for o in ok]
    print("CONC " + json.dumps(rec), flush = True)

#!/usr/bin/env python3
"""Single-stream speed check against the running server (stdlib only).

Each prompt is sent streamed, one request at a time, greedy unless --temperature is given. Two kinds of number come
back and they are printed separately:

  server  prefill and decode tok/s from the server's own `usage` counters
  client  time to first streamed event, and wall time for the whole request

    python3 scripts/bench.py                       # 3 short prompts, 300 tokens each
    python3 scripts/bench.py --prompt-tokens 8000  # adds one long-prompt request
    python3 scripts/bench.py --json out.json
"""
import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and three unit tests.",
    "prose": "Explain how DNS caching works, from the stub resolver to the authoritative server, in about 250 words.",
    "devops": "Write a systemd unit file for a Python web service with automatic restart, then explain each directive.",
}
WORDS = ("harbor", "copper", "lantern", "meadow", "quartz", "willow", "ember", "falcon", "garnet", "isle")


def long_prompt(base, n_tokens, nonce):
    # Varied filler (not one repeated sentence), with a per-run nonce up front so a rerun cannot
    # be answered from a reused prompt cache. Sized by the server's own token count.
    lines, i = [f"Run id {nonce}. Reference log follows."], 0
    while True:
        lines.append(f"Entry {i}: the {WORDS[i % 10]} team moved item {i * 7919 % 100003} to bay {i % 97} at step {i * 31 % 1009}.")
        i += 1
        if i % 200 == 0:
            text = "\n".join(lines)
            req = urllib.request.Request(base + "/token/encode", json.dumps({"text": text}).encode(), {"Content-Type": "application/json"})
            if json.load(urllib.request.urlopen(req))["length"] >= n_tokens:
                return text + "\n\nSummarize the pattern in the log above in three sentences."


def run(base, model, prompt, max_tokens, key, temperature=0.0):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    ttft, usage, finish, chars = None, None, None, 0
    with urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), headers), timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece and ttft is None:
                    ttft = time.time() - t0
                chars += len(piece)
                finish = ch.get("finish_reason") or finish
    wall = time.time() - t0
    u = usage or {}
    return {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "server_prefill_tok_s": u.get("prompt_tokens_per_sec"), "server_decode_tok_s": u.get("completion_tokens_per_sec"),
            "client_ttft_s": round(ttft, 3) if ttft else None, "client_wall_s": round(wall, 3),
            "finish_reason": finish, "chars": chars}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Inkling-2-Mini-SAGE-REXL3-4.23bpw")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default); 1 = the pack's own sampling default")
    ap.add_argument("--prompt-tokens", type=int, default=0, help="also send one filler prompt of about this many tokens")
    ap.add_argument("--key", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    run(a.base, a.model, "Say hi.", 16, a.key)  # warm-up, not reported
    rows = []
    for rep in range(a.repeats):
        for name, prompt in PROMPTS.items():
            r = run(a.base, a.model, prompt, a.max_tokens, a.key, a.temperature)
            r.update(name=name, rep=rep)
            rows.append(r)
            print(json.dumps(r), flush=True)
    if a.prompt_tokens:
        r = run(a.base, a.model, long_prompt(a.base, a.prompt_tokens, int(time.time())), a.max_tokens, a.key, a.temperature)
        r.update(name=f"long-{a.prompt_tokens}", rep=0)
        rows.append(r)
        print(json.dumps(r), flush=True)
    dec = [float(r["server_decode_tok_s"]) for r in rows if r["name"] in PROMPTS and r["server_decode_tok_s"] not in (None, "Infinity")]
    if dec:
        print(f"SUMMARY temperature {a.temperature:g} server decode tok/s over {len(dec)} short requests: median {statistics.median(dec):.1f}, min {min(dec):.1f}, max {max(dec):.1f}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

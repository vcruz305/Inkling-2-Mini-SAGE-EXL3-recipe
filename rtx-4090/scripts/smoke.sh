#!/usr/bin/env bash
# The serve is only "up" if a fresh chat request returns the expected answer with
# finish_reason=stop.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
RESP=$(curl -s -m 180 "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  ${TABBY_API_KEY:+-H "Authorization: Bearer $TABBY_API_KEY"} -d '{
  "model": "'"$PACK_NAME"'",
  "messages": [{"role": "user", "content": "What is the capital of France? Answer with the city name only."}],
  "max_tokens": 400, "temperature": 0
}')
python3 - "$RESP" <<'PY'
import json, sys
try:
    r = json.loads(sys.argv[1]); c = r["choices"][0]
except Exception as e:
    print("SMOKE FAIL: no completion in response:", e, sys.argv[1][:300]); sys.exit(1)
msg = c["message"]; text = (msg.get("content") or "").strip(); finish = c.get("finish_reason")
leaked = [t for t in ("<|content_", "<|message_", "<|end_message|>") if t in text]
if finish == "stop" and "Paris" in text and not leaked:
    print(f"SMOKE PASS: {text[:60]!r}, finish_reason=stop"); sys.exit(0)
print(f"SMOKE FAIL: finish_reason={finish!r} leaked={leaked} content={text[:200]!r}"); sys.exit(1)
PY

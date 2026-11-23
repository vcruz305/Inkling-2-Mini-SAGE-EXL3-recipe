#!/usr/bin/env bash
# Status watcher for the serve.   Modes: status | watch | stop
#   status: one snapshot (process, endpoint, GPU memory, log tail)
#   watch:  poll until the endpoint answers or the process dies (MAX_MIN, default 15)
#   stop:   stop the server started by serve.sh
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
MODE="${1:-status}"
PIDFILE="$RUN_DIR/server.pid"
pid=$(cat "$PIDFILE" 2>/dev/null || true)
alive() { [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; }
endpoint() { curl -s -m 5 -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/v1/models" 2>/dev/null; }
SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)

snapshot() {
  echo "date: $(date -Is)"
  if alive; then echo "process: running (pid $pid)"; else echo "process: not running"; fi
  echo "endpoint: $(endpoint)"
  "$SMI" --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
  echo "log_tail:"; tail -n 5 "$RUN_DIR/server.log" 2>/dev/null | cut -c1-200
}

case "$MODE" in
  status) snapshot ;;
  stop)
    if alive; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
      for _ in $(seq 30); do alive || break; sleep 1; done
      alive && { kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"; }
      echo "stopped pid $pid"
    else echo "not running"; fi
    rm -f "$PIDFILE" ;;
  watch)
    start=$(date +%s)
    while :; do
      el=$(( $(date +%s) - start ))
      if [[ "$(endpoint)" == 200 ]]; then echo "MILESTONE=SERVE_UP after ${el}s"; snapshot; exit 0; fi
      if ! alive; then echo "MILESTONE=SERVE_DEAD after ${el}s"; snapshot; exit 1; fi
      if (( el > ${MAX_MIN:-15} * 60 )); then echo "MILESTONE=TIMEOUT after ${el}s"; snapshot; exit 1; fi
      sleep 3
    done ;;
  *) echo "usage: guard.sh status|watch|stop"; exit 2 ;;
esac

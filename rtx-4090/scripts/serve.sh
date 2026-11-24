#!/usr/bin/env bash
# Start TabbyAPI on the ExLlamaV3 fork, detached, serving the pack at http://$HOST:$PORT/v1.
#   bash scripts/serve.sh              start (returns once the process is launched)
#   DRY_RUN=1 bash scripts/serve.sh    print the config and environment, start nothing
#   FOREGROUND=1 bash scripts/serve.sh run attached
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

(( CACHE_SIZE % 256 == 0 )) || { echo "CACHE_SIZE must be a multiple of 256 (got $CACHE_SIZE)"; exit 1; }
(( CACHE_SIZE >= MAX_SEQ_LEN )) || { echo "CACHE_SIZE ($CACHE_SIZE) must be >= MAX_SEQ_LEN ($MAX_SEQ_LEN)"; exit 1; }
case "$CACHE_MODE" in FP16) ;; Q8|Q6|Q4|*,*) echo "NOTE: CACHE_MODE=$CACHE_MODE is untested with the Inkling2Loop port; FP16 is the measured setting." ;; *) echo "bad CACHE_MODE=$CACHE_MODE"; exit 1 ;; esac
(( MAX_BATCH_SIZE == 1 )) || echo "NOTE: MAX_BATCH_SIZE=$MAX_BATCH_SIZE: the draft runs only while a single stream is active (see README, Concurrency)."
[[ "$DRAFT" == dflash2 ]] && (( MAX_BATCH_SIZE > 1 )) && echo "NOTE: DRAFT=dflash2 with MAX_BATCH_SIZE=$MAX_BATCH_SIZE: concurrent streams measured slower than one stream with this drafter loaded. Use DRAFT=mtp or off for concurrency."
case "$DRAFT" in mtp|dflash2|off) ;; *) echo "bad DRAFT=$DRAFT (mtp, dflash2 or off)"; exit 1 ;; esac
if [[ "$HOST" == 127.0.0.1 || "$HOST" == localhost ]]; then DISABLE_AUTH=true; else DISABLE_AUTH=false; fi

CONFIG=$(cat <<YAML
network:
  host: $HOST
  port: $PORT
  disable_auth: $DISABLE_AUTH
logging:
  log_generation_params: false
  log_requests: false
model:
  model_dir: $RUN_DIR/models
  model_name: $PACK_NAME
  backend: exllamav3
  max_seq_len: $MAX_SEQ_LEN
  cache_size: $CACHE_SIZE
  cache_mode: $CACHE_MODE
  chunk_size: $CHUNK_SIZE
  max_batch_size: $MAX_BATCH_SIZE
# Fallbacks for clients that send no sampler values (agent frameworks). The pack's
# generation_config.json sets temperature 1.0 and nothing else; client values still win.
sampling:
  temperature: {override: 1.0, force: false}
YAML
)
# The draft head is a component of the pack itself: no draft_model_name, no second download.
[[ "$DRAFT" == mtp ]] && CONFIG+=$'\ndraft_model:\n  draft_mode: mtp'
# DFlash2 is a separate model directory; it loads alongside the target and reads its hidden states.
[[ "$DRAFT" == dflash2 ]] && CONFIG+=$'\ndraft_model:\n  draft_mode: model\n  draft_model_dir: '"$RUN_DIR/draft"$'\n  draft_model_name: '"$DRAFT_NAME"

knob_line=""
for k in "${INKLING_KNOBS[@]}"; do
  if [[ -n "${!k+x}" ]]; then export "${k?}"; knob_line+=" $k=${!k}"; else knob_line+=" $k=(fork default)"; fi
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "# config.yml"; echo "$CONFIG"
  echo "# env: TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR CUDA_HOME=${CUDA_HOME:-}$knob_line"
  echo "# cmd: (cd $RECIPE_HOME/tabbyAPI && $PY main.py --config $RUN_DIR/config.yml)"
  exit 0
fi

[[ -x "$PY" ]] || { echo "Run scripts/setup.sh first."; exit 1; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "Pack not found at $MODEL_DIR. Run scripts/download_and_verify.sh."; exit 1; }
if [[ "$DRAFT" == dflash2 && ! -f "$DRAFT_DIR/model.safetensors" ]]; then
  echo "DFlash2 drafter not found at $DRAFT_DIR. Run: bash scripts/download_and_verify.sh --drafter"; exit 1
fi
if [[ -f "$RECIPE_HOME/.exl3_ref" && "$(cat "$RECIPE_HOME/.exl3_ref")" != "$EXL3_REF" ]]; then
  echo "WARNING: runtime was built at $(cat "$RECIPE_HOME/.exl3_ref"), env.sh pins $EXL3_REF. Re-run scripts/setup.sh."
fi
if [[ -f "$RUN_DIR/server.pid" ]] && kill -0 "$(cat "$RUN_DIR/server.pid")" 2>/dev/null; then
  echo "Already running (pid $(cat "$RUN_DIR/server.pid")). Stop it first: bash scripts/guard.sh stop"; exit 1
fi

if (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; then
  echo "Port $PORT on $HOST is already in use by another process. Pick another: PORT=8001 bash scripts/serve.sh"; exit 1
fi

# Serve from a directory holding one symlink, so /v1/models lists exactly this pack.
mkdir -p "$RUN_DIR/models"
ln -sfn "$MODEL_DIR" "$RUN_DIR/models/$PACK_NAME"
if [[ "$DRAFT" == dflash2 ]]; then mkdir -p "$RUN_DIR/draft"; ln -sfn "$DRAFT_DIR" "$RUN_DIR/draft/$DRAFT_NAME"; fi
echo "$CONFIG" > "$RUN_DIR/config.yml"

# WSL2 and many desktops default to 1024 open files; the CUDA driver then fails mid-load with
# "CUDA driver error: unknown error". The fork raises the limit at import too; this covers Tabby.
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

echo "serving $PACK_NAME on http://$HOST:$PORT/v1  ctx=$MAX_SEQ_LEN cache=$CACHE_SIZE $CACHE_MODE batch=$MAX_BATCH_SIZE draft=$DRAFT"
echo "knobs:$knob_line"
cd "$RECIPE_HOME/tabbyAPI"
if [[ "${FOREGROUND:-0}" == 1 ]]; then exec "$PY" main.py --config "$RUN_DIR/config.yml"; fi
setsid nohup "$PY" main.py --config "$RUN_DIR/config.yml" > "$RUN_DIR/server.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/server.pid"
echo "LAUNCHED pid $(cat "$RUN_DIR/server.pid"), log $RUN_DIR/server.log"
echo "Wait for it: bash scripts/guard.sh watch"

#!/usr/bin/env bash
# Build the runtime into $RECIPE_HOME: its own venv, the ExLlamaV3 fork at the pinned commit
# (CUDA extension compiled here), and the latest TabbyAPI. Touches nothing outside $RECIPE_HOME.
#   bash scripts/setup.sh           full install (about 15 minutes, most of it the CUDA build)
#   bash scripts/setup.sh --check   only verify an existing install
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

if [[ "${1:-}" == "--check" ]]; then verify_runtime; exit; fi

BASE_PYTHON="${BASE_PYTHON:-python3}"
command -v "$BASE_PYTHON" >/dev/null || { echo "python not found: $BASE_PYTHON (set BASE_PYTHON=)"; exit 1; }
command -v git >/dev/null || { echo "git is required"; exit 1; }
command -v g++ >/dev/null || { echo "a C++ compiler is required (sudo apt install build-essential)"; exit 1; }
[[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]] || {
  echo "CUDA toolkit not found. Install the CUDA 13.0 toolkit or set CUDA_HOME to one that has bin/nvcc."; exit 1; }
"$BASE_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { echo "Python 3.10+ required"; exit 1; }
"$BASE_PYTHON" -c 'import venv, ensurepip' 2>/dev/null || { echo "python venv module missing (sudo apt install python3-venv)"; exit 1; }

mkdir -p "$RECIPE_HOME"
free_gb=$(df -BG --output=avail "$RECIPE_HOME" | tail -1 | tr -dc 0-9)
(( free_gb >= 12 )) || { echo "Need about 12 GB free under $RECIPE_HOME for the runtime, have ${free_gb} GB"; exit 1; }

[[ -x "$PY" ]] || "$BASE_PYTHON" -m venv "$VENV"
"$PY" -m pip install -q --upgrade pip wheel "setuptools<81"
"$PY" -m pip install -q "$TORCH_SPEC" --index-url "$TORCH_INDEX"

fetch() {  # fetch <dir> <repo> <ref>
  local dir=$1 repo=$2 ref=$3
  [[ -d "$dir/.git" ]] || git clone -q "$repo" "$dir"
  git -C "$dir" fetch -q origin "$ref"
  git -C "$dir" checkout -q --detach FETCH_HEAD
  echo "$(basename "$dir") at $(git -C "$dir" rev-parse --short HEAD)"
}
fetch "$RECIPE_HOME/exllamav3" "$EXL3_REPO" "$EXL3_REF"
fetch "$RECIPE_HOME/tabbyAPI" "$TABBY_REPO" "$TABBY_REF"

"$PY" -m pip uninstall -q -y exllamav3 2>/dev/null || true
"$PY" -m pip install -q -r "$RECIPE_HOME/exllamav3/requirements.txt"
# EXLLAMA_NOCOMPILE: skip the setuptools precompile; the extension is built once below, into
# $TORCH_EXTENSIONS_DIR, by the same loader the server uses.
EXLLAMA_NOCOMPILE=1 "$PY" -m pip install -q --no-build-isolation --no-deps -e "$RECIPE_HOME/exllamav3"
# TabbyAPI base dependencies only: its cu12 extra would replace torch and install a stock exllamav3.
(cd "$RECIPE_HOME/tabbyAPI" && "$PY" -m pip install -q .)
"$PY" -m pip install -q ninja uvloop hf_xet "huggingface_hub>=0.34"
# Read the whole output first: under pipefail, `pip show | grep -q` fails when grep exits early.
_show="$("$PY" -m pip show exllamav3 2>/dev/null || true)"
[[ "$_show" == *"Editable project location: $RECIPE_HOME/exllamav3"* ]] || {
  echo "exllamav3 is no longer the editable fork install (a dependency replaced it)"; exit 1; }

echo "Compiling the CUDA extension (about 10 minutes the first time)..."
mkdir -p "$TORCH_EXTENSIONS_DIR"
"$PY" -c 'import exllamav3' 
verify_runtime
echo "$EXL3_REF" > "$RECIPE_HOME/.exl3_ref"
echo "SETUP DONE. Next: bash scripts/download_and_verify.sh"

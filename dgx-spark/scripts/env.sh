#!/usr/bin/env bash
# Pins and paths for the Inkling-2-Mini EXL3 recipe. Sourced by every script; installs nothing.
# Override any value by exporting it before calling a script.

# --- runtime pins -----------------------------------------------------------------------------
# ExLlamaV3 fork master (upstream v1.6.7 plus the Inkling2Loop port, its decode kernels and the
# MTP draft head). The speed numbers in the README were measured on this commit.
export EXL3_REPO="${EXL3_REPO:-https://github.com/vcruz305/exllamav3.git}"
export EXL3_REF="${EXL3_REF:-10a85129180145c888693f222655f79e7d403216}"
# TabbyAPI fork main (upstream main plus the Inkling 2 message format). Stock TabbyAPI has no
# parser for this model's replies and returns raw message markup.
export TABBY_REPO="${TABBY_REPO:-https://github.com/vcruz305/tabbyAPI.git}"
export TABBY_REF="${TABBY_REF:-a2898191eed710b55066d699ce829a976b4b3272}"
# PyTorch row the numbers were measured on (aarch64, CUDA 13.0 toolkit on the host).
export TORCH_SPEC="${TORCH_SPEC:-torch==2.13.0}"
export TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

# --- pack -------------------------------------------------------------------------------------
export PACK_REPO="${PACK_REPO:-vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw}"
export PACK_REV="${PACK_REV:-44c78c9d229b43dd45e41506071b9f5f939bed44}"
export PACK_NAME="${PACK_NAME:-Inkling-2-Mini-SAGE-REXL3-4.23bpw}"

# --- paths ------------------------------------------------------------------------------------
_RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RECIPE_ROOT="${RECIPE_ROOT:-$_RECIPE_ROOT}"
export RECIPE_HOME="${RECIPE_HOME:-$HOME/inkling2-exl3}"       # venv, fork, TabbyAPI, logs
export MODEL_DIR="${MODEL_DIR:-$HOME/models/$PACK_NAME}"       # the pack itself (15 GB)
export VENV="$RECIPE_HOME/venv"
export PY="$VENV/bin/python"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$RECIPE_HOME/torch_ext}"
export RUN_DIR="$RECIPE_HOME/run"
if [[ -z "${CUDA_HOME:-}" ]]; then
  for _c in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda; do
    [[ -x "$_c/bin/nvcc" ]] && { export CUDA_HOME="$_c"; break; }
  done
fi
# The extension loader needs ninja (in the venv) and nvcc on PATH, at build time and at import.
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$VENV/bin:$PATH"
unset PYTHONPATH   # tool shells sometimes export one for another environment

# --- serve settings (the measured set; see README "The measured flag set") --------------------
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-24576}"     # per-request context; the measured setting (larger is untested here)
export CACHE_SIZE="${CACHE_SIZE:-$MAX_SEQ_LEN}" # KV pool, multiple of 256, >= MAX_SEQ_LEN
export CACHE_MODE="${CACHE_MODE:-FP16}"        # quantized KV (Q8/Q6/Q4) is untested with this port
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"   # the fast decode path covers one stream only
export CHUNK_SIZE="${CHUNK_SIZE:-2048}"
# Speculative decode with the MTP draft head that ships inside the pack (mtp.safetensors, 237 MB).
# mtp = on (the measured default), off = plain one-token-per-step decode.
export DRAFT="${DRAFT:-mtp}"

# Fork switches. serve.sh forwards one only when you set it; unset means the fork default, which
# is what was measured. The first group selects the fast decode path (all on; 0 falls back to a
# slower path). The second group tunes the draft: tokens drafted per round (5), the joint
# confidence below which a round's tail is dropped (0.3), and speculative sampling for sampled
# requests (on).
INKLING_KNOBS=(INKLING2_TOKEN INKLING2_GRAPH INKLING2_TOK_MLIN INKLING2_T1 EXL3_MOE_ROW
               INKLING2_TOKEN_T INKLING2_TOKEN_BATCH
               INKLING2_MTP_BLOCK INKLING2_MTP_CONF INKLING2_MTP_SPEC)

verify_runtime() {
  "$PY" - <<'PY'
import os, sys
import exllamav3, torch
from importlib.metadata import version
from exllamav3.ext import exllamav3_ext as ext
home = os.path.realpath(os.environ["RECIPE_HOME"])
where = os.path.realpath(exllamav3.__file__)
problems = []
if not where.startswith(home):
    problems.append(f"exllamav3 imported from {where}, not from {home} (stock wheel or another venv?)")
for sym in ("InkToken", "exl3_mlin", "MoeRow"):
    if not hasattr(ext, sym):
        problems.append(f"extension lacks {sym}: not the Inkling2Loop fork build")
from exllamav3.architecture import architectures
if "Inkling2Loop" not in open(architectures.__file__).read():
    problems.append("Inkling2Loop architecture is not registered in this exllamav3")
try:
    from exllamav3.architecture import inkling2_mtp
    if "spec_sampling" not in open(inkling2_mtp.__file__).read():
        problems.append("the MTP drafter predates the pinned commit (no speculative sampling)")
except ImportError:
    problems.append("this exllamav3 has no Inkling 2 MTP drafter")
if not torch.cuda.is_available():
    problems.append("torch reports no CUDA device")
if problems:
    print("RUNTIME CHECK FAIL"); [print(" -", p) for p in problems]; sys.exit(1)
print(f"RUNTIME CHECK PASS: exllamav3 {version('exllamav3')} at {os.path.dirname(where)}, torch {torch.__version__}")
PY
}

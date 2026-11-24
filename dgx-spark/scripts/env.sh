#!/usr/bin/env bash
# Pins and paths for the Inkling-2-Mini EXL3 recipe. Sourced by every script; installs nothing.
# Override any value by exporting it before calling a script.

# --- runtime pins -----------------------------------------------------------------------------
# ExLlamaV3 fork master (upstream v1.6.7 plus the Inkling2Loop port, its decode kernels, the MTP
# draft head, the DFlash2 drafter hookup and the multi-stream token step). The speed numbers in
# the README were measured on this commit.
export EXL3_REPO="${EXL3_REPO:-https://github.com/vcruz305/exllamav3.git}"
export EXL3_REF="${EXL3_REF:-819b039c5703ece8dec7c0ac0f4fe4e6ec0ec9c1}"
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
# Optional separate drafter (DRAFT=dflash2): a 990 MB block-diffusion draft model for this target.
export DRAFT_REPO="${DRAFT_REPO:-incoai/Inkling-2-Mini-DFlash2}"
export DRAFT_REV="${DRAFT_REV:-4f9088ea66c21f2b99ed657d1941c306b03e84d2}"
export DRAFT_NAME="${DRAFT_NAME:-Inkling-2-Mini-DFlash2}"

# --- paths ------------------------------------------------------------------------------------
_RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RECIPE_ROOT="${RECIPE_ROOT:-$_RECIPE_ROOT}"
export RECIPE_HOME="${RECIPE_HOME:-$HOME/inkling2-exl3}"       # venv, fork, TabbyAPI, logs
export MODEL_DIR="${MODEL_DIR:-$HOME/models/$PACK_NAME}"       # the pack itself (15 GB)
export DRAFT_DIR="${DRAFT_DIR:-$HOME/models/$DRAFT_NAME}"     # the DFlash2 drafter, only for DRAFT=dflash2
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
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"   # concurrent streams; above 1 the draft runs only while one stream is active
export CHUNK_SIZE="${CHUNK_SIZE:-2048}"
# Sampling temperature used only when a client sends none (the pack's generation_config.json says 1.0).
export TEMPERATURE="${TEMPERATURE:-0.7}"
# Speculative decode. dflash2 = the separate DFlash2 drafter in $DRAFT_DIR (the default on this
# setup for one stream; download_and_verify.sh fetches it). mtp = the draft head that ships inside
# the pack (mtp.safetensors, 237 MB; the default when MAX_BATCH_SIZE is above 1, because DFlash2
# slows concurrent streams). off = plain one-token-per-step decode.
if (( MAX_BATCH_SIZE > 1 )); then export DRAFT="${DRAFT:-mtp}"; else export DRAFT="${DRAFT:-dflash2}"; fi

# Fork switches. serve.sh forwards one only when you set it; unset means the fork default, which
# is what was measured. The first group selects the fast decode path (all on; 0 falls back to a
# slower path; INKLING2_SEQS is the one token step shared by concurrent streams). The second
# group tunes the MTP draft: tokens drafted per round (5), the joint confidence below which a
# round's tail is dropped (0.3), and speculative sampling for sampled requests (on). The third
# does the same for DFlash2: confidence cut (0.3) and speculative sampling (on).
INKLING_KNOBS=(INKLING2_TOKEN INKLING2_GRAPH INKLING2_TOK_MLIN INKLING2_T1 EXL3_MOE_ROW
               INKLING2_TOKEN_T INKLING2_TOKEN_BATCH INKLING2_SEQS
               INKLING2_MTP_BLOCK INKLING2_MTP_CONF INKLING2_MTP_SPEC
               INKLING2_DFLASH_CONF INKLING2_DFLASH_SPEC)

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
from exllamav3.modules.arch_specific import dflash2
if not hasattr(getattr(dflash2, "DFlash2Selector", None), "walk_scored"):
    problems.append("the DFlash2 drafter predates the pinned commit (no Inkling 2 target support)")
if "step_seqs" not in open(os.path.join(os.path.dirname(where), "architecture", "inkling2_loop.py")).read():
    problems.append("this exllamav3 predates the multi-stream token step")
if not torch.cuda.is_available():
    problems.append("torch reports no CUDA device")
if problems:
    print("RUNTIME CHECK FAIL"); [print(" -", p) for p in problems]; sys.exit(1)
print(f"RUNTIME CHECK PASS: exllamav3 {version('exllamav3')} at {os.path.dirname(where)}, torch {torch.__version__}")
PY
}

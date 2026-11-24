#!/usr/bin/env bash
# Download the pack with hf_xet high performance into $MODEL_DIR, then verify every file
# against SHA256SUMS (the authority; taken from the Hub's own LFS hashes at the pinned revision).
# The repo is public and ungated: no token needed. If HF_TOKEN is set it is used, never printed.
#   bash scripts/download_and_verify.sh            the pack
#   bash scripts/download_and_verify.sh --drafter  only the DFlash2 drafter (990 MB), for DRAFT=dflash2
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
[[ -x "$PY" ]] || { echo "Run scripts/setup.sh first (it provides huggingface_hub + hf_xet)."; exit 1; }

if [[ "${1:-}" == "--drafter" ]]; then
  mkdir -p "$DRAFT_DIR"
  export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0
  "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["DRAFT_REPO"], revision=os.environ["DRAFT_REV"], local_dir=os.environ["DRAFT_DIR"],
                  allow_patterns=[l.split("  ", 1)[1].strip() for l in open(os.path.join(os.environ["RECIPE_ROOT"], "SHA256SUMS.dflash2"))])
print("downloaded to", os.environ["DRAFT_DIR"])
PY
  ( cd "$DRAFT_DIR" && sha256sum -c --quiet "$RECIPE_ROOT/SHA256SUMS.dflash2" ) \
    && echo "VERIFY PASS: the drafter matches SHA256SUMS.dflash2" \
    || { echo "VERIFY FAIL: drafter files differ from the measured ones"; exit 1; }
  exit 0
fi

NEED=15070275721
mkdir -p "$MODEL_DIR"
have=$(du -sb "$MODEL_DIR" | cut -f1)
free=$(df -B1 --output=avail "$MODEL_DIR" | tail -1)
if (( free + have < NEED + 1073741824 )); then
  echo "Not enough disk: the pack is 15.1 GB, $(( (free + have) / 1073741824 )) GiB usable at $MODEL_DIR"; exit 1
fi

export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
"$PY" - <<'PY'
import os, sys
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RevisionNotFoundError
repo, rev, dest = os.environ["PACK_REPO"], os.environ["PACK_REV"], os.environ["MODEL_DIR"]
try:
    HfApi().model_info(repo, revision=rev)
except RevisionNotFoundError:
    print(f"WARNING: pinned revision {rev[:12]} no longer resolves; downloading main. "
          "SHA256SUMS decides whether these are the measured files.")
    rev = "main"
snapshot_download(repo_id=repo, revision=rev, local_dir=dest, max_workers=8,
                  allow_patterns=[l.split("  ", 1)[1].strip() for l in open(os.path.join(os.environ["RECIPE_ROOT"], "SHA256SUMS"))])
print("downloaded to", dest)
PY

echo "Verifying sha256 (two 7 GB shards, about a minute)..."
( cd "$MODEL_DIR" && sha256sum -c --quiet "$RECIPE_ROOT/SHA256SUMS" ) \
  && echo "VERIFY PASS: every file matches SHA256SUMS" \
  || { echo "VERIFY FAIL: files differ from the measured pack"; exit 1; }

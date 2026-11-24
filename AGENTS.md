# Inkling 2 Mini SAGE EXL3 serving recipe

Serve vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw through TabbyAPI on the
ExLlamaV3 fork pinned in `scripts/env.sh`. Pick ONE setup folder (`rtx-4090/`
or `dgx-spark/`), read its README first, and run its scripts from inside it.

Do NOT:
- `pip install exllamav3` or use a stock wheel (the architecture and the MTP
  draft exist only in the pinned fork commit; `bash scripts/setup.sh --check`
  verifies the import)
- install TabbyAPI's GPU extras (they replace torch and the fork)
- hand-edit `draft_model_name` (the default draft is `mtp` in `rtx-4090/` and
  `dflash2` in `dgx-spark/`; `download_and_verify.sh` fetches the DFlash2
  drafter when it is selected; `DRAFT=off` disables drafting)
- serve concurrent streams with `DRAFT=dflash2`, or quote the single-stream
  numbers for `MAX_BATCH_SIZE` above 1
- raise `MAX_SEQ_LEN` above 24576 without measuring memory first
- print or commit HF tokens anywhere

Exact commands: see the setup folder's README, "Quick start".

# Inkling-2-Mini-SAGE-EXL3-recipe

Serving recipes for **Inkling 2 Mini SAGE EXL3 4.23 bpw** on the ExLlamaV3 fork behind TabbyAPI,
with speculative decode from the pack's built-in MTP draft head (the default in `rtx-4090/`) or the separate DFlash2 drafter (the default in `dgx-spark/`). One folder per setup; each folder is
self-contained and its README carries the numbers measured on that setup.

**Model (Hugging Face):** <https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw> (revision `44c78c9d229b`)

**Derisked model (Hugging Face):** <https://huggingface.co/vcruz305/Inkling-2-Mini-Derisked-SAGE-REXL3-4.24bpw> (revision `118975225f9f`)

**DFlash2 drafter:** <https://huggingface.co/incoai/Inkling-2-Mini-DFlash2> (revision `4f9088ea66c2`)

**Engine (ExLlamaV3 fork):** <https://github.com/vcruz305/exllamav3> (commit `c702887ba7be`)

## Status

| Setup | Folder | Engine | Status |
|---|---|---|---|
| One 24 GB card | [`rtx-4090/`](rtx-4090/README.md) | ExLlamaV3 fork + TabbyAPI, MTP draft (DFlash2 optional) | **Measured, serving** |
| One DGX Spark | [`dgx-spark/`](dgx-spark/README.md) | ExLlamaV3 fork + TabbyAPI, DFlash2 draft (MTP optional) | **Measured, serving** |

## Quick start

Pick the folder for your hardware and run from inside it:

```bash
cd rtx-4090        # or dgx-spark
bash scripts/setup.sh
bash scripts/download_and_verify.sh
bash scripts/serve.sh
bash scripts/guard.sh watch
bash scripts/smoke.sh
```

The API is OpenAI-compatible at `http://127.0.0.1:8000/v1`. Requests that set no temperature get 0.7.

## Measured speed

FP16 cache, server-side counters, 300 new tokens per request. `rtx-4090/` serves 24,576 tokens of
context and `dgx-spark/` 262,144. Decode figures are single stream; the short-prompt column is the
median of 12 requests (code, devops, prose), and the code and prose columns are per-prompt medians
of 4. Long-prompt rows, memory and the per-prompt figures are in each folder.

| Setup | Draft | Temp 0, all | Temp 0, code | Temp 0, prose | Temp 1, all | Temp 1, code | Temp 1, prose |
|---|---|---|---|---|---|---|---|
| `rtx-4090/` | MTP | 148.6 | 217.3 | 128.7 | 132.3 | 210.1 | 122.0 |
| `rtx-4090/` | DFlash2 | 107.5 | 170.9 | 82.9 | 80.4 | 153.4 | 74.5 |
| `rtx-4090/` | off | 113.0 | 112.9 | 113.1 | 113.5 | 113.4 | 113.6 |
| `dgx-spark/` | DFlash2 | 75.4 | 112.8 | 61.4 | 62.3 | 94.8 | 54.9 |
| `dgx-spark/` | MTP | 66.4 | 93.2 | 59.0 | 59.3 | 90.4 | 57.9 |
| `dgx-spark/` | off | 51.2 | 51.0 | 51.2 | 51.2 | 51.0 | 51.2 |

All figures in tok/s. MTP is the faster drafter in `rtx-4090/`; DFlash2 is the faster one at
temperature 0 in `dgx-spark/`.

| Setup | Prefill, 9,159-token prompt | Prefill, 22,839-token prompt | Prefill, 94,235-token prompt | 2 streams, total | 4 streams, total |
|---|---|---|---|---|---|
| `rtx-4090/` | 1,253–1,489 tok/s | 1,476–1,507 tok/s | does not fit | 164–172 tok/s | 226–270 tok/s |
| `dgx-spark/` | 656–675 tok/s | 623–626 tok/s | 425 tok/s | 70–74 tok/s | 95–100 tok/s |

Concurrency rows are `MAX_BATCH_SIZE=4` with any draft setting; either drafter stands down while
more than one stream is active.

## Quants

The packs live at <https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw> and <https://huggingface.co/vcruz305/Inkling-2-Mini-Derisked-SAGE-REXL3-4.24bpw> (the quant of `Blackfrost-AI/Inkling-2-Mini-Derisked`); this table mirrors their model cards.

| bpw | size | smallest card | top-1 vs original | mean KLD | p99 KLD | download |
|---|---|---|---|---|---|---|
| 4.23 | 14.00 GiB | 24 GB | 98.33% | 0.0088 | — | `HF_XET_HIGH_PERFORMANCE=1 hf download vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` |
| 4.24 (Derisked) | 14.00 GiB | 24 GB | 98.68% | 0.0028 | — | `HF_XET_HIGH_PERFORMANCE=1 hf download vcruz305/Inkling-2-Mini-Derisked-SAGE-REXL3-4.24bpw` |

Top-1 is next-token agreement with the original checkpoint; KLD is KL(reference ‖ pack) on the
same positions. Both come from the model card, not from this recipe's runs. Each speed figure
above is tied to one card, one pack (4.23 bpw) and one set of server settings. Each setup's README shows how to
serve the Derisked pack with the same scripts.

## Cards tested on

| Card | Memory | Folder | Status |
|---|---|---|---|
| NVIDIA RTX 4090 | 24 GB | `rtx-4090/` | Measured |
| NVIDIA DGX Spark (GB10) | 128 GB unified | `dgx-spark/` | Measured |

Nothing else has been verified.

## Credits and license

- Pack, ExLlamaV3 fork work (Inkling2Loop port, decode kernels, MTP draft, DFlash2 hookup, multi-stream decode) and TabbyAPI fork: Victor Cruz.
- DFlash2 drafter: [incoai](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2).
- ExLlamaV3 and TabbyAPI: their upstream authors.
- Recipe scripts: MIT.

## Layout

| Path | What |
|---|---|
| `rtx-4090/` | scripts, `SHA256SUMS` and README for a single 24 GB card |
| `dgx-spark/` | the same for one DGX Spark |
| `AGENTS.md` | short instructions for coding agents |
| `LICENSE` | MIT |

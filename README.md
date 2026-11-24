# Inkling-2-Mini-SAGE-EXL3-recipe

Serving recipes for **Inkling 2 Mini SAGE EXL3 4.23 bpw** on the ExLlamaV3 fork behind TabbyAPI,
with speculative decode from the pack's built-in MTP draft head (the default in `rtx-4090/`) or the separate DFlash2 drafter (the default in `dgx-spark/`). One folder per setup; each folder is
self-contained and its README carries the numbers measured on that setup.

**Model (Hugging Face):** <https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw> (revision `44c78c9d229b`)

**DFlash2 drafter:** <https://huggingface.co/incoai/Inkling-2-Mini-DFlash2> (revision `4f9088ea66c2`)

**Engine (ExLlamaV3 fork):** <https://github.com/vcruz305/exllamav3> (commit `819b039c5703`)

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

24,576 context, FP16 cache, server-side counters, 300 new tokens per request. Decode figures are
single stream; the short-prompt column is the median of 12 requests (code, devops, prose), and
the code and prose columns are per-prompt medians of 4. Long-prompt rows, memory and the
temperature-1 per-prompt figures are in each folder.

| Setup | Draft | Temp 0, all | Temp 0, code | Temp 0, prose | Temp 1, all | Temp 1, code | Temp 1, prose |
|---|---|---|---|---|---|---|---|
| `rtx-4090/` | MTP | **152.8** | **217.7** | 128.0 | **138.7** | **204.0** | 121.2 |
| `rtx-4090/` | DFlash2 | 105.9 | 178.3 | 77.8 | 80.8 | 138.5 | 72.7 |
| `rtx-4090/` | off | 113.6 | 112.9 | 113.8 | 113.8 | 113.7 | 113.8 |
| `dgx-spark/` | DFlash2 | **66.6** | **108.8** | 57.5 | 61.4 | **101.8** | 55.2 |
| `dgx-spark/` | MTP | 61.3 | 90.6 | 56.7 | 61.9 | 89.3 | 54.5 |
| `dgx-spark/` | off | 50.2 | 50.0 | 50.2 | 50.2 | 50.0 | 50.2 |

All figures in tok/s. MTP is the faster drafter in `rtx-4090/`; DFlash2 is the faster one on
code in `dgx-spark/` and level with MTP elsewhere there.

| Setup | Prefill, 9,159-token prompt | Prefill, 22,839-token prompt | 2 streams, total | 4 streams, total |
|---|---|---|---|---|
| `rtx-4090/` | 1,329–1,440 tok/s | 1,162–1,245 tok/s | 164–171 tok/s | 207–266 tok/s |
| `dgx-spark/` | 582–587 tok/s | 444–445 tok/s | 69–73 tok/s | 92–102 tok/s |

Concurrency rows are `MAX_BATCH_SIZE=4` with `DRAFT=mtp` or `off`; the draft stands down while
more than one stream is active.

## Quants

The pack lives at <https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw>; this table mirrors its model card.

| bpw | size | smallest card | top-1 vs original | mean KLD | p99 KLD | download |
|---|---|---|---|---|---|---|
| 4.23 | 14.00 GiB | 24 GB | 98.33% | 0.0088 | — | `HF_XET_HIGH_PERFORMANCE=1 hf download vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` |

Top-1 is next-token agreement with the original checkpoint; KLD is KL(reference ‖ pack) on the
same positions. Both come from the model card, not from this recipe's runs. Each speed figure
above is tied to one card, one pack and one set of server settings.

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

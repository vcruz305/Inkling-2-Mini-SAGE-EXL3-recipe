# Inkling-2-Mini-SAGE-EXL3-recipe

Serving recipes for **Inkling 2 Mini SAGE EXL3 4.23 bpw** on the ExLlamaV3 fork behind TabbyAPI,
with the pack's built-in MTP draft head on by default. One folder per setup; each folder is
self-contained and its README carries the numbers measured on that setup.

**Model (Hugging Face):** <https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw> (revision `44c78c9d229b`)

## Status

| Setup | Folder | Engine | Status |
|---|---|---|---|
| One 24 GB card | [`rtx-4090/`](rtx-4090/README.md) | ExLlamaV3 fork + TabbyAPI, MTP draft | **Measured, serving** |
| One DGX Spark | [`dgx-spark/`](dgx-spark/README.md) | ExLlamaV3 fork + TabbyAPI, MTP draft | **Measured, serving** |

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

The API is OpenAI-compatible at `http://127.0.0.1:8000/v1`.

## Measured speed

Single stream, 24,576 context, FP16 cache, server-side counters, median of 12 short requests of
300 new tokens. Per-prompt figures, long-prompt rows and memory are in each folder.

| Setup | Temperature | Draft on | Draft off | Ratio |
|---|---|---|---|---|
| `rtx-4090/` | 0 | 150.6 tok/s | 113.5 tok/s | 1.33× |
| `rtx-4090/` | 1 | 136.4 tok/s | 113.4 tok/s | 1.20× |
| `dgx-spark/` | 0 | 63.1 tok/s | 51.4 tok/s | 1.23× |
| `dgx-spark/` | 1 | 62.3 tok/s | 51.0 tok/s | 1.22× |

Draft-on speed varies with the text: at temperature 0 individual requests ranged 121.3–219.4
tok/s in `rtx-4090/` and 57.0–93.7 in `dgx-spark/`. Prefill on 9k and 23k-token prompts measured
1,180 to 1,320 tok/s and 445 to 585 tok/s respectively.

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

- Pack, ExLlamaV3 fork work (Inkling2Loop port, decode kernels, MTP draft) and TabbyAPI fork: Victor Cruz.
- ExLlamaV3 and TabbyAPI: their upstream authors.
- Recipe scripts: MIT.

## Layout

| Path | What |
|---|---|
| `rtx-4090/` | scripts, `SHA256SUMS` and README for a single 24 GB card |
| `dgx-spark/` | the same for one DGX Spark |
| `AGENTS.md` | short instructions for coding agents |
| `LICENSE` | MIT |

# Inkling 2 Mini SAGE EXL3 4.23 bpw on one 24 GB card

Serves [`vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw`](https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw) through TabbyAPI on the
[ExLlamaV3 fork](https://github.com/vcruz305/exllamav3), with speculative decode from either the pack's own MTP draft head or the separate
[DFlash2 drafter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2). Every number here was
measured by running these scripts from an empty directory on the card named at the bottom.

## Quick start

Needs a 24 GB NVIDIA card with nothing else on it, Python 3.10+, `git`, `g++` and the CUDA 12.8 toolkit (`nvcc`).
Run from this folder:

```bash
bash scripts/setup.sh                # venv + fork + TabbyAPI, compiles the CUDA extension
bash scripts/download_and_verify.sh  # 15.1 GB pack via hf_xet, then sha256 of every file
bash scripts/serve.sh                # starts detached, MTP draft on
bash scripts/guard.sh watch          # returns when the endpoint answers
bash scripts/smoke.sh                # one chat request, must print SMOKE PASS
python3 scripts/bench.py             # optional: the speed table below
```

To serve with the DFlash2 drafter instead (990 MB, public, Apache-2.0):

```bash
bash scripts/download_and_verify.sh --drafter
bash scripts/guard.sh stop
DRAFT=dflash2 bash scripts/serve.sh
```

The API is OpenAI-compatible at `http://127.0.0.1:8000/v1`. Clients that append
`/chat/completions` themselves need the base URL **including** `/v1`.

Everything installs under `RECIPE_HOME` (default `~/inkling2-exl3`) and the pack goes to
`MODEL_DIR` (default `~/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw`). The pack is public and
ungated, so no token is needed.

### The Derisked pack

The same scripts serve [`vcruz305/Inkling-2-Mini-Derisked-SAGE-REXL3-4.24bpw`](https://huggingface.co/vcruz305/Inkling-2-Mini-Derisked-SAGE-REXL3-4.24bpw), the quant of
`Blackfrost-AI/Inkling-2-Mini-Derisked`. Export these before `download_and_verify.sh` and `serve.sh`:

```bash
export PACK_REPO=vcruz305/Inkling-2-Mini-Derisked-SAGE-REXL3-4.24bpw
export PACK_REV=118975225f9f21ebddf590d2705280ba7072479f
export PACK_NAME=Inkling-2-Mini-Derisked-SAGE-REXL3-4.24bpw
export PACK_SUMS=SHA256SUMS.derisked
```

The speed tables in this README were measured on the 4.23 bpw pack.

## The measured settings

| Setting | Value | Why |
|---|---|---|
| `DRAFT` | `mtp` | Speculative decode with the draft head inside the pack (`mtp.safetensors`). `DRAFT=dflash2` uses the separate DFlash2 drafter; `DRAFT=off` serves one token per step. |
| `MAX_SEQ_LEN` / `CACHE_SIZE` | 24576 / 24576 | The context every number below was taken at, and the most a 24 GB card holds with the FP16 cache. |
| `CACHE_MODE` | `FP16` | The only KV cache type tested with this architecture. |
| `CHUNK_SIZE` | 2048 | Prefill chunk used for the prefill figures. |
| `MAX_BATCH_SIZE` | 1 | The single-stream numbers were taken at 1. Raise it for concurrent streams (see Concurrency, measured at 4). |
| sampling fallback | temperature 0.7 | `TEMPERATURE`, applied only when a client sends none. The pack's own `generation_config.json` value is 1.0. The speed tables below were taken at 0 and 1. |

`scripts/env.sh` holds these and the pins. The fork's own switches (`INKLING2_*`) are left at
their defaults, which is what was measured; `serve.sh` forwards one only if you export it and
prints the full set on start. `DRY_RUN=1 bash scripts/serve.sh` prints the config without
starting anything.

## Measured bring-up

| Phase | Measured |
|---|---|
| `setup.sh` from an empty directory (includes the CUDA extension build; timed at the previous engine pin) | 8 min 22 s |
| `serve.sh` to endpoint answering, pack already in the page cache | 22 s (MTP), 22 s (DFlash2), 21 s (off) |

A first start after a download or reboot reads 15 GB from disk and takes longer. On this host the first `serve.sh` call after setup spent 2 min 14 s before launching the server (cause not traced); the second call took 3 s.

## Measured memory

| Reading | MTP | DFlash2 | Off |
|---|---|---|---|
| Peak over the run (during the 22,839-token prompt) | 22,306 MiB | 22,442 MiB | 21,120 MiB |

Readings are `nvidia-smi` totals for the card, sampled about once a second, and include about 200 MiB the desktop was already using. The card has 24,564 MiB, so the draft-on peaks leave about 2 GiB free. `MAX_SEQ_LEN=24576` is the most this card holds with the FP16 cache: the cache costs about 0.19 GiB per 1,024 tokens. If the same card carries other GPU work, start with `DRAFT=off` (about 1.2 GiB lower at peak) or a smaller `MAX_SEQ_LEN`; a smaller `MAX_SEQ_LEN` was not measured.

## Measured speed

Server-side counters from TabbyAPI's `usage` block, one request at a time, streamed, at `MAX_SEQ_LEN=24576`.
`scripts/bench.py` produced all of them: three short prompts (code, devops, prose), 300 new
tokens each, 4 repeats. Each draft setting was its own server start in the same session.

| Decode, short prompts | `DRAFT=mtp` | `DRAFT=dflash2` | `DRAFT=off` |
|---|---|---|---|
| Temperature 0, median of 12 | **148.6** tok/s | 107.5 tok/s | 113.0 tok/s |
| Temperature 1, median of 12 | **132.3** tok/s | 80.4 tok/s | 113.5 tok/s |

With a draft on, speed depends on how predictable the text is. Per prompt, median of 4, tok/s:

| Prompt | Temp 0, MTP | Temp 0, DFlash2 | Temp 0, off | Temp 1, MTP | Temp 1, DFlash2 | Temp 1, off |
|---|---|---|---|---|---|---|
| code | **217.3** | 170.9 | 112.9 | **210.1** | 153.4 | 113.4 |
| devops | **148.6** | 107.5 | 113.1 | **130.8** | 79.0 | 113.8 |
| prose | **128.7** | 82.9 | 113.1 | **122.0** | 74.5 | 113.6 |

The 12 requests ranged 118.5–218.3 tok/s (MTP) and 79.3–188.9 (DFlash2) at temperature 0, and
116.5–214.1 and 69.2–161.1 at temperature 1. **On this card MTP is the drafter to use.** DFlash2
runs correctly but is slower than MTP on every prompt and slower than no draft on devops and
prose: each of its draft rounds costs more than this card saves.

Prefill and long prompts, temperature 0, one request each, in the order sent:

| Prompt tokens | Draft | Prefill | Decode | Time to first token |
|---|---|---|---|---|
| 9,159 (first) | mtp | 1,413 tok/s | 141.6 tok/s | 6.64 s |
| 9,159 (first) | dflash2 | 1,381 tok/s | 101.0 tok/s | 6.77 s |
| 9,159 (first) | off | 1,407 tok/s | 108.3 tok/s | 6.59 s |
| 22,839 | mtp | 1,476 tok/s | 156.0 tok/s | 15.64 s |
| 22,839 | dflash2 | 1,507 tok/s | 112.1 tok/s | 15.33 s |
| 22,839 | off | 1,501 tok/s | 102.2 tok/s | 15.32 s |
| 9,159 (after the 22,839 one) | mtp | 1,363 tok/s | 168.4 tok/s | 6.81 s |
| 9,159 (after the 22,839 one) | dflash2 | 1,253 tok/s | 88.4 tok/s | 7.49 s |
| 9,159 (after the 22,839 one) | off | 1,489 tok/s | 108.0 tok/s | 6.23 s |

Prefill is the server counter; time to first token is client-side and includes it.

## Concurrency

`MAX_BATCH_SIZE=4`, same context and cache, temperature 0, 300 new tokens per stream.
`scripts/bench_concurrency.py` produced these; totals are all streams together, first token to
last token, client-side.

| Streams at once | `DRAFT=mtp` total | `DRAFT=dflash2` total | `DRAFT=off` total | Per stream |
|---|---|---|---|---|
| 1 | 123.8–200.3 tok/s | 65.1–156.1 tok/s | 114.5–115.4 tok/s | the same |
| 2 | 166.6–169.6 tok/s | 164.0–168.2 tok/s | 167.5–172.2 tok/s | about 82–86 tok/s |
| 4 | 228.5–235.9 tok/s | 225.9–258.4 tok/s | 231.4–269.8 tok/s | about 57–67 tok/s |

Either drafter runs while one stream is active, stands down when a second arrives, and resumes
afterwards, so with two or more streams the draft setting makes no difference. The first request
after a start is slower and is left out of the single-stream row. Time to first token was
0.84–0.99 s at 2 streams and 1.25–2.87 s at 4.
GPU memory peaked at 21,090 MiB (MTP), 21,462 MiB (DFlash2) and 19,992 MiB (off) in these short-prompt rounds; long
prompts at `MAX_BATCH_SIZE=4` were not measured.

## Correctness checks in the same run

- `smoke.sh` passed with DFlash2, MTP and the draft off (answer `Paris`, `finish_reason: stop`, no leaked control tokens).
- With DFlash2 and with MTP, a tool request returned one parsed `tool_calls` entry (`get_weather`, `{"city": "Paris"}`) with the reasoning in `reasoning_content`.
- With DFlash2 and with MTP, a temperature-1 chat request returned reasoning and answer in separate channels, with the right answer.
- No traceback in any server log, including the 2- and 4-stream rounds; every concurrent stream completed.

## Known limits

- **Sampler settings beyond temperature draft the old way.** Requests that set only a temperature use speculative sampling (a draft is kept with probability min(1, p/q)). Requests that add top-k, top-p, min-p or penalties still draft, but greedily with exact-match acceptance, so they gain less above temperature 0.
- **Draft-on output is not token-identical to draft-off.** Greedy text can diverge because verification runs several positions in one batch.
- **Constrained output (JSON schema, grammar) was not tested** through this recipe.
- **No drafting under concurrency.** With two or more streams active either drafter stands down and decode is one token per step per stream (see Concurrency).
- **Concurrent streams are not token-identical to a single stream.** Greedy text can diverge partway, because several streams share one batched step.
- **DFlash2 uses speculative sampling only for temperature-only requests**, the same rule as MTP in the first item.
- **Image input was not tested** through this recipe.
- **Context above 24,576 does not fit** this card with the FP16 cache (about 0.19 GiB per 1,024 tokens). `dgx-spark/` serves 262,144.

## Do not

- `pip install exllamav3` or use a stock wheel: the Inkling2Loop architecture and the draft head exist only in the fork commit pinned in `scripts/env.sh`. `bash scripts/setup.sh --check` verifies the import.
- Install TabbyAPI's GPU extras: they replace torch and the fork.
- Hand-edit `draft_model_name`: `DRAFT=mtp` needs no second model, and `DRAFT=dflash2` wires the drafter in for you.

## Runtime identity

| Component | Pin |
|---|---|
| Pack | `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` @ `44c78c9d229b43dd45e41506071b9f5f939bed44` |
| DFlash2 drafter (optional) | [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) @ `4f9088ea66c21f2b99ed657d1941c306b03e84d2` |
| ExLlamaV3 fork | [`vcruz305/exllamav3`](https://github.com/vcruz305/exllamav3) @ `c702887ba7be368e1714339dd755a7c3c9b263d5` |
| TabbyAPI | `vcruz305/tabbyAPI` @ `a2898191eed710b55066d699ce829a976b4b3272` |
| PyTorch / CUDA | torch 2.10.0+cu128, CUDA 12.8 toolkit |
| Host | Ubuntu 22.04 under WSL2, Python 3.10 |

`SHA256SUMS` is the authority for the pack contents. If the pinned pack revision stops resolving,
`download_and_verify.sh` falls back to `main` with a warning and the checksums still decide.

## Cards tested on

| Card | Memory | Status |
|---|---|---|
| NVIDIA RTX 4090 | 24 GB | Measured (this folder) |

Nothing else has been verified with these scripts.

## Layout

| Path | What |
|---|---|
| `scripts/env.sh` | pins, paths, settings, runtime check |
| `scripts/setup.sh` | venv, fork at the pinned commit, TabbyAPI, extension build; `--check` verifies an install |
| `scripts/download_and_verify.sh` | hf_xet download and sha256 verification against `SHA256SUMS`; `--drafter` fetches DFlash2 against `SHA256SUMS.dflash2` |
| `scripts/serve.sh` | writes the TabbyAPI config and starts it detached |
| `scripts/guard.sh` | `watch`, `status`, `stop` |
| `scripts/smoke.sh` | one chat request that must answer cleanly |
| `scripts/bench_concurrency.py` | the concurrency measurements above (`--n` streams at once) |
| `scripts/bench.py` | the speed measurements above (`--temperature`, `--prompt-tokens`, `--json`) |

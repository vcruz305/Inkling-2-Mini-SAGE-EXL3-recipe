# Inkling 2 Mini SAGE EXL3 4.23 bpw on one DGX Spark

Serves [`vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw`](https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw) through TabbyAPI on the
[ExLlamaV3 fork](https://github.com/vcruz305/exllamav3), with speculative decode from either the pack's own MTP draft head or the separate
[DFlash2 drafter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2). Every number here was
measured by running these scripts from an empty directory on the card named at the bottom.

## Quick start

Needs one DGX Spark (MemAvailable fell by about 68 GiB at the measured settings, `MAX_SEQ_LEN=262144`), Python 3.10+, `git`, `g++` and the CUDA 13.0 toolkit (`nvcc`).
Run from this folder:

```bash
bash scripts/setup.sh                # venv + fork + TabbyAPI, compiles the CUDA extension
bash scripts/download_and_verify.sh  # 15.1 GB pack and the 990 MB DFlash2 drafter via hf_xet, then sha256 of every file
bash scripts/serve.sh                # starts detached, DFlash2 draft on
bash scripts/guard.sh watch          # returns when the endpoint answers
bash scripts/smoke.sh                # one chat request, must print SMOKE PASS
python3 scripts/bench.py             # optional: the speed table below
```

The [DFlash2 drafter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (public, Apache-2.0) is the default here because it measured fastest for one stream. To use the pack's built-in MTP head instead, with no second download:

```bash
bash scripts/guard.sh stop
DRAFT=mtp bash scripts/serve.sh
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
| `DRAFT` | `dflash2` | Speculative decode with the separate DFlash2 drafter. `DRAFT=mtp` uses the draft head inside the pack (`mtp.safetensors`); `DRAFT=off` serves one token per step. |
| `MAX_SEQ_LEN` / `CACHE_SIZE` | 262144 / 262144 | The context every number below was taken at. The model allows 1,048,576; larger values were not run here. Lower it to save memory (about 0.19 GiB per 1,024 tokens). |
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
| `setup.sh` from an empty directory (includes the CUDA extension build; timed at the previous engine pin) | 3 min 42 s |
| `serve.sh` to endpoint answering, pack already in the page cache | 9 s (any draft setting) |

A first start after a download or reboot reads 15 GB from disk and takes longer.

## Measured memory

| Reading | DFlash2 | MTP | Off |
|---|---|---|---|
| MemAvailable before load | 114.7 GiB | 118.2 GiB | 118.2 GiB |
| Lowest MemAvailable over the run | 48.1 GiB | 49.9 GiB | 51.1 GiB |
| Swap growth | none | none | none |

At `MAX_SEQ_LEN=262144` the server holds about 68 GiB from load onward: the pack plus an FP16 cache of about 0.19 GiB per 1,024 tokens of `CACHE_SIZE`. The DFlash2 column includes the 94,235-token prompt. Unified memory: `nvidia-smi` reports `[N/A]` for memory on this part, so the figure to watch is `MemAvailable`, sampled once a second (`guard.sh status` prints it). If the Spark carries other work, lower `MAX_SEQ_LEN`: 24,576 holds about 25 GiB.

## Measured speed

Server-side counters from TabbyAPI's `usage` block, one request at a time, streamed, at `MAX_SEQ_LEN=262144`.
`scripts/bench.py` produced all of them: three short prompts (code, devops, prose), 300 new
tokens each, 4 repeats. Each draft setting was its own server start in the same session.

| Decode, short prompts | `DRAFT=dflash2` | `DRAFT=mtp` | `DRAFT=off` |
|---|---|---|---|
| Temperature 0, median of 12 | **75.4** tok/s | 66.4 tok/s | 51.2 tok/s |
| Temperature 1, median of 12 | **62.3** tok/s | 59.3 tok/s | 51.2 tok/s |

With a draft on, speed depends on how predictable the text is. Per prompt, median of 4, tok/s:

| Prompt | Temp 0, DFlash2 | Temp 0, MTP | Temp 0, off | Temp 1, DFlash2 | Temp 1, MTP | Temp 1, off |
|---|---|---|---|---|---|---|
| code | **112.8** | 93.2 | 51.0 | **94.8** | 90.4 | 51.0 |
| devops | **75.4** | 66.4 | 51.4 | **62.3** | 59.2 | 51.5 |
| prose | **61.4** | 59.0 | 51.2 | 54.9 | **57.9** | 51.2 |

The 12 requests ranged 60.9–113.3 tok/s (DFlash2) and 58.8–93.4 (MTP) at temperature 0, and
51.8–109.3 and 55.5–98.1 at temperature 1. On this setup DFlash2 is the faster drafter at temperature 0 and on code;
at temperature 1 the two are close on devops and prose.

Prefill and long prompts, temperature 0, one request each, in the order sent:

| Prompt tokens | Draft | Prefill | Decode | Time to first token |
|---|---|---|---|---|
| 9,159 (first) | dflash2 | 656 tok/s | 59.5 tok/s | 14.19 s |
| 9,159 (first) | mtp | 670 tok/s | 67.5 tok/s | 13.80 s |
| 9,159 (first) | off | 672 tok/s | 46.4 tok/s | 13.72 s |
| 22,839 | dflash2 | 623 tok/s | 47.1 tok/s | 36.90 s |
| 22,839 | mtp | 624 tok/s | 56.7 tok/s | 36.79 s |
| 22,839 | off | 626 tok/s | 42.5 tok/s | 36.63 s |
| 9,159 (after the 22,839 one) | dflash2 | 670 tok/s | 60.5 tok/s | 13.88 s |
| 9,159 (after the 22,839 one) | mtp | 671 tok/s | 66.9 tok/s | 13.75 s |
| 9,159 (after the 22,839 one) | off | 675 tok/s | 46.3 tok/s | 13.68 s |
| 94,235 | dflash2 | 425 tok/s | 26.8 tok/s | 222.41 s |
| 9,159 (after the 94,235 one) | dflash2 | 658 tok/s | 69.9 tok/s | 14.13 s |

Prefill is the server counter; time to first token is client-side and includes it.

The 94,235-token request was sent with `DRAFT=dflash2` only. Prefill slows as the context fills,
because every new token attends to all earlier ones on the global layers: a 2,048-token chunk
runs at about 620 tok/s at 8k of context, 355 at 65k and 105 at 262k (direct engine measurement,
not through the server). Decode slows the same way: about 46, 33 and 17 tok/s without a draft at
those depths. A prompt that fills the whole 262,144 context was not run end to end.

## Concurrency

`MAX_BATCH_SIZE=4`, same context and cache, temperature 0, 300 new tokens per stream.
`scripts/bench_concurrency.py` produced these; totals are all streams together, first token to
last token, client-side.

| Streams at once | `DRAFT=mtp` total | `DRAFT=dflash2` total | `DRAFT=off` total | Per stream |
|---|---|---|---|---|
| 1 | 59.3–93.6 tok/s | 61.1–117.1 tok/s | 51.2–51.9 tok/s | the same |
| 2 | 69.7–72.9 tok/s | 71.0–73.0 tok/s | 69.9–74.2 tok/s | about 35–37 tok/s |
| 4 | 99.1–99.3 tok/s | 95.4–100.4 tok/s | 96.6–100.0 tok/s | about 24–26 tok/s |

Either drafter runs while one stream is active, stands down when a second arrives, and resumes
afterwards, so with two or more streams the draft setting makes no difference. The first request
after a start is slower and is left out of the single-stream row. Time to first token was
0.32–0.70 s at 2 streams and 0.54–1.43 s at 4.

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
- **Long prompts take minutes.** A 94,235-token prompt took 222 s to first token; a prompt filling the whole 262,144 context was not run end to end (estimated at about 25 minutes from the per-depth rates).

## Do not

- `pip install exllamav3` or use a stock wheel: the Inkling2Loop architecture and the draft head exist only in the fork commit pinned in `scripts/env.sh`. `bash scripts/setup.sh --check` verifies the import.
- Install TabbyAPI's GPU extras: they replace torch and the fork.
- Hand-edit `draft_model_name`: `DRAFT=dflash2` wires the drafter in for you, and `DRAFT=mtp` needs no second model.

## Runtime identity

| Component | Pin |
|---|---|
| Pack | `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` @ `44c78c9d229b43dd45e41506071b9f5f939bed44` |
| DFlash2 drafter (default draft) | [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) @ `4f9088ea66c21f2b99ed657d1941c306b03e84d2` |
| ExLlamaV3 fork | [`vcruz305/exllamav3`](https://github.com/vcruz305/exllamav3) @ `c702887ba7be368e1714339dd755a7c3c9b263d5` |
| TabbyAPI | `vcruz305/tabbyAPI` @ `a2898191eed710b55066d699ce829a976b4b3272` |
| PyTorch / CUDA | torch 2.13.0+cu130, CUDA 13.0 toolkit |
| Host | Ubuntu 24.04 aarch64, Python 3.12 |

`SHA256SUMS` is the authority for the pack contents. If the pinned pack revision stops resolving,
`download_and_verify.sh` falls back to `main` with a warning and the checksums still decide.

## Cards tested on

| Card | Memory | Status |
|---|---|---|
| NVIDIA DGX Spark (GB10) | 128 GB unified | Measured (this folder) |

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

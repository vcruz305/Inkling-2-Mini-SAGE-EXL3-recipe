# Inkling 2 Mini SAGE EXL3 4.23 bpw on one DGX Spark

Serves [`vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw`](https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw) through TabbyAPI on the
[ExLlamaV3 fork](https://github.com/vcruz305/exllamav3), with the pack's own MTP draft head doing speculative decode. Every number here was
measured by running these scripts from an empty directory on the card named at the bottom.

## Quick start

Needs one DGX Spark (MemAvailable fell by about 22 GiB at the measured settings), Python 3.10+, `git`, `g++` and the CUDA 13.0 toolkit (`nvcc`).
Run from this folder:

```bash
bash scripts/setup.sh                # venv + fork + TabbyAPI, compiles the CUDA extension
bash scripts/download_and_verify.sh  # 15.1 GB pack via hf_xet, then sha256 of every file
bash scripts/serve.sh                # starts detached, draft on
bash scripts/guard.sh watch          # returns when the endpoint answers
bash scripts/smoke.sh                # one chat request, must print SMOKE PASS
python3 scripts/bench.py             # optional: the speed table below
```

The API is OpenAI-compatible at `http://127.0.0.1:8000/v1`. Clients that append
`/chat/completions` themselves need the base URL **including** `/v1`.

Everything installs under `RECIPE_HOME` (default `~/inkling2-exl3`) and the pack goes to
`MODEL_DIR` (default `~/models/Inkling-2-Mini-SAGE-REXL3-4.23bpw`). The pack is public and
ungated, so no token is needed.

## The measured settings

| Setting | Value | Why |
|---|---|---|
| `DRAFT` | `mtp` | Speculative decode with the draft head inside the pack (`mtp.safetensors`). `DRAFT=off` serves one token per step. |
| `MAX_SEQ_LEN` / `CACHE_SIZE` | 24576 / 24576 | The context every number below was taken at. Larger values are untested here. |
| `CACHE_MODE` | `FP16` | The only KV cache type tested with this architecture. |
| `CHUNK_SIZE` | 2048 | Prefill chunk used for the prefill figures. |
| `MAX_BATCH_SIZE` | 1 | Two or more concurrent streams leave the fast decode path and the draft. |
| sampling fallback | temperature 1.0 | The pack's `generation_config.json` value, applied only when a client sends none. |

`scripts/env.sh` holds these and the pins. The fork's own switches (`INKLING2_*`) are left at
their defaults, which is what was measured; `serve.sh` forwards one only if you export it and
prints the full set on start. `DRY_RUN=1 bash scripts/serve.sh` prints the config without
starting anything.

## Measured bring-up

| Phase | Measured |
|---|---|
| `setup.sh` from an empty directory (includes the CUDA extension build) | 3 min 42 s |
| `serve.sh` to endpoint answering, pack already in the page cache | 9 s (draft on), 6 s (draft off) |

A first start after a download or reboot reads 15 GB from disk and takes longer.

## Measured memory

| Reading | Value |
|---|---|
| MemAvailable before load, draft on | 104.5 GiB |
| Lowest MemAvailable over the run, draft on | 82.1 GiB |
| MemAvailable before load, draft off | 107.2 GiB |
| Lowest MemAvailable over the run, draft off | 83.6 GiB |
| Swap growth | none |

Unified memory: `nvidia-smi` reports `[N/A]` for memory on this part, so the figure to watch is `MemAvailable`, sampled once a second (`guard.sh status` prints it). Two unrelated GPU services (about 9 GiB) were running on the box during the measurement, so the before-load figures are not an idle machine's.

## Measured speed

Server-side counters from TabbyAPI's `usage` block, one request at a time, streamed.
`scripts/bench.py` produced all of them: three short prompts (code, devops, prose), 300 new
tokens each, 4 repeats. Draft on and draft off were two server starts in the same session.

| Decode, short prompts | Draft on | Draft off | Ratio |
|---|---|---|---|
| Temperature 0, median of 12 | 63.1 tok/s | 51.4 tok/s | 1.23× |
| Temperature 1, median of 12 | 62.3 tok/s | 51.0 tok/s | 1.22× |

With the draft on, speed depends on how predictable the text is: the 12 requests ranged
57.0–93.7 tok/s at temperature 0 and 57.3–90.5 at temperature 1. Per prompt, median of 4, tok/s:

| Prompt | Temp 0, draft on | Temp 0, draft off | Temp 1, draft on | Temp 1, draft off |
|---|---|---|---|---|
| code | 93.2 | 51.1 | 86.9 | 50.7 |
| devops | 63.1 | 51.8 | 62.3 | 51.1 |
| prose | 58.3 | 51.5 | 58.7 | 50.8 |

The server logged 57–69% of drafted tokens accepted per request (51 requests).

Long prompts, temperature 0, one request each, in the order sent:

| Prompt tokens | Draft | Prefill | Decode | Time to first token |
|---|---|---|---|---|
| 9,159 (first) | on | 580 tok/s | 61.1 tok/s | 15.96 s |
| 9,159 (first) | off | 584 tok/s | 46.1 tok/s | 15.8 s |
| 22,839 | on | 445 tok/s | 55.7 tok/s | 51.52 s |
| 22,839 | off | 447 tok/s | 42.4 tok/s | 51.18 s |
| 9,159 (after the 22,839 one) | on | 583 tok/s | 69.8 tok/s | 15.81 s |
| 9,159 (after the 22,839 one) | off | 585 tok/s | 46.1 tok/s | 15.76 s |

Prefill is the server counter; time to first token is client-side and includes it. The draft does
not change prefill beyond run-to-run variation. Short-prompt decode re-measured after the long
prompts was within 5% of the first pass.

## Correctness checks in the same run

- `smoke.sh` passed with the draft on and off (answer `Paris`, `finish_reason: stop`, no leaked control tokens).
- A tool request returned one parsed `tool_calls` entry (`get_weather`, `{"city": "Paris"}`) with the reasoning in `reasoning_content` and `finish_reason: tool_calls`.
- A temperature-1 chat request returned reasoning and answer in separate channels, with the right answer.
- No traceback in either server log.

## Known limits

- **Sampler settings beyond temperature draft the old way.** Requests that set only a temperature use speculative sampling (a draft is kept with probability min(1, p/q)). Requests that add top-k, top-p, min-p or penalties still draft, but greedily with exact-match acceptance, so they gain less above temperature 0.
- **Draft-on output is not token-identical to draft-off.** Greedy text can diverge because verification runs several positions in one batch.
- **Constrained output (JSON schema, grammar) was not tested** through this recipe.
- **Concurrency is untested.** `MAX_BATCH_SIZE` above 1 leaves the fast decode path.
- **Image input was not tested** through this recipe.
- **Context above 24,576 was not measured** on this setup.

## Do not

- `pip install exllamav3` or use a stock wheel: the Inkling2Loop architecture and the draft head exist only in the fork commit pinned in `scripts/env.sh`. `bash scripts/setup.sh --check` verifies the import.
- Install TabbyAPI's GPU extras: they replace torch and the fork.
- Set `draft_model_name` or download a separate draft model: the draft is part of the pack.
- Raise `MAX_BATCH_SIZE` and expect the single-stream numbers.

## Runtime identity

| Component | Pin |
|---|---|
| Pack | `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` @ `44c78c9d229b43dd45e41506071b9f5f939bed44` |
| ExLlamaV3 fork | [`vcruz305/exllamav3`](https://github.com/vcruz305/exllamav3) @ `10a85129180145c888693f222655f79e7d403216` |
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
| `scripts/download_and_verify.sh` | hf_xet download and sha256 verification against `SHA256SUMS` |
| `scripts/serve.sh` | writes the TabbyAPI config and starts it detached |
| `scripts/guard.sh` | `watch`, `status`, `stop` |
| `scripts/smoke.sh` | one chat request that must answer cleanly |
| `scripts/bench.py` | the speed measurements above (`--temperature`, `--prompt-tokens`, `--json`) |

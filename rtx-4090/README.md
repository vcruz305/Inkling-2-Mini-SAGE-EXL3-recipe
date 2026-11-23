# Inkling 2 Mini SAGE EXL3 4.23 bpw on one 24 GB card

Serves [`vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw`](https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw) through TabbyAPI on the
ExLlamaV3 fork, with the pack's own MTP draft head doing speculative decode. Every number here was
measured by running these scripts from an empty directory on the card named at the bottom.

## Quick start

Needs a 24 GB NVIDIA card with nothing else on it, Python 3.10+, `git`, `g++` and the CUDA 12.8 toolkit (`nvcc`).
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
| `setup.sh` from an empty directory (includes the CUDA extension build) | about 9 minutes |
| `serve.sh` to endpoint answering, pack already in the page cache | 22 s |

A first start after a download or reboot reads 15 GB from disk and takes longer.

## Measured memory

| Reading | Value |
|---|---|
| Loaded, draft on | 20,449 MiB |
| Peak over the run, draft on (during the 22,839-token prompt) | 23,989 MiB |
| Loaded, draft off | 19,972 MiB |
| Peak over the run, draft off | 22,154 MiB |

Readings are `nvidia-smi` totals for the card, sampled once a second, and include about 200 MiB the desktop was already using. The card has 24,564 MiB, so the draft-on peak left about 575 MiB free. If the same card carries other GPU work, start with `DRAFT=off` (about 1.8 GB lower at peak) or a smaller `MAX_SEQ_LEN`; a smaller `MAX_SEQ_LEN` was not measured.

## Measured speed

Server-side counters from TabbyAPI's `usage` block, one request at a time, streamed.
`scripts/bench.py` produced all of them: three short prompts (code, devops, prose), 300 new
tokens each, 4 repeats. Draft on and draft off were two server starts in the same session.

| Decode, short prompts | Draft on | Draft off | Ratio |
|---|---|---|---|
| Temperature 0, median of 12 | 150.6 tok/s | 113.5 tok/s | 1.33× |
| Temperature 1, median of 12 | 136.4 tok/s | 113.4 tok/s | 1.20× |

With the draft on, speed depends on how predictable the text is: the 12 requests ranged
121.3–219.4 tok/s at temperature 0 and 118.8–205.0 at temperature 1. Per prompt, median of 4, tok/s:

| Prompt | Temp 0, draft on | Temp 0, draft off | Temp 1, draft on | Temp 1, draft off |
|---|---|---|---|---|
| code | 215.3 | 112.9 | 196.4 | 112.7 |
| devops | 150.6 | 114.2 | 136.4 | 113.5 |
| prose | 128.0 | 113.4 | 121.2 | 113.6 |

The server logged 59–69% of drafted tokens accepted per request.

Long prompts, temperature 0, one request each, in the order sent:

| Prompt tokens | Draft | Prefill | Decode | Time to first token |
|---|---|---|---|---|
| 9,159 (first) | on | 1,267 tok/s | 171.4 tok/s | 7.42 s |
| 9,159 (first) | off | 1,189 tok/s | 108.1 tok/s | 7.79 s |
| 22,839 | on | 1,182 tok/s | 143.0 tok/s | 19.58 s |
| 22,839 | off | 1,207 tok/s | 101.0 tok/s | 19.03 s |
| 9,159 (after the 22,839 one) | on | 1,322 tok/s | 163.8 tok/s | 7.01 s |
| 9,159 (after the 22,839 one) | off | 1,277 tok/s | 107.4 tok/s | 7.25 s |

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
| ExLlamaV3 fork | `vcruz305/exllamav3` @ `10a85129180145c888693f222655f79e7d403216` |
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
| `scripts/download_and_verify.sh` | hf_xet download and sha256 verification against `SHA256SUMS` |
| `scripts/serve.sh` | writes the TabbyAPI config and starts it detached |
| `scripts/guard.sh` | `watch`, `status`, `stop` |
| `scripts/smoke.sh` | one chat request that must answer cleanly |
| `scripts/bench.py` | the speed measurements above (`--temperature`, `--prompt-tokens`, `--json`) |

# Inkling 2 Mini SAGE EXL3 4.23 bpw on one DGX Spark

Serves [`vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw`](https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw) through TabbyAPI on the
[ExLlamaV3 fork](https://github.com/vcruz305/exllamav3), with speculative decode from either the pack's own MTP draft head or the separate
[DFlash2 drafter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2). Every number here was
measured by running these scripts from an empty directory on the card named at the bottom.

## Quick start

Needs one DGX Spark (MemAvailable fell by about 25 GiB at the measured settings), Python 3.10+, `git`, `g++` and the CUDA 13.0 toolkit (`nvcc`).
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
| `DRAFT` | `dflash2` | Speculative decode with the separate DFlash2 drafter. `DRAFT=mtp` uses the draft head inside the pack (`mtp.safetensors`) and becomes the default when `MAX_BATCH_SIZE` is above 1; `DRAFT=off` serves one token per step. |
| `MAX_SEQ_LEN` / `CACHE_SIZE` | 24576 / 24576 | The context every number below was taken at. Larger values are untested here. |
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
| MemAvailable before load | 115.3 GiB | 118.9 GiB | 118.8 GiB |
| Lowest MemAvailable over the run | 93.2 GiB | 93.8 GiB | 95.3 GiB |
| Swap growth | none | none | none |

Unified memory: `nvidia-smi` reports `[N/A]` for memory on this part, so the figure to watch is `MemAvailable`, sampled once a second (`guard.sh status` prints it). The lowest reading falls during the 22,839-token prompt.

## Measured speed

Server-side counters from TabbyAPI's `usage` block, one request at a time, streamed.
`scripts/bench.py` produced all of them: three short prompts (code, devops, prose), 300 new
tokens each, 4 repeats. Each draft setting was its own server start in the same session.

| Decode, short prompts | `DRAFT=dflash2` | `DRAFT=mtp` | `DRAFT=off` |
|---|---|---|---|
| Temperature 0, median of 12 | **66.6 tok/s** | 61.3 tok/s | 50.2 tok/s |
| Temperature 1, median of 12 | 61.4 tok/s | 61.9 tok/s | 50.2 tok/s |

With a draft on, speed depends on how predictable the text is. Per prompt, median of 4, tok/s:

| Prompt | Temp 0, DFlash2 | Temp 0, MTP | Temp 0, off | Temp 1, DFlash2 | Temp 1, MTP | Temp 1, off |
|---|---|---|---|---|---|---|
| code | **108.8** | 90.6 | 50.0 | **101.8** | 89.3 | 50.0 |
| devops | 66.6 | 61.3 | 50.5 | 60.7 | 61.9 | 50.4 |
| prose | 57.5 | 56.7 | 50.2 | 55.2 | 54.5 | 50.2 |

The 12 requests ranged 57.0–108.9 tok/s (DFlash2) and 56.3–90.8 (MTP) at temperature 0, and
50.1–112.1 and 54.2–92.4 at temperature 1. On this setup DFlash2 is the faster drafter on code
and about level with MTP on prose and at temperature 1.

Prefill and long prompts, temperature 0, one request each, in the order sent:

| Prompt tokens | Draft | Prefill | Decode | Time to first token |
|---|---|---|---|---|
| 9,159 (first) | dflash2 | 583 tok/s | 67.4 tok/s | 15.94 s |
| 9,159 (first) | mtp | 584 tok/s | 62.4 tok/s | 15.84 s |
| 9,159 (first) | off | 586 tok/s | 45.5 tok/s | 15.75 s |
| 22,839 | dflash2 | 444 tok/s | 50.4 tok/s | 51.72 s |
| 22,839 | mtp | 444 tok/s | 58.5 tok/s | 51.61 s |
| 22,839 | off | 445 tok/s | 41.5 tok/s | 51.43 s |
| 9,159 (after the 22,839 one) | dflash2 | 584 tok/s | 59.0 tok/s | 15.90 s |
| 9,159 (after the 22,839 one) | mtp | 582 tok/s | 59.9 tok/s | 15.86 s |
| 9,159 (after the 22,839 one) | off | 587 tok/s | 45.5 tok/s | 15.70 s |

Prefill is the server counter; time to first token is client-side and includes it. Neither draft
changes prefill beyond run-to-run variation.

## Concurrency

`MAX_BATCH_SIZE=4`, same context and cache, temperature 0, 300 new tokens per stream.
`scripts/bench_concurrency.py` produced these; totals are all streams together, first token to
last token, client-side.

| Streams at once | `DRAFT=mtp` total | `DRAFT=off` total | Per stream |
|---|---|---|---|
| 1 | 56.8–90.6 tok/s | 50.4–51.3 tok/s | the same |
| 2 | 68.8–72.5 tok/s | 69.0–73.2 tok/s | about 35–36 tok/s |
| 4 | 91.9–97.7 tok/s | 98.0–101.7 tok/s | about 24–25 tok/s |

The MTP draft runs while one stream is active, stands down when a second arrives, and resumes
afterwards (single-stream speed after the 4-stream rounds: 56.8–90.5 tok/s). With two or more
streams the draft setting therefore makes no difference.

**Concurrent serving uses `DRAFT=mtp` (automatic when `MAX_BATCH_SIZE` is above 1) or `off`, not `dflash2`.** With the DFlash2 drafter
loaded, concurrent streams were slower than one stream: 38.2–52.3 tok/s total at 2 streams and
46.4–50.3 at 4. Its single-stream speed at `MAX_BATCH_SIZE=4` was unaffected (57.8–111.4 tok/s).

## Correctness checks in the same run

- `smoke.sh` passed with DFlash2, MTP and the draft off (answer `Paris`, `finish_reason: stop`, no leaked control tokens).
- With DFlash2 and with MTP, a tool request returned one parsed `tool_calls` entry (`get_weather`, `{"city": "Paris"}`) with the reasoning in `reasoning_content`.
- With DFlash2 and with MTP, a temperature-1 chat request returned reasoning and answer in separate channels, with the right answer.
- No traceback in any server log, including the 2- and 4-stream rounds; every concurrent stream completed.

## Known limits

- **Sampler settings beyond temperature draft the old way.** Requests that set only a temperature use speculative sampling (a draft is kept with probability min(1, p/q)). Requests that add top-k, top-p, min-p or penalties still draft, but greedily with exact-match acceptance, so they gain less above temperature 0.
- **Draft-on output is not token-identical to draft-off.** Greedy text can diverge because verification runs several positions in one batch.
- **Constrained output (JSON schema, grammar) was not tested** through this recipe.
- **No drafting under concurrency.** With two or more streams active the MTP draft stands down, and DFlash2 slows concurrent streams (see Concurrency).
- **Concurrent streams are not token-identical to a single stream.** Greedy text can diverge partway, because several streams share one batched step.
- **DFlash2 uses speculative sampling only for temperature-only requests**, the same rule as MTP in the first item.
- **Image input was not tested** through this recipe.
- **Context above 24,576 was not measured** on this setup.

## Do not

- `pip install exllamav3` or use a stock wheel: the Inkling2Loop architecture and the draft head exist only in the fork commit pinned in `scripts/env.sh`. `bash scripts/setup.sh --check` verifies the import.
- Install TabbyAPI's GPU extras: they replace torch and the fork.
- Hand-edit `draft_model_name`: `DRAFT=dflash2` wires the drafter in for you, and `DRAFT=mtp` needs no second model.
- Force `DRAFT=dflash2` together with `MAX_BATCH_SIZE` above 1.

## Runtime identity

| Component | Pin |
|---|---|
| Pack | `vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw` @ `44c78c9d229b43dd45e41506071b9f5f939bed44` |
| DFlash2 drafter (default draft) | [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) @ `4f9088ea66c21f2b99ed657d1941c306b03e84d2` |
| ExLlamaV3 fork | [`vcruz305/exllamav3`](https://github.com/vcruz305/exllamav3) @ `819b039c5703ece8dec7c0ac0f4fe4e6ec0ec9c1` |
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

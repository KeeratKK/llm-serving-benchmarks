# LLM Serving Performance Lab

> **Status: v0.1 in progress.**

## The question

On a fixed GPU and model, how does the throughput/latency frontier move as
concurrency and context length change, and where does TTFT break down?

## Environment

Every number in this README was measured on the setup below. Nothing is
extrapolated from other hardware.

| | |
|---|---|
| GPU | TODO (count, model, driver, CUDA) |
| Libraries | TODO (torch, vLLM, transformers) |
| Model | TODO (name, dtype, quantization, TP degree) |
| Attention backend | TODO (from the vLLM startup log; T4 is sm75, so not FA2) |
| Server flags | TODO (all set explicitly, none left at default) |

## Architecture

TODO: diagram (workload generator → vLLM OpenAI server → per-request JSONL;
GPU sampler alongside).

## Methodology

Decisions made so far:

- **Closed-loop load.** Exactly N requests are in flight at a time. When one
  finishes, the next starts.
- **Steady-state window.** A request counts only if it *started* after
  every request in the first wave (the first N) had finished, and it
  finished before sending stopped. Having N requests in flight isn't
  enough on its own: the first wave all arrives together, and its prefill
  backlog is still clearing after all N are sent. Requests from the
  ramp-up and the drain at the end are discarded.
- **Staggered start.** The first N requests are spaced out slightly, so
  the closed loop doesn't synchronize into waves.
- **Stopping rule per point.** Each concurrency level stops sending at
  whichever comes first:
  - **K = max(150, 5N)** in-window successful requests. 5N means about
    five full turnovers of the worker pool, and at N=256 there are about
    13 samples behind TTFT p99.
  - **T = 20 minutes** from the start of the level. This is a safety cap,
    about 2× the slowest point in the pre-run estimate. A point that hits
    T was more than 2× slower than predicted, and that gets investigated.
  - **M = max(10, N)** consecutive errors or validation failures. This
    means the server is dead or misconfigured, so the whole sweep aborts.
    It scales with N because at high N, timeouts from genuine overload
    arrive in bursts, and those are data.

  Errors and failed validations never count toward K. They are still
  recorded.
- **Server warmup.** Warmup requests are sent and discarded after each
  server start.
- **Prompts.** Random token IDs, sent to `/v1/completions`. Special-token
  IDs are excluded, and each request gets its own prompt from a seeded
  RNG.
- **Fixed output length.** Set with `ignore_eos` / `min_tokens`. Every
  response is checked against `usage`.
- **Prefix caching off**, so shared prefixes can't make TTFT look better
  than it is.
- **The client is plain `aiohttp` with its own SSE parsing, not the OpenAI
  SDK.** The SDK builds a typed Python object for every streamed token, and
  at high concurrency that made the client the bottleneck (see Harness
  validation). The harness reads each `data:` line, timestamps it before
  parsing, and `json.loads` it into a plain dict. Other client settings,
  each chosen because the default would distort the measurement:
  - Connection limit sized from the largest N in the sweep. aiohttp's
    default of 100 would silently cap concurrency.
  - Keep-alive of 60 s, so reconnect cost doesn't land in TTFT.
  - The stream is always read to EOF, even after `[DONE]`. aiohttp closes
    a connection whose body wasn't fully read instead of reusing it.
  - The timeout is a cap on the whole request (E2E), set to about 2× the
    slowest predicted E2E.
- **The harness only records.** Each request writes its send time, every
  token's arrival time, and the server's `usage`, and any mismatch is
  flagged rather than dropped. TTFT, TPOT, the ITL distribution, and the
  steady-state window are all computed from that raw data during analysis.
- **No client retries.** A failed request is recorded as a failure, never
  resent.
- **Every run records its own environment.** Each run writes to
  `results/<run_id>/`:

  | File | Contents |
  |---|---|
  | `meta.json` | Git commit (plus a flag for uncommitted changes), every harness arg, client and server package versions, GPU and driver, the exact server launch command, and, parsed from the vLLM log, the attention backend, KV cache capacity, and effective engine config |
  | `requests.jsonl` | One row per request: send time, every token's arrival time, `usage`, and errors or validation failures. Against the mock, also the mock's send time for every token (`sent_timestamps`) |
  | `events.jsonl` | Level start, N-th request sent, first wave finished, sending stopped (and why), drain finished |
  | `warmup.jsonl` | The discarded warmup requests, kept to compare cold and warm behavior |
  | `server.log` | The vLLM log, copied at the end. It includes the server's periodic running / waiting / KV-usage lines |
- **Repeats.** TODO: count, and how spread is reported.
- **Sweep grid.** TODO: concurrency levels, prompt lengths, output
  length, and the reason for each value.

## Harness validation

Before measuring a GPU, the harness was checked against a mock server
(`mock/server.py`). The mock streams vLLM's wire format with fixed delays
(TTFT 200 ms, then one token every 20 ms) and has no capacity limit. So any
latency that grows with N has to come from the client. Every token chunk
also carries the mock's send time. On Linux, `time.perf_counter()` reads
`CLOCK_MONOTONIC`, which is shared by every process on the machine, so the
harness's arrival time minus the mock's send time is the client's own delay
for that token (`analysis/stall_check.py`).

**Setup:** WSL2 (kernel 6.18.40.1) on an Intel i7-1260P (16 logical CPUs),
Python 3.12.3, aiohttp 3.14.3, numpy 2.5.3, mock on fastapi 0.141.1 /
uvicorn 0.54.0 (`--log-level warning`). N=256, prompt 128, output 256,
K=1280 in-window requests. Commit `7c77717`, run `results/20261001-003329`.
**This checks the harness and isn't a benchmark result.** A second run on
the same code agreed within the ranges shown in brackets.

| N=256, 20 ms ITL | Expected | Measured [second run] |
|---|---|---|
| Throughput | ~46 req/s (N / E2E) | 45.9 [45.8] req/s |
| TTFT p50 / p99 | ~200 ms | 204 / 215 [204 / 221] ms |
| E2E p50 / p99 | ~5.6 s (5.65 s measured at N=1) | 5.57 / 5.62 [5.60 / 5.65] s |
| ITL p10 / p50 / p99 | the mock's own send gaps: 20.4 / 20.9 / 24.5 ms | 19.2 / 20.9 / 28.4 [19.3 / 20.9 / 29.4] ms |
| Client delay (arrival − send), p50 / p99 / max | small | 1.2 / 8.4 / 34.2 [1.1 / 10.5 / 26.1] ms |
| Longest gap with no tokens on any stream | — | 10 ms, the same on the mock's send side (both runs) |

That's 12,800 tokens/s through one client process. At that rate, the
harness timestamps tokens about 1 ms late at the median and 8–11 ms late
at p99. That adds about 4 ms to ITL p99. At N=16 the median delay is
1.2 ms, so load mostly affects the tail. This is the harness's measurement
error, and the vLLM results should be read with it in mind.

**What failed before this passed:**

- **OpenAI SDK client** (same test, no profiler attached, run
  `results/20260930-210638`): 28 req/s,
  E2E p50 9.1 s, ITL p50 0.1 ms / p99 1,161 ms. The client process sat
  at about 95% of a core. Chunks queued up in the socket and were read in
  bursts, so ITL showed stalls that weren't there. A py-spy profile put
  about half of the client's CPU in the SDK building per-token response
  objects. Switching to raw SSE parsing on aiohttp fixed it.
- **Profiling and measuring in the same run.** By default, `py-spy record`
  pauses the target process for each sample. Under py-spy, the ITL
  distribution showed gaps of about 45 ms with no tokens on any stream,
  and those gaps disappeared once py-spy was off. Profiles and
  measurements now come from separate runs.

## Results

TODO: throughput vs concurrency, TTFT vs concurrency, TPOT vs context
length, and latency percentiles.

## Bottleneck analysis

TODO: an explanation of why at least one of the curves bends.

## What surprised me

TODO

## Limitations

- Random-token prompts: valid for throughput and latency, not for
  anything content-dependent.
- Closed-loop load can't show unbounded queueing.
- Client timestamp error, measured against the mock: about 1 ms at the
  median and 8–11 ms at p99, at 12,800 tokens/s. The vLLM points
  stream far fewer tokens per second, but ITL p99 differences of a few ms
  are within this error.
- TODO

## Reproducing

```bash
pip install -r requirements.txt
```

**Against the mock server** (no GPU; checks the harness itself):

```bash
uvicorn mock.server:app --port 8001 --log-level warning   # TTFT_DELAY / ITL_DELAY env vars set its timing
python harness/loadgen.py --concurrencies 256 --prompt-lens 128
python analysis/stall_check.py results/<run_id>
```

Against the mock, TTFT should match `TTFT_DELAY` (0.2 s by default) and
the harness's ITL should match the mock's own send gaps, at any N. If
either one rises with N, the client is the bottleneck. Don't run a
profiler during this check.

**Against vLLM:** TODO (the Colab launch command with every server flag, and
the harness invocation with `--server-cmd`, `--server-log`, and
`--results-dir`).

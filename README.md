# LLM Serving Performance Lab

> **Status: v0.1 in progress. Target ship date: Oct 5, 2026.** No results yet.

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

TODO: diagram (workload generator → vLLM OpenAI server → per-request CSV;
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
  | `requests.jsonl` | One row per request: send time, every token's arrival time, `usage`, and errors or validation failures |
  | `events.jsonl` | Level start, N-th request sent, first wave finished, sending stopped (and why), drain finished |
  | `warmup.jsonl` | The discarded warmup requests, kept to compare cold and warm behavior |
  | `server.log` | The vLLM log, copied at the end. It includes the server's periodic running / waiting / KV-usage lines |
- **Repeats.** TODO: count, and how spread is reported.
- **Sweep grid.** TODO: concurrency levels, prompt lengths, output
  length, and the reason for each value.

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
- TODO

## Reproducing

```bash
pip install -r requirements.txt
```

**Against the mock server** (no GPU; checks the harness itself):

```bash
uvicorn mock.server:app --port 8001          # TTFT_DELAY / ITL_DELAY env vars set its timing
python harness/loadgen.py --concurrencies 1 256 --prompt-lens 128
```

Against the mock, TTFT should match `TTFT_DELAY` (0.2 s by default) at both
N=1 and N=256. If it rises with N, the client is the bottleneck.

**Against vLLM:** TODO (the Colab launch command with every server flag, and
the harness invocation with `--server-cmd`, `--server-log`, and
`--results-dir`).

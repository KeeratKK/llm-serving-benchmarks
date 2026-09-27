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
- **Steady-state window.** Only requests whose whole lifetime falls inside
  the window with N requests in flight are counted. Requests from the
  ramp-up and the drain at the end are discarded.
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

TODO

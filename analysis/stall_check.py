"""Mock-only diagnostic: is the mock stalling, or is the harness?

Compares the mock's per-token send time (t_sent) against the harness's arrival time,
both on CLOCK_MONOTONIC. Needs a run made against the mock (vLLM rows have no sent_timestamps).

    python analysis/stall_check.py results/<run_id> [--concurrency 256]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

BUCKET_MS = 5

def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def pct(xs, ps=(10, 50, 99)):
    return "  ".join(f"p{p}={np.percentile(xs, p):7.1f}" for p in ps)

def empty_bucket_runs(times, start, end):
    """Longest run of empty BUCKET_MS buckets, plus the fraction of buckets that are empty."""
    n = int((end - start) * 1000 // BUCKET_MS)
    counts = Counter(int((t - start) * 1000 // BUCKET_MS) for t in times if start <= t < end)
    longest = cur = empty = 0
    for i in range(n):
        if counts.get(i, 0) == 0:
            cur += 1
            empty += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest * BUCKET_MS, empty / max(n, 1), sum(counts.values()) / max(n, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=256)
    args = parser.parse_args()

    events = load_jsonl(args.run_dir / "events.jsonl")
    requests = load_jsonl(args.run_dir / "requests.jsonl")

    w = {e["event"]: e["timestamp"] for e in events if e.get("concurrency") == args.concurrency}
    start, end = w["first_wave_finished"], w["sending_stopped"]

    rows = [r for r in requests
            if r["concurrency"] == args.concurrency and not r["error_type"]
            and r["start"] >= start and r["end"] <= end and r.get("sent_timestamps")]
    if not rows:
        raise SystemExit("No in-window rows with sent_timestamps (was this run against the mock?)")

    # Stall buckets use every request's tokens, not just in-window ones: in-window requests all start
    # after the window opens, so the first TTFT_DELAY of the window would always look empty
    all_rows = [r for r in requests
                if r["concurrency"] == args.concurrency and not r["error_type"] and r.get("sent_timestamps")]
    sent = [t for r in all_rows for t in r["sent_timestamps"]]
    recv = [t for r in all_rows for t in r["token_timestamps"]]
    delay_ms = [(rv - s) * 1000 for r in rows for s, rv in zip(r["sent_timestamps"], r["token_timestamps"])]
    sent_gaps = [(b - a) * 1000 for r in rows for a, b in zip(r["sent_timestamps"], r["sent_timestamps"][1:])]
    recv_gaps = [(b - a) * 1000 for r in rows for a, b in zip(r["token_timestamps"], r["token_timestamps"][1:])]

    print(f"N={args.concurrency}  in-window requests={len(rows)}  tokens={len(delay_ms)}  window={end - start:.1f}s\n")

    print("Per-token gaps (ms) — what each side sees")
    print(f"  mock send gaps       {pct(sent_gaps)}")
    print(f"  harness arrival gaps {pct(recv_gaps)}")
    print(f"  arrival - send delay {pct(delay_ms)}  max={max(delay_ms):.1f}\n")

    print(f"Whole-process stalls ({BUCKET_MS} ms buckets across all streams; a steady stream has ~none empty)")
    for name, times in (("mock sends", sent), ("harness arrivals", recv)):
        longest, frac_empty, mean = empty_bucket_runs(times, start, end)
        print(f"  {name:17s} longest empty run={longest:4d} ms  empty buckets={frac_empty:6.1%}  mean/bucket={mean:.1f}")


if __name__ == "__main__":
    main()

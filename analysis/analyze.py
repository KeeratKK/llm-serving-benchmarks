import argparse
import csv
import gzip
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class RunKey:
    concurrency: int
    prompt_len: int
    repeat: int

@dataclass
class RunGroup:
    meta: dict[str, Any]
    events: list[dict[str, Any]]
    requests: list[dict[str, Any]]

@dataclass
class Window:
    start: float
    end: float
    stop_reason: str

@dataclass
class RequestMetrics:
    request_idx: int
    ttft: float
    tpot: float
    e2e: float
    itl_gaps: list[float]
    completion_tokens: int

@dataclass
class MetricSummary:
    key: RunKey
    window: Window | None

    error_count: int
    validation_failures: int

    latency_count: int
    throughput_count: int

    # All latencies in ms
    ttft: dict[str, float | int]
    tpot: dict[str, float | int]
    e2e: dict[str, float | int]
    itl: dict[str, float | int]

    requests_rate: float
    output_tokens_rate: float
    # Little's law: request rate x E2E p50 should come out close to N
    littles_law_n: float

def get_window(events: list[dict[str, Any]]):

    first_wave = next(
        (event for event in events if event.get("event") == "first_wave_finished"), None
    )

    stop_sending = next(
        (event for event in events if event.get("event") == "sending_stopped"), None
    )

    # A level that aborted during the first wave never has a steady-state window
    if first_wave is None or stop_sending is None:
        return None

    return Window(start=first_wave["timestamp"], end=stop_sending["timestamp"], stop_reason=stop_sending["reason"])

def is_successful(request: dict[str, Any]):

    return ( request["error_type"] is None and not request["validation_errors"] )

def get_latency_set(requests: list[dict[str, Any]], window: Window):

    return [
        request for request in requests if is_successful(request)
        and request["start"] >= window.start
        and request["end"] <= window.end
    ]

def get_request_metrics(request: dict[str, Any]):

    timestamps = request["token_timestamps"]

    if len(timestamps) < 2:
        raise ValueError(f"Request {request['request_idx']} has less than 2 token timestamps")

    ttft = timestamps[0] - request["start"]
    e2e = request["end"] - request["start"]

    tpot = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)

    itl_gaps = [ahead - prev for prev, ahead in zip(timestamps, islice(timestamps, 1, None))]

    return RequestMetrics(
        request_idx=request["request_idx"],
        ttft=ttft,
        tpot=tpot,
        e2e=e2e,
        itl_gaps=itl_gaps,
        completion_tokens=request["completion_tokens"]
    )

def get_percentiles(values: list[float]):

    if not values:
        return {
            "n": 0,
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }

    # Timestamps are in seconds, every reported latency is in ms
    values_ms = np.asarray(values) * 1000

    return {
        "n": len(values),
        "p50": float(np.percentile(values_ms, 50)),
        "p90": float(np.percentile(values_ms, 90)),
        "p99": float(np.percentile(values_ms, 99)),
        "mean": float(np.mean(values_ms)),
        "std": float(np.std(values_ms))
    }

def summarize_latency(requests: list[dict[str, Any]]):

    metrics = [ get_request_metrics(request) for request in requests ]

    ttft = get_percentiles([ metric.ttft for metric in metrics ])
    tpot = get_percentiles([ metric.tpot for metric in metrics ])
    e2e = get_percentiles([ metric.e2e for metric in metrics ])

    itl_gaps = [ itl_gap for metric in metrics for itl_gap in metric.itl_gaps]
    itl = get_percentiles(itl_gaps)

    return ttft, tpot, e2e, itl

def successful_throughput_requests(requests: list[dict[str, Any]], window: Window):

    return [ request for request in requests if is_successful(request) and window.start <= request["end"] <= window.end ]

def get_throughput(requests: list[dict[str, Any]], window: Window):

    successful_reqs = successful_throughput_requests(requests, window)
    duration = window.end - window.start

    request_rate = len(successful_reqs) / duration

    output_tokens = sum( request["completion_tokens"] for request in successful_reqs )
    output_rate = output_tokens / duration

    return request_rate, output_rate, len(successful_reqs)

def summarize_group(key: RunKey, group: RunGroup):

    window = get_window(group.events)

    error_count =  sum( request.get("error_type") is not None for request in group.requests )
    validation_failures = sum( bool(request.get("validation_errors")) for request in group.requests )

    if window is None:
        empty = get_percentiles([])
        return MetricSummary(
            key=key, 
            window=None, 
            error_count=error_count, 
            validation_failures=validation_failures,
            latency_count=0, 
            throughput_count=0, 
            ttft=empty, 
            tpot=empty, 
            e2e=empty, 
            itl=empty,
            requests_rate=float("nan"), 
            output_tokens_rate=float("nan"), 
            littles_law_n=float("nan")
        )

    latency_requests = get_latency_set(group.requests, window)
    ttft, tpot, e2e, itl = summarize_latency(latency_requests)

    request_rate, output_token_rate, throughput_count = get_throughput(group.requests, window)

    littles_law_n = request_rate * e2e["p50"] / 1000

    return MetricSummary(
        key=key,
        window=window,
        error_count=error_count,
        validation_failures=validation_failures,
        latency_count=len(latency_requests),
        throughput_count=throughput_count,
        ttft=ttft,
        tpot=tpot,
        e2e=e2e,
        itl=itl,
        requests_rate=request_rate,
        output_tokens_rate=output_token_rate,
        littles_law_n=littles_law_n
    )

def get_env(meta: dict[str, Any]):

    run = meta.get("run", {})
    git = run.get("git", {})
    args = run.get("args", {})
    server = meta.get("server_environment", {})
    client = meta.get("client_environment", {})

    return {
        "run_id": run.get("run_id"),
        "commit": git.get("commit_hash"),
        "uncommitted_changes": git.get("uncommitted_changes"),
        "model": args.get("model"),
        "dtype": args.get("dtype"),
        "output_len": args.get("output_len"),
        "gpu": server.get("gpu"),
        "gpu_count": server.get("gpu_count"),
        "driver_version": server.get("driver_version"),
        "cuda_version": server.get("cuda_version"),
        "torch_version": server.get("torch_version"),
        "vllm_version": server.get("vllm_version"),
        "attention_backend": server.get("attention_backend"),
        "python_version": client.get("python_version"),
    }

def summary_to_row(summary: MetricSummary, meta: dict[str, Any]):

    window = summary.window

    row = {
        "concurrency": summary.key.concurrency,
        "prompt_len": summary.key.prompt_len,
        "repeat": summary.key.repeat,

        "window_start": window.start if window else None,
        "window_end": window.end if window else None,
        "window_duration_s": (window.end - window.start) if window else None,
        "stop_reason": window.stop_reason if window else "no_window",

        "error_count": summary.error_count,
        "validation_failure_count": summary.validation_failures,

        "latency_n": summary.latency_count,
        "throughput_n": summary.throughput_count,

        "request_per_sec": summary.requests_rate,
        "output_tokens_per_sec": summary.output_tokens_rate,
        "littles_law_n": summary.littles_law_n,
    }

    for name, stats in (
        ("ttft", summary.ttft),
        ("tpot", summary.tpot),
        ("e2e", summary.e2e),
        ("itl", summary.itl),
    ):
        for stat_name, value in stats.items():
            suffix = "" if stat_name == "n" else "_ms"
            row[f"{name}_{stat_name}{suffix}"] = value

    row.update(get_env(meta))

    return row

def write_summary(summaries: list[MetricSummary], meta: dict[str, Any], path: Path):

    rows = [ summary_to_row(summary, meta) for summary in summaries ]

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def print_summaries(summaries: list[MetricSummary]):

    header = (f"{'N':>5} {'P':>5} {'rep':>3} {'stop':>5} {'n':>5} {'req/s':>8} {'tok/s':>9} "
              f"{'TTFT p50':>9} {'p99':>8} {'TPOT p50':>9} {'ITL p50':>8} {'p99':>8} {'E2E p50':>9} {'Little N':>9}  errs")
    print(header + "\n" + "-" * len(header))

    for s in summaries:
        k = s.key
        if s.window is None:
            print(f"{k.concurrency:>5} {k.prompt_len:>5} {k.repeat:>3}  !! no steady-state window "
                  f"(level ended before the first wave finished)  errors={s.error_count}")
            continue

        stop = s.window.stop_reason
        print(f"{k.concurrency:>5} {k.prompt_len:>5} {k.repeat:>3} {stop[:5]:>5} {s.latency_count:>5} "
              f"{s.requests_rate:>8.2f} {s.output_tokens_rate:>9.0f} "
              f"{s.ttft['p50']:>9.1f} {s.ttft['p99']:>8.1f} {s.tpot['p50']:>9.2f} "
              f"{s.itl['p50']:>8.2f} {s.itl['p99']:>8.1f} {s.e2e['p50']:>9.0f} {s.littles_law_n:>9.1f}  "
              f"{s.error_count}+{s.validation_failures}")

        if stop != "K":
            print(f"      !! N={k.concurrency} P={k.prompt_len} repeat={k.repeat} stopped on {stop}, not K")

    print("\nLatencies in ms. errs = errors + validation failures. Little N = req/s x E2E p50, should be close to N.")

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def load_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def load_run(dir: str | Path):
    dir = Path(dir)

    meta = load_json(dir / "meta.json")
    events = load_jsonl(dir / "events.jsonl")

    requests_path = dir / "requests.jsonl"

    # Raw jsonl is gitignored, so after a fresh clone only the .gz exists
    if requests_path.exists():
        requests = load_jsonl(requests_path)
    else:
        requests = load_jsonl_gz(dir / "requests.jsonl.gz")

    return meta, events, requests

def get_key(item: dict[str, Any]):
    return RunKey(
        concurrency=int(item["concurrency"]),
        prompt_len=int(item["prompt_len"]),
        repeat=int(item["repeat"])
    )

def group_run(meta: dict[str, Any], events: list[dict[str, Any]], requests: list[dict[str, Any]]):

    groups: dict[RunKey, RunGroup] = defaultdict( lambda: RunGroup(meta=meta, events=[], requests=[]))

    for event in events:
        # These are run-level events we dont group
        if event["event"].startswith("warmup") or event["event"].startswith("sweep"):
            continue
        groups[get_key(event)].events.append(event)

    for request in requests:
        groups[get_key(request)].requests.append(request)

    return dict(groups)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)

    args = parser.parse_args()

    meta, events, requests = load_run(args.run_dir)
    groups = group_run(meta, events, requests)

    summaries = [ summarize_group(key, group) for key, group in sorted(groups.items(), key=lambda kv: (kv[0].repeat, kv[0].prompt_len, kv[0].concurrency)) ]

    print_summaries(summaries)

    summary_path = args.run_dir / "summary.csv"
    write_summary(summaries=summaries, meta=meta, path=summary_path)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

import asyncio
import aiohttp
import argparse
import json
import time
import sys
import numpy as np
import platform
import re
import subprocess
import shutil
import urllib.request
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import TextIO, Any
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

BASE_SEED = 42

# Defaults for the CLI args
CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PROMPT_LENGTHS = [128, 2048] # Eventually: [128, 512, 1024, 2048]
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
OUTPUT_LEN = 256

# Extra connections over max N so the pool never caps concurrency
CONNECTION_MARGIN = 24

# Run-wide: set once in main, fixed for the whole run, shared by every level
@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    meta_path: Path

    requests_file: TextIO
    events_file: TextIO
    warmup_file: TextIO

    client: aiohttp.ClientSession

    model: str
    output_len: int
    temperature: float
    stagger_interval: float

    # Base values; K and M get scaled per level in run_concurrency_level
    k_min: int
    level_timeout: int
    max_consecutive_errors: int

# Per-level: created fresh for each (repeat, prompt_len, concurrency) and mutated by the workers
@dataclass
class RunState:
    concurrency: int
    prompt_len: int
    repeat: int
    k: int
    deadline: float
    max_consecutive_errors: int

    # This is how workers get their request idx
    next_request_idx: int

    # Helps us track the first wave 0..N-1 request idxs
    first_wave_num: int = 0
    first_wave_done_time: float | None = None

    # For knowing when we reach K
    window_successes: int = 0

    # Keep track of consecutive errors
    consecutive_errors: int = 0

    # Keeps track of sending state and if we should still be sending requests
    sending_stopped: bool = False
    stopped_reason: str | None = None

async def send_single_request(client: aiohttp.ClientSession, model: str, temperature: float, prompt_len: int, output_len: int, concurrency: int, repeat: int, request_idx: int):

    token_timestamps = []
    # Only the mock sends t_sent (a diagnostic for client-side delay); stays empty against vLLM
    sent_timestamps = []
    validation_errors = []

    finish_reason = None
    prompt_tokens = None
    completion_tokens = None
    error_type = None
    error_msg = None

    # Makes it so that we can ensure that no matter which worker sends this request, the configuration gets the same prompt
    # This way it doesnt matter which worker asks first and its not based on timing
    rand_gen = np.random.default_rng([BASE_SEED, concurrency, prompt_len, repeat, request_idx])

    # Tokenizer vocab size is 128_256 but last 256 are special tokens and we are simulating base vocabulary
    prompt = rand_gen.integers(0, 128_000, size=prompt_len).tolist()

    wall_start = time.time()  # for lining rows up against the nvidia-smi log
    start = time.perf_counter()

    request_body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": temperature,
        "seed": BASE_SEED,
        "stream_options": {"include_usage": True},
        "stream": True,
        "min_tokens": output_len,
        "ignore_eos": True
    }

    try:

        # No leading slash: aiohttp joins it onto the session's base_url (.../v1/), a leading slash would drop the /v1
        async with client.post("completions", json=request_body) as response:

            if not 200 <= response.status < 300:
                body = await response.read()
                error_msg = body.decode("utf-8", errors="replace")

                response.raise_for_status()

            # Split the byte stream into lines
            async for line in response.content:        

                line = line.decode("utf-8").strip()

                # We only care about data: lines
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue

                data = line[len("data:"):].strip()

                # Only for OpenAI, not part of SSE. Keep reading to EOF instead of breaking:
                # aiohttp closes a connection whose body wasn't fully read instead of pooling it,
                # and the reconnect would land in the next request's TTFT
                if data == "[DONE]":
                    continue

                now = time.perf_counter()

                # Loads it into a dictionary (which is cheap since we are no longer constructing a full python object)
                chunk = json.loads(data)

                if "error" in chunk:
                    error = chunk["error"]

                    if isinstance(error, dict):
                        error_type = error.get("type", "StreamError")
                        error_msg = error.get("message", str(error))
                    else:
                        error_type = "StreamError"
                        error_msg = str(error)

                    break

                choices = chunk.get("choices", [])

                # This is a token, meanwhile usage chunk has choices=[]
                if choices:
                    token_timestamps.append(now)

                    if "t_sent" in chunk:
                        sent_timestamps.append(chunk["t_sent"])

                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0].get("finish_reason")

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")

                
    except Exception as e:
        if error_msg is None:
            error_msg = str(e)
        error_type = type(e).__name__

    end = time.perf_counter()

    # Only validate streams that have finished since an errored one is already flagged by error_type
    if error_type is None:
        if finish_reason != "length":
            validation_errors.append(f"finish_reason={finish_reason!r}, expected 'length'")

        if prompt_tokens != prompt_len:
            validation_errors.append(f"prompt_tokens={prompt_tokens}, expected {prompt_len}")

        if completion_tokens != output_len:
            validation_errors.append(f"completion_tokens={completion_tokens}, expected {output_len}")

        if len(token_timestamps) != output_len:
            validation_errors.append(f"token_chunks={len(token_timestamps)}, expected {output_len}")

    return {
        "request_idx": request_idx,
        "prompt_len": prompt_len,
        "output_len": output_len,
        "concurrency": concurrency,
        "repeat": repeat,
        "wall_start": wall_start,
        "start": start,
        "end": end,
        "token_timestamps": token_timestamps,
        "sent_timestamps": sent_timestamps or None,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error_type": error_type,
        "error_msg": error_msg,
        "validation_errors": validation_errors
    }

async def worker(worker_id: int, ctx: RunContext, state: RunState):

    # Stagger the appraoch of the first N requests
    await asyncio.sleep(worker_id * ctx.stagger_interval)

    request_idx = worker_id

    # Want to log the event when the Nth request is sent
    if request_idx == state.concurrency - 1:
        write_event(
            ctx,
            {
                "event": "nth_request_sent",
                "timestamp": time.perf_counter(),
                "request_idx": request_idx
            },
            state
        )

    result = await send_single_request(
        client=ctx.client,
        model=ctx.model,
        temperature=ctx.temperature,
        prompt_len=state.prompt_len,
        output_len=ctx.output_len,
        concurrency=state.concurrency,
        repeat=state.repeat,
        request_idx=request_idx
    )

    write_request(ctx.requests_file, result)

    request_completion(ctx, state, result)

    while not state.sending_stopped:

        next_request_idx = get_next_request(ctx, state)

        if next_request_idx is None:
            return

        result = await send_single_request(
            client=ctx.client,
            model=ctx.model,
            temperature=ctx.temperature,
            prompt_len=state.prompt_len,
            output_len=ctx.output_len,
            concurrency=state.concurrency,
            repeat=state.repeat,
            request_idx=next_request_idx
        )

        write_request(ctx.requests_file, result)

        request_completion(ctx, state, result)

async def run_concurrency_level(ctx: RunContext, prompt_len: int, concurrency: int, repeat: int):

    level_start = time.perf_counter()

    state = RunState(
        concurrency=concurrency,
        prompt_len=prompt_len,
        repeat=repeat,
        k=max(ctx.k_min, 5 * concurrency),
        deadline=level_start + ctx.level_timeout,
        # At high N, overload timeouts arrive in bursts; that's data, not a dead server
        max_consecutive_errors=max(ctx.max_consecutive_errors, concurrency),
        next_request_idx=concurrency
    )

    write_event(
        ctx,
        {
            "event": "concurrency_level_start",
            "timestamp": level_start,
            "K": state.k,
            "M": state.max_consecutive_errors,
            "deadline": state.deadline
        },
        state
    )

    workers = [ asyncio.create_task(
        worker(
            worker_id=idx,
            ctx=ctx,
            state=state
        )
    ) for idx in range(concurrency) ]

    await asyncio.gather(*workers)

    write_event(
        ctx,
        {
            "event": "drain_finished",
            "timestamp": time.perf_counter()
        },
        state
    )

    return state.stopped_reason

def get_next_request(ctx: RunContext, state: RunState):

    if state.sending_stopped:
        return None

    cur_time = time.perf_counter()

    if cur_time >= state.deadline:
        state.sending_stopped = True
        state.stopped_reason = "T"

        write_event(
            ctx,
            {
                "event": "sending_stopped",
                "timestamp": cur_time,
                "reason": "T"
            },
            state
        )

        return None

    new_idx = state.next_request_idx
    state.next_request_idx += 1

    return new_idx

def request_completion(ctx: RunContext, state: RunState, result: dict):

    request_idx = result["request_idx"]

    if request_idx < state.concurrency:

        state.first_wave_num += 1

        if state.first_wave_num == state.concurrency:

            state.first_wave_done_time = time.perf_counter()

            write_event(
                ctx,
                {
                    "event": "first_wave_finished",
                    "timestamp": state.first_wave_done_time
                },
                state
            )

    if result["error_type"] is not None or result["validation_errors"]:

        state.consecutive_errors += 1

        if state.consecutive_errors >= state.max_consecutive_errors:

            if not state.sending_stopped:

                state.sending_stopped = True
                state.stopped_reason = "max_consecutive_errors"

                write_event(
                    ctx,
                    {
                        "event": "sending_stopped",
                        "timestamp": time.perf_counter(),
                        "reason": "max_consecutive_errors"
                    },
                    state
                )

    if state.first_wave_done_time and result["start"] > state.first_wave_done_time and not result["validation_errors"] and result["error_type"] is None:

        state.window_successes += 1

        if state.window_successes == state.k:

            if not state.sending_stopped:

                state.sending_stopped = True
                state.stopped_reason = "K"
        
                write_event(
                    ctx,
                    {
                        "event": "sending_stopped",
                        "timestamp": time.perf_counter(),
                        "reason": "K"
                    },
                    state
                )

def write_event(ctx: RunContext, event: dict, state: RunState | None = None):
    # Level events carry their level so analysis doesn't depend on line order
    # Run-level events (warmup, sweep aborted) have no state and go untagged
    if state is not None:
        event = {
            "concurrency": state.concurrency,
            "prompt_len": state.prompt_len,
            "repeat": state.repeat,
            **event
        }
    ctx.events_file.write(json.dumps(event) + "\n")
    ctx.events_file.flush()

def write_request(requests_file: TextIO, request: dict):
    requests_file.write(json.dumps(request) + "\n")
    requests_file.flush()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001/v1",
        help="OpenAI-compatible server base url"
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Model name the server was launched with"
    )
    parser.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        default=CONCURRENCIES,
        help="Concurrency levels N to sweep, e.g. --concurrencies 1 256"
    )
    parser.add_argument(
        "--prompt-lens",
        nargs="+",
        type=int,
        default=PROMPT_LENGTHS,
        help="Prompt lengths to sweep, e.g. --prompt-lens 128 2048"
    )
    parser.add_argument(
        "--output-len",
        default=OUTPUT_LEN,
        type=int,
        help="Fixed number of output tokens per request"
    )
    parser.add_argument(
        "--temperature",
        default=0.0,
        type=float,
        help="Sampling temperature, set explicitly so it's recorded"
    )
    parser.add_argument(
        "--client-timeout",
        default=180,
        type=int,
        help="Client Timeout"
    )
    parser.add_argument(
        "--K",
        default=150,
        type=int,
        help="Number of requests to be completed in the steady state window"
    )
    parser.add_argument(
        "--T",
        default=20*60,
        type=int,
        help="Max seconds per concurrency level, measured from the level's start. A safety cap; K is the normal stop"
    )
    parser.add_argument(
        "--repeat",
        default=1,
        type=int,
        help="Number of times to repeat each experiment"
    )
    parser.add_argument(
        "--stagger-interval",
        default=0.03,
        type=float,
        help="Interval for which we stagger the initial N requests"
    )
    parser.add_argument(
        "--M",
        default=10,
        type=int,
        help="Max number of consecutive errors before aborting the entire sweep"
    )
    parser.add_argument(
        "--num-requests",
        default=100,
        type=int,
        help="Number of throwaway requests to warmup the server"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Parent directory; each run writes to <results-dir>/<run_id>/"
    )
    parser.add_argument(
        "--server-cmd",
        type=str,
        default=None,
        help="Command used to launch the server"
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        default=None,
        help="VLLM server log"
    )

    return parser.parse_args()

async def fetch_json(url: str):
    def _fetch():
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.load(response)
        except Exception:
            return None

    return await asyncio.to_thread(_fetch)

def server_root(base_url: str):

    url = base_url.rstrip("/")

    if url.endswith("/v1"):
        url = url[:-3]

    return url.rstrip("/")

async def get_vllm_environment(base_url: str):

    root = server_root(base_url)

    vllm_version = None
    max_model_len = None
    served_models = None

    version_data, models_data = await asyncio.gather(
        fetch_json(f"{root}/version"),
        fetch_json(f"{root}/v1/models")
    )

    if version_data is not None:
        vllm_version = version_data.get("version")

    if models_data is not None:

        served_models = models_data.get("data", [])

        if served_models:
            max_model_len = served_models[0].get("max_model_len")

    return {
        "version": vllm_version,
        "served_models": served_models,
        "max_model_len": max_model_len
    }

def get_vllm_log(log_path: Path):

    attention_backend = None
    kv_cache_tokens = None
    max_concurrency = None
    effective_config = None

    if log_path is None or not log_path.exists():
        return {
            "attention_backend": attention_backend,
            "kv_cache_tokens": kv_cache_tokens,
            "max_concurrency": max_concurrency,
            "effective_config": effective_config
        }

    text = log_path.read_text(encoding="utf-8", errors="replace")

    match = re.search(r"Using\s+(\S+)\s+attention backend", text)
    attention_backend = match.group(1) if match else None

    match = re.search(
        r"GPU KV cache size:\s*([\d,]+)\s*tokens,\s*"
        r"Maximum concurrency for\s*([\d,]+)\s*"
        r"tokens per request:\s*([\d.]+)x",
        text,
    )

    if match:
        kv_cache_tokens = int(match.group(1).replace(",", ""))
        max_concurrency = float(match.group(3))

    match = re.search(
        r"^.*Initializing a V1 LLM engine.*$",
        text,
        re.MULTILINE
    )

    effective_config = match.group(0) if match else None

    return {
        "attention_backend": attention_backend,
        "kv_cache_tokens": kv_cache_tokens,
        "max_concurrency": max_concurrency,
        "effective_config": effective_config
    }

def get_package_version(package: str):
    try:
        return version(package)
    except PackageNotFoundError:
        return None

def get_git_info():

    cwd = Path(__file__).resolve().parent

    try:
        commit_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        return {
            "commit_hash": commit_hash,
            "uncommitted_changes": bool(git_status)
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "commit_hash": None,
            "uncommitted_changes": None
        }

def get_nvidia_smi_info():

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []

        for line in result.stdout.splitlines():

            if not line.strip():
                continue

            name, driver_version = [ x.strip() for x in line.split(",", maxsplit=1) ]

            gpus.append({
                "name": name,
                "driver_version": driver_version
            })

        cuda = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True
        ).stdout

        cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", cuda)
        cuda_version = (cuda_match.group(1) if cuda_match else None)

        return {
            "available": True,
            "count": len(gpus),
            "gpus": gpus,
            "driver_version": (gpus[0]["driver_version"] if gpus else None),
            "cuda_version": cuda_version
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "available": False,
            "count": None,
            "gpus": None,
            "driver_version": None,
            "cuda_version": None
        }

async def write_metadata(ctx: RunContext, args):

    git_info = get_git_info()
    gpu_info = get_nvidia_smi_info()
    vllm_info = await get_vllm_environment(args.base_url)
    vllm_log = get_vllm_log(args.server_log)

    # Read back to back. Every event/request timestamp is perf_counter, which only means something
    # inside this process; wall = wall_time + (perf - perf_counter) puts them on the nvidia-smi / server log clock
    clock_reference = {
        "wall_time": time.time(),
        "perf_counter": time.perf_counter()
    }

    meta = {
        "run": {
            "run_id": ctx.run_id,
            "start_time": datetime.now().astimezone().isoformat(),
            "clock_reference": clock_reference,
            "git": git_info,
            "argv": sys.argv,
            "args": vars(args)
        },
        "client_environment": {
            "python_version": platform.python_version(),
            "packages": {
                "aiohttp": get_package_version("aiohttp"),
                "numpy": get_package_version("numpy")
            }
        },
        "server_environment": {
            "torch_version": get_package_version("torch"),
            "vllm_version": vllm_info["version"],
            "gpu": gpu_info["gpus"],
            "gpu_count": gpu_info["count"],
            "cuda_version": gpu_info["cuda_version"],
            "driver_version": gpu_info["driver_version"],
            "server_cmd": args.server_cmd,
            "attention_backend": vllm_log["attention_backend"],
            "kv_cache_tokens": vllm_log["kv_cache_tokens"],
            "max_concurrency": vllm_log["max_concurrency"],
            "max_model_len": vllm_info["max_model_len"],
            "served_models": vllm_info["served_models"],
        },
        "config": {
            "effective_config": vllm_log["effective_config"]
        }
    }

    with ctx.meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, default=str)
        file.write("\n")

async def warmup_server(ctx: RunContext, num_requests: int, prompt_len: int, concurrency: int, repeat: int):

    write_event(
        ctx,
        {
            "event": "warmup_start",
            "timestamp": time.perf_counter()
        }
    )

    # Throwaway requests just to warm up the server so we dont factor in warmup costs in TTFT
    warmup_tasks = [ asyncio.create_task(
        send_single_request(
            client=ctx.client,
            model=ctx.model,
            temperature=ctx.temperature,
            prompt_len=prompt_len,
            output_len=ctx.output_len,
            concurrency=concurrency,
            repeat=repeat,
            request_idx=idx
        )
    ) for idx in range(num_requests) ]

    try:
        for task in asyncio.as_completed(warmup_tasks):
            result = await task
            write_request(ctx.warmup_file, result)
    finally:
        write_event(
            ctx,
            {
                "event": "warmup_finished",
                "timestamp": time.perf_counter()
            }
        )

async def main():
    args = parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Sized from the largest N in this sweep, and reused across every level
    max_connections = max(args.concurrencies) + CONNECTION_MARGIN

    # aiohttp's default connector caps at 100 connections, which would silently cap N
    client = aiohttp.ClientSession(
        base_url=args.base_url.rstrip("/") + "/",  # aiohttp requires the trailing slash when base_url has a path
        connector=aiohttp.TCPConnector(
            limit=max_connections,
            keepalive_timeout=60
        ),
        # total covers the whole request including the stream, so this is a cap on E2E
        timeout=aiohttp.ClientTimeout(total=args.client_timeout)
    )

    ctx = RunContext(
        run_id=run_id,
        run_dir=run_dir,
        meta_path=run_dir / "meta.json",
        requests_file=open(run_dir / "requests.jsonl", "w"),
        events_file=open(run_dir / "events.jsonl", "w"),
        warmup_file=open(run_dir / "warmup.jsonl", "w"),
        client=client,
        model=args.model,
        output_len=args.output_len,
        temperature=args.temperature,
        stagger_interval=args.stagger_interval,
        k_min=args.K,
        level_timeout=args.T,
        max_consecutive_errors=args.M
    )

    try:
        await write_metadata(ctx, args)

        await warmup_server(ctx, args.num_requests, max(args.prompt_lens), 0, args.repeat)

        for repeat, prompt_len, concurrency in product(range(args.repeat), args.prompt_lens, args.concurrencies):
            stopped_reason = await run_concurrency_level(
                ctx,
                prompt_len=prompt_len,
                concurrency=concurrency,
                repeat=repeat
            )

            if stopped_reason == "max_consecutive_errors":
                break
    finally:

        if args.server_log and args.server_log.exists():
            shutil.copy(args.server_log, run_dir / "server.log")

        ctx.requests_file.close()
        ctx.events_file.close()
        ctx.warmup_file.close()

        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
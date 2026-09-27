import asyncio
import argparse
import httpx2
import time
import numpy as np
from openai import AsyncOpenAI

BASE_SEED = 42
CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PROMPT_LENGTHS = [128, 512, 1024, 2048]
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

http_client = httpx2.AsyncClient(
    limits=httpx2.Limits(
        max_connections=280, # 256 + some margin
        max_keepalive_connections=280,
        keepalive_expiry=60
    )
)

async def send_single_request(client: AsyncOpenAI, prompt_len: int, output_len: int, concurrency: int, repeat: int, request_idx: int):

    token_timestamps = []
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

    try:
        stream = await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=output_len,
            seed=BASE_SEED,
            stream_options={"include_usage": True},
            stream=True,
            extra_body={
                "min_tokens": output_len,
                "ignore_eos": True
            }
        )

        # 'Async with' gives the connection back to the pool even if we fail mid-stream
        async with stream:
            async for chunk in stream:

                now = time.perf_counter()

                # Any chunk with choices is a token and the usage chunk has none
                if chunk.choices:
                    token_timestamps.append(now)
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)

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
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error_type": error_type,
        "error_msg": error_msg,
        "validation_errors": validation_errors
    }

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001/v1",
        help="OpenAI-compatible server base url"
    )

    return parser.parse_args()

async def main():
    args = parse_args()

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key="dummy",
        max_retries=0,
        http_client=http_client,
        timeout=60
    )
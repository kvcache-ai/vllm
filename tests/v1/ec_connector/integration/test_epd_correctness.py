# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
EPD Correctness Test

Tests that EPD (Encoder-Prefill-Decode) disaggregation produces the same
outputs as a baseline single instance.

Usage:
    # Baseline mode (saves outputs):
    python test_epd_correctness.py \
        --service_url http://localhost:8000 \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --mode baseline \
        --baseline_file .vllm_epd_baseline.txt

    # Disagg mode (compares outputs):
    python test_epd_correctness.py \
        --service_url http://localhost:8000 \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --mode disagg \
        --baseline_file .vllm_epd_baseline.txt
"""

import argparse
import json
import os
import time

import openai
import requests

from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_url

MAX_OUTPUT_LEN = 256

# Sample prompts with multimodal content
image_1 = ImageAsset("stop_sign").pil_image.resize((1280, 720))
image_2 = ImageAsset("cherry_blossom").pil_image.resize((1280, 720))

image_local_path = f"{os.path.dirname(os.path.abspath(__file__))}/hato.jpg"

SAMPLE_PROMPTS_MM: list[dict] = [
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_url(image_1)},
                    },
                    {"type": "text", "text": "What's in this image?"},
                ],
            }
        ],
        "description": "Single image query",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_url(image_2)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{image_local_path}"},
                    },
                    {"type": "text", "text": "Describe these 2 images in detail."},
                ],
            }
        ],
        "description": "2 images with detailed query",
    },
]

# Text-only prompts for mixed testing
SAMPLE_PROMPTS_TEXT: list[dict] = [
    {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "description": "Simple text-only query",
    },
    {
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        "description": "Text-only explanation request",
    },
]

# Short prompts for Mooncake full E2E (1P+1D vs 1E+1P+1D must match with seed=0).
SAMPLE_PROMPTS_MOONCAKE_TEXT: list[dict] = [
    {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "description": "Simple text-only query",
    },
    {
        "messages": [{"role": "user", "content": "What is 1+1? Answer with one number."}],
        "description": "Short numeric reply",
    },
]


def check_vllm_server(url: str, timeout=5, retries=10) -> bool:
    """Check if the vLLM server is ready.

    Args:
        url: The URL to check (usually /health or /healthcheck endpoint)
        timeout: Timeout in seconds for each request
        retries: Number of retries if the server is not ready

    Returns:
        True if the server is ready, False otherwise
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                print(f"Server is ready at {url}")
                return True
            else:
                print(
                    f"Attempt {attempt + 1}/{retries}: Server returned "
                    f"status code {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{retries}: Error connecting: {e}")
        time.sleep(2)  # Wait before retrying
    return False


def check_mooncake_proxy_ready(
    base_url: str,
    model_name: str,
    timeout: int = 30,
    retries: int = 10,
) -> bool:
    """Mooncake PD proxy has no GET /health; probe POST /v1/chat/completions."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "seed": 42,
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code == 200:
                print(f"Mooncake proxy ready at {url}")
                return True
            print(
                f"Attempt {attempt + 1}/{retries}: POST {url} "
                f"status {response.status_code}"
            )
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{retries}: POST {url} error: {e}")
        time.sleep(2)
    return False


def run_chat_completion(
    base_url: str,
    model_name: str,
    messages: list,
    max_tokens: int = MAX_OUTPUT_LEN,
) -> str:
    """Run a chat completion request.

    Args:
        base_url: Base URL of the vLLM server
        model_name: Name of the model
        messages: Messages for chat completion
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text content
    """
    client = openai.OpenAI(api_key="EMPTY", base_url=base_url)

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=42,
    )

    return completion.choices[0].message.content


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="EPD correctness test - compare disagg vs baseline"
    )

    parser.add_argument(
        "--service_url",
        type=str,
        required=True,
        help="The vLLM service URL (e.g., http://localhost:8000)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "baseline_pd", "disagg"],
        help="Mode: baseline/baseline_pd (saves outputs) or disagg (compares outputs)",
    )

    parser.add_argument(
        "--baseline_file",
        type=str,
        default=".vllm_epd_baseline.txt",
        help="File to save/load baseline outputs",
    )

    parser.add_argument(
        "--use_mm_prompts",
        action="store_true",
        help="Use multimodal prompts (default: use text-only for quick testing)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max_tokens per request (default: MAX_OUTPUT_LEN)",
    )

    parser.add_argument(
        "--mooncake-full-e2e",
        action="store_true",
        help="Use short deterministic text prompts for Mooncake 1P+1D vs 1E+1P+1D",
    )

    args = parser.parse_args()
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = int(os.environ.get("VLLM_EPD_MAX_TOKENS", str(MAX_OUTPUT_LEN)))

    print(f"Service URL: {args.service_url}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {args.mode}")
    print(f"Output file: {args.baseline_file}")
    print(f"Use MM prompts: {args.use_mm_prompts}")

    # Determine health check endpoint
    if args.mode == "baseline":
        health_check_url = f"{args.service_url}/health"
    elif args.mode == "baseline_pd":
        # Nixl toy proxy: GET /healthcheck; Mooncake proxy: POST /v1/chat/completions
        health_check_url = None
        for suffix in ("/healthcheck", "/health", "/v1/models"):
            candidate = f"{args.service_url}{suffix}"
            if check_vllm_server(candidate, retries=3):
                health_check_url = candidate
                break
        if health_check_url is None and not check_mooncake_proxy_ready(
            args.service_url, args.model_name
        ):
            raise RuntimeError(
                f"vLLM PD proxy at {args.service_url} is not ready "
                "(tried GET /healthcheck,/health,/v1/models and POST /v1/chat/completions)"
            )
        if health_check_url is None:
            health_check_url = f"{args.service_url}/v1/chat/completions"
    else:
        # Disagg EPD proxy uses /health
        health_check_url = f"{args.service_url}/health"
        if not os.path.exists(args.baseline_file):
            raise ValueError(
                f"In disagg mode, the output file {args.baseline_file} from "
                "baseline does not exist. Run baseline mode first."
            )

    if args.mode != "baseline_pd" and health_check_url is not None:
        if not check_vllm_server(health_check_url):
            raise RuntimeError(f"vLLM server at {args.service_url} is not ready!")

    # Select prompts to use
    if args.use_mm_prompts:
        test_prompts = SAMPLE_PROMPTS_MM
        print("Using multimodal prompts")
    elif args.mooncake_full_e2e or os.environ.get("MOONCAKE_FULL_E2E") == "1":
        test_prompts = SAMPLE_PROMPTS_MOONCAKE_TEXT
        print("Using Mooncake full-E2E short text prompts")
    else:
        test_prompts = SAMPLE_PROMPTS_TEXT
        print("Using text-only prompts for quick testing")

    # Run completions
    service_url = f"{args.service_url}/v1"
    output_strs = {}

    for i, prompt_data in enumerate(test_prompts):
        print(
            f"\nRunning prompt {i + 1}/{len(test_prompts)}: "
            f"{prompt_data['description']}"
        )

        output_str = run_chat_completion(
            base_url=service_url,
            model_name=args.model_name,
            messages=prompt_data["messages"],
            max_tokens=max_tokens,
        )

        # Use description as key for comparison
        key = prompt_data["description"]
        output_strs[key] = output_str
        print(f"Output: {output_str}")

    if args.mode in ("baseline", "baseline_pd"):
        # Baseline mode: Save outputs
        print(f"\nSaving baseline outputs to {args.baseline_file}")
        try:
            with open(args.baseline_file, "w") as json_file:
                json.dump(output_strs, json_file, indent=4)
            print("✅ Baseline outputs saved successfully")
        except OSError as e:
            print(f"Error writing to file: {e}")
            raise
    else:
        # Disagg mode: Load and compare outputs
        print(f"\nLoading baseline outputs from {args.baseline_file}")
        baseline_outputs = None
        try:
            with open(args.baseline_file) as json_file:
                baseline_outputs = json.load(json_file)
        except OSError as e:
            print(f"Error reading from file: {e}")
            raise

        # Verify outputs match
        print("\nComparing disagg outputs with baseline...")
        assert isinstance(baseline_outputs, dict), "Baseline outputs should be a dict"
        assert len(baseline_outputs) == len(output_strs), (
            f"Length mismatch: baseline has {len(baseline_outputs)}, "
            f"disagg has {len(output_strs)}"
        )

        all_match = True
        for key, baseline_output in baseline_outputs.items():
            assert key in output_strs, f"{key} not in disagg outputs"

            disagg_output = output_strs[key]
            if baseline_output == disagg_output:
                print(f"✅ {key}: MATCH")
            else:
                print(f"❌ {key}: MISMATCH")
                print(f"  Baseline: {baseline_output}")
                print(f"  Disagg:   {disagg_output}")
                all_match = False

        assert all_match, "❌❌Disagg outputs do not match baseline!❌❌"
        if all_match:
            print("\n✅ All outputs match! Test PASSED")


if __name__ == "__main__":
    main()

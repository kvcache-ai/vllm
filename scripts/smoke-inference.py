#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal GPU inference smoke test (facebook/opt-125m)."""

from vllm import LLM, SamplingParams

MODEL = "facebook/opt-125m"
PROMPT = "Hello, my name is"


def main() -> None:
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        max_model_len=128,
        gpu_memory_utilization=0.5,
    )
    outputs = llm.generate([PROMPT], SamplingParams(max_tokens=32, temperature=0.0))
    text = outputs[0].outputs[0].text
    print(f"Prompt: {PROMPT!r}")
    print(f"Output: {text!r}")
    if not text.strip():
        raise SystemExit("Empty generation output")


if __name__ == "__main__":
    main()

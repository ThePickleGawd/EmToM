#!/usr/bin/env python3
"""Quick smoke test: send one benchmark-style prompt to every campaign model."""

import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf

MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "o3",
    "opus",
    "sonnet",
    "haiku",
    "kimi-k2.5",
    "deepseek-v3.2",
    "qwen3-vl-235b",
    "llama-4-maverick",
    "gemini-pro",
    "gemini-flash",
]

# Benchmark-style planning prompt
SYSTEM_MSG = "You are an expert at task planning."
USER_MSG = """You are agent_0, a robot in an apartment. Your task is to move the cup from the kitchen table to the bedroom table.

You can see: kitchen, kitchen_table_01, cup_01 (on kitchen_table_01), fridge_01 (closed).

Available actions: Navigate[room], Pick[object], Place[object, furniture], Open[furniture], Close[furniture], Wait[], Communicate[message]

What is your next action? Respond with exactly one action."""


def test_model(model_short: str) -> dict:
    """Test a single model using the same config as benchmark."""
    from emtom.examples.run_habitat_benchmark import (
        expand_model_name,
        resolve_model_spec,
    )

    spec = resolve_model_spec(model_short)
    model_id = expand_model_name(model_short)
    provider = spec["llm_provider"]

    # Build config matching the benchmark YAML
    if provider == "anthropic_claude":
        conf = OmegaConf.create({
            "llm": {"_target_": "habitat_llm.llm.AnthropicClaude", "_partial_": True},
            "verbose": True,
            "system_message": SYSTEM_MSG,
            "system_tag": "", "user_tag": "", "assistant_tag": "", "eot_tag": "",
            "use_image_input": False, "save_prompt_images": False,
            "keep_message_history": False,
            "generation_params": {
                "model": model_id,
                "max_tokens": 250,
                "temperature": 1,
                "top_p": 1,
                "stream": False,
                "stop": "Assigned!",
            },
        })
        from habitat_llm.llm import AnthropicClaude
        llm_cls = AnthropicClaude
    else:
        conf = OmegaConf.create({
            "llm": {"_target_": "habitat_llm.llm.OpenAIChat", "_partial_": True},
            "verbose": True,
            "system_message": SYSTEM_MSG,
            "system_tag": "", "user_tag": "", "assistant_tag": "", "eot_tag": "",
            "use_image_input": False, "save_prompt_images": False,
            "keep_message_history": False,
            "generation_params": {
                "model": model_id,
                "max_tokens": 250,
                "temperature": 1,
                "top_p": 1,
                "stream": False,
                "stop": "Assigned!",
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "request_timeout": 30,
            },
        })
        from habitat_llm.llm import OpenAIChat
        llm_cls = OpenAIChat

    start = time.time()
    try:
        llm = llm_cls(conf)
        response = llm.generate(USER_MSG, stop="Assigned!")
        elapsed = time.time() - start
        text = response.strip()[:200]
        return {"status": "OK", "elapsed": f"{elapsed:.1f}s", "response": text}
    except Exception as e:
        elapsed = time.time() - start
        return {"status": "FAIL", "elapsed": f"{elapsed:.1f}s", "error": str(e)[:200]}


def main():
    print(f"Testing {len(MODELS)} models...\n")
    results = {}
    for model in MODELS:
        print(f"  {model:20s} ... ", end="", flush=True)
        result = test_model(model)
        status = result["status"]
        elapsed = result["elapsed"]
        if status == "OK":
            print(f"OK ({elapsed}) — {result['response'][:80]}")
        else:
            print(f"FAIL ({elapsed}) — {result['error'][:80]}")
        results[model] = result

    print(f"\n{'='*60}")
    ok = sum(1 for r in results.values() if r["status"] == "OK")
    print(f"Results: {ok}/{len(MODELS)} passed")
    for model, r in results.items():
        icon = "OK" if r["status"] == "OK" else "XX"
        print(f"  [{icon}] {model:20s} {r['elapsed']}")


if __name__ == "__main__":
    main()

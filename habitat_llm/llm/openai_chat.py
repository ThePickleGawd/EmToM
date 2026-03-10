#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

# Suppress verbose httpx logs from OpenAI client
logging.getLogger("httpx").setLevel(logging.WARNING)

from habitat_llm.llm.base_llm import BaseLLM, Prompt

# Load .env file if it exists (for API keys)
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        # Fallback: manually parse .env if dotenv not installed
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def generate_message(multimodal_prompt, image_detail="auto"):
    # Converts the multimodal prompt to the OpenAI format.
    content = []
    for prompt_type, prompt_value in multimodal_prompt:
        if prompt_type == "text":
            message_item = {"type": "text", "text": prompt_value}
        else:
            message_item = {
                "type": "image_url",
                "image_url": {
                    "url": prompt_value,
                    "detail": image_detail,
                },
            }
        content.append(message_item)
    return {"role": "user", "content": content}


class OpenAIChat(BaseLLM):
    """
    LLM implementation using OpenAI's Chat API.
    Uses environment variable: OPENAI_API_KEY

    Supports model aliases for convenience:
        gpt5, gpt-5           -> gpt-5
        gpt5-mini, gpt-5-mini -> gpt-5-mini
        gpt5.1, gpt-5.1       -> gpt-5.1
        gpt5.2, gpt-5.2       -> gpt-5.2
    """

    # Model aliases mapping short names to full OpenAI model IDs
    MODEL_ALIASES: Dict[str, str] = {
        # GPT-5
        "gpt5": "gpt-5",
        # GPT-5 Mini
        "gpt5-mini": "gpt-5-mini",
        # GPT-5.1
        "gpt5.1": "gpt-5.1",
        # GPT-5.2
        "gpt5.2": "gpt-5.2",
        # Fireworks-hosted Kimi K2.5
        "kimi-k2.5": "accounts/fireworks/models/kimi-k2p5",
    }

    @classmethod
    def resolve_model_alias(cls, model: str) -> str:
        """Resolve a model alias to the full OpenAI model ID."""
        return cls.MODEL_ALIASES.get(model.lower(), model)

    @staticmethod
    def _is_fireworks_model(model: str) -> bool:
        normalized = (model or "").strip().lower()
        return (
            normalized.startswith("accounts/fireworks/models/")
            or normalized.startswith("kimi-k2.5")
        )

    def __init__(self, conf: DictConfig):
        """
        Initialize the chat model.
        :param conf: the configuration of the language model
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        model_name = self.resolve_model_alias(self.generation_params.model)
        if self._is_fireworks_model(model_name):
            api_key = (os.getenv("FIREWORKS_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("No FIREWORKS_API_KEY provided")
            base_url = (
                os.getenv("FIREWORKS_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.fireworks.ai/inference/v1"
            ).strip()
        else:
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("No OPENAI_API_KEY provided")
            base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._validate_conf()
        self.verbose = self.llm_conf.verbose
        self.verbose = True
        self.message_history: List[Dict] = []
        self.keep_message_history = self.llm_conf.keep_message_history

    def _validate_conf(self):
        if self.generation_params.stream:
            raise ValueError("Streaming not supported")

    # @retry(Timeout, tries=3)
    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: int = 40,
    ):
        """
        Generate a response autoregressively.
        :param prompt: A string with the input to the language model.
        :param image: Image input
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate.
        :param request_timeout: maximum time before timeout.
        :param generation_args: contains arguments like the grammar definition. We don't use this here
        """

        params = OmegaConf.to_object(self.generation_params)

        # Resolve model alias
        params["model"] = self.resolve_model_alias(params["model"])

        # Override stop if provided
        if stop is None and len(self.generation_params.stop) > 0:
            stop = self.generation_params.stop
        params["stop"] = stop

        # Override max_length if provided
        if max_length is not None:
            params["max_tokens"] = max_length

        messages = self.message_history.copy()
        # Add system message if no messages
        if len(messages) == 0:
            messages.append({"role": "system", "content": self.llm_conf.system_message})

        params["request_timeout"] = request_timeout
        if type(prompt) is str:
            # Add current message
            messages.append({"role": "user", "content": prompt})

        else:
            # Multimodal prompt
            image_detail = "low"  # high/low/auto
            messages.append(generate_message(prompt, image_detail=image_detail))

        # Pass temperature to ensure non-deterministic exploration
        # Note: Some models (o1, o3, gpt-5) only support temperature=1
        model_name = params["model"].lower()
        temperature = params.get("temperature", 0.7)

        # Models that don't support custom temperature
        fixed_temp_models = ["o1", "o3", "gpt-5"]
        uses_fixed_temp = any(m in model_name for m in fixed_temp_models)

        if uses_fixed_temp:
            # These models only support temperature=1, don't pass it
            text_response = self.client.chat.completions.create(
                model=params["model"],
                messages=messages,
            )
        else:
            text_response = self.client.chat.completions.create(
                model=params["model"],
                messages=messages,
                temperature=temperature,
            )
        text_response = text_response.choices[0].message.content
        self.response = text_response

        # Update message history
        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append({"role": "assistant", "content": text_response})

        if stop is not None:
            text_response = text_response.split(stop)[0]
        return text_response

#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from omegaconf import DictConfig

from habitat_llm.llm.base_llm import BaseLLM, Prompt

# Load .env file if it exists (for AWS credentials)
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


class BedrockClaude(BaseLLM):
    """
    LLM implementation using AWS Bedrock with Claude models.
    Uses environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

    Supports model aliases for convenience:
        sonnet, sonnet-4.5, sonnet4.5  -> us.anthropic.claude-sonnet-4-5-20250929-v1:0
        haiku, haiku-4.5, haiku4.5     -> us.anthropic.claude-haiku-4-5-20251001-v1:0
        opus, opus-4.5, opus4.5        -> us.anthropic.claude-opus-4-5-20251101-v1:0
    """

    # Model aliases mapping short names to full Bedrock inference profile IDs
    # Note: Newer Claude models require regional inference profiles (us. prefix)
    MODEL_ALIASES: Dict[str, str] = {
        # Claude Sonnet 4.5
        "sonnet": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "sonnet4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        # Claude Haiku 4.5
        "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "haiku4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        # Claude Opus 4.5
        "opus": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "opus-4.5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "opus4.5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    }

    @classmethod
    def resolve_model_alias(cls, model: str) -> str:
        """Resolve a model alias to the full Bedrock model ID."""
        return cls.MODEL_ALIASES.get(model.lower(), model)

    def __init__(self, conf: DictConfig):
        """
        Initialize the Bedrock Claude model.
        :param conf: the configuration of the language model
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params

        # Initialize boto3 client - uses env vars automatically
        region = os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=region)

        self.verbose = getattr(self.llm_conf, "verbose", True)
        self.message_history: List[Dict] = []
        self.keep_message_history = getattr(self.llm_conf, "keep_message_history", False)
        self.system_message = getattr(self.llm_conf, "system_message", "You are an expert at task planning.")

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: int = 40,
    ):
        """
        Generate a response using AWS Bedrock Claude.
        :param prompt: A string with the input to the language model.
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate.
        :param request_timeout: maximum time before timeout (not used directly by boto3)
        :param generation_args: contains arguments like the grammar definition. Not used here.
        """
        # Get model ID (resolve alias if provided)
        model_id = self.resolve_model_alias(self.generation_params.model)

        # Override stop if provided
        if stop is None and hasattr(self.generation_params, "stop") and self.generation_params.stop:
            stop = self.generation_params.stop

        # Override max_length if provided
        max_tokens = max_length if max_length is not None else self.generation_params.max_tokens

        # Build messages
        messages = self.message_history.copy()

        # Add current message
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            # Multimodal prompt - convert to Claude format
            content = []
            for prompt_type, prompt_value in prompt:
                if prompt_type == "text":
                    content.append({"type": "text", "text": prompt_value})
                else:
                    # Image - extract base64 data
                    # Expecting format: "data:image/jpeg;base64,<base64_data>"
                    if prompt_value.startswith("data:"):
                        parts = prompt_value.split(",", 1)
                        if len(parts) == 2:
                            media_type = parts[0].split(";")[0].replace("data:", "")
                            base64_data = parts[1]
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data,
                                }
                            })
            messages.append({"role": "user", "content": content})

        # Build request body for Claude
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
        }

        # Add system message
        if self.system_message:
            request_body["system"] = self.system_message

        # Add optional parameters
        # Note: Claude Sonnet 4.5 doesn't allow both temperature and top_p
        if hasattr(self.generation_params, "temperature"):
            request_body["temperature"] = self.generation_params.temperature
        elif hasattr(self.generation_params, "top_p"):
            request_body["top_p"] = self.generation_params.top_p
        if stop:
            request_body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        # Call Bedrock
        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )

        # Parse response
        response_body = json.loads(response["body"].read())
        text_response = response_body["content"][0]["text"]
        self.response = text_response

        # Update message history
        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append({"role": "assistant", "content": text_response})

        # Handle stop sequence
        if stop is not None and stop in text_response:
            text_response = text_response.split(stop)[0]

        return text_response

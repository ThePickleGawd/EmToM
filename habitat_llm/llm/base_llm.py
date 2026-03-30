# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig

Prompt = Union[str, List[Tuple[str, str]]]


class LLMRequestError(RuntimeError):
    """Normalized provider error with retry metadata preserved."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.retryable = retryable


class BaseLLM:
    """
    Base LLM Class
    """

    def __init__(self, conf: DictConfig):
        """
        Initialize the HF Language Model
        :param conf: The Language Model config
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
    ):
        """
        Generate a response autoregressively.
        :param prompt: A string with the input to the language model.
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate
        """
        raise NotImplementedError

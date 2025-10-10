from abc import ABC, abstractmethod
from typing import Optional

from macrec.utils.token_tracker import get_token_tracker

class BaseLLM(ABC):
    def __init__(self) -> None:
        self.model_name: str
        self.max_tokens: int
        self.max_context_length: int
        self.json_mode: bool
        self.token_tracker = get_token_tracker()

    @property
    def tokens_limit(self) -> int:
        """Limit of tokens that can be fed into the LLM under the current context length.

        Returns:
            `int`: The limit of tokens that can be fed into the LLM under the current context length.
        """
        return self.max_context_length - 2 * self.max_tokens - 50  # single round need 2 agent prompt steps: thought and action

    def track_tokens(self, input_tokens: int, output_tokens: int, call_type: str = "unknown") -> None:
        """Track token usage for this LLM call.
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            call_type: Type of call (e.g., "manager", "analyst", "searcher")
        """
        self.token_tracker.track_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=self.model_name,
            call_type=call_type
        )

    @abstractmethod
    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """Forward pass of the LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `str`: The LLM output.
        """
        raise NotImplementedError("BaseLLM.__call__() not implemented")

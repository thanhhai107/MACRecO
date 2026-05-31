from abc import ABC, abstractmethod
from typing import Optional

from macrec.utils.token_tracker import token_tracker


class SampleDeferredError(RuntimeError):
    """Signal that the current sample should be retried after the main pass."""


class BaseLLM(ABC):
    def __init__(self) -> None:
        self.model_name: str
        self.max_tokens: int
        self.max_context_length: int
        self.json_mode: bool
        self.token_tracker = token_tracker
        # Time spent in failed attempts and retry backoff for the latest call.
        self.last_call_retry_overhead: float = 0.0
        self.last_call_retry_count: int = 0
        self.sample_retry_count: int = 0

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

    def reset_call_metrics(self) -> None:
        """Reset per-call timing metadata before starting a new LLM request."""
        self.last_call_retry_overhead = 0.0
        self.last_call_retry_count = 0

    def reset_sample_metrics(self) -> None:
        """Reset retry metadata before processing a new dataset sample."""
        self.sample_retry_count = 0
        self.reset_call_metrics()

    def record_retry_overhead(self, failed_attempt_duration: float, backoff_seconds: float = 0.0) -> None:
        """Accumulate time that should be excluded from runtime metrics.

        The excluded time includes failed API attempt time and any retry sleep.
        """
        overhead = max(failed_attempt_duration, 0.0) + max(backoff_seconds, 0.0)
        self.last_call_retry_overhead += overhead
        self.last_call_retry_count += 1
        self.sample_retry_count = max(self.sample_retry_count, self.last_call_retry_count)
        self.token_tracker.record_retry_overhead(failed_attempt_duration, backoff_seconds)

    def adjusted_call_duration(self, wall_time: float) -> float:
        """Return wall time with retry overhead removed.

        If there were no retries, this is identical to the wall time.
        """
        return max(wall_time - self.last_call_retry_overhead, 0.0)

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

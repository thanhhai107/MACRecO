import json
import os
import time
from typing import Optional

import httpx
from google import genai
from google.oauth2 import service_account
from google.genai import types
from loguru import logger
import tiktoken

from macrec.llms.basellm import BaseLLM, SampleDeferredError


class VertexTokenizerWrapper:
    """Provide a tokenizer-like interface backed by Vertex AI token counting."""

    def __init__(self, client: genai.Client, model_name: str):
        self.client = client
        self.model_name = model_name
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str):
        """Count tokens locally to avoid a network round-trip during prompt checks."""
        return types.CountTokensResponse(total_tokens=len(self._encoding.encode(text)))

    def encode(self, text: str) -> list:
        response = self.count_tokens(text)
        return [0] * int(getattr(response, "total_tokens", 0) or 0)

    def decode(self, tokens: list) -> str:
        return ""


class VertexLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        json_mode: bool = False,
        service_account_path: str = "config/vertex-service-account.json",
        project_id: Optional[str] = None,
        location: str = "global",
        *args,
        **kwargs,
    ):
        """Initialize a Vertex AI Gemini model.

        Args:
            model_name: Vertex AI Gemini model id, e.g. `gemini-2.5-flash`.
            json_mode: Whether to request JSON output.
            service_account_path: Path to the service account JSON key.
            project_id: Optional Google Cloud project id override.
            location: Vertex AI region.
        """
        super().__init__()
        self.model_name = model_name
        self.json_mode = json_mode
        self.max_tokens: int = kwargs.get("max_tokens", 256)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.8)
        self.top_k = kwargs.get("top_k", 40)
        # google.genai HttpOptions.timeout is expressed in milliseconds.
        self.request_timeout_ms = int(
            kwargs.get(
                "request_timeout_ms",
                kwargs.get("request_timeout_seconds", 30) * 1000,
            )
        )
        self.max_retries = int(kwargs.get("max_retries", 100))
        self.defer_after_retries = int(kwargs.get("defer_after_retries", 12))
        self.service_account_path = service_account_path
        self.location = location

        context_lengths = {
            "gemini-2.5-pro": 1048576,
            "gemini-2.5-flash": 1048576,
            "gemini-2.5-flash-lite": 1048576,
            "gemini-1.5-flash": 1048576,
            "gemini-1.5-pro": 2097152,
            "gemini-1.0-pro": 32768,
        }
        self.max_context_length = 32768
        for model_prefix, context_length in context_lengths.items():
            if model_name.startswith(model_prefix):
                self.max_context_length = context_length
                break

        resolved_service_account_path = self._resolve_service_account_path(service_account_path)
        if not os.path.exists(resolved_service_account_path):
            raise FileNotFoundError(
                f"Vertex service account file not found: {resolved_service_account_path}"
            )

        with open(resolved_service_account_path, "r", encoding="utf-8") as f:
            service_account_info = json.load(f)

        resolved_project_id = project_id or service_account_info.get("project_id") or os.getenv(
            "GOOGLE_CLOUD_PROJECT"
        )
        if not resolved_project_id:
            raise ValueError(
                "Vertex project id must be provided either in the service account JSON, "
                "via project_id, or via GOOGLE_CLOUD_PROJECT."
            )

        credentials = service_account.Credentials.from_service_account_file(
            resolved_service_account_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        self.client = genai.Client(
            vertexai=True,
            credentials=credentials,
            project=resolved_project_id,
            location=location,
            http_options=types.HttpOptions(timeout=self.request_timeout_ms),
        )
        self.project_id = resolved_project_id
        self.resolved_service_account_path = resolved_service_account_path
        # Keep a tokenizer-compatible surface for existing agents.
        self.model = VertexTokenizerWrapper(self.client, self.model_name)

        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
        ]

        self.generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            topP=self.top_p,
            topK=self.top_k,
            maxOutputTokens=self.max_tokens,
            safetySettings=safety_settings,
            responseMimeType="application/json" if json_mode else None,
        )

        if json_mode:
            logger.info("Using JSON mode for Vertex AI Gemini.")

        logger.info(
            f"[VERTEX] VertexLLM '{model_name}' initialized: project={resolved_project_id}, "
            f"location={location}, max_context={self.max_context_length}, json_mode={json_mode}, "
            f"request_timeout_ms={self.request_timeout_ms}, max_retries={self.max_retries}"
        )

    @staticmethod
    def _resolve_service_account_path(service_account_path: str) -> str:
        if os.path.isabs(service_account_path):
            return service_account_path
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            service_account_path,
        )

    def __call__(self, prompt: str, call_type: str = "unknown", *args, **kwargs) -> str:
        retry_count = 0
        self.reset_call_metrics()

        while retry_count <= self.max_retries:
            try:
                attempt_start = time.time()
                request_prompt = prompt
                if self.json_mode:
                    request_prompt = (
                        request_prompt + "\n\nPlease respond with valid JSON format only."
                    )

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=request_prompt,
                    config=self.generation_config,
                )

                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                    self.track_tokens(input_tokens, output_tokens, call_type)

                output = getattr(response, "text", "") or ""
                if not output:
                    logger.warning("No content generated by Vertex AI Gemini")
                    return ""

                if self.json_mode:
                    output = self._normalize_json_output(output)
                    return output.strip()

                return output.replace("\n", " ").strip()

            except Exception as e:
                attempt_time = time.time() - attempt_start
                error_str = str(e) if e else ""
                error_str_lower = error_str.lower() if isinstance(error_str, str) else ""
                is_retryable = (
                    "429" in error_str
                    or "503" in error_str
                    or "504" in error_str
                    or "499" in error_str
                    or "cancelled" in error_str_lower
                    or "canceled" in error_str_lower
                    or "server disconnected without sending a response" in error_str_lower
                    or "deadline exceeded" in error_str_lower
                    or "timed out" in error_str_lower
                    or "timeout" in error_str_lower
                    or isinstance(e, (TimeoutError, httpx.TimeoutException))
                    or "rate limit" in error_str_lower
                    or "too many requests" in error_str_lower
                    or "resource exhausted" in error_str_lower
                )

                if is_retryable and retry_count < self.max_retries:
                    retry_count += 1
                    if self.defer_after_retries > 0 and retry_count >= self.defer_after_retries:
                        self.record_retry_overhead(attempt_time, 0)
                        raise SampleDeferredError(
                            f"Vertex call reached {retry_count} retryable failures; deferring sample. "
                            f"Last error: {e}"
                        ) from e

                    wait_time = min(2 ** (retry_count - 1), 100)
                    self.record_retry_overhead(attempt_time, wait_time)
                    if "timeout" in error_str_lower or "deadline exceeded" in error_str_lower or "timed out" in error_str_lower or isinstance(e, (TimeoutError, httpx.TimeoutException)):
                        logger.warning(
                            f"Vertex request timed out after ~{self.request_timeout_ms / 1000:.0f}s "
                            f"(attempt {retry_count}/{self.max_retries}): {e}. "
                            f"Retrying in {wait_time} second..."
                        )
                    else:
                        logger.warning(
                            f"Rate limit/error calling Vertex model (attempt {retry_count}/{self.max_retries}): "
                            f"{e}. Retrying in {wait_time} second..."
                        )
                    time.sleep(wait_time)
                    continue

                if retry_count >= self.max_retries:
                    logger.error(
                        f"Error calling Vertex model after {self.max_retries} retries: {e}. "
                        "Giving up on this call."
                    )
                else:
                    logger.error(f"Non-retryable error calling Vertex model: {e}")
                raise e

        logger.error("Unexpected exit from retry loop")
        return ""

    @staticmethod
    def _normalize_json_output(output: str) -> str:
        output = output.strip()
        if output.startswith("```json"):
            start = output.find("```json") + 7
            end = output.rfind("```")
            if end > start:
                output = output[start:end].strip()
            else:
                output = output[start:].strip()
        elif output.startswith("```"):
            start = output.find("```") + 3
            end = output.rfind("```")
            if end > start:
                output = output[start:end].strip()
            else:
                output = output[start:].strip()

        try:
            json.loads(output)
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                error_msg = str(e)
                if "char" in error_msg:
                    char_pos = int(error_msg.split("char ")[1].rstrip(")"))
                    output = output[:char_pos].strip()
                    logger.debug(
                        f"Extracted first valid JSON object (truncated at char {char_pos})"
                    )
        return output

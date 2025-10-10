from loguru import logger
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage
from typing import Optional

from macrec.llms.basellm import BaseLLM

class AnyOpenAILLM(BaseLLM):
    def __init__(self, model_name: str = 'gpt-3.5-turbo', json_mode: bool = False, *args, **kwargs):
        """Initialize the OpenAI LLM.

        Args:
            `model_name` (`str`, optional): The name of the OpenAI model. Defaults to `gpt-3.5-turbo`.
            `json_mode` (`bool`, optional): Whether to use the JSON mode of the OpenAI API. Defaults to `False`.
        """
        super().__init__()
        self.model_name = model_name
        self.json_mode = json_mode
        if json_mode and self.model_name not in ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']:
            raise ValueError("json_mode is only available for gpt-3.5-turbo-1106 and gpt-4-1106-preview")
        self.max_tokens: int = kwargs.get('max_tokens', 256)
        self.max_context_length: int = 16384 if '16k' in model_name else 32768 if '32k' in model_name else 4096
        if model_name.split('-')[0] == 'text' or model_name == 'gpt-3.5-turbo-instruct':
            self.model = OpenAI(model_name=model_name, *args, **kwargs)
            self.model_type = 'completion'
        else:
            if json_mode:
                logger.info("Using JSON mode of OpenAI API.")
                if 'model_kwargs' in kwargs:
                    kwargs['model_kwargs']['response_format'] = {
                        "type": "json_object"
                    }
                else:
                    kwargs['model_kwargs'] = {
                        "response_format": {
                            "type": "json_object"
                        }
                    }
            self.model = ChatOpenAI(model_name=model_name, *args, **kwargs)
            self.model_type = 'chat'

    def __call__(self, prompt: str, call_type: str = "unknown", *args, **kwargs) -> str:
        """Forward pass of the OpenAI LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
            `call_type` (`str`): Type of call for token tracking.
        Returns:
            `str`: The OpenAI LLM output.
        """
        if self.model_type == 'completion':
            response = self.model.invoke(prompt)
            output = response.content.replace('\n', ' ').strip()
            
            # Track tokens if available in response
            if hasattr(response, 'usage_metadata'):
                usage_metadata = response.usage_metadata
                input_tokens = getattr(usage_metadata, 'input_tokens', 0)
                output_tokens = getattr(usage_metadata, 'output_tokens', 0)
                self.track_tokens(input_tokens, output_tokens, call_type)
            elif hasattr(response, 'usage'):
                usage = response.usage
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
                self.track_tokens(input_tokens, output_tokens, call_type)
            
            return output
        else:
            response = self.model.invoke(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            )
            output = response.content.replace('\n', ' ').strip()
            
            # Track tokens if available in response
            if hasattr(response, 'usage_metadata'):
                usage_metadata = response.usage_metadata
                input_tokens = getattr(usage_metadata, 'input_tokens', 0)
                output_tokens = getattr(usage_metadata, 'output_tokens', 0)
                self.track_tokens(input_tokens, output_tokens, call_type)
            elif hasattr(response, 'usage'):
                usage = response.usage
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
                self.track_tokens(input_tokens, output_tokens, call_type)
            
            return output

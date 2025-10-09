from loguru import logger
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage

from macrec.llms.basellm import BaseLLM

class OpenRouterLLM(BaseLLM):
    def __init__(self, model_name: str = 'openai/gpt-3.5-turbo', json_mode: bool = False, api_key: str = None, *args, **kwargs):
        """Initialize the OpenRouter LLM.

        Args:
            `model_name` (`str`, optional): The name of the model on OpenRouter (e.g., 'openai/gpt-3.5-turbo'). Defaults to `openai/gpt-3.5-turbo`.
            `json_mode` (`bool`, optional): Whether to use the JSON mode. Defaults to `False`.
            `api_key` (`str`, optional): The API key for OpenRouter. If not provided, will use environment variable OPENROUTER_API_KEY.
        """
        self.model_name = model_name
        self.json_mode = json_mode
        self.max_tokens: int = kwargs.get('max_tokens', 256)
        
        # Set up context lengths for common models
        context_lengths = {
            'openai/gpt-3.5-turbo': 4096,
            'openai/gpt-3.5-turbo-16k': 16384,
            'openai/gpt-3.5-turbo-1106': 4096,
            'openai/gpt-4': 8192,
            'openai/gpt-4-32k': 32768,
            'openai/gpt-4-1106-preview': 128000,
            'anthropic/claude-3-sonnet': 200000,
            'anthropic/claude-3-haiku': 200000,
            'anthropic/claude-3-opus': 200000,
            'meta-llama/llama-2-70b-chat': 4096,
            'meta-llama/llama-2-13b-chat': 4096,
            'meta-llama/llama-2-7b-chat': 4096,
        }
        self.max_context_length: int = context_lengths.get(model_name, 4096)
        
        # Set up API configuration
        if api_key:
            api_key_value = api_key
        else:
            # Try to read from api-config.json first
            import os
            import json
            api_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'api-config.json')
            if os.path.exists(api_config_path):
                with open(api_config_path, 'r') as f:
                    api_config = json.load(f)
                    api_key_value = api_config.get('api_key')
            else:
                # Fallback to environment variable
                api_key_value = os.getenv('OPENROUTER_API_KEY')
            
            if not api_key_value:
                raise ValueError("API key must be provided either as parameter, in config/api-config.json, or as OPENROUTER_API_KEY environment variable")
        
        # Configure the model arguments for OpenRouter
        model_kwargs = kwargs.get('model_kwargs', {})
        model_kwargs.update({
            'api_key': api_key_value,
            'base_url': 'https://openrouter.ai/api/v1',
            'http_client': kwargs.get('http_client', None),
        })
        
        # Handle JSON mode for supported models
        if json_mode and self.model_name not in ['openai/gpt-3.5-turbo-1106', 'openai/gpt-4-1106-preview']:
            logger.warning(f"JSON mode may not be supported for model {self.model_name}")
        
        if json_mode:
            logger.info("Using JSON mode for OpenRouter API.")
            model_kwargs['response_format'] = {"type": "json_object"}
        
        # Determine if this is a completion model or chat model
        completion_models = [
            'openai/text-davinci-003',
            'openai/text-davinci-002',
            'openai/text-curie-001',
            'openai/text-babbage-001',
            'openai/text-ada-001',
        ]
        
        if any(model in model_name for model in completion_models):
            # Use completion model
            self.model = OpenAI(
                model_name=model_name,
                openai_api_key=api_key_value,
                openai_api_base='https://openrouter.ai/api/v1',
                **{k: v for k, v in kwargs.items() if k != 'model_kwargs'}
            )
            self.model_type = 'completion'
        else:
            # Use chat model
            self.model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key_value,
                openai_api_base='https://openrouter.ai/api/v1',
                model_kwargs=model_kwargs,
                **{k: v for k, v in kwargs.items() if k != 'model_kwargs'}
            )
            self.model_type = 'chat'

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """
        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
        Returns:
            `str`: The OpenRouter LLM output.
        """
        try:
            if self.model_type == 'completion':
                return self.model.invoke(prompt).content.replace('\n', ' ').strip()
            else:
                return self.model.invoke(
                    [
                        HumanMessage(
                            content=prompt,
                        )
                    ]
                ).content.replace('\n', ' ').strip()
        except Exception as e:
            logger.error(f"Error calling OpenRouter model: {e}")
            raise e

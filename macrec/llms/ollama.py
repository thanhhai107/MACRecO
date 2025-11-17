from loguru import logger
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
import tiktoken

from macrec.llms.basellm import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = 'llama3.2', json_mode: bool = False, base_url: str = 'http://localhost:11434', *args, **kwargs):
        """Initialize the Ollama LLM.

        Args:
            `model_name` (`str`, optional): The name of the Ollama model (e.g., 'llama3.2', 'mistral', 'phi3'). Defaults to `llama3.2`.
            `json_mode` (`bool`, optional): Whether to use JSON mode for structured output. Defaults to `False`.
            `base_url` (`str`, optional): The base URL for the Ollama server. Defaults to `http://localhost:11434`.
        """
        super().__init__()
        self.model_name = model_name
        self.json_mode = json_mode
        self.max_tokens: int = kwargs.get('max_tokens', 256)
        
        # Set up context lengths for common Ollama models
        # These are typical values, adjust based on the specific model
        context_lengths = {
            'llama3.2': 131072,        # 128K tokens
            'llama3.1': 131072,        # 128K tokens
            'llama3': 8192,            # 8K tokens
            'llama2': 4096,            # 4K tokens
            'mistral': 32768,          # 32K tokens
            'mixtral': 32768,          # 32K tokens
            'phi3': 131072,            # 128K tokens
            'gemma': 8192,             # 8K tokens
            'gemma2': 8192,            # 8K tokens
            'qwen2': 32768,            # 32K tokens
            'codellama': 16384,        # 16K tokens
            'deepseek-coder': 16384,   # 16K tokens
            'neural-chat': 8192,       # 8K tokens
            'starling-lm': 8192,       # 8K tokens
            'vicuna': 16384,           # 16K tokens
        }
        
        # Try to find the context length by matching model name prefix
        self.max_context_length: int = 4096  # default
        for model_prefix, context_length in context_lengths.items():
            if model_name.startswith(model_prefix):
                self.max_context_length = context_length
                break
        
        # Try to read base_url from api-config.json if not provided
        if base_url == 'http://localhost:11434':
            import os
            import json
            api_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'api-config.json')
            if os.path.exists(api_config_path):
                try:
                    with open(api_config_path, 'r') as f:
                        api_config = json.load(f)
                        if 'ollama_base_url' in api_config:
                            base_url = api_config.get('ollama_base_url', base_url)
                except Exception as e:
                    logger.debug(f"Could not read Ollama base URL from api-config.json: {e}")
        
        self.base_url = base_url
        
        # Configure the model
        model_kwargs = {
            'base_url': base_url,
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.8),
            'num_predict': self.max_tokens,  # Ollama uses num_predict instead of max_tokens
        }
        
        # Add JSON mode if requested
        if json_mode:
            logger.info("Using JSON mode for Ollama.")
            # Ollama supports format parameter for JSON output
            model_kwargs['format'] = 'json'
        
        # Initialize the ChatOllama model
        try:
            self.model = ChatOllama(
                model=model_name,
                **model_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model '{model_name}'. Make sure Ollama is running and the model is pulled. Error: {e}")
            raise e

    def __call__(self, prompt: str, call_type: str = "unknown", *args, **kwargs) -> str:
        """Forward pass of the Ollama LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
            `call_type` (`str`): Type of call for token tracking.
        Returns:
            `str`: The Ollama LLM output.
        """
        try:
            # For JSON mode, add instruction to ensure JSON output
            if self.json_mode:
                prompt = prompt + "\n\nPlease respond with valid JSON format only."
            
            response = self.model.invoke(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            )
            
            output = response.content
            
            # Track tokens - Ollama may not always provide token counts
            # so we estimate using tiktoken
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                input_tokens = len(encoding.encode(prompt))
                output_tokens = len(encoding.encode(output))
                self.track_tokens(input_tokens, output_tokens, call_type)
            except Exception as e:
                # Fallback to rough estimation: ~4 characters per token
                input_tokens = len(prompt) // 4
                output_tokens = len(output) // 4
                self.track_tokens(input_tokens, output_tokens, call_type)
                logger.debug(f"Token estimation fallback used for Ollama: {e}")
            
            # For JSON mode, clean up markdown code blocks if present
            if self.json_mode:
                output = output.strip()
                # Remove markdown code blocks if present
                if output.startswith('```json'):
                    start = output.find('```json') + 7
                    end = output.rfind('```')
                    if end > start:
                        output = output[start:end].strip()
                    else:
                        output = output[start:].strip()
                elif output.startswith('```'):
                    start = output.find('```') + 3
                    end = output.rfind('```')
                    if end > start:
                        output = output[start:end].strip()
                    else:
                        output = output[start:].strip()
                
                # Try to extract just the first complete JSON object if there's extra text
                import json
                try:
                    # Attempt to parse - if it works, we're good
                    json.loads(output)
                except json.JSONDecodeError as e:
                    if "Extra data" in str(e):
                        # Extract character position where valid JSON ends
                        error_msg = str(e)
                        if "char" in error_msg:
                            char_pos = int(error_msg.split("char ")[1].rstrip(")"))
                            output = output[:char_pos].strip()
                            logger.debug(f"Extracted first valid JSON object (truncated at char {char_pos})")
                
                # For JSON, preserve formatting
                return output.strip()
            else:
                # For non-JSON mode, replace newlines with spaces (old behavior)
                return output.replace('\n', ' ').strip()
                
        except Exception as e:
            logger.error(f"Error calling Ollama model '{self.model_name}': {e}")
            logger.error(f"Make sure Ollama is running at {self.base_url} and model '{self.model_name}' is available.")
            logger.error(f"You can pull the model with: ollama pull {self.model_name}")
            raise e

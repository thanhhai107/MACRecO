from loguru import logger
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
import tiktoken
import time

from macrec.llms.basellm import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = 'llama3.2', json_mode: bool = False, base_url: str = 'http://localhost:11434', *args, **kwargs):
        """Initialize the Ollama LLM.

        Args:
            `model_name` (`str`, optional): The name of the Ollama model (e.g., 'llama3.2', 'mistral', 'phi3'). Defaults to `llama3.2`.
            `json_mode` (`bool`, optional): Whether to use JSON mode for structured output. Defaults to `False`.
            `base_url` (`str`, optional): The base URL for the Ollama server. Defaults to `http://localhost:11434`.
        """
        init_start = time.time()
        super().__init__()
        self.model_name = model_name
        self.json_mode = json_mode
        self.max_tokens: int = kwargs.get('max_tokens', 256)
        logger.debug(f"[OLLAMA] Initializing Ollama LLM: model={model_name}, json_mode={json_mode}, max_tokens={self.max_tokens}")
        
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
            model_init_start = time.time()
            self.model = ChatOllama(
                model=model_name,
                **model_kwargs
            )
            model_init_time = time.time() - model_init_start
            logger.debug(f"[OLLAMA] ChatOllama model initialized in {model_init_time:.3f}s")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model '{model_name}'. Make sure Ollama is running and the model is pulled. Error: {e}")
            raise e
        
        init_time = time.time() - init_start
        logger.info(f"[OLLAMA] Ollama LLM '{model_name}' initialized in {init_time:.3f}s - max_context={self.max_context_length}, json_mode={json_mode}")

    def __call__(self, prompt: str, call_type: str = "unknown", *args, **kwargs) -> str:
        """Forward pass of the Ollama LLM.

        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
            `call_type` (`str`): Type of call for token tracking.
        Returns:
            `str`: The Ollama LLM output.
        """
        try:
            call_start = time.time()
            prompt_len = len(prompt)
            logger.debug(f"[OLLAMA] Starting LLM call: type={call_type}, prompt_len={prompt_len}")
            
            # For JSON mode, add instruction to ensure JSON output
            if self.json_mode:
                prompt = prompt + "\n\nPlease respond with valid JSON format only."
            
            invoke_start = time.time()
            response = self.model.invoke(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            )
            invoke_time = time.time() - invoke_start
            logger.debug(f"[OLLAMA] LLM invoke completed in {invoke_time:.3f}s")
            
            output = response.content
            output_len = len(output)
            logger.debug(f"[OLLAMA] LLM response received: output_len={output_len}")
            
            # Track tokens - Ollama may not always provide token counts
            # so we estimate using tiktoken
            token_start = time.time()
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                input_tokens = len(encoding.encode(prompt))
                output_tokens = len(encoding.encode(output))
                self.track_tokens(input_tokens, output_tokens, call_type)
                logger.debug(f"[OLLAMA] Tokens tracked: input={input_tokens}, output={output_tokens}")
            except Exception as e:
                # Fallback to rough estimation: ~4 characters per token
                input_tokens = len(prompt) // 4
                output_tokens = len(output) // 4
                self.track_tokens(input_tokens, output_tokens, call_type)
                logger.debug(f"[OLLAMA] Token estimation fallback: input~{input_tokens}, output~{output_tokens}, error: {e}")
            
            token_time = time.time() - token_start
            
            # For JSON mode, clean up markdown code blocks if present
            if self.json_mode:
                json_parse_start = time.time()
                output = output.strip()
                logger.debug(f"[OLLAMA] Processing JSON output: original_len={len(output)}")
                
                # Remove markdown code blocks if present
                if output.startswith('```json'):
                    start = output.find('```json') + 7
                    end = output.rfind('```')
                    if end > start:
                        output = output[start:end].strip()
                    else:
                        output = output[start:].strip()
                    logger.debug(f"[OLLAMA] Removed ```json blocks")
                elif output.startswith('```'):
                    start = output.find('```') + 3
                    end = output.rfind('```')
                    if end > start:
                        output = output[start:end].strip()
                    else:
                        output = output[start:].strip()
                    logger.debug(f"[OLLAMA] Removed ``` blocks")
                
                # Try to extract just the first complete JSON object if there's extra text
                import json
                try:
                    # Attempt to parse - if it works, we're good
                    json.loads(output)
                    logger.debug(f"[OLLAMA] JSON validation passed")
                except json.JSONDecodeError as e:
                    if "Extra data" in str(e):
                        # Extract character position where valid JSON ends
                        error_msg = str(e)
                        if "char" in error_msg:
                            char_pos = int(error_msg.split("char ")[1].rstrip(")"))
                            output = output[:char_pos].strip()
                            logger.debug(f"[OLLAMA] Extracted valid JSON (truncated at char {char_pos})")
                
                json_parse_time = time.time() - json_parse_start
                call_time = time.time() - call_start
                logger.info(f"[OLLAMA] LLM call completed: type={call_type}, total_time={call_time:.3f}s (invoke={invoke_time:.3f}s, token={token_time:.3f}s, json_parse={json_parse_time:.3f}s)")
                
                # For JSON, preserve formatting
                return output.strip()
            else:
                # For non-JSON mode, replace newlines with spaces (old behavior)
                call_time = time.time() - call_start
                logger.info(f"[OLLAMA] LLM call completed: type={call_type}, total_time={call_time:.3f}s (invoke={invoke_time:.3f}s, token={token_time:.3f}s)")
                return output.replace('\n', ' ').strip()
                
        except Exception as e:
            call_time = time.time() - call_start
            logger.error(f"[OLLAMA] Error calling Ollama model '{self.model_name}' after {call_time:.3f}s: {e}")
            logger.error(f"[OLLAMA] Make sure Ollama is running at {self.base_url} and model '{self.model_name}' is available.")
            logger.error(f"[OLLAMA] You can pull the model with: ollama pull {self.model_name}")
            raise e

from loguru import logger
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain.schema import HumanMessage

from macrec.llms.basellm import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = 'gemini-2.0-flash', json_mode: bool = False, api_key: str = None, *args, **kwargs):
        """Initialize the Gemini LLM.

        Args:
            `model_name` (`str`, optional): The name of the Gemini model. Defaults to `gemini-2.0-flash`.
            `json_mode` (`bool`, optional): Whether to use structured output mode. Defaults to `False`.
            `api_key` (`str`, optional): The API key for Gemini. If not provided, will use environment variable GOOGLE_API_KEY.
        """
        super().__init__()
        self.model_name = model_name
        self.json_mode = json_mode
        self.max_tokens: int = kwargs.get('max_tokens', 256)
        
        # Set up context lengths for different Gemini models
        context_lengths = {
            'gemini-2.0-flash': 1048576,      # 1M tokens
            'gemini-1.5-flash': 1048576,      # 1M tokens
            'gemini-1.5-pro': 2097152,        # 2M tokens
            'gemini-1.0-pro': 32768,          # 32K tokens
        }
        self.max_context_length: int = context_lengths.get(model_name, 32768)
        
        # Configure the API
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to read from api-config.json first
            import os
            import json
            api_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'api-config.json')
            if os.path.exists(api_config_path):
                with open(api_config_path, 'r') as f:
                    api_config = json.load(f)
                    api_key = api_config.get('api_key')
            else:
                # Fallback to environment variable
                api_key = os.getenv('GOOGLE_API_KEY')
            
            if not api_key:
                raise ValueError("API key must be provided either as parameter, in config/api-config.json, or as GOOGLE_API_KEY environment variable")
            genai.configure(api_key=api_key)
        
        # Initialize the model
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_tokens,
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.8),
            top_k=kwargs.get('top_k', 40),
        )
        
        safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
        ]
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # For JSON mode, we'll add instructions to the prompt
        if json_mode:
            logger.info("Using JSON mode for Gemini.")

    def __call__(self, prompt: str, call_type: str = "unknown", *args, **kwargs) -> str:
        """
        Args:
            `prompt` (`str`): The prompt to feed into the LLM.
            `call_type` (`str`): Type of call for token tracking.
        Returns:
            `str`: The Gemini LLM output.
        """
        try:
            # Add JSON formatting instruction if in json_mode
            if self.json_mode:
                prompt = prompt + "\n\nPlease respond with valid JSON format only."
            
            response = self.model.generate_content(prompt)
            
            # Check if response was truncated due to token limit
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason = response.candidates[0].finish_reason
                finish_reason_str = str(finish_reason)
                # Only warn if it's actually a truncation (MAX_TOKENS, etc), not just STOP or OTHER
                # Gemini FinishReason enum values: STOP=1, MAX_TOKENS=2, SAFETY=3, RECITATION=4, OTHER=5
                # After str(): 'FinishReason.STOP', 'FinishReason.MAX_TOKENS', etc.
                if 'MAX_TOKENS' in finish_reason_str:
                    logger.info(f"Gemini response ended with finish_reason={finish_reason_str}. Response may be incomplete due to token limit.")
            
            # Track tokens if available in response
            if hasattr(response, 'usage_metadata'):
                usage_metadata = response.usage_metadata
                input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
                self.track_tokens(input_tokens, output_tokens, call_type)
            
            if response.candidates and response.candidates[0].content:
                content = response.candidates[0].content.parts[0].text
                
                # For JSON mode, clean up markdown code blocks before processing
                if self.json_mode:
                    content = content.strip()
                    # Remove markdown code blocks if present
                    if content.startswith('```json'):
                        start = content.find('```json') + 7
                        end = content.rfind('```')
                        if end > start:
                            content = content[start:end].strip()
                        else:
                            content = content[start:].strip()
                    elif content.startswith('```'):
                        start = content.find('```') + 3
                        end = content.rfind('```')
                        if end > start:
                            content = content[start:end].strip()
                        else:
                            content = content[start:].strip()
                    
                    # Try to extract just the first complete JSON object if there's extra text
                    # This handles cases where Gemini adds explanation after the JSON
                    import json
                    try:
                        # Attempt to parse - if it works, we're good
                        json.loads(content)
                    except json.JSONDecodeError as e:
                        if "Extra data" in str(e):
                            # Extract character position where valid JSON ends
                            error_msg = str(e)
                            if "char" in error_msg:
                                char_pos = int(error_msg.split("char ")[1].rstrip(")"))
                                content = content[:char_pos].strip()
                                logger.debug(f"Extracted first valid JSON object (truncated at char {char_pos})")
                    
                    # For JSON, preserve newlines but normalize spacing
                    return content.strip()
                else:
                    # For non-JSON mode, replace newlines with spaces (old behavior)
                    return content.replace('\n', ' ').strip()
            else:
                logger.warning("No content generated by Gemini model")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Gemini model: {e}")
            raise e

import tiktoken
from enum import Enum
from loguru import logger
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate

from macrec.agents.base import Agent
from macrec.llms import AnyOpenAILLM, GeminiLLM
from macrec.utils import format_step, format_reflections, format_last_attempt, read_json, get_rm

class GeminiTokenizerWrapper:
    """
    Wrapper class to provide a tokenizer-like interface for Gemini models.
    Uses Gemini's native count_tokens API for accurate token counting.
    """
    def __init__(self, model):
        """
        Args:
            model: The GenerativeModel instance from google.generativeai
        """
        self.model = model
    
    def encode(self, text: str) -> list:
        """
        Encode text and return a list representing tokens.
        For compatibility, returns a list with length equal to token count.
        
        Args:
            text: The text to tokenize
            
        Returns:
            list: A list with length equal to the token count
        """
        try:
            result = self.model.count_tokens(text)
            # Return a list with length equal to token count for compatibility
            return [0] * result.total_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens with Gemini API: {e}. Falling back to approximation.")
            # Fallback to rough estimation: ~4 chars per token
            return [0] * (len(text) // 4)
    
    def decode(self, tokens: list) -> str:
        """
        Dummy decode method for compatibility.
        Not actually used in the reflector.
        """
        return ""

class ReflectionStrategy(Enum):
    """
    Reflection strategies for the `Reflector` agent. `NONE` means no reflection. `REFLEXION` is the default strategy for the `Reflector` agent, which prompts the LLM to reflect on the input and the scratchpad. `LAST_ATTEMPT` simply store the input and the scratchpad of the last attempt. `LAST_ATTEMPT_AND_REFLEXION` combines the two strategies.
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial'
    REFLEXION = 'reflection'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflection'

class Reflector(Agent):
    """
    The reflector agent. The reflector agent prompts the LLM to reflect on the input and the scratchpad as default. Other reflection strategies are also supported. See `ReflectionStrategy` for more details.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        """Initialize the reflector agent. The reflector agent prompts the LLM to reflect on the input and the scratchpad as default.

        Args:
            `config_path` (`str`): The path to the config file of the reflector LLM.
        """
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        keep_reflections = get_rm(config, 'keep_reflections', True)
        reflection_strategy = get_rm(config, 'reflection_strategy', ReflectionStrategy.REFLEXION.value)
        self.llm = self.get_LLM(config=config)
        if isinstance(self.llm, AnyOpenAILLM):
            self.enc = tiktoken.encoding_for_model(self.llm.model_name)
        elif isinstance(self.llm, GeminiLLM):
            # Use Gemini's native count_tokens API for accurate token counting
            self.enc = GeminiTokenizerWrapper(self.llm.model)
            logger.info(f"Using Gemini native tokenizer for model: {self.llm.model_name}")
        else:
            self.enc = AutoTokenizer.from_pretrained(self.llm.model_name)
        self.json_mode = self.llm.json_mode
        self.keep_reflections = keep_reflections
        for strategy in ReflectionStrategy:
            if strategy.value == reflection_strategy:
                self.reflection_strategy = strategy
                break
        assert self.reflection_strategy is not None, f'Unknown reflection strategy: {reflection_strategy}'
        self.reflections: list[str] = []
        self.reflections_str: str = ''

    @property
    def reflector_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['reflect_prompt_json']
        else:
            return self.prompts['reflect_prompt']

    @property
    def reflect_examples(self) -> str:
        prompt_name = 'reflect_examples_json' if self.json_mode else 'reflect_examples'
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        else:
            return ''

    def _build_reflector_prompt(self, input: str, scratchpad: str) -> str:
        return self.reflector_prompt.format(
            examples=self.reflect_examples,
            input=input,
            scratchpad=scratchpad
        )

    def _prompt_reflection(self, input: str, scratchpad: str) -> str:
        reflection_prompt = self._build_reflector_prompt(input, scratchpad)
        reflection_response = self.llm(reflection_prompt, call_type="reflector")
        if self.keep_reflections:
            self.reflection_input = reflection_prompt
            self.reflection_output = reflection_response
            logger.trace(f'Reflection input length: {len(self.enc.encode(self.reflection_input))}')
            logger.trace(f"Reflection input: {self.reflection_input}")
            logger.trace(f'Reflection output length: {len(self.enc.encode(self.reflection_output))}')
            if self.json_mode:
                self.system.log(f"[:violet[Reflection]]:\n- `{self.reflection_output}`", agent=self, logging=False)
            else:
                self.system.log(f"[:violet[Reflection]]:\n- {self.reflection_output}", agent=self, logging=False)
            logger.debug(f"Reflection output: {self.reflection_output}")
        return format_step(reflection_response)

    def forward(self, input: str, scratchpad: str, *args, **kwargs) -> str:
        logger.trace('Running Reflecion strategy...')
        if self.reflection_strategy == ReflectionStrategy.LAST_ATTEMPT:
            self.reflections = [scratchpad]
            self.reflections_str = format_last_attempt(input, scratchpad, self.prompts['last_trial_header'])
        elif self.reflection_strategy == ReflectionStrategy.REFLEXION:
            self.reflections.append(self._prompt_reflection(input=input, scratchpad=scratchpad))
            self.reflections_str = format_reflections(self.reflections, header=self.prompts['reflection_header'])
        elif self.reflection_strategy == ReflectionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(input, scratchpad, self.prompts['last_trial_header'])
            self.reflections = self._prompt_reflection(input=input, scratchpad=scratchpad)
            self.reflections_str += format_reflections(self.reflections, header=self.prompts['reflection_last_trial_header'])
        elif self.reflection_strategy == ReflectionStrategy.NONE:
            self.reflections = []
            self.reflections_str = ''
        else:
            raise ValueError(f'Unknown reflection strategy: {self.reflection_strategy}')
        logger.trace(self.reflections_str)
        return self.reflections_str

# Description: Package for large language models
from macrec.llms.basellm import BaseLLM
from macrec.llms.openai import AnyOpenAILLM
from macrec.llms.gemini import GeminiLLM
from macrec.llms.vertex import VertexLLM
from macrec.llms.openrouter import OpenRouterLLM
from macrec.llms.ollama import OllamaLLM

try:
    from macrec.llms.opensource import OpenSourceLLM
except ImportError as e:
    # OpenSourceLLM requires jsonformer which may have compatibility issues
    # with certain transformers versions. Create a dummy class if import fails.
    from loguru import logger
    logger.warning(f"Could not import OpenSourceLLM: {e}. OpenSourceLLM will not be available.")
    class OpenSourceLLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("OpenSourceLLM is not available due to dependency issues.")

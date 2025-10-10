"""
Token tracking utility for monitoring LLM usage across the system.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class TokenUsage:
    """Track token usage for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""
    call_type: str = ""  # e.g., "manager", "analyst", "searcher", etc.
    
    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens


class TokenTracker:
    """Global token tracker for monitoring all LLM calls in the system."""
    
    def __init__(self):
        self.usage_history: list[TokenUsage] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_tokens: int = 0
        self.model_breakdown: Dict[str, TokenUsage] = {}
        self.call_type_breakdown: Dict[str, TokenUsage] = {}
    
    def track_usage(self, input_tokens: int, output_tokens: int, 
                   model_name: str, call_type: str = "unknown") -> None:
        """Track a single LLM call's token usage."""
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=model_name,
            call_type=call_type
        )
        
        self.usage_history.append(usage)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += usage.total_tokens
        
        # Update model breakdown
        if model_name not in self.model_breakdown:
            self.model_breakdown[model_name] = TokenUsage(model_name=model_name)
        
        model_usage = self.model_breakdown[model_name]
        model_usage.input_tokens += input_tokens
        model_usage.output_tokens += output_tokens
        model_usage.total_tokens += usage.total_tokens
        
        # Update call type breakdown
        if call_type not in self.call_type_breakdown:
            self.call_type_breakdown[call_type] = TokenUsage(call_type=call_type)
        
        call_usage = self.call_type_breakdown[call_type]
        call_usage.input_tokens += input_tokens
        call_usage.output_tokens += output_tokens
        call_usage.total_tokens += usage.total_tokens
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of token usage."""
        return {
            "total_calls": len(self.usage_history),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "model_breakdown": {
                model: {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "call_count": len([u for u in self.usage_history if u.model_name == model])
                }
                for model, usage in self.model_breakdown.items()
            },
            "call_type_breakdown": {
                call_type: {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "call_count": len([u for u in self.usage_history if u.call_type == call_type])
                }
                for call_type, usage in self.call_type_breakdown.items()
            }
        }
    
    def log_summary(self) -> None:
        """Log a detailed summary of token usage."""
        summary = self.get_summary()
        
        logger.success("=" * 80)
        logger.success("TOKEN USAGE SUMMARY")
        logger.success("=" * 80)
        
        logger.info(f"Total LLM Calls: {summary['total_calls']}")
        logger.info(f"Total Input Tokens: {summary['total_input_tokens']:,}")
        logger.info(f"Total Output Tokens: {summary['total_output_tokens']:,}")
        logger.info(f"Total Tokens: {summary['total_tokens']:,}")
        
        logger.info("\n" + "-" * 40 + " BY MODEL " + "-" * 40)
        for model, usage in summary['model_breakdown'].items():
            logger.info(f"Model: {model}")
            logger.info(f"  Calls: {usage['call_count']}")
            logger.info(f"  Input Tokens: {usage['input_tokens']:,}")
            logger.info(f"  Output Tokens: {usage['output_tokens']:,}")
            logger.info(f"  Total Tokens: {usage['total_tokens']:,}")
            logger.info("")
        
        logger.info("\n" + "-" * 40 + " BY CALL TYPE " + "-" * 40)
        for call_type, usage in summary['call_type_breakdown'].items():
            logger.info(f"Call Type: {call_type}")
            logger.info(f"  Calls: {usage['call_count']}")
            logger.info(f"  Input Tokens: {usage['input_tokens']:,}")
            logger.info(f"  Output Tokens: {usage['output_tokens']:,}")
            logger.info(f"  Total Tokens: {usage['total_tokens']:,}")
            logger.info("")
        
        logger.success("=" * 80)
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.usage_history.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.model_breakdown.clear()
        self.call_type_breakdown.clear()


# Global token tracker instance
global_token_tracker = TokenTracker()


def get_token_tracker() -> TokenTracker:
    """Get the global token tracker instance."""
    return global_token_tracker

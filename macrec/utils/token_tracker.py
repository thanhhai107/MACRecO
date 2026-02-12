"""Token tracking utility for monitoring LLM usage across the system.
"""

import time
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
    timestamp: float = 0.0  # Unix timestamp when this call was made
    
    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class TokenTracker:
    """Global token tracker for monitoring all LLM calls in the system."""
    
    def __init__(self):
        self.usage_history: list[TokenUsage] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_tokens: int = 0
        self.model_breakdown: Dict[str, TokenUsage] = {}
        self.call_type_breakdown: Dict[str, TokenUsage] = {}
        
        # Duration tracking
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.sample_durations: list[float] = []  # Duration for each sample
        self.agent_durations: Dict[str, list[float]] = {}  # Duration per agent call
        self.current_sample_start: Optional[float] = None
    
    def start_tracking(self) -> None:
        """Mark the start of tracking session."""
        self.start_time = time.time()
    
    def end_tracking(self) -> None:
        """Mark the end of tracking session."""
        self.end_time = time.time()
    
    def start_sample(self) -> None:
        """Mark the start of processing a sample."""
        self.current_sample_start = time.time()
    
    def end_sample(self) -> None:
        """Mark the end of processing a sample."""
        if self.current_sample_start is not None:
            duration = time.time() - self.current_sample_start
            self.sample_durations.append(duration)
            self.current_sample_start = None
    
    def track_agent_duration(self, agent_name: str, duration: float) -> None:
        """Track duration of an agent call.
        
        Args:
            agent_name: Name of the agent (e.g., 'manager', 'analyst')
            duration: Duration of the call in seconds
        """
        if agent_name not in self.agent_durations:
            self.agent_durations[agent_name] = []
        self.agent_durations[agent_name].append(duration)
    
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
        """Get a comprehensive summary of token usage and duration."""
        # Calculate duration statistics
        total_duration = None
        if self.start_time is not None and self.end_time is not None:
            total_duration = self.end_time - self.start_time
        elif self.start_time is not None:
            total_duration = time.time() - self.start_time
        
        avg_sample_duration = sum(self.sample_durations) / len(self.sample_durations) if self.sample_durations else 0
        min_sample_duration = min(self.sample_durations) if self.sample_durations else 0
        max_sample_duration = max(self.sample_durations) if self.sample_durations else 0
        
        agent_stats = {}
        for agent, durations in self.agent_durations.items():
            agent_stats[agent] = {
                "total_calls": len(durations),
                "total_duration": sum(durations),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
            }
        
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
            },
            "duration": {
                "total_duration": total_duration,
                "sample_count": len(self.sample_durations),
                "total_sample_duration": sum(self.sample_durations),
                "avg_sample_duration": avg_sample_duration,
                "min_sample_duration": min_sample_duration,
                "max_sample_duration": max_sample_duration,
                "agent_breakdown": agent_stats
            }
        }
    
    def log_summary(self, metrics_data: Optional[Dict[str, Any]] = None) -> None:
        """Log a detailed summary of token usage, duration, and optionally metrics.
        
        Args:
            metrics_data: Optional dictionary containing evaluation metrics to include in summary
        """
        summary = self.get_summary()
        duration_info = summary['duration']
        
        logger.info("")
        logger.success("=" * 80)
        logger.success("RUN STATISTICS SUMMARY")
        logger.success("=" * 80)
        
        # Duration Statistics
        logger.info("")
        logger.info("-" * 40 + " DURATION STATISTICS " + "-" * 40)
        if duration_info['total_duration'] is not None:
            total_sec = duration_info['total_duration']
            hours = int(total_sec // 3600)
            minutes = int((total_sec % 3600) // 60)
            seconds = total_sec % 60
            logger.info(f"Total Duration: {hours:02d}h {minutes:02d}m {seconds:06.3f}s ({total_sec:.3f}s)")
        else:
            logger.info("Total Duration: Not available")
        
        if duration_info['sample_count'] > 0:
            logger.info(f"\nSample Processing:")
            logger.info(f"  Total Samples: {duration_info['sample_count']}")
            logger.info(f"  Total Sample Duration: {duration_info['total_sample_duration']:.3f}s")
            logger.info(f"  Average per Sample: {duration_info['avg_sample_duration']:.3f}s")
            logger.info(f"  Min Sample Duration: {duration_info['min_sample_duration']:.3f}s")
            logger.info(f"  Max Sample Duration: {duration_info['max_sample_duration']:.3f}s")
        
        # Token Statistics
        logger.info("")
        logger.info("-" * 40 + " TOKEN STATISTICS " + "-" * 40)
        logger.info(f"Total LLM Calls: {summary['total_calls']}")
        logger.info(f"Total Input Tokens: {summary['total_input_tokens']:,}")
        logger.info(f"Total Output Tokens: {summary['total_output_tokens']:,}")
        logger.info(f"Total Tokens: {summary['total_tokens']:,}")
        
        if duration_info['total_duration'] and duration_info['total_duration'] > 0:
            tokens_per_sec = summary['total_tokens'] / duration_info['total_duration']
            logger.info(f"Tokens per Second: {tokens_per_sec:.2f}")
        
        logger.info("\n" + "-" * 40 + " BY MODEL " + "-" * 40)
        for model, usage in summary['model_breakdown'].items():
            logger.info(f"Model: {model}")
            logger.info(f"  Calls: {usage['call_count']}")
            logger.info(f"  Input Tokens: {usage['input_tokens']:,}")
            logger.info(f"  Output Tokens: {usage['output_tokens']:,}")
            logger.info(f"  Total Tokens: {usage['total_tokens']:,}")
            logger.info("")
        
        # Separate LLM-level and invocation-level statistics
        logger.info("\n" + "-" * 40 + " LLM CALL STATISTICS " + "-" * 40)
        logger.info("(Individual API calls to language models)")
        logger.info("")
        
        # Group LLM-level stats (those without '_invocation' suffix)
        llm_agents = {k: v for k, v in summary['call_type_breakdown'].items() if not k.endswith('_invocation')}
        llm_durations = {k: v for k, v in duration_info['agent_breakdown'].items() if not k.endswith('_invocation')}
        
        for agent_name in sorted(llm_agents.keys()):
            token_usage = llm_agents[agent_name]
            logger.info(f"{agent_name}:")
            logger.info(f"  LLM API Calls: {token_usage['call_count']}")
            logger.info(f"  Input Tokens: {token_usage['input_tokens']:,}")
            logger.info(f"  Output Tokens: {token_usage['output_tokens']:,}")
            logger.info(f"  Total Tokens: {token_usage['total_tokens']:,}")
            
            # Duration stats if available
            if agent_name in llm_durations:
                dur_stats = llm_durations[agent_name]
                logger.info(f"  Total Duration: {dur_stats['total_duration']:.3f}s")
                logger.info(f"  Average Duration: {dur_stats['avg_duration']:.3f}s")
                logger.info(f"  Min Duration: {dur_stats['min_duration']:.3f}s")
                logger.info(f"  Max Duration: {dur_stats['max_duration']:.3f}s")
                
                if dur_stats['total_duration'] > 0:
                    tps = token_usage['total_tokens'] / dur_stats['total_duration']
                    logger.info(f"  Tokens/Second: {tps:.2f}")
            
            logger.info("")
        
        # Show invocation-level stats separately
        invocation_agents = {k: v for k, v in duration_info['agent_breakdown'].items() if k.endswith('_invocation')}
        
        if invocation_agents:
            logger.info("\n" + "-" * 40 + " AGENT INVOCATION STATISTICS " + "-" * 40)
            logger.info("(Full agent execution including multiple LLM calls and tool usage)")
            logger.info("")
            
            for agent_name in sorted(invocation_agents.keys()):
                dur_stats = invocation_agents[agent_name]
                # Remove '_invocation' suffix for cleaner display
                display_name = agent_name.replace('_invocation', '')
                logger.info(f"{display_name}:")
                logger.info(f"  Total Invocations: {dur_stats['total_calls']}")
                logger.info(f"  Total Duration: {dur_stats['total_duration']:.3f}s")
                logger.info(f"  Average Duration: {dur_stats['avg_duration']:.3f}s")
                logger.info(f"  Min Duration: {dur_stats['min_duration']:.3f}s")
                logger.info(f"  Max Duration: {dur_stats['max_duration']:.3f}s")
                
                # Calculate average LLM calls per invocation if possible
                base_agent_name = agent_name.replace('_invocation', '')
                if base_agent_name in llm_agents:
                    llm_calls = llm_agents[base_agent_name]['call_count']
                    invocations = dur_stats['total_calls']
                    avg_llm_per_inv = llm_calls / invocations if invocations > 0 else 0
                    logger.info(f"  Avg LLM Calls per Invocation: {avg_llm_per_inv:.2f}")
                
                logger.info("")
        
        # Metrics Statistics (if provided)
        if metrics_data:
            logger.info("")
            logger.info("-" * 40 + " EVALUATION METRICS " + "-" * 40)
            for metric_name, metric_value in metrics_data.items():
                if isinstance(metric_value, dict):
                    logger.info(f"{metric_name}:")
                    for k, v in metric_value.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                elif isinstance(metric_value, (int, float)):
                    logger.info(f"{metric_name}: {metric_value:.4f}")
                else:
                    logger.info(f"{metric_name}: {metric_value}")
        
        logger.success("=" * 80)
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.usage_history.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.model_breakdown.clear()
        self.call_type_breakdown.clear()
        
        # Reset duration tracking
        self.start_time = None
        self.end_time = None
        self.sample_durations.clear()
        self.agent_durations.clear()
        self.current_sample_start = None


# Global token tracker instance
global_token_tracker = TokenTracker()

# Alias for convenience
token_tracker = global_token_tracker


def get_token_tracker() -> TokenTracker:
    """Get the global token tracker instance."""
    return global_token_tracker

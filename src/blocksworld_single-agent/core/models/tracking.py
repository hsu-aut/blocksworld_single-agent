"""
Token Tracking - Simple usage monitoring and cost calculation.
"""

from .config import AVAILABLE_MODELS


class TokenTrackingCallback:
    """
    Simple token tracking for all LLM providers with cost calculation.
    
    Tracks token usage and calculates estimated API costs based on 
    pricing information from AVAILABLE_MODELS.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
    
    def update_from_usage(self, prompt_tokens: int, completion_tokens: int):
        """Update token counts and calculate costs."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        
        # Calculate cost
        if self.model_name in AVAILABLE_MODELS:
            model_config = AVAILABLE_MODELS[self.model_name]
            if 'input_cost_per_1m' in model_config and 'output_cost_per_1m' in model_config:
                input_cost = (prompt_tokens / 1_000_000) * model_config['input_cost_per_1m']
                output_cost = (completion_tokens / 1_000_000) * model_config['output_cost_per_1m']
                self.total_cost += input_cost + output_cost
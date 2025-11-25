"""
Model Factory - Model creation and API key management.

This module handles the creation of LangChain chat models with proper
provider configuration and API key management.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from .config import AVAILABLE_MODELS, MODELS_BY_PROVIDER_AND_SERIES


def get_api_key_for_provider(provider: str) -> str:
    """
    Get the appropriate API key for the given LLM provider.
    
    Args:
        provider (str): The LLM provider name (openai, google-genai, anthropic)
        
    Returns:
        str: The API key from environment variables
        
    Raises:
        ValueError: If provider is unsupported or API key is missing
        
    Example:
        >>> get_api_key_for_provider("openai")
        "sk-..."
    """
    
    # Map provider names to their corresponding environment variable names
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "google-genai": "GOOGLE_API_KEY", 
        "anthropic": "ANTHROPIC_API_KEY"
    }
    
    # Validate provider is supported
    if provider not in key_mapping:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Retrieve API key from environment
    api_key = os.getenv(key_mapping[provider])
    if not api_key:
        raise ValueError(f"Missing API key: {key_mapping[provider]}")
    
    return api_key


def create_model(model_name: str, temperature: float = 0.1):
    """
    Create a LangChain chat model using init_chat_model with environment configuration.
    
    This function initializes any supported LLM provider (OpenAI, Google, Anthropic)
    with the appropriate API key and configuration from environment variables.
    
    Args:
        model_name (str): Model identifier from AVAILABLE_MODELS (e.g., "gpt-4o-mini")
        temperature (float, optional): Sampling temperature (0.0-1.0). Defaults to 0.1.
        
    Returns:
        ChatModel: Initialized LangChain chat model ready for use
        
    Raises:
        ValueError: If model_name is not in AVAILABLE_MODELS
        ValueError: If required API key is missing from environment
        
    Example:
        >>> model = create_model("gpt-4o-mini", temperature=0.5)
        >>> model = create_model("claude-3-haiku")
    """
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Validate model is available
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Choose from: {list(AVAILABLE_MODELS.keys())}")
    
    # Extract model configuration
    model_config = AVAILABLE_MODELS[model_name]
    provider = model_config["provider"]          # e.g., "openai", "google-genai"
    model_name_actual = model_config["model_name"]  # e.g., "gpt-4o-mini"
    
    # Initialize the model with provider-specific configuration
    model_kwargs = {}

    if provider == "openai":
        # Enable stream_usage for OpenAI models (Anthropic/Google provide usage_metadata automatically)
        model_kwargs["stream_usage"] = True
        
        # OpenAI reasoning models (O-series) require temperature=1.0 and don't allow custom temperature
        openai_models = MODELS_BY_PROVIDER_AND_SERIES.get("OpenAI", {})
        o_series_models = openai_models.get("O-Series", {})
        if model_name in o_series_models:
            temperature = 1.0
        if model_name == "o3":
            model_kwargs["disable_streaming"] = True
    # elif provider == "anthropic":
    #     # Add thinking capabilities for Claude models that support it
    #     if "sonnet-4" in model_name_actual or "claude-3-7" in model_name_actual:
    #         model_kwargs["thinking"] = {
    #             "type": "enabled", 
    #             "budget_tokens": 10000
    #         }
    
    return init_chat_model(
        model_provider=provider,
        model=model_name_actual,
        api_key=get_api_key_for_provider(provider),
        temperature=temperature,
        **model_kwargs
    )
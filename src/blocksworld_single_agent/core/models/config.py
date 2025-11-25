"""
Model Configuration - Available models and their settings.

This module defines all supported LLM models with their provider information,
descriptions, and pricing data.

Each model entry contains:
- provider: The LangChain provider name (openai, google-genai, anthropic)
- model_name: The internal model identifier used by the provider
- description: Human-readable description of the model
- input_cost_per_1m: Cost per 1 million input tokens in USD
- output_cost_per_1m: Cost per 1 million output tokens in USD

Pricing is updated as of January 2025 and should be verified against
current provider pricing pages for accuracy.
"""

from typing import Dict, Any

# Models organized by provider and series for better structure and maintainability  
MODELS_BY_PROVIDER_AND_SERIES: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "OpenAI": {
        "GPT-4.1 Series": {
            "gpt-4.1": {
                "provider": "openai",
                "model_name": "gpt-4.1",
                "description": "Latest GPT-4 model with extended context and enhanced reasoning",
                "input_cost_per_1m": 2.00,
                "output_cost_per_1m": 8.00
            },
            "gpt-4.1-mini": {
                "provider": "openai",
                "model_name": "gpt-4.1-mini",
                "description": "Compact version of GPT-4.1 for faster, cost-effective tasks",
                "input_cost_per_1m": 0.40,
                "output_cost_per_1m": 1.60
            },
            "gpt-4.1-nano": {
                "provider": "openai",
                "model_name": "gpt-4.1-nano",
                "description": "Ultra-lightweight GPT-4 model for high-volume applications",
                "input_cost_per_1m": 0.10,
                "output_cost_per_1m": 0.40
            }
        },
        
        "GPT-4o Series": {
            "gpt-4o": {
                "provider": "openai",
                "model_name": "gpt-4o",
                "description": "Multimodal GPT-4 model with vision and advanced reasoning",
                "input_cost_per_1m": 2.50,
                "output_cost_per_1m": 10.00
            },
            "chatgpt-4o-latest": {
                "provider": "openai", 
                "model_name": "chatgpt-4o-latest",
                "description": "Latest ChatGPT-4o model with current improvements",
                "input_cost_per_1m": 5.00,
                "output_cost_per_1m": 15.00
            },
            "gpt-4o-mini": {
                "provider": "openai",
                "model_name": "gpt-4o-mini",
                "description": "Faster, cheaper GPT-4 model",
                "input_cost_per_1m": 0.15,
                "output_cost_per_1m": 0.60
            }
        },
        
        "O-Series": {
            "o1": {
                "provider": "openai",
                "model_name": "o1",
                "description": "OpenAI's flagship reasoning model with deep thinking capabilities",
                "input_cost_per_1m": 15.00,
                "output_cost_per_1m": 60.00
            },
            "o1-pro": {
                "provider": "openai",
                "model_name": "o1-pro",
                "description": "Premium O-series model with maximum reasoning performance",
                "input_cost_per_1m": 150.00,
                "output_cost_per_1m": 600.00
            },
            "o3-pro": {
                "provider": "openai",
                "model_name": "o3-pro",
                "description": "Professional-grade O3 model with enhanced reasoning and safety",
                "input_cost_per_1m": 20.00,
                "output_cost_per_1m": 90.00
            },
            "o3": {
                "provider": "openai",
                "model_name": "o3",
                "description": "Latest O-series model with improved reasoning and safety alignment",
                "input_cost_per_1m": 2.00,
                "output_cost_per_1m": 8.00
            },
            "o4-mini": {
                "provider": "openai",
                "model_name": "o4-mini",
                "description": "Compact reasoning model with enhanced math and coding capabilities",
                "input_cost_per_1m": 1.10,
                "output_cost_per_1m": 4.40
            },
            "o3-mini": {
                "provider": "openai",
                "model_name": "o3-mini",
                "description": "Efficient O-series model optimized for reasoning tasks",
                "input_cost_per_1m": 1.10,
                "output_cost_per_1m": 4.40
            },
            "o1-mini": {
                "provider": "openai",
                "model_name": "o1-mini",
                "description": "Original mini O-series model for lightweight reasoning",
                "input_cost_per_1m": 1.10,
                "output_cost_per_1m": 4.40
            }
        },
        
        "Legacy": {
            "gpt-4-turbo": {
                "provider": "openai", 
                "model_name": "gpt-4-turbo",
                "description": "High-performance GPT-4 with 128K context window",
                "input_cost_per_1m": 10.00,
                "output_cost_per_1m": 30.00
            },
            "gpt-4": {
                "provider": "openai", 
                "model_name": "gpt-4",
                "description": "Original GPT-4 model with strong reasoning capabilities",
                "input_cost_per_1m": 30.00,
                "output_cost_per_1m": 60.00
            },
            "gpt-3.5-turbo": {
                "provider": "openai", 
                "model_name": "gpt-3.5-turbo",
                "description": "Fast and cost-effective legacy model",
                "input_cost_per_1m": 0.50,
                "output_cost_per_1m": 1.50
            }
        }
    },
    
    "Google": {
        "Gemini 2.5": {
            "gemini-2.5-pro": {
                "provider": "google-genai",
                "model_name": "gemini-2.5-pro",
                "description": "Gemini 2.5 Pro with advanced thinking and reasoning capabilities",
                "input_cost_per_1m": 1.25,
                "output_cost_per_1m": 10.00
            },
            "gemini-2.5-flash": {
                "provider": "google-genai",
                "model_name": "gemini-2.5-flash",
                "description": "Optimized Gemini 2.5 for speed and efficiency",
                "input_cost_per_1m": 0.30,
                "output_cost_per_1m": 2.50
            },
            "gemini-2.5-flash-lite": {
                "provider": "google-genai",
                "model_name": "gemini-2.5-flash-lite",
                "description": "Lightweight version of Gemini 2.5 Flash for cost-effective tasks",
                "input_cost_per_1m": 0.30,
                "output_cost_per_1m": 0.40
            }
        },
        
        "Gemini 2.0": {
            "gemini-2.0-flash": {
                "provider": "google-genai",
                "model_name": "gemini-2.0-flash",
                "description": "Fast Gemini 2.0 model with multimodal capabilities",
                "input_cost_per_1m": 0.10,
                "output_cost_per_1m": 0.40
            },
            "gemini-2.0-flash-lite": {
                "provider": "google-genai",
                "model_name": "gemini-2.0-flash-lite",
                "description": "Ultra-efficient Gemini 2.0 for high-volume processing",
                "input_cost_per_1m": 0.075,
                "output_cost_per_1m": 0.30
            }
        }
    },
    
    "Anthropic": {
        "Claude 4": {
            "claude-opus-4.1": {
                "provider": "anthropic",
                "model_name": "claude-opus-4-1-20250805",
                "description": "Most capable Claude model with advanced reasoning and thinking",
                "input_cost_per_1m": 15.00,
                "output_cost_per_1m": 75.00
            },
            "claude-opus-4": {
                "provider": "anthropic",
                "model_name": "claude-opus-4-20250514",
                "description": "Top-tier Claude 4 model for complex reasoning and analysis",
                "input_cost_per_1m": 15.00,
                "output_cost_per_1m": 75.00
            },
            "claude-sonnet-4": {
                "provider": "anthropic",
                "model_name": "claude-sonnet-4-20250514",
                "description": "Claude Sonnet 4 with extended thinking and enhanced reasoning",
                "input_cost_per_1m": 3.00,
                "output_cost_per_1m": 15.00
            }
        },
        
        "Claude 3": {
            "claude-3-7-sonnet": {
                "provider": "anthropic",
                "model_name": "claude-3-7-sonnet-20250219",
                "description": "Claude 3.7 Sonnet with improved performance and capabilities",
                "input_cost_per_1m": 3.00,
                "output_cost_per_1m": 15.00
            },
            "claude-3-5-sonnet": {
                "provider": "anthropic",
                "model_name": "claude-3-5-sonnet-20240620",
                "description": "Claude 3.5 Sonnet with balanced performance and reasoning",
                "input_cost_per_1m": 3.00,
                "output_cost_per_1m": 15.00
            },
            "claude-3-5-haiku": {
                "provider": "anthropic",
                "model_name": "claude-3-5-haiku-20241022",
                "description": "Claude 3.5 Haiku optimized for speed and cost-effectiveness",
                "input_cost_per_1m": 0.80,
                "output_cost_per_1m": 4.00
            },
            "claude-3-haiku": {
                "provider": "anthropic", 
                "model_name": "claude-3-haiku-20240307",
                "description": "Fast Claude model",
                "input_cost_per_1m": 0.25,
                "output_cost_per_1m": 1.25
            }
        }
    }
}

# Create flat structures for backward compatibility
MODELS_BY_PROVIDER: Dict[str, Dict[str, Dict[str, Any]]] = {}
for provider, series_dict in MODELS_BY_PROVIDER_AND_SERIES.items():
    MODELS_BY_PROVIDER[provider] = {}
    for series, models in series_dict.items():
        MODELS_BY_PROVIDER[provider].update(models)

# Flatten the structure for backward compatibility
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {}
for provider_models in MODELS_BY_PROVIDER.values():
    AVAILABLE_MODELS.update(provider_models)
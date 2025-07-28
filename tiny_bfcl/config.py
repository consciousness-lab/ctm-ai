"""
Configuration module for Tiny BFCL
Contains model configurations, constants, and settings
"""

import os
from pathlib import Path
from typing import Any, Dict

# Constants
RETRY_LIMIT = 3
RETRY_DELAY = 65  # Delay in seconds

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / 'result'
DATA_PATH = PROJECT_ROOT / 'data'

# Ensure directories exist
RESULT_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    'gpt-4o-mini-2024-07-18': {
        'handler': 'openai',
        'model_name': 'gpt-4o-mini-2024-07-18',
        'display_name': 'GPT-4o-mini-2024-07-18',
        'is_fc_model': True,
        'temperature': 0.001,
        'org': 'OpenAI',
        'license': 'Proprietary',
        'input_price': 0.15,
        'output_price': 0.6,
    },
    'claude-3-5-sonnet-20241022': {
        'handler': 'anthropic',
        'model_name': 'claude-3-5-sonnet-20241022',
        'display_name': 'Claude-3-5-Sonnet-20241022',
        'is_fc_model': True,
        'temperature': 0.001,
        'url': 'https://www.anthropic.com/news/claude-3-5-sonnet',
        'org': 'Anthropic',
        'license': 'Proprietary',
        'input_price': 3.0,
        'output_price': 15.0,
    },
    'gemini-1.5-flash': {
        'handler': 'gemini',
        'model_name': 'gemini-1.5-flash',
        'display_name': 'Gemini 1.5 Flash',
        'is_fc_model': True,
        'temperature': 0.001,
        'url': 'https://ai.google.dev/models/gemini',
        'org': 'Google',
        'license': 'Proprietary',
        'input_price': 0.075,
        'output_price': 0.3,
    },
}

# Test categories
TEST_CATEGORIES = {
    'simple': 'Simple function calling tests',
    'complex': 'Complex multi-function tests',
    'math': 'Mathematical computation tests',
    'weather': 'Weather API tests',
    'search': 'Search functionality tests',
}

# Default test cases
DEFAULT_TEST_CASES = [
    {
        'id': 'simple_101',
        'category': 'simple',
        'function': [
            {
                'name': 'get_weather',
                'description': 'Get weather information for a specified city',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'city': {'type': 'string', 'description': 'City name'}
                    },
                    'required': ['city'],
                },
            }
        ],
        'question': [
            [
                {
                    'role': 'user',
                    'content': 'Please help me get weather information for Beijing',
                }
            ]
        ],
    },
    {
        'id': 'simple_102',
        'category': 'simple',
        'function': [
            {
                'name': 'calculate',
                'description': 'Perform mathematical calculations',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'expression': {
                            'type': 'string',
                            'description': 'Mathematical expression',
                        }
                    },
                    'required': ['expression'],
                },
            }
        ],
        'question': [
            [{'role': 'user', 'content': 'Please calculate the result of 15 + 27'}]
        ],
    },
    {
        'id': 'complex_101',
        'category': 'complex',
        'function': [
            {
                'name': 'search_web',
                'description': 'Search the web for information',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'Search query'}
                    },
                    'required': ['query'],
                },
            },
            {
                'name': 'summarize_text',
                'description': 'Summarize given text',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Text to summarize'}
                    },
                    'required': ['text'],
                },
            },
        ],
        'question': [
            [
                {
                    'role': 'user',
                    'content': 'Search for information about AI and summarize the results',
                }
            ]
        ],
    },
]


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration by name"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f'Unsupported model: {model_name}')
    return MODEL_CONFIGS[model_name].copy()


def get_test_cases(category: str = None) -> list:
    """Get test cases, optionally filtered by category"""
    if category is None:
        return DEFAULT_TEST_CASES

    if category not in TEST_CATEGORIES:
        raise ValueError(f'Unknown test category: {category}')

    return [case for case in DEFAULT_TEST_CASES if case.get('category') == category]


def validate_environment():
    """Validate that required environment variables are set"""
    missing_vars = []

    if not os.getenv('OPENAI_API_KEY'):
        missing_vars.append('OPENAI_API_KEY')

    if not os.getenv('ANTHROPIC_API_KEY'):
        missing_vars.append('ANTHROPIC_API_KEY')

    if not os.getenv('GOOGLE_API_KEY'):
        missing_vars.append('GOOGLE_API_KEY')

    if missing_vars:
        raise EnvironmentError(
            f'Missing required environment variables: {", ".join(missing_vars)}'
        )

    return True

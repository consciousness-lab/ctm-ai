"""
Core module for Tiny BFCL
Contains main TinyBFCL class and inference logic
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .config import MODEL_CONFIGS, RETRY_DELAY, RETRY_LIMIT, validate_environment
from .model_handler import AnthropicHandler, GeminiHandler, OpenAIHandler


class TinyBFCL:
    """Main Tiny BFCL class for running evaluations"""

    def __init__(self):
        """Initialize Tiny BFCL"""
        self.handlers = {}

    def build_handler(self, model_name: str, temperature: float = 0.001):
        """Build handler for a specific model"""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f'Unsupported model: {model_name}')

        config = MODEL_CONFIGS[model_name].copy()
        config['temperature'] = temperature

        if config['handler'] == 'openai':
            return OpenAIHandler(config)
        elif config['handler'] == 'anthropic':
            return AnthropicHandler(config)
        elif config['handler'] == 'gemini':
            return GeminiHandler(config)
        else:
            raise ValueError(f'Unsupported handler type: {config["handler"]}')

    def multi_threaded_inference(
        self,
        handler,
        test_case: Dict[str, Any],
        include_input_log: bool = False,
        exclude_state_log: bool = False,
    ):
        """Multi-threaded inference with retry logic"""
        retry_count = 0

        while True:
            try:
                result, metadata = handler.inference(
                    deepcopy(test_case), include_input_log, exclude_state_log
                )
                break
            except Exception as e:
                if retry_count < RETRY_LIMIT and (
                    'rate limit reached' in str(e).lower()
                    or (
                        hasattr(e, 'status_code') and (e.status_code in {429, 503, 500})
                    )
                ):
                    print(
                        f'Rate limit reached. Sleeping for {RETRY_DELAY} seconds. Retry {retry_count + 1}/{RETRY_LIMIT}'
                    )
                    time.sleep(RETRY_DELAY)
                    retry_count += 1
                else:
                    print('-' * 100)
                    print(
                        'Error occurred during inference. Maximum retries reached for rate limit or other error. Continuing to next test case.'
                    )
                    print(f'Test case ID: {test_case["id"]}, Error: {str(e)}')
                    traceback.print_exc(limit=10)
                    print('-' * 100)

                    return {
                        'id': test_case['id'],
                        'result': f'Error during inference: {str(e)}',
                        'traceback': traceback.format_exc(),
                    }

        result_to_write = {
            'id': test_case['id'],
            'result': result,
        }
        result_to_write.update(metadata)

        return result_to_write

    def generate_results(
        self,
        model_name: str,
        test_cases: List[Dict[str, Any]],
        num_threads: int = 1,
        result_dir: Optional[str] = None,
    ):
        """Generate results for a model"""
        handler = self.build_handler(model_name)

        futures = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            with tqdm(
                total=len(test_cases), desc=f'Generating results for {model_name}'
            ) as pbar:
                for test_case in test_cases:
                    future = executor.submit(
                        self.multi_threaded_inference,
                        handler,
                        test_case,
                        False,  # include_input_log
                        False,  # exclude_state_log
                    )
                    futures.append(future)

                for future in futures:
                    result = future.result()
                    handler.write(result, result_dir=result_dir, update_mode=True)
                    pbar.update()

    def run_evaluation(
        self,
        models: List[str],
        test_cases: List[Dict[str, Any]],
        num_threads: int = 1,
        result_dir: Optional[str] = None,
    ):
        """Run evaluation for multiple models"""
        print('🚀 Tiny BFCL - Simplified Berkeley Function Call Leaderboard')
        print('=' * 60)

        # Validate environment
        try:
            validate_environment()
        except EnvironmentError as e:
            print(f'Environment validation failed: {e}')
            return

        # Load test data
        print('Loading test data...')
        print(f'Loaded {len(test_cases)} test cases')

        # Run evaluation for each model
        for model_name in models:
            print(f'\nStarting evaluation for model: {model_name}')
            print('-' * 40)

            try:
                self.generate_results(
                    model_name,
                    test_cases,
                    num_threads=num_threads,
                    result_dir=result_dir,
                )
                print(f'✅ {model_name} evaluation completed')
            except Exception as e:
                print(f'❌ {model_name} evaluation failed: {str(e)}')
                traceback.print_exc()

        print('\nAll evaluations completed!')
        print('Results saved in result/ directory')

    def list_models(self):
        """List all supported models"""
        print('Supported Models:')
        print('-' * 40)
        for model_name, config in MODEL_CONFIGS.items():
            print(f'• {model_name} ({config["display_name"]})')
            print(f'  - Handler: {config["handler"]}')
            print(f'  - FC Model: {config["is_fc_model"]}')
            print(f'  - Organization: {config["org"]}')
            print(f'  - License: {config["license"]}')
            print()

    def list_test_categories(self):
        """List all test categories"""
        from .config import DEFAULT_TEST_CASES, TEST_CATEGORIES

        print('Test Categories:')
        print('-' * 40)
        for category, description in TEST_CATEGORIES.items():
            count = len(
                [
                    case
                    for case in DEFAULT_TEST_CASES
                    if case.get('category') == category
                ]
            )
            print(f'• {category}: {description} ({count} test cases)')
        print()

    def get_test_cases(self, category: str = None):
        """Get test cases, optionally filtered by category"""
        from .config import get_test_cases

        return get_test_cases(category)


def create_tiny_bfcl() -> TinyBFCL:
    """Factory function to create TinyBFCL instance"""
    return TinyBFCL()

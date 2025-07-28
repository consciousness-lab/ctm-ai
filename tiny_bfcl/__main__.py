#!/usr/bin/env python3
"""
Main entry point for Tiny BFCL
Run this module directly to execute evaluations
"""

import sys
from typing import List

from .config import get_test_cases
from .core import TinyBFCL


def main():
    """Main function for command line execution"""
    if len(sys.argv) < 2:
        print('Usage: python -m tiny_bfcl <command> [options]')
        print('\nCommands:')
        print('  run <model1> <model2> ...  - Run evaluation for specified models')
        print('  list-models               - List all supported models')
        print('  list-categories           - List all test categories')
        print('  test                      - Run test evaluation with default models')
        print('\nExamples:')
        print(
            '  python -m tiny_bfcl run gpt-4o-mini-2024-07-18 claude-3-5-sonnet-20241022'
        )
        print('  python -m tiny_bfcl list-models')
        print('  python -m tiny_bfcl test')
        return

    command = sys.argv[1]
    bfcl = TinyBFCL()

    if command == 'run':
        if len(sys.argv) < 3:
            print('Error: Please specify at least one model to run')
            return

        models = sys.argv[2:]
        test_cases = get_test_cases()

        print(f'Running evaluation for models: {", ".join(models)}')
        bfcl.run_evaluation(models, test_cases)

    elif command == 'list-models':
        bfcl.list_models()

    elif command == 'list-categories':
        bfcl.list_test_categories()

    elif command == 'test':
        # Run test evaluation with default models
        models = [
            'gpt-4o-mini-2024-07-18',
            'claude-3-5-sonnet-20241022',
            'gemini-1.5-flash',
        ]
        test_cases = get_test_cases('simple')  # Only simple test cases for testing

        print('Running test evaluation with default models...')
        bfcl.run_evaluation(models, test_cases)

    else:
        print(f'Unknown command: {command}')
        print("Use 'python -m tiny_bfcl' to see available commands")


if __name__ == '__main__':
    main()

"""
Base handler class for Tiny BFCL
Abstract base class that all model handlers must inherit from
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseHandler(ABC):
    """Base handler class for all model handlers"""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model_name = model_config['model_name']
        self.temperature = model_config['temperature']
        self.is_fc_model = model_config['is_fc_model']

    def inference(
        self,
        test_entry: Dict[str, Any],
        include_input_log: bool = False,
        exclude_state_log: bool = False,
    ):
        """Execute inference based on model type"""
        if self.is_fc_model:
            return self.inference_single_turn_FC(test_entry, include_input_log)
        else:
            return self.inference_single_turn_prompting(test_entry, include_input_log)

    @abstractmethod
    def inference_single_turn_FC(
        self, test_entry: Dict[str, Any], include_input_log: bool = False
    ):
        """Single turn inference for Function Calling mode"""
        pass

    @abstractmethod
    def inference_single_turn_prompting(
        self, test_entry: Dict[str, Any], include_input_log: bool = False
    ):
        """Single turn inference for Prompting mode"""
        pass

    def write(
        self,
        result: Dict[str, Any],
        result_dir: Optional[str] = None,
        update_mode: bool = False,
    ):
        """Write results to file"""
        if result_dir is None:
            result_dir = Path('result')
        else:
            result_dir = Path(result_dir)

        result_dir.mkdir(parents=True, exist_ok=True)
        model_dir = result_dir / self.model_name
        model_dir.mkdir(exist_ok=True)

        # Write result file
        result_file = model_dir / 'tiny_bfcl_result.json'

        if update_mode and result_file.exists():
            # Update mode: read existing file, update or add results
            with open(result_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)

            # Find and update existing result
            updated = False
            for i, existing_result in enumerate(existing_results):
                if existing_result.get('id') == result['id']:
                    existing_results[i] = result
                    updated = True
                    break

            if not updated:
                existing_results.append(result)

            results_to_write = existing_results
        else:
            # New file or non-update mode
            results_to_write = [result]

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_write, f, indent=2, ensure_ascii=False)

        print(f'Results written to: {result_file}')

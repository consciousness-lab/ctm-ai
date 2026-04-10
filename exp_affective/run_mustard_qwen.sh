#!/bin/bash

python run_unified.py --dataset_name mustard --provider qwen --output unified_mustard_qwen.jsonl
python run_unified_ensemble.py --dataset_name mustard --provider qwen --n_votes 3 --output unified_ensemble_mustard_qwen.jsonl
python run_ensemble.py --dataset_name mustard --provider qwen --output ensemble_mustard_qwen.jsonl
python run_debate.py --dataset_name mustard --provider qwen --rounds 3 --output debate_mustard_qwen.jsonl
python run_orchestra.py --dataset_name mustard --provider qwen --rounds 3 --output orchestra_mustard_qwen.jsonl

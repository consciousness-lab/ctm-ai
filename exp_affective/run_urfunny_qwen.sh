#!/bin/bash

python run_unified.py --dataset_name urfunny --provider qwen --output unified_urfunny_qwen.jsonl
python run_unified_ensemble.py --dataset_name urfunny --provider qwen --n_votes 3 --output unified_ensemble_urfunny_qwen.jsonl
python run_ensemble.py --dataset_name urfunny --provider qwen --output ensemble_urfunny_qwen.jsonl
python run_debate.py --dataset_name urfunny --provider qwen --rounds 3 --output debate_urfunny_qwen.jsonl
python run_orchestra.py --dataset_name urfunny --provider qwen --rounds 3 --output orchestra_urfunny_qwen.jsonl

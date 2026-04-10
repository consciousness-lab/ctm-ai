#!/bin/bash

# mustard
python run_unified.py --dataset_name mustard --provider gemini --output unified_mustard_gemini.jsonl
python run_unified_ensemble.py --dataset_name mustard --provider gemini --n_votes 3 --output unified_ensemble_mustard_gemini.jsonl
python run_ensemble.py --dataset_name mustard --provider gemini --output ensemble_mustard_gemini.jsonl
python run_debate.py --dataset_name mustard --provider gemini --rounds 3 --output debate_mustard_gemini.jsonl
python run_orchestra.py --dataset_name mustard --provider gemini --rounds 3 --output orchestra_mustard_gemini.jsonl

# urfunny
python run_unified.py --dataset_name urfunny --provider gemini --output unified_urfunny_gemini.jsonl
python run_unified_ensemble.py --dataset_name urfunny --provider gemini --n_votes 3 --output unified_ensemble_urfunny_gemini.jsonl
python run_ensemble.py --dataset_name urfunny --provider gemini --output ensemble_urfunny_gemini.jsonl
python run_debate.py --dataset_name urfunny --provider gemini --rounds 3 --output debate_urfunny_gemini.jsonl
python run_orchestra.py --dataset_name urfunny --provider gemini --rounds 3 --output orchestra_urfunny_gemini.jsonl

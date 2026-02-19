"""
Close-domain QA Pipeline
"""

import sys

sys.path.append("..")

import argparse

from ctm_ai.apis import pipeline_runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tool_root_dir", type=str, default="your_tools_path/", required=True, help=""
    )
    parser.add_argument(
        "--max_observation_length",
        type=int,
        default=1024,
        required=False,
        help="maximum observation length",
    )
    parser.add_argument(
        "--observ_compress_method",
        type=str,
        default="truncate",
        choices=["truncate", "filter", "random"],
        required=False,
        help="observation compress method",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ctm",
        required=False,
        help="method for answer generation: CoT@n,Reflexion@n,BFS,DFS,UCT_vote",
    )
    parser.add_argument(
        "--input_query_file", type=str, default="", required=False, help="input path"
    )
    parser.add_argument(
        "--output_answer_file", type=str, default="", required=False, help="output path"
    )
    parser.add_argument(
        "--toolbench_key",
        type=str,
        default="",
        required=False,
        help="your toolbench key to request rapidapi service",
    )
    parser.add_argument(
        "--rapidapi_key",
        type=str,
        default="",
        required=False,
        help="your rapidapi key to request rapidapi service",
    )
    parser.add_argument(
        "--use_rapidapi_key",
        action="store_true",
        help="To use customized rapidapi service or not.",
    )
    parser.add_argument(
        "--api_customization", action="store_true", help="To use customized api or not."
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        default="",
        required=False,
        help="your openai key to request openai service",
    )
    parser.add_argument(
        "--test", type=bool, default=False, help="To use test mode or not."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes to run.",
    )
    parser.add_argument(
        "--query_id",
        type=int,
        default=None,
        help="Specific query ID to run. If not specified, run all queries.",
    )
    parser.add_argument(
        "--num_additional_questions",
        type=int,
        default=3,
        help="Number of additional questions to generate per processor (default: 3, set to 0 to disable).",
    )
    parser.add_argument(
        "--ctm_name",
        type=str,
        default=None,
        help="CTM config name to load from ctm_conf/{ctm_name}_config.json (e.g. 'tooluse_ctm').",
    )

    args = parser.parse_args()

    pipeline_runner = pipeline_runner(args)
    pipeline_runner.run(num_processes=args.num_processes)

import json
import os
import random
import multiprocessing

from termcolor import colored

from .api_manager import contain, get_white_list, rapidapi_wrapper
from .api_server import standardize


def method_converter(
    env,
    query,
    num_additional_questions: int = 3,
    ctm_name: str = None,
):
    """
    Convert method using CTM.
    
    Args:
        env: The API environment/manager
        query: The query to process
        num_additional_questions: Number of additional questions to generate (default: 3)
        ctm_name: Optional CTM config name to load from ctm_conf/{ctm_name}_config.json
    """
    from ctm_ai.ctms import ToolCTM

    ctm = ToolCTM(
        api_manager=env,
        ctm_name=ctm_name,
        num_additional_questions=num_additional_questions,
    )
    answer = ctm(
        query=query,
    )
    return answer


def run_single_task(
    method,
    query_id,
    data_dict,
    args,
    output_dir_path,
    tool_des,
    process_id=0,
    callbacks=None,
    server=None,
):
    """Run a single task using CTM"""
    if server is None:
        server = False
    if callbacks is None:
        if server:
            print("Warning: no callbacks are defined for server mode")
        callbacks = []

    splits = output_dir_path.split("/")
    os.makedirs("/".join(splits[:-1]), exist_ok=True)
    os.makedirs("/".join(splits), exist_ok=True)
    output_file_path = os.path.join(output_dir_path, f"{query_id}_{method}.json")
    if (not server) and os.path.exists(output_file_path):
        return

    # Set up logging for this query_id
    from ctm_ai.utils import set_iteration_log_file

    log_file = os.path.join(output_dir_path, f"ctm_iterations_{query_id}.jsonl")
    set_iteration_log_file(str(query_id), log_file)

    [callback.on_tool_retrieval_start() for callback in callbacks]
    env = rapidapi_wrapper(data_dict, tool_des, args, process_id=process_id)

    if (
        env.funcs_to_all_info is None
        or len(env.funcs_to_all_info) == 0
        or len(env.tool_names) == 0
        or len(env.functions) == 0
    ):
        if process_id == 0:
            print(
                colored(
                    f"[process({process_id})]Skipping query {query_id}: no available tools/functions",
                    "yellow",
                )
            )
        return None

    [callback.on_tool_retrieval_end(tools=env.functions) for callback in callbacks]
    query = data_dict["query"]

    if process_id == 0:
        print(
            colored(
                f"[process({process_id})]now playing {query}, with {len(env.functions)} APIs",
                "green",
            )
        )

    [
        callback.on_request_start(
            user_input=query,
            method=method,
        )
        for callback in callbacks
    ]

    num_additional_questions = getattr(args, "num_additional_questions", 3)
    ctm_name = getattr(args, "ctm_name", None)

    answer, weight_score, parsed_answer = method_converter(
        env=env,
        query=query,
        num_additional_questions=num_additional_questions,
        ctm_name=ctm_name,
    )

    [
        callback.on_request_end(
            chain=answer,
            outputs=answer,
        )
        for callback in callbacks
    ]
    if output_dir_path is not None:
        with open(output_file_path, "w") as writer:
            data = {}
            data["query"] = query
            data["query_id"] = query_id
            data["final_answer"] = answer
            data["weight_score"] = weight_score
            data["parsed_answer"] = parsed_answer
            json.dump(data, writer, indent=2)
            success = True
            print(colored(f"[process({process_id})]valid={success}", "green"))
    return answer


class pipeline_runner:
    def __init__(self, args, process_id=0, server=False, test=False):
        self.args = args
        self.process_id = process_id
        self.server = server
        if not self.server:
            self.task_list = self.generate_task_list()
        else:
            self.task_list = []
        self.test = test

    def get_args(self):
        return self.args

    def generate_task_list(self):
        args = self.args
        query_dir = args.input_query_file
        answer_dir = args.output_answer_file
        if not os.path.exists(answer_dir):
            os.mkdir(answer_dir)
        method = args.method
        white_list = get_white_list(args.tool_root_dir)
        task_list = []
        querys = json.load(open(query_dir, "r"))
        for query_id, data_dict in enumerate(querys):
            if "query_id" in data_dict:
                query_id = data_dict["query_id"]

            if hasattr(args, "query_id") and args.query_id is not None:
                if query_id != args.query_id:
                    continue

            if "api_list" in data_dict:
                origin_tool_names = [
                    standardize(cont["tool_name"]) for cont in data_dict["api_list"]
                ]
                tool_des = contain(origin_tool_names, white_list)
                if not tool_des:
                    continue
                tool_des = [
                    [cont["standard_tool_name"], cont["description"]]
                    for cont in tool_des
                ]
            else:
                tool_des = None
            task_list.append(
                (
                    method,
                    query_id,
                    data_dict,
                    args,
                    answer_dir,
                    tool_des,
                )
            )
        if self.args.test:
            task_list = [task_list[0]]
            print("========================TEST MODE=========================")
            print(f"task_list: {task_list}")

        if hasattr(self.args, "query_id") and self.args.query_id is not None:
            if not task_list:
                print(f"Warning: No task found with query_id={self.args.query_id}")
            else:
                print(
                    f"Found {len(task_list)} task(s) with query_id={self.args.query_id}"
                )

        return task_list

    def run(self, num_processes=1):
        """Run the pipeline with the task list"""
        random.seed(42)
        random.shuffle(self.task_list)
        print(f"total tasks: {len(self.task_list)}")

        new_task_list = []
        for task in self.task_list:
            out_dir_path = task[-2]
            query_id = task[2]
            output_file_path = os.path.join(
                out_dir_path, f"{query_id}_{self.args.method}.json"
            )
            if not os.path.exists(output_file_path):
                new_task_list.append(task)

        task_list = new_task_list
        print(f"undo tasks: {len(task_list)}")

        if num_processes > 1:
            print(f"Running in parallel with {num_processes} processes.")
            tasks_for_pool = [
                task + (i % num_processes,) for i, task in enumerate(task_list)
            ]
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.starmap(run_single_task, tasks_for_pool)
        else:
            for k, task in enumerate(task_list):
                print(
                    f"process[{self.process_id}] doing task {k}/{len(task_list)}: real_task_id_{task[1]}"
                )
                run_single_task(*task, process_id=self.process_id)

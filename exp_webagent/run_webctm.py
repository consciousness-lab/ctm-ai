"""
# run all categories
python run_webctm.py --all --use_screenshot True

# run specified categories
python run_webctm.py --categories shopping shopping_admin wikipedia map reddit --use_screenshot True

# enable saving screenshots with SOM overlay
python run_webctm.py --all --use_screenshot True --save_som True

# disable saving screenshots
python run_webctm.py --all --use_screenshot True --save_screenshot False
"""

import argparse
import json
import traceback
from pathlib import Path

from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
from browsergym.experiments.loop import (
    StepInfo,
    _is_debugging,
    _send_chat_info,
    logger,
    save_package_versions,
)
from ctm_webagent import CTMAgentArgs
from task_by_category import gitlab, map, reddit, shopping, shopping_admin, wikipedia

CATEGORY_TASKS = {
    'gitlab': gitlab,
    'map': map,
    'reddit': reddit,
    'shopping': shopping,
    'shopping_admin': shopping_admin,
    'wikipedia': wikipedia,
}

ALL_CATEGORIES = list(CATEGORY_TASKS.keys())


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch run CTM web agent experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all categories
  python run_batch.py --all --use_screenshot True

  # Run specified categories
  python run_batch.py --categories gitlab shopping --use_screenshot True

  # Run single category
  python run_batch.py --categories gitlab --use_screenshot True

  # Specify custom results directory
  python run_batch.py --categories shopping --use_screenshot True --results_dir ./results_1222/ctm-ai/shopping

  # Start from a specific index (e.g., start from the 5th task)
  python run_batch.py --categories shopping --use_screenshot True --start_idx 5

  # Run specific tasks by ID (must specify a single category)
  python run_batch.py --categories shopping_admin --task_ids 109 115 4 --use_screenshot True
        """,
    )

    # Category selection
    category_group = parser.add_mutually_exclusive_group(required=True)
    category_group.add_argument(
        '--all',
        action='store_true',
        help='Run all categories',
    )
    category_group.add_argument(
        '--categories',
        nargs='+',
        choices=ALL_CATEGORIES,
        help=f'Specify categories to run, options: {", ".join(ALL_CATEGORIES)}',
    )

    # CTM configuration parameters
    parser.add_argument(
        '--ctm_name',
        type=str,
        default='web',
        help='CTM configuration name',
    )
    parser.add_argument(
        '--visual_effects',
        type=str2bool,
        default=True,
        help='Add visual effects when performing actions',
    )
    parser.add_argument(
        '--use_html',
        type=str2bool,
        default=True,
        help="Use HTML in agent's observation space",
    )
    parser.add_argument(
        '--use_axtree',
        type=str2bool,
        default=True,
        help="Use AXTree in agent's observation space",
    )
    parser.add_argument(
        '--use_screenshot',
        type=str2bool,
        default=True,
        help="Use screenshot in agent's observation space",
    )

    # Environment parameters
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10,
        help='Maximum steps per task',
    )
    parser.add_argument(
        '--headless',
        type=str2bool,
        default=True,
        help='Whether to run browser in headless mode',
    )

    # Experiment saving parameters
    parser.add_argument(
        '--save_screenshot',
        type=str2bool,
        default=True,
        help='Whether to save screenshots for each step',
    )
    parser.add_argument(
        '--save_som',
        type=str2bool,
        default=True,
        help='Whether to save screenshots with SOM (Set of Marks) overlay',
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results_',
        help='Base directory to save experiment results',
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Start index for tasks in each category (default: 0, starts from the beginning)',
    )
    parser.add_argument(
        '--task_ids',
        type=int,
        nargs='+',
        default=None,
        help='Specific task IDs to run (must specify a single category when using this option)',
    )
    parser.add_argument(
        '--action_timeout',
        type=float,
        default=None,
        help='Action timeout in seconds. If an action takes longer than this time (from extraction to execution end), the task will be marked as timeout. Action time excludes model inference time.',
    )

    return parser.parse_args()


def run_with_action_timeout(exp_args: ExpArgs, action_timeout: float = None):
    """Run experiment with action timeout checking.

    Args:
        exp_args: Experiment arguments
        action_timeout: Maximum time in seconds for action execution (from extraction to end).
                       If None, uses default ExpArgs.run() behavior.
    """
    if action_timeout is None:
        # Use default behavior
        exp_args.run()
        return

    # Custom run with action timeout checking
    exp_args._set_logger()
    save_package_versions(exp_args.exp_dir)

    episode_info = []
    env, step_info, err_msg, stack_trace = None, None, None, None
    try:
        logger.info(f'Running experiment {exp_args.exp_name} in:\n  {exp_args.exp_dir}')
        agent = exp_args.agent_args.make_agent()
        logger.debug(f'Agent created.')

        env = exp_args.env_args.make_env(
            action_mapping=agent.action_set.to_python_code,
            exp_dir=exp_args.exp_dir,
        )

        logger.debug(f'Environment created.')

        step_info = StepInfo(step=0)
        episode_info = [step_info]
        step_info.from_reset(
            env,
            seed=exp_args.env_args.task_seed,
            obs_preprocessor=agent.obs_preprocessor,
        )
        logger.debug(f'Environment reset.')

        while not step_info.is_done:
            logger.debug(f'Starting step {step_info.step}.')
            action = step_info.from_action(agent)
            logger.debug(f'Agent chose action:\n {action}')

            if action is None:
                step_info.truncated = True

            step_info.save_step_info(
                exp_args.exp_dir,
                save_screenshot=exp_args.save_screenshot,
                save_som=exp_args.save_som,
            )
            logger.debug(f'Step info saved.')

            _send_chat_info(env.unwrapped.chat, action, step_info.agent_info)
            logger.debug(f'Chat info sent.')

            if action is None:
                logger.debug(f'Agent returned None action. Ending episode.')
                break

            step_info = StepInfo(step=step_info.step + 1)
            episode_info.append(step_info)

            logger.debug(f'Sending action to environment.')
            step_info.from_step(env, action, obs_preprocessor=agent.obs_preprocessor)
            logger.debug(f'Environment stepped.')

            # Check action execution time (from extraction to execution end)
            # action_exec_start is set in pre_step(), action_exec_stop is set in post_step()
            # The time excludes model inference time (which happens in from_action)
            if (
                step_info.profiling.action_exec_start > 0
                and step_info.profiling.action_exec_stop > 0
            ):
                action_exec_time = (
                    step_info.profiling.action_exec_stop
                    - step_info.profiling.action_exec_start
                )

                if action_exec_time > action_timeout:
                    logger.warning(
                        f'Action timeout exceeded: {action_exec_time:.2f}s > {action_timeout}s. '
                        f'Marking task as timeout.'
                    )
                    step_info.truncated = True
                    step_info.stats = step_info.stats or {}
                    step_info.stats['action_timeout'] = True
                    step_info.stats['action_exec_time'] = action_exec_time
                    step_info.stats['action_timeout_limit'] = action_timeout
                    break

    except Exception as e:
        err_msg = f'Exception uncaught by agent or environment in task {exp_args.env_args.task_name}.\n{type(e).__name__}:\n{e}'
        stack_trace = traceback.format_exc()

        exp_args.err_msg = err_msg
        exp_args.stack_trace = stack_trace

        logger.warning(err_msg + '\n' + stack_trace)
        if _is_debugging() and exp_args.enable_debug:
            logger.warning('Debug mode is enabled. Raising the error.')
            raise

    finally:
        try:
            if step_info is not None:
                step_info.save_step_info(
                    exp_args.exp_dir,
                    save_screenshot=exp_args.save_screenshot,
                    save_som=exp_args.save_som,
                )
        except Exception as e:
            logger.error(f'Error while saving step info in the finally block: {e}')
        try:
            if (
                not err_msg
                and len(episode_info) > 0
                and not (episode_info[-1].terminated or episode_info[-1].truncated)
            ):
                e = KeyboardInterrupt('Early termination??')
                err_msg = f'Exception uncaught by agent or environment in task {exp_args.env_args.task_name}.\n{type(e).__name__}:\n{e}'
            logger.info(f'Saving summary info.')
            exp_args.save_summary_info(
                episode_info, exp_args.exp_dir, err_msg, stack_trace
            )
        except Exception as e:
            logger.error(f'Error while saving summary info in the finally block: {e}')
        try:
            if env is not None:
                env.close()
        except Exception as e:
            logger.error(
                f'Error while closing the environment in the finally block: {e}'
            )
        try:
            exp_args._unset_logger()
        except Exception as e:
            logger.error(f'Error while unsetting the logger in the finally block: {e}')


def run_single_task(
    task_id: int,
    category: str,
    result_base_dir: Path,
    agent_args: CTMAgentArgs,
    max_steps: int,
    headless: bool,
    save_screenshot: bool = True,
    save_som: bool = False,
    action_timeout: float = None,
):
    """Run a single task using CTM agent (same logic as run_web.py)"""
    task_name = f'webarena.{task_id}'
    print(f'\n{"=" * 80}')
    print(f'Running task: {task_name} (category: {category})')
    print(f'{"=" * 80}')

    # Create environment arguments (same as run_web.py)
    env_args = EnvArgs(
        task_name=task_name,
        task_seed=None,
        max_steps=max_steps,
        headless=headless,
        wait_for_user_message=False,
    )

    # Create experiment arguments with CTM agent (same as run_web.py)
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,  # This uses CTMAgentArgs which creates CTM agent
        save_screenshot=save_screenshot,
        save_som=save_som,
    )

    # Use category-specific result folder (already created in main)
    category_result_dir = result_base_dir / category
    if not category_result_dir.exists():
        category_result_dir.mkdir(parents=True, exist_ok=True)

    exp_args.prepare(str(category_result_dir))

    try:
        # Run the experiment with action timeout checking
        run_with_action_timeout(exp_args, action_timeout=action_timeout)
        exp_result = get_exp_result(exp_args.exp_dir)
        exp_record = exp_result.get_exp_record()

        # Extract detailed step information
        action_history = []
        step_details = []
        try:
            steps_info = exp_result.steps_info
            exp_dir_path = Path(exp_args.exp_dir)

            for step_info in steps_info:
                if step_info.action is not None:
                    action_history.append(step_info.action)

                # Extract detailed step information
                step_detail = {
                    'step': step_info.step,
                    'action': step_info.action,
                }

                # Try to get observation from step_info first
                obs = None
                if hasattr(step_info, 'obs') and step_info.obs:
                    obs = step_info.obs
                else:
                    # If not available, try to load from saved step file
                    step_file = exp_dir_path / f'step_{step_info.step}.pkl.gz'
                    if step_file.exists():
                        try:
                            import gzip
                            import pickle

                            with gzip.open(step_file, 'rb') as f:
                                saved_step = pickle.load(f)
                                if hasattr(saved_step, 'obs') and saved_step.obs:
                                    obs = saved_step.obs
                        except Exception as e:
                            pass  # Silently continue if loading fails

                # Extract reference_url and query from observation
                if obs and isinstance(obs, dict):
                    # Get current page URL
                    if 'open_pages_urls' in obs and 'active_page_index' in obs:
                        urls = obs['open_pages_urls']
                        active_idx = obs['active_page_index']
                        if urls and 0 <= active_idx < len(urls):
                            step_detail['reference_url'] = urls[active_idx]

                    # Get query/goal
                    if 'goal_object' in obs and obs['goal_object']:
                        if (
                            isinstance(obs['goal_object'], list)
                            and len(obs['goal_object']) > 0
                        ):
                            goal = obs['goal_object'][0]
                            if isinstance(goal, dict) and 'text' in goal:
                                step_detail['query'] = goal['text']

                step_details.append(step_detail)

        except Exception as e:
            print(f'  Warning: Failed to extract step details: {e}')
            import traceback

            traceback.print_exc()

        # Update summary_info.json with detailed information
        summary_info_path = Path(exp_args.exp_dir) / 'summary_info.json'
        if summary_info_path.exists():
            try:
                with open(summary_info_path, 'r', encoding='utf-8') as f:
                    summary_info = json.load(f)

                # Add detailed step information
                summary_info['action_history'] = action_history
                summary_info['step_details'] = step_details

                # Save updated summary_info
                with open(summary_info_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_info, f, indent=4, ensure_ascii=False)

                print(
                    f'  Added {len(action_history)} actions and {len(step_details)} step details to summary_info.json'
                )

                # Print query and reference_url for each step
                print(f'\n  {"-" * 76}')
                print(f'  Query and Reference URLs:')
                print(f'  {"-" * 76}')

                # Get query from first step or summary_info
                query = None
                if step_details and len(step_details) > 0:
                    query = step_details[0].get('query')
                if not query:
                    query = (
                        summary_info.get('query')
                        or summary_info.get('task_name')
                        or '未找到 query'
                    )

                print(f'  Query: {query}')
                print(f'\n  Step Details:')

                # Print each step's reference_url
                for step_detail in step_details:
                    step_num = step_detail.get('step', 'N/A')
                    reference_url = step_detail.get(
                        'reference_url', '未找到 reference_url'
                    )
                    action = step_detail.get('action', 'N/A')
                    print(f'    步骤 {step_num}: {reference_url}')
                    print(f'      动作: {action}')

                print(f'  {"-" * 76}')
            except Exception as e:
                print(f'  Warning: Failed to update summary_info.json: {e}')
        else:
            print(f'  Warning: summary_info.json not found at {summary_info_path}')

        print(f'\nTask {task_name} completed!')
        print(f'Result directory: {exp_args.exp_dir}')
        for key, val in exp_record.items():
            if key not in ['exp_dir']:  # exp_dir already printed
                print(f'  {key}: {val}')

        return True, exp_record
    except Exception as e:
        print(f'\nTask {task_name} failed: {e}')
        import traceback

        traceback.print_exc()
        return False, {'error': str(e)}


def main():
    args = parse_args()

    # Validate task_ids usage
    if args.task_ids is not None:
        if args.all:
            raise ValueError(
                'Cannot use --task_ids with --all. Please specify a single category with --categories.'
            )
        if args.categories is None or len(args.categories) == 0:
            raise ValueError(
                'Must specify a category with --categories when using --task_ids.'
            )
        if len(args.categories) > 1:
            raise ValueError(
                'Cannot use --task_ids with multiple categories. Please specify a single category.'
            )

    # Determine categories to run
    if args.all:
        categories_to_run = ALL_CATEGORIES
    else:
        categories_to_run = args.categories

    print(f'Will run the following categories: {", ".join(categories_to_run)}')
    print(f'Total number of categories: {len(categories_to_run)}')

    if args.task_ids:
        print(f'Will run specific task IDs: {args.task_ids}')

    # Create CTM agent arguments (same as run_web.py)
    # This ensures each task uses CTM agent with the same configuration
    agent_args = CTMAgentArgs(
        ctm_name=args.ctm_name,
        chat_mode=False,  # Same as run_web.py
        demo_mode='default' if args.visual_effects else 'off',  # Same as run_web.py
        use_html=args.use_html,
        use_axtree=args.use_axtree,
        use_screenshot=args.use_screenshot,
    )

    # Base result directory - each category will have its own subfolder
    result_base_dir = Path(args.results_dir)
    result_base_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    total_tasks = 0
    completed_tasks = 0
    failed_tasks = 0
    category_stats = {}

    # Iterate through each category
    for category in categories_to_run:
        task_ids = CATEGORY_TASKS[category]

        # Create category-specific result directory
        category_result_dir = result_base_dir / category
        category_result_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n{"#" * 80}')
        print(f'Category: {category} (total {len(task_ids)} tasks)')
        print(f'Results will be saved to: {category_result_dir.absolute()}')

        # Determine which tasks to run
        if args.task_ids is not None:
            # Run only specified task IDs
            task_ids_to_run = []
            invalid_task_ids = []
            for task_id in args.task_ids:
                if task_id in task_ids:
                    task_ids_to_run.append(task_id)
                else:
                    invalid_task_ids.append(task_id)

            if invalid_task_ids:
                print(
                    f"  Warning: The following task IDs are not in category '{category}': {invalid_task_ids}"
                )

            if not task_ids_to_run:
                print(
                    f"  Warning: No valid task IDs found for category '{category}', skipping"
                )
                category_stats[category] = {
                    'total': 0,
                    'completed': 0,
                    'failed': 0,
                }
                continue

            print(f'  Running specified task IDs: {task_ids_to_run}')
        else:
            # Apply start_idx to slice the task list
            start_idx = args.start_idx
            if start_idx > 0:
                if start_idx >= len(task_ids):
                    print(
                        f'  Warning: start_idx ({start_idx}) >= total tasks ({len(task_ids)}), skipping this category'
                    )
                    category_stats[category] = {
                        'total': len(task_ids),
                        'completed': 0,
                        'failed': 0,
                    }
                    continue
                task_ids_to_run = task_ids[start_idx:]
                print(
                    f'  Starting from index {start_idx}, will run {len(task_ids_to_run)} tasks'
                )
            else:
                task_ids_to_run = task_ids

        print(f'{"#" * 80}')

        category_completed = 0
        category_failed = 0

        # Iterate through tasks in this category (starting from start_idx)
        for task_id in task_ids_to_run:
            total_tasks += 1
            success, record = run_single_task(
                task_id=task_id,
                category=category,
                result_base_dir=result_base_dir,
                agent_args=agent_args,
                max_steps=args.max_steps,
                headless=args.headless,
                save_screenshot=args.save_screenshot,
                save_som=args.save_som,
                action_timeout=args.action_timeout,
            )

            if success:
                completed_tasks += 1
                category_completed += 1
            else:
                failed_tasks += 1
                category_failed += 1

        category_stats[category] = {
            'total': len(task_ids_to_run),
            'completed': category_completed,
            'failed': category_failed,
        }

    # Print summary
    print(f'\n{"=" * 80}')
    print('Batch run summary')
    print(f'{"=" * 80}')
    print(f'Total tasks: {total_tasks}')
    print(f'Success: {completed_tasks}')
    print(f'Failed: {failed_tasks}')
    print('\nStatistics by category:')
    for category, stats in category_stats.items():
        print(
            f'  {category}: {stats["completed"]}/{stats["total"]} success, '
            f'{stats["failed"]}/{stats["total"]} failed'
        )
    print(f'\nResults saved in: {result_base_dir.absolute()}')
    print('\nFolder structure:')
    for category in categories_to_run:
        category_dir = result_base_dir / category
        print(f'  {category}/ -> {category_dir.absolute()}')
    print(f'{"=" * 80}\n')


if __name__ == '__main__':
    main()

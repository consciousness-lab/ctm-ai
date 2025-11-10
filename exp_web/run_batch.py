"""
# run all categories
python run_batch.py --all --use_screenshot True

# run specified categories
python run_batch.py --categories shopping shopping_admin wikipedia map reddit --use_screenshot True
"""

import argparse
from pathlib import Path

from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
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

    return parser.parse_args()


def run_single_task(
    task_id: int,
    category: str,
    result_base_dir: Path,
    agent_args: CTMAgentArgs,
    max_steps: int,
    headless: bool,
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
    )

    # Use category-specific result folder (already created in main)
    category_result_dir = result_base_dir / category
    if not category_result_dir.exists():
        category_result_dir.mkdir(parents=True, exist_ok=True)

    exp_args.prepare(str(category_result_dir))

    try:
        # Run the experiment (same as run_web.py)
        exp_args.run()
        exp_result = get_exp_result(exp_args.exp_dir)
        exp_record = exp_result.get_exp_record()

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

    # Determine categories to run
    if args.all:
        categories_to_run = ALL_CATEGORIES
    else:
        categories_to_run = args.categories

    print(f'Will run the following categories: {", ".join(categories_to_run)}')
    print(f'Total number of categories: {len(categories_to_run)}')

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
    result_base_dir = Path('./results_1111')
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
        print(f'{"#" * 80}')

        category_completed = 0
        category_failed = 0

        # Iterate through all tasks in this category
        for task_id in task_ids:
            total_tasks += 1
            success, record = run_single_task(
                task_id=task_id,
                category=category,
                result_base_dir=result_base_dir,
                agent_args=agent_args,
                max_steps=args.max_steps,
                headless=args.headless,
            )

            if success:
                completed_tasks += 1
                category_completed += 1
            else:
                failed_tasks += 1
                category_failed += 1

        category_stats[category] = {
            'total': len(task_ids),
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

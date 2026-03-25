"""
Gemini 2.5 Flash Lite baseline runner for WebArena Lite.

Runs all tasks in parallel using ProcessPoolExecutor.
Uses Gemini's OpenAI-compatible endpoint so no new dependencies needed.

Usage:
  # Single task test
  python run_baseline_gemini.py --categories shopping_admin --task_ids 4 --num_workers 1

  # All non-map categories in parallel
  python run_baseline_gemini.py --categories shopping shopping_admin gitlab reddit wikipedia --num_workers 5

  # All categories
  python run_baseline_gemini.py --all --num_workers 5
"""

import argparse
import base64
import dataclasses
import io
import json
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import openai
from PIL import Image

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import AbstractAgentArgs, Agent, get_exp_result
from browsergym.experiments.loop import EnvArgs, ExpArgs
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task loading (same pattern as run_webctm.py)
# ---------------------------------------------------------------------------
_TASKS_JSON = Path(__file__).parent / 'test_webarena_lite.raw.json'


def _load_category_tasks(json_path: Path) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        all_tasks = json.load(f)
    groups: dict = {}
    for task in all_tasks:
        for site in task['sites']:
            groups.setdefault(site, []).append(task)
    return groups


CATEGORY_TASKS = _load_category_tasks(_TASKS_JSON)
ALL_CATEGORIES = sorted(CATEGORY_TASKS.keys())


# ---------------------------------------------------------------------------
# Gemini baseline agent
# ---------------------------------------------------------------------------
def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')
    with io.BytesIO() as buffer:
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/jpeg;base64,{image_base64}'


class GeminiBaselineAgent(Agent):
    """Baseline agent using Gemini 2.5 Flash Lite via OpenAI-compatible API."""

    def obs_preprocessor(self, obs: dict) -> dict:
        return {
            'chat_messages': obs['chat_messages'],
            'screenshot': obs['screenshot'],
            'goal_object': obs['goal_object'],
            'last_action': obs['last_action'],
            'last_action_error': obs['last_action_error'],
            'open_pages_urls': obs['open_pages_urls'],
            'open_pages_titles': obs['open_pages_titles'],
            'active_page_index': obs['active_page_index'],
            'axtree_txt': flatten_axtree_to_str(obs['axtree_object']),
            'pruned_html': prune_html(flatten_dom_to_str(obs['dom_object'])),
        }

    def __init__(
        self,
        model_name: str = 'gemini-2.5-flash-lite',
        use_axtree: bool = True,
        use_html: bool = False,
        use_screenshot: bool = True,
        demo_mode: str = 'off',
        task_info: str = '',
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.use_axtree = use_axtree
        self.use_html = use_html
        self.use_screenshot = use_screenshot
        self.task_info = task_info

        self.client = openai.OpenAI(
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
            api_key=os.environ['GEMINI_API_KEY'],
        )

        self.action_set = HighLevelActionSet(
            subsets=['chat', 'tab', 'nav', 'bid', 'infeas'],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode,
        )

        self.action_history = []

    def get_action(self, obs: dict) -> tuple[str, dict]:
        system_msgs = []
        user_msgs = []

        # System prompt
        system_msgs.append(
            {
                'type': 'text',
                'text': """\
# Instructions

Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
            }
        )

        # Goal
        assert obs['goal_object'], 'The goal is missing.'
        user_msgs.append({'type': 'text', 'text': '# Goal\n'})
        user_msgs.extend(obs['goal_object'])

        # Task type info
        if self.task_info:
            user_msgs.append(
                {
                    'type': 'text',
                    'text': f'# Task Info\n\n{self.task_info}\n\n',
                }
            )

        # Open tabs
        user_msgs.append({'type': 'text', 'text': '# Currently open tabs\n'})
        for page_index, (page_url, page_title) in enumerate(
            zip(obs['open_pages_urls'], obs['open_pages_titles'])
        ):
            user_msgs.append(
                {
                    'type': 'text',
                    'text': f'Tab {page_index}'
                    f'{" (active tab)" if page_index == obs["active_page_index"] else ""}\n'
                    f'  Title: {page_title}\n'
                    f'  URL: {page_url}\n',
                }
            )

        # AXTree
        if self.use_axtree:
            user_msgs.append(
                {
                    'type': 'text',
                    'text': f'# Current page Accessibility Tree\n\n{obs["axtree_txt"]}\n\n',
                }
            )

        # HTML
        if self.use_html:
            user_msgs.append(
                {
                    'type': 'text',
                    'text': f'# Current page DOM\n\n{obs["pruned_html"]}\n\n',
                }
            )

        # Screenshot
        if self.use_screenshot:
            user_msgs.append({'type': 'text', 'text': '# Current page Screenshot\n'})
            user_msgs.append(
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': image_to_jpg_base64_url(obs['screenshot']),
                        'detail': 'auto',
                    },
                }
            )

        # Action space
        user_msgs.append(
            {
                'type': 'text',
                'text': f"""\
# Action Space

{self.action_set.describe(with_long_description=False, with_examples=True)}

Here are examples of actions with chain-of-thought reasoning:

I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
```click("12")```

I found the information requested by the user, I will send it to the chat.
```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

""",
            }
        )

        # Action history
        if self.action_history:
            user_msgs.append({'type': 'text', 'text': '# History of past actions\n'})
            user_msgs.extend(
                [{'type': 'text', 'text': f'\n{action}\n'} for action in self.action_history]
            )

            if obs['last_action_error']:
                user_msgs.append(
                    {
                        'type': 'text',
                        'text': f'# Error message from last action\n\n{obs["last_action_error"]}\n\n',
                    }
                )

        # Next action prompt
        user_msgs.append(
            {
                'type': 'text',
                'text': '# Next action\n\nYou will now think step by step and produce your next best action. '
                'Reflect on your past actions, any resulting error message, and the current state of the page '
                'before deciding on your next action.\n',
            }
        )

        # Call Gemini via OpenAI-compatible API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system_msgs},
                {'role': 'user', 'content': user_msgs},
            ],
        )
        action = response.choices[0].message.content

        self.action_history.append(action)
        return action, {}


@dataclasses.dataclass
class GeminiBaselineAgentArgs(AbstractAgentArgs):
    model_name: str = 'gemini-2.5-flash-lite'
    use_axtree: bool = True
    use_html: bool = False
    use_screenshot: bool = True
    demo_mode: str = 'off'
    task_info: str = ''

    def make_agent(self):
        return GeminiBaselineAgent(
            model_name=self.model_name,
            use_axtree=self.use_axtree,
            use_html=self.use_html,
            use_screenshot=self.use_screenshot,
            demo_mode=self.demo_mode,
            task_info=self.task_info,
        )


# ---------------------------------------------------------------------------
# Single task runner (runs in a subprocess)
# ---------------------------------------------------------------------------
def run_single_task(
    task_record: dict,
    category: str,
    result_base_dir: str,
    agent_args_dict: dict,
    max_steps: int,
    headless: bool,
):
    """Run a single WebArena Lite task. Designed to be called in a subprocess."""
    # Must import and register tasks in each subprocess
    import browsergym.webarenalite  # noqa: F401 — registers webarenalite tasks

    old_task_id = task_record['old_task_id']
    task_name = f'webarenalite.{old_task_id}'

    print(f'[{category}] Starting task: {task_name}')

    # Build task type info from task_record
    sites = ', '.join(task_record.get('sites', []))
    eval_types = ', '.join(task_record.get('eval', {}).get('eval_types', []))
    start_url = task_record.get('start_url', '')
    intent_template = task_record.get('intent_template', '')
    task_info_parts = [
        f'Site(s): {sites}',
        f'Evaluation type: {eval_types}',
        f'Start URL pattern: {start_url}',
    ]
    if intent_template:
        task_info_parts.append(f'Task template: {intent_template}')
    task_info_str = '\n'.join(task_info_parts)

    # Reconstruct agent args from dict (dataclasses aren't always pickle-friendly)
    agent_args_dict = dict(agent_args_dict)  # copy
    agent_args_dict['task_info'] = task_info_str
    agent_args = GeminiBaselineAgentArgs(**agent_args_dict)

    env_args = EnvArgs(
        task_name=task_name,
        task_seed=None,
        max_steps=max_steps,
        headless=headless,
        wait_for_user_message=False,
    )

    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
        save_screenshot=True,
        save_som=False,
    )

    category_result_dir = Path(result_base_dir) / category
    category_result_dir.mkdir(parents=True, exist_ok=True)
    exp_args.prepare(str(category_result_dir))

    try:
        exp_args.run()
        exp_result = get_exp_result(exp_args.exp_dir)
        exp_record = exp_result.get_exp_record()
        cum_reward = exp_record.get('cum_reward', 0)
        print(f'[{category}] Task {task_name} done — reward={cum_reward}')
        return {
            'task_name': task_name,
            'category': category,
            'old_task_id': old_task_id,
            'success': True,
            'cum_reward': cum_reward,
            'exp_dir': exp_args.exp_dir,
        }
    except Exception as e:
        print(f'[{category}] Task {task_name} FAILED: {e}')
        traceback.print_exc()
        return {
            'task_name': task_name,
            'category': category,
            'old_task_id': old_task_id,
            'success': False,
            'cum_reward': 0,
            'error': str(e),
        }


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Gemini 2.5 Flash Lite baseline on WebArena Lite (parallel)',
    )

    cat_group = parser.add_mutually_exclusive_group(required=True)
    cat_group.add_argument('--all', action='store_true', help='Run all categories')
    cat_group.add_argument(
        '--categories',
        nargs='+',
        choices=ALL_CATEGORIES,
        help=f'Categories to run: {", ".join(ALL_CATEGORIES)}',
    )

    parser.add_argument('--task_ids', type=int, nargs='+', default=None, help='Specific old_task_ids to run')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of parallel workers (default: 5)')
    parser.add_argument('--max_steps', type=int, default=15, help='Max steps per task (default: 15)')
    parser.add_argument('--headless', type=str2bool, default=True, help='Run browser headless (default: True)')
    parser.add_argument('--use_screenshot', type=str2bool, default=True, help='Use screenshot (default: True)')
    parser.add_argument('--use_axtree', type=str2bool, default=True, help='Use AXTree (default: True)')
    parser.add_argument('--use_html', type=str2bool, default=False, help='Use HTML (default: False)')
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results_baseline_gemini',
        help='Results directory (default: ./results_baseline_gemini)',
    )
    parser.add_argument('--model_name', type=str, default='gemini-2.5-flash-lite', help='Gemini model name')

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate
    if args.task_ids and args.all:
        raise ValueError('Cannot use --task_ids with --all')

    # Check API keys
    if 'GEMINI_API_KEY' not in os.environ:
        raise EnvironmentError('GEMINI_API_KEY not set. Export it before running.')
    if 'OPENAI_API_KEY' not in os.environ:
        print('WARNING: OPENAI_API_KEY not set. Some BrowserGym internals may need it.')

    categories = ALL_CATEGORIES if args.all else args.categories
    print(f'Categories: {", ".join(categories)}')
    print(f'Model: {args.model_name}')
    print(f'Workers: {args.num_workers}')
    print(f'Max steps: {args.max_steps}')
    print(f'Results dir: {args.results_dir}')

    # Collect all tasks to run
    tasks_to_run = []
    for category in categories:
        task_records = CATEGORY_TASKS.get(category, [])
        if args.task_ids is not None:
            task_records = [t for t in task_records if t['old_task_id'] in args.task_ids]
        for t in task_records:
            tasks_to_run.append((t, category))

    # Deduplicate by (old_task_id, category) in case of overlaps
    seen = set()
    deduped = []
    for t, cat in tasks_to_run:
        key = (t['old_task_id'], cat)
        if key not in seen:
            seen.add(key)
            deduped.append((t, cat))
    tasks_to_run = deduped

    print(f'Total tasks: {len(tasks_to_run)}')
    if not tasks_to_run:
        print('No tasks to run.')
        return

    # Agent args as dict for pickling across processes
    agent_args_dict = {
        'model_name': args.model_name,
        'use_axtree': args.use_axtree,
        'use_html': args.use_html,
        'use_screenshot': args.use_screenshot,
        'demo_mode': 'off',
    }

    result_base_dir = str(Path(args.results_dir))
    Path(result_base_dir).mkdir(parents=True, exist_ok=True)

    # Run tasks in parallel
    results = []
    start_time = datetime.now()
    print(f'\nStarting at {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for task_record, category in tasks_to_run:
            future = executor.submit(
                run_single_task,
                task_record=task_record,
                category=category,
                result_base_dir=result_base_dir,
                agent_args_dict=agent_args_dict,
                max_steps=args.max_steps,
                headless=args.headless,
            )
            futures[future] = (task_record['old_task_id'], category)

        for future in as_completed(futures):
            task_id, category = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f'[{category}] Task {task_id} executor error: {e}')
                results.append({
                    'task_name': f'webarenalite.{task_id}',
                    'category': category,
                    'old_task_id': task_id,
                    'success': False,
                    'cum_reward': 0,
                    'error': str(e),
                })

    end_time = datetime.now()
    elapsed = end_time - start_time

    # Aggregate results
    print('\n' + '=' * 80)
    print('RESULTS SUMMARY')
    print('=' * 80)

    # Per-category breakdown
    cat_results = {}
    for r in results:
        cat = r['category']
        cat_results.setdefault(cat, []).append(r)

    total_tasks = len(results)
    total_successes = sum(1 for r in results if r.get('cum_reward', 0) > 0)
    total_errors = sum(1 for r in results if not r.get('success', False))

    for cat in sorted(cat_results.keys()):
        cat_list = cat_results[cat]
        cat_successes = sum(1 for r in cat_list if r.get('cum_reward', 0) > 0)
        cat_total = len(cat_list)
        rate = cat_successes / cat_total if cat_total > 0 else 0
        print(f'  {cat:20s}: {cat_successes}/{cat_total} = {rate:.3f}')

    overall_rate = total_successes / total_tasks if total_tasks > 0 else 0
    print(f'\n  {"OVERALL":20s}: {total_successes}/{total_tasks} = {overall_rate:.3f}')
    print(f'  Errors (crashed):    {total_errors}')
    print(f'  Elapsed time:        {elapsed}')
    print('=' * 80)

    # Save summary to JSON
    summary_path = Path(result_base_dir) / 'baseline_summary.json'
    summary = {
        'model': args.model_name,
        'total_tasks': total_tasks,
        'total_successes': total_successes,
        'overall_success_rate': overall_rate,
        'total_errors': total_errors,
        'elapsed_seconds': elapsed.total_seconds(),
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'per_category': {
            cat: {
                'tasks': len(cat_results[cat]),
                'successes': sum(1 for r in cat_results[cat] if r.get('cum_reward', 0) > 0),
                'success_rate': sum(1 for r in cat_results[cat] if r.get('cum_reward', 0) > 0)
                / len(cat_results[cat])
                if len(cat_results[cat]) > 0
                else 0,
            }
            for cat in sorted(cat_results.keys())
        },
        'task_results': results,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSummary saved to: {summary_path}')


if __name__ == '__main__':
    main()

import argparse


from ctm_webagent import CTMAgentArgs

from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CTM web agent experiment with hyperparameters."
    )
    parser.add_argument(
        "--ctm_name",
        type=str,
        default="web",
        help="CTM configuration name.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="webarena.310",
        help="Name of the BrowserGym task to run (e.g., webarena.310, miniwob.click-test).",
    )
    parser.add_argument(
        "--visual_effects",
        type=str2bool,
        default=True,
        help="Add visual effects when the agent performs actions.",
    )
    parser.add_argument(
        "--use_html",
        type=str2bool,
        default=True,
        help="Use HTML in the agent's observation space.",
    )
    parser.add_argument(
        "--use_axtree",
        type=str2bool,
        default=True,
        help="Use AXTree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=True,
        help="Use screenshot in the agent's observation space.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    agent_args = CTMAgentArgs(
        ctm_name=args.ctm_name,
        chat_mode=False,
        demo_mode="default" if args.visual_effects else "off",
        use_html=args.use_html,
        use_axtree=args.use_axtree,
        use_screenshot=args.use_screenshot,
    )

    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=100,
        headless=False,
        wait_for_user_message=False,
    )

    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
    )

    exp_args.prepare("./results")

    exp_args.run()

    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()

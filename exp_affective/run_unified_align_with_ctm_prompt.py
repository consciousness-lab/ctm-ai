"""
Unified baseline using the CTM-merged system prompt (zero-shot, no few-shot).

Per-sample pipeline: single MultimodalAgent call with the merged prompt in the
``system`` role and only the target text (+ full video) in the ``user`` role.

Difference from ``run_unified.py``:

- ``system_prompt`` is taken from ``CTM_MERGED_PROMPTS`` below — a merge of the
  CTM config's language_processor / video_processor / audio_processor system
  prompts with confidence-scoring rubrics stripped (those were designed for
  CTM's uptree fusion and have no role in a single-shot call). This gives the
  unified baseline the same domain knowledge CTM gives each of its specialist
  processors, so "CTM vs unified" measures pipeline structure rather than
  prompt quality. ``dataset_configs.py``'s short ``get_system_prompt()`` is
  NOT used here.
- The merged prompt is passed through ``create_agent(system_prompt=...)`` and
  lands in the real ``system`` role — it is NOT inlined into the user turn —
  so the user content is just the target text alongside the video.

Examples:
python run_unified_align_with_ctm_prompt.py --dataset_name urfunny --provider qwen --model qwen3-omni-flash     --output unified_ctmprompt_urfunny_qwen3omni.jsonl
python run_unified_align_with_ctm_prompt.py --dataset_name urfunny --provider qwen --model qwen3-vl-8b-thinking --output unified_ctmprompt_urfunny_qwen3vl_thinking.jsonl
python run_unified_align_with_ctm_prompt.py --dataset_name urfunny --provider qwen --model qwen3-vl-8b-instruct --output unified_ctmprompt_urfunny_qwen3vl_instruct.jsonl
python run_unified_align_with_ctm_prompt.py --dataset_name mustard  --provider qwen --model qwen3-vl-8b-thinking --output unified_ctmprompt_mustard_qwen3vl_thinking.jsonl
"""

import argparse
import sys
import time
from typing import Dict

import litellm
from dataset_configs import get_dataset_config
from llm_utils import (
    MultimodalAgent,
    StatsTracker,
    build_frames_content,
    build_text_content,
    check_api_key,
    create_agent,
    load_data,
    load_processed_keys,
    load_sample_inputs,
    normalize_label,
    save_result_to_jsonl,
)

sys.path.append('..')

COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30


class FramesOnlyMultimodalAgent(MultimodalAgent):
    """Variant of :class:`MultimodalAgent` that always sends extracted video
    frames instead of the raw video file.

    Used when we want all models (omni + vl-*) to receive exactly the same
    input modality — text + 8 image frames — so that null rates and decoder
    behavior differences don't contaminate cross-model comparisons. This
    eliminates the server-side "video too short" rejections that affect VL
    models but not omni, and removes omni's audio advantage (since audio
    tracks are never sent).
    """

    AGENT_TYPE = 'frames_only'

    def _build_content(self, query, **kwargs):
        context = kwargs.get('context')
        video_path = kwargs.get('video_path')
        content = build_text_content(query, context)
        content.extend(build_frames_content(video_path, self.provider))
        return content


# Multimodal system prompts merged from the CTM processor prompts for each
# dataset (language + video + audio processors concatenated, scoring/confidence
# rubrics stripped since they were designed for CTM's uptree fusion and have no
# role in a single-shot call). Sources:
#   urfunny: ctm_conf/urfunny_qwen3vl_thinking_v28_config.json (== v28; the
#     instruct_v28 variant has character-identical system_prompts, and
#     URFunnyConfig.DEFAULT_CTM_CONFIGS['qwen'] points to the thinking_v28 one)
#   mustard: ctm_conf/mustard_qwen3vl_thinking_config.json (no v28 variant
#     exists for mustard; thinking/instruct variants have character-identical
#     system_prompts, and MustardConfig.DEFAULT_CTM_CONFIGS['qwen'] points to
#     the thinking one). audio_processor is missing from both mustard configs,
#     so audio is borrowed from ctm_conf/sarcasm_ctm_config.json, matching
#     MustardConfig.MODALITY_FALLBACK_CONFIGS.
CTM_MERGED_PROMPTS: Dict[str, str] = {
    'urfunny': (
        "You are an expert at detecting humor in talks and presentations. "
        "You will receive a text transcript (context sentences followed by a "
        "punchline) together with a video that contains both visual frames and "
        "audio. Your job is to judge whether the punchline is humorous.\n\n"
        "[TEXT] A punchline is HUMOROUS if it uses a recognizable humor "
        "technique with the intent to amuse:\n"
        "- Self-deprecation: speaker mocks themselves in a clearly JOKING way\n"
        "- Ironic reveal: serious buildup then a contradictory, trivial, or "
        "absurd punchline meant as a JOKE\n"
        "- Absurd comparison: comparing wildly different things PURELY for "
        "comic effect\n"
        "- Wordplay or double meanings used for laughs\n"
        "- Incongruity: punchline deliberately subverts expectations in a "
        "funny way\n"
        "- Misdirection: setup implies one thing, the punchline reveals "
        "something completely different and unexpected\n"
        "- Deadpan understatement: a brief, plain, matter-of-fact statement "
        "that is funny BECAUSE of how understated it is after an elaborate "
        "or dramatic setup\n"
        "- Ironic self-reference or circular logic presented as wisdom\n"
        "- Bathetic contrast: a deliberately grand or poetic description "
        "followed by a deliberately mundane punchline, where the speaker "
        "clearly intends the letdown as comedy\n\n"
        "CRITICAL — what is NOT humor (common in TED talks):\n"
        "- Rhetorical contrasts that make a serious point\n"
        "- Technical or factual descriptions, even if surprising or "
        "counterintuitive\n"
        "- Describing something as spectacular/powerful/amazing — enthusiasm, "
        "not comedy\n"
        "- Presenting unexpected data — surprises alone are NOT jokes\n"
        "- Serious observations about society, even if phrased cleverly\n"
        "- Philosophical paradoxes or observations — intellectual insights, "
        "NOT jokes\n"
        "- Using 'literally' for emphasis in a sincere narrative\n"
        "- A straightforward factual progression (she went to school, he "
        "joined the company)\n"
        "- Vivid analogies used for EXPLANATION rather than comedy\n\n"
        "KEY TEST: Does the punchline have comedic INTENT — is the speaker "
        "trying to amuse? Clever phrasing for a serious point is NOT humor.\n\n"
        "[VIDEO] Analyze body language, facial expressions, gestures, and "
        "audience reactions. Look for smiles, laughter, exaggerated gestures, "
        "comedic timing pauses, audience engagement or amusement. NOTE: "
        "deadpan delivery is common — a speaker can deliver humorous content "
        "with a straight face, so a serious expression does not rule out "
        "humor.\n\n"
        "[AUDIO] Listen for two things: (1) the spoken content — the full "
        "setup that leads to the punchline and whether it contains a humor "
        "technique; (2) AUDIENCE REACTIONS — audience laughter, chuckles, or "
        "amusement after the punchline is the single strongest signal for "
        "humor. Also note vocal comedic timing, tone shifts, and emphasis.\n\n"
        "Provide a single judgment based on all three modalities combined. "
        "Your answer MUST begin with 'Yes' or 'No', followed by a brief "
        "explanation. If you are not sure, give your best guess — do not say "
        "you are unsure."
    ),
    'mustard': (
        "You are an expert at detecting sarcasm and irony in sitcom dialogue. "
        "You will receive a text transcript (conversation context followed by "
        "the target utterance) together with a video that contains both "
        "visual frames and audio. The LAST line of the transcript is the "
        "target utterance; preceding lines are the conversation context. "
        "Your job is to judge whether the target utterance is sarcastic.\n\n"
        "[TEXT] First identify what the CONTEXT is about, then analyze the "
        "target utterance in that context: consider what was said before, "
        "whether the statement contradicts the situation, and whether the "
        "speaker is saying the opposite of what they mean. Look for verbal "
        "irony, understatement, hyperbole, and absurd or impossible scenarios "
        "presented as serious explanations.\n\n"
        "Key distinctions:\n"
        "1. Genuine anger, frustration, or direct insults are NOT sarcasm "
        "even if they sound harsh. A rhetorical question expressing real "
        "annoyance is not sarcastic.\n"
        "2. Sarcasm requires the speaker to mean the OPPOSITE of their "
        "literal words, or to use deliberate incongruity with the situation.\n"
        "3. Absurd statements that cannot be taken literally in context are "
        "likely sarcastic.\n"
        "4. Deadpan or dry humor — where the speaker says something absurd, "
        "exaggerated, or contextually inappropriate with a straight-faced "
        "delivery — IS a form of sarcasm. A statement that sounds normal on "
        "its own but is clearly ridiculous given the conversation context is "
        "likely sarcastic.\n"
        "5. Playful teasing where the speaker genuinely means what they say "
        "is NOT sarcasm. Calling someone bad at something when they actually "
        "did something bad is genuine, not sarcastic.\n"
        "6. Genuine self-deprecation or honest emotional reactions "
        "(frustration, surprise, amusement) are NOT sarcasm.\n"
        "7. Very short or simple responses ('Oh', 'Yeah', 'No') should NOT "
        "be classified as sarcastic unless the context makes the meaning "
        "reversal unmistakable.\n\n"
        "[VIDEO] Analyze body language, facial expressions, gestures, and "
        "context. Ground your observations in specific visual evidence — eye "
        "rolls, smirks, exaggerated reactions, mismatched expressions between "
        "what is said and how the speaker looks.\n\n"
        "[AUDIO] Listen to tone, pitch, speed, emphasis on specific words, "
        "and pauses/timing. A flat or dry delivery style does NOT "
        "automatically mean sarcasm — some people naturally speak in a "
        "deadpan manner. Only flag flat tone as sarcastic if it clearly "
        "contrasts with emotionally charged content. Teasing, playful, or "
        "amused tones are not the same as sarcastic tones.\n\n"
        "Provide a single judgment based on all three modalities combined. "
        "Your answer MUST begin with 'Yes' or 'No', followed by a brief "
        "explanation. If you are not sure, give your best guess — do not say "
        "you are unsure."
    ),
}


def run_instance(
    test_file,
    dataset,
    dataset_name,
    agent,
    tracker,
    output_file,
):
    """Process one sample: load inputs -> call agent -> save result.

    The merged CTM prompt lives in ``agent.system_prompt`` (the real system
    role), so the user content only carries the target text. No few-shot.
    """
    start_time = time.time()

    inputs = load_sample_inputs(test_file, dataset, dataset_name)
    target_sentence = inputs['target_sentence']
    label = inputs['label']
    full_video_path = inputs['full_video_path']

    query = f"target text: '{target_sentence}'"

    print(f'[{test_file}] target: {target_sentence[:80]}...')

    answer, usage = agent.call(query, video_path=full_video_path)

    end_time = time.time()
    duration = end_time - start_time
    tracker.add(
        duration,
        usage.get('prompt_tokens', 0),
        usage.get('completion_tokens', 0),
        1,
    )

    print(
        f'[{test_file}] answer: {answer[:80] if answer else "None"}... ({duration:.1f}s)'
    )

    result = {
        test_file: {
            'answer': [answer],
            'label': label,
            'label_normalized': normalize_label(label),
            'method': 'unified_align_with_ctm_prompt',
            'usage': {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'api_calls': 1,
            },
            'latency': duration,
        }
    }
    save_result_to_jsonl(result, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified baseline using CTM-merged system prompt (zero-shot)'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='urfunny',
        choices=['urfunny', 'mustard'],
        help='Dataset name (default: urfunny)',
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='qwen',
        choices=['gemini', 'qwen'],
        help='LLM provider (default: qwen)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name for litellm (default: auto based on provider)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset JSON file (default: auto based on dataset_name)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL file path',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0)',
    )
    parser.add_argument(
        '--frames_only',
        action='store_true',
        help=(
            'Force all samples to use 8 extracted frames instead of the raw '
            'video. Produces identical inputs across omni/vl models, at the '
            'cost of losing audio (omni) and temporal dynamics.'
        ),
    )
    args = parser.parse_args()

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()
    output_file = (
        args.output
        or f'unified_ctmprompt_{args.dataset_name}_{args.provider}.jsonl'
    )

    check_api_key(args.provider)
    litellm.set_verbose = False

    tracker = StatsTracker(
        cost_input_per_1m=COST_INPUT_PER_1M, cost_output_per_1m=COST_OUTPUT_PER_1M
    )

    if args.dataset_name not in CTM_MERGED_PROMPTS:
        raise ValueError(
            f'No CTM_MERGED_PROMPTS entry for dataset {args.dataset_name!r}. '
            f'Known: {list(CTM_MERGED_PROMPTS)}'
        )
    system_prompt = CTM_MERGED_PROMPTS[args.dataset_name]

    if args.frames_only:
        agent = FramesOnlyMultimodalAgent(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            system_prompt=system_prompt,
        )
    else:
        agent = create_agent(
            'multimodal',
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            system_prompt=system_prompt,
        )

    modality = 'frames (8 per sample)' if args.frames_only else 'video'
    print(
        f'Dataset: {args.dataset_name} | Provider: {args.provider} | '
        f'Model: {agent.model} | Modality: {modality}'
    )
    print(f'System prompt: CTM-merged ({len(system_prompt)} chars, in system role)')

    dataset = load_data(args.dataset)
    test_list = list(dataset.keys())
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(
            f'Resuming: {len(processed_keys)} done, {len(test_list) - len(processed_keys)} remaining'
        )

    try:
        for test_file in test_list:
            if test_file in processed_keys:
                continue
            try:
                run_instance(
                    test_file,
                    dataset,
                    args.dataset_name,
                    agent,
                    tracker,
                    output_file,
                )
            except Exception as e:
                print(f'[ERROR] {test_file}: {e}')
                continue
            time.sleep(2)
    finally:
        tracker.print_summary('Unified (CTM-merged prompt)')

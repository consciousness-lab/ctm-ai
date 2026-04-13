"""
LiteLLM-backed BaseLLM provider for MetaGPT.

Lets MetaGPT Actions call the LLM through the canonical `self.llm.aask(...)`
/ `await self._aask(...)` interface (same as every other metagpt example)
while routing under the hood through `litellm.completion`, so that:

1. All three baselines (MoA, MetaGPT, AutoGen) share the exact same LLM
   call surface (litellm + identical retry/temperature/model) and can be
   compared apples-to-apples on token cost and latency.

2. Audio / video multimodal inputs are supported via a local extension
   `aask_multimodal(prompt, audio_path=..., video_path=...)`, which builds
   the same `{"type": "image_url", "image_url": {"url": "data:audio/..."}}`
   content payload we were using directly before — only now it's reachable
   from Actions via `self.llm.aask_multimodal(...)` rather than by
   bypassing the LLM layer entirely.

3. MetaGPT's cost_manager sees every request (via `self._update_costs`),
   so `team.invest(...)` is no longer purely decorative.

The provider is registered under `LLMType.OPENAI`, which overrides
metagpt's stock OpenAIChatCompletion provider in the LLM_REGISTRY. This
means any config with `api_type: "openai"` — including our existing
auto-generated dummy `~/.metagpt/config2.yaml` — automatically routes
through this provider without touching config plumbing.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from typing import Any, Dict, List, Optional

import litellm
from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import LLM_REGISTRY


class LiteLLMProvider(BaseLLM):
    """BaseLLM subclass routing all completions through litellm."""

    # BaseLLM class attributes
    use_system_prompt: bool = True

    def __init__(self, config: LLMConfig):
        self.config = config
        # config.model holds the litellm model string, e.g.
        # "gemini/gemini-2.5-flash-lite" or "gpt-4o". litellm routes based on
        # the prefix.
        self.model: str = config.model or 'gemini/gemini-2.5-flash-lite'
        self.pricing_plan = getattr(config, 'pricing_plan', None)
        self.cost_manager = None  # set by metagpt Context when LLM is wired in
        # Populated by _sync_litellm_call on every request so per-Action
        # usage tracking (Action.last_usage) can read the last call's stats.
        self.last_response_usage: Dict[str, Any] = {}
        self.aclient = None  # unused; metagpt keeps this for openai-style clients

    # ------------------------------------------------------------------
    # Abstract methods required by BaseLLM
    # ------------------------------------------------------------------

    async def _achat_completion(
        self, messages: List[dict], timeout: int = USE_CONFIG_TIMEOUT,
    ) -> dict:
        """Run one non-streaming chat completion via litellm. Returns an
        OpenAI-compatible dict so BaseLLM.get_choice_text can extract the
        text without any custom parsing."""
        loop = asyncio.get_running_loop()
        text, usage = await loop.run_in_executor(
            None, self._sync_litellm_call, messages,
        )
        return {
            'choices': [
                {'message': {'content': text or ''}, 'finish_reason': 'stop'}
            ],
            'usage': {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
            },
        }

    async def acompletion(
        self, messages: List[dict], timeout: int = USE_CONFIG_TIMEOUT,
    ):
        return await self._achat_completion(messages, timeout)

    async def _achat_completion_stream(
        self, messages: List[dict], timeout: int = USE_CONFIG_TIMEOUT,
    ) -> str:
        """We don't stream; fall through to a non-streaming call and
        return just the text, matching BaseLLM.acompletion_text's contract."""
        resp = await self._achat_completion(messages, timeout)
        return self.get_choice_text(resp)

    # ------------------------------------------------------------------
    # Underlying litellm call with retry + usage capture
    # ------------------------------------------------------------------

    def _sync_litellm_call(self, messages: List[dict]) -> tuple[str, Dict[str, Any]]:
        """Single blocking litellm.completion call with bounded retry.
        Captures usage onto self.last_response_usage AND feeds metagpt's
        cost_manager (via BaseLLM._update_costs) so cost tracking works
        through the standard metagpt machinery."""
        start = time.time()
        max_retries = 3
        last_error: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=getattr(self.config, 'temperature', 0.0) or 0.0,
                )
                text = resp.choices[0].message.content or ''
                usage = {
                    'prompt_tokens': getattr(resp.usage, 'prompt_tokens', 0) or 0,
                    'completion_tokens': getattr(resp.usage, 'completion_tokens', 0) or 0,
                    'total_tokens': getattr(resp.usage, 'total_tokens', 0) or 0,
                    'api_calls': attempt,
                    'latency': time.time() - start,
                }
                self.last_response_usage = usage
                # Feed metagpt's cost_manager (silently no-ops if unconfigured
                # or if model name isn't in its pricing table).
                try:
                    self._update_costs(usage)
                except Exception:
                    pass
                return text.strip(), usage
            except Exception as e:
                last_error = str(e)
                print(f'  LiteLLMProvider attempt {attempt}/{max_retries} failed: {e}')
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        # Exhausted retries
        self.last_response_usage = {
            'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
            'api_calls': max_retries, 'latency': time.time() - start,
            'error': last_error,
        }
        return '', self.last_response_usage

    # ------------------------------------------------------------------
    # Local extension: multimodal aask
    # ------------------------------------------------------------------

    async def aask_multimodal(
        self,
        prompt: str,
        *,
        system_msgs: Optional[List[str]] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> str:
        """Ask with optional audio and/or video file attachments.

        Analogous to BaseLLM.aask() but builds an OpenAI-style multimodal
        `user` message with `{type: text}` + `{type: image_url, image_url:
        {url: "data:audio/...;base64,..."}}` content parts. Routes through
        the same _sync_litellm_call path so retry and usage tracking are
        shared with the text-only aask().
        """
        messages: List[dict] = []
        if system_msgs:
            messages.extend(self._system_msgs(system_msgs))
        elif self.use_system_prompt:
            messages.append(self._default_system_msg())

        content: list = [{'type': 'text', 'text': prompt}]
        if audio_path and os.path.exists(audio_path):
            ext = audio_path.rsplit('.', 1)[-1].lower()
            mime = {
                'mp3': 'audio/mp3', 'wav': 'audio/wav',
                'mp4': 'audio/mp4', 'aac': 'audio/aac',
            }.get(ext, 'audio/mp4')
            with open(audio_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:{mime};base64,{encoded}'},
            })
        if video_path and os.path.exists(video_path):
            with open(video_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:video/mp4;base64,{encoded}'},
            })

        messages.append({'role': 'user', 'content': content})

        loop = asyncio.get_running_loop()
        text, _usage = await loop.run_in_executor(
            None, self._sync_litellm_call, messages,
        )
        return text


# Register under LLMType.OPENAI. This overrides metagpt's stock
# OpenAIChatCompletion for any config with `api_type: "openai"`, which is
# the default we ship in our auto-generated ~/.metagpt/config2.yaml.
LLM_REGISTRY.register(LLMType.OPENAI, LiteLLMProvider)

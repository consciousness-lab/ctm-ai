from typing import Any, List, Union

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..utils import multi_info_exponential_backoff
from .executor_base import BaseExecutor


@BaseExecutor.register_executor('gpt4v_executor')
class GPT4VExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        self.model = OpenAI()

    @multi_info_exponential_backoff()
    def ask(
        self,
        messages: list[ChatCompletionMessageParam],
        max_token: int = 300,
        return_num: int = 5,
        *args: Any,
        **kwargs: Any,
    ) -> List[Union[str, None]]:
        response = self.model.chat.completions.create(
            model='gpt-4-vision-preview',
            messages=messages,
            max_tokens=max_token,
            n=return_num,
        )
        gists = [response.choices[i].message.content for i in range(return_num)]
        return gists

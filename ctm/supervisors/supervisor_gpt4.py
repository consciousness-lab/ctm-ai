from typing import Optional

import openai

from ..utils import exponential_backoff
from .supervisor_base import BaseSupervisor


@BaseSupervisor.register_supervisor("gpt4_supervisor")
class GPT4Supervisor(BaseSupervisor):
    def __init__(self, *args, **kwargs):
        self.init_supervisor()

    def init_supervisor(self) -> None:
        self.model = (
            openai.ChatCompletion.create()
        )  # Assuming you're using the OpenAI API client

    @exponential_backoff(retries=5, base_wait_time=1)
    def ask_info(self, query: str, context: Optional[str] = None) -> str:
        prompt = [
            {
                "role": "user",
                "content": f"The following is detailed information on the topic: {context}. Based on this information, answer the question: {query}. Answer with a few words:",
            }
        ]
        responses = self.model(
            model="gpt-4-turbo-preview", messages=prompt, max_tokens=300, n=1
        )
        answer = responses.choices[0].message.content
        return answer

    @exponential_backoff(retries=5, base_wait_time=1)
    def ask_score(
        self, query: str, gist: str, verbose: bool = False, *args, **kwargs
    ) -> float:
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.model(
                    model="gpt-4-0125-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": f"How related is the information ({gist}) with the query ({query})? We want to make sure that the information includes a person's name as the answer. Answer with a number from 0 to 5 and do not add any other thing.",
                        },
                    ],
                    max_tokens=50,
                )
                score = float(response.choices[0].message.content.strip()) / 5
                return score
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                else:
                    print("Max attempts reached. Returning default score.")
        return 0.0

from typing import Any, Optional
import json
from ..utils import (
    info_exponential_backoff,
    logprobs_to_softmax,
    score_exponential_backoff,
)
from .supervisor_base import BaseSupervisor
from ..utils.llm_generator import LLMGenerator

@BaseSupervisor.register_supervisor('language_supervisor')
class LanguageSupervisor(BaseSupervisor):
    def init_supervisor(self, model_name: str = "gpt-4o-mini", *args: Any, **kwargs: Any) -> None:
        self.llm_generator = LLMGenerator(generator_name=model_name)
    
    @info_exponential_backoff(retries=5, base_wait_time=1)
    def ask_info(self, query: str, context: Optional[str] = None) -> Optional[str]:
        prompt = f"The following is detailed information on the topic: {context}. Based on this information, answer the question: {query}. Answer with a few words:"
        system_prompt = "You are a concise assistant that provides brief, accurate answers."
        
        response = self.llm_generator.generate_response(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        try:
            content = json.loads(response["content"])
            return content.get("answer", None)
        except (json.JSONDecodeError, KeyError):
            return response["content"] or None
    
    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_score(self, query: str, gist: str, *args: Any, **kwargs: Any) -> float:
        if not gist:
            return 0.0
        
        prompt = f"Is the information ({gist}) related to the query ({query})? Analyze carefully and return a JSON with two fields: 'is_relevant' (true/false) and 'confidence' (float between 0 and 1)."
        system_prompt = "You are an analytical assistant. Always return JSON format with the fields 'is_relevant' (boolean) and 'confidence' (float)."
        
        original_temp = self.llm_generator.temperature
        self.llm_generator.temperature = 0.1
        
        try:
            response = self.llm_generator.generate_response(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            content = json.loads(response["content"])
            is_relevant = content.get("is_relevant", False)
            confidence = content.get("confidence", 0.5)
            score = confidence if is_relevant else 0.0
            return score
            
        except (json.JSONDecodeError, KeyError):
            return 0.0
        finally:
            self.llm_generator.temperature = original_temp
from typing import List

import numpy as np
from litellm import completion, embedding
from sklearn.metrics.pairwise import cosine_similarity

from ..messengers import Message
from ..utils import score_exponential_backoff
from .scorer_base import BaseScorer


@BaseScorer.register_scorer('hybrid_scorer')
class HybridRelevanceScorer(BaseScorer):
    def init_scorer(self, *args, **kwargs) -> None:
        """Initialize the scorer using LiteLLM."""
        # Use specific models for different purposes
        kwargs.setdefault('model', 'gpt-4o')
        kwargs.setdefault('embedding_model', 'text-embedding-3-small')
        super().init_scorer(*args, **kwargs)
        
        self.embedding_model = kwargs.get('embedding_model', 'text-embedding-3-small')

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using LiteLLM."""
        try:
            response = embedding(model=self.embedding_model, input=[text])
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(1536, dtype=np.float32)  # Default embedding size

    def get_directness_score(self, query: str, gist: str) -> float:
        """Get directness score using LiteLLM."""
        try:
            response = completion(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': (
                            f'Question: {query}\n'
                            f'Answer: {gist}\n\n'
                            'Evaluate how relevant does the answer respond to the question?.\n'
                            'Give a score from 0 to 10 based on the following criteria:\n'
                            '- 10 = Fully answers the question clearly and directly.\n'
                            '- 5 = Partially answers the question, but vague or lacks detail.\n'
                            '- 0 = Does not answer the question or is completely off-topic.\n'
                            'Respond with a single number only. Do not explain.'
                        ),
                    }
                ],
                max_tokens=10,
                temperature=0.0,
            )

            score_text = response.choices[0].message.content.strip()
            print(f'[Directness LLM Response] {score_text}')
            
            try:
                score = float(score_text)
                return min(max(score / 10.0, 0.0), 1.0)
            except (ValueError, TypeError):
                return 0.0
                
        except Exception as e:
            print(f'[Directness LLM Error] {e}')
            return 0.0

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(self, messages: List[Message]) -> float:
        """Hybrid relevance scoring using both embeddings and LLM assessment."""
        if not messages or not messages[-1].query or not messages[-1].gist:
            return 0.0
            
        query = messages[-1].query
        gist = messages[-1].gist

        # Method 1: Topical similarity using embeddings
        try:
            query_emb = self.get_embedding(query)
            gist_emb = self.get_embedding(gist)
            
            # Check for zero vectors
            if np.allclose(query_emb, 0) or np.allclose(gist_emb, 0):
                topical_score = 0.0
            else:
                topical_score = cosine_similarity([query_emb], [gist_emb])[0][0]
        except Exception as e:
            print(f'[Embedding Error] {e}')
            topical_score = 0.0

        # Method 2: Directness assessment using LLM
        directness_score = self.get_directness_score(query, gist)

        # Combine both scores with weighted average
        # TODO: These weights could be made configurable
        final_score = 0.3 * topical_score + 0.7 * directness_score
        
        return float(np.clip(final_score, 0.0, 1.0))

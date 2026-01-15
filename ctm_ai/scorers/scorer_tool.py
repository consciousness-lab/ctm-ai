import math
from typing import Any, Dict, List

import numpy as np
from litellm import completion, embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordfreq import word_frequency

from ..utils import configure_litellm, score_exponential_backoff
from .scorer_base import BaseScorer


class ToolScorer(BaseScorer):
    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_relevance(
        self, query: str, messages: Dict[str, Any], use_llm: bool = True
    ) -> float:
        query = query
        gist = messages['response']

        final_relevance = self._ask_llm_relevance(query, gist)

        return float(np.clip(final_relevance, 0.0, 1.0))

    def _ask_llm_relevance(self, query: str, gist: str) -> float:
        relevance_prompt = [
            {
                'role': 'user',
                'content': f"""Please evaluate how relevant the answer is to the question on a scale from 0.0 to 1.0. 
Here, "relevant" means that the answer engages with the question and provides specific information that is useful or connected to addressing it. Even if the answer only provides parts of the information, it should be considered highyly relevant. 
Only answers that completely refuse, ignore, or go off-topic should be scored as 0.0. 

Question: {query}
Answer: {gist}

Consider:
- 1.0 = Perfectly relevant, provide answers answer part of the question with specific information.
- 0.8 = Highly relevant, mostly answers the question and provides useful supporting details
- 0.6 = Moderately relevant, engages with the question but is limited, uncertain, or incomplete
- 0.4 = Somewhat relevant, loosely connected to the question but not very helpful
- 0.2 = Barely relevant, only a weak or indirect connection
- 0.0 = Not relevant, completely unrelated, refuses to answer, or provides information irrelevant to the question, can not answer the question. 

IMPORTANT: Respond with ONLY a single number between 0.0 and 1.0 (e.g., 0.85). Do not include any other text, explanations, or formatting.
""",
            }
        ]

        try:
            responses = completion(
                messages=relevance_prompt,
                model=self.relevance_model,
                max_tokens=10,
                temperature=0.0,
                n=1,
            )

            score_text = responses.choices[0].message.content.strip()
            import re

            number_match = re.search(r'(\d+\.?\d*)', score_text)
            if number_match:
                score = float(number_match.group(1))
            else:
                score = float(score_text)

            return max(0.0, min(1.0, score))

        except (ValueError, TypeError, IndexError):
            raise ValueError('Error getting relevance score')

    def _ask_statistical_relevance(self, query: str, gist: str) -> float:
        try:
            query_emb = self.get_embedding(query)
            gist_emb = self.get_embedding(gist)

            if np.allclose(query_emb, 0.0) or np.allclose(gist_emb, 0.0):
                topical_score = 0.0
            else:
                topical_score = cosine_similarity([query_emb], [gist_emb])[0][0]
        except Exception as e:
            print(f'[Embedding Error] {e}')
            topical_score = 0.0
        return float(topical_score)

    @score_exponential_backoff(retries=5, base_wait_time=1)
    def ask_confidence(
        self, query: str, messages: Dict[str, Any], use_llm: bool = True
    ) -> float:
        gist = messages['response']

        if use_llm:
            final_confidence = self._ask_llm_confidence(query=query, gist=gist)
        else:
            final_confidence = self._ask_statistical_confidence(query=query, gist=gist)

        return float(np.clip(final_confidence, 0.0, 1.0))

    def _ask_llm_confidence(self, query: str, gist: str) -> float:
        confidence_prompt = [
            {
                'role': 'user',
                'content': f"""Please evaluate how confident this response appears to be on a scale from 0.0 to 1.0. 

Response: {gist}

Consider:
- 1.0 = Very confident, definitive statements, clear and certain
- 0.8 = Confident, mostly certain with minor qualifications
- 0.6 = Moderately confident, some uncertainty expressed
- 0.4 = Somewhat uncertain, many qualifications or hedging
- 0.2 = Very uncertain, lots of "maybe", "possibly", "might be"
- 0.0 = Completely uncertain, says "cannot determine", "I don't know", or no definitive information


If the response says "I cannot determine", "cannot answer", "I don't know", or refuses to provide information, score it as 0.0.

IMPORTANT: Respond with ONLY a single number between 0.0 and 1.0 (e.g., 0.85). Do not include any other text, explanations, or formatting.

""",
            }
        ]

        try:
            responses = completion(
                messages=confidence_prompt,
                model=self.confidence_model,
                max_tokens=10,
                temperature=0.0,
                n=1,
            )

            score_text = responses.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))

        except (ValueError, TypeError, IndexError):
            return 0.5

    def _ask_statistical_confidence(self, gists: List[str]) -> float:
        if len(gists) <= 1:
            return 1.0

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(gists)
            cos_sim_matrix = cosine_similarity(tfidf_matrix)

            upper_triangle_indices = np.triu_indices_from(cos_sim_matrix, k=1)
            upper_triangle_values = cos_sim_matrix[upper_triangle_indices]

            avg_similarity = np.mean(upper_triangle_values)

            return float(avg_similarity)

        except Exception:
            return 0.5

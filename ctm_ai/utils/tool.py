from typing import List

import numpy as np


def logprobs_to_softmax(logprobs: List[float]) -> List[float]:
    max_logprob = max(logprobs)
    exp_probs = np.exp(np.array(logprobs) - max_logprob)
    softmax_probs = exp_probs / np.sum(exp_probs)
    return list(softmax_probs)

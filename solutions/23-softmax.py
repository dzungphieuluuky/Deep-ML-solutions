import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores)
    max_score = np.max(scores)
    sum_exp = np.sum(np.exp(scores - max_score))
    softmax_scores = np.exp(scores - max_score) / sum_exp
    return softmax_scores
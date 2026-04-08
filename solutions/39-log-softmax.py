import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	scores = np.array(scores)
	max_score = np.max(scores)
	sum_exp = np.sum(np.exp(scores - max_score))
	log_softmax = scores - max_score - np.log(sum_exp)
	return log_softmax
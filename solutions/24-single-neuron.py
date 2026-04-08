import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features = np.array(features)
	labels = np.array(labels)
	weights = np.array(weights)
	bias = np.array(bias)

	pre_bias = features @ weights
	after_bias = pre_bias + bias
	probabilities = 1 / (1 + np.exp(-after_bias))
	mse = np.mean((probabilities - labels)**2)
	return probabilities, mse
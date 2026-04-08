import numpy as np
def calculate_variance(X: list, Y: list):
	X = np.array(X)
	Y = np.array(Y)
	n = len(X)
	mean_X = np.mean(X)
	mean_Y = np.mean(Y)
	return np.sum((X - mean_X)*(Y - mean_Y))/(n - 1)

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	n = len(vectors)
	cov_matrix = [[0 for i in range(n)] for j in range(n)]
	for i in range(n):
		for j in range(n):
			var = calculate_variance(vectors[i], vectors[j])
			cov_matrix[i][j] = var
	return cov_matrix
import numpy as np
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	scaled_matrix = np.array(matrix.copy())
	scaled_matrix = scaled_matrix * scalar
	return scaled_matrix
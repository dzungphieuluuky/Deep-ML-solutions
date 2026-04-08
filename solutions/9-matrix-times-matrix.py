import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    array_a = np.array(a)
    array_b = np.array(b)
    if array_a.shape[1] != array_b.shape[0]:
        return -1
    return array_a @ array_b
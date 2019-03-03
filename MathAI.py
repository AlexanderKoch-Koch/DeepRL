import math
import numpy as np


def sigmoid(x):
    return 1/(1 + math.pow(math.e, -x))


def euclidean_distance(x, y):
    #x = np.array(x)
    #y = np.array(y)
    if x.shape[0] != y.shape[0]:
        return None

    sum = 0
    for i in range(len(x)):
        sum += math.pow(x[i] * y[i], 2)

    return math.sqrt(sum)


import math

def get_distance(x, y):
    assert len(x) == len(y)
    cum_sum = 0
    for i in range(len(x)):
        cum_sum += (x[i] - y[i])**2
    return cum_sum**0.5

def sigmoid(x):
    return 1 / (1 + math.exp(x))
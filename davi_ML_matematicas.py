import math 

def DE(x, y):
    """distancia euclidiana"""
    return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))

def gini(s):
    """
    impureza de gini
    """
    n = len(s)
    if n == 0:
        return 0
    p = [s.count(i) / n for i in range(max(s) + 1)]
    return sum([p[i] ** 2 for i in range(len(p))])



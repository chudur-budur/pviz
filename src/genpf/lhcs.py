import sys
import math
import random

sys.path.append("./")
from paretoviz.utils import vectorops as vops
from paretoviz.utils import fmt

"""
This script implements a Latin hypercube sampling method.
"""

def lhcs(n, m):
    """
    Latin hypercube sampling n samples of m-dimensional points
    """
    d = 1.0 / float(n) ;
    samp = []
    for i in range(m):
        temp = []
        for j in range(n):
            val = (j * d) + ((((j + 1.0) * d) - (j * d)) * random.random())
            temp.append(val)
        random.shuffle(temp)
        if len(samp) == 0:
            for item in temp:
                samp.append([item])
        else:
            for idx,item in enumerate(temp):
                samp[idx].append(item)
    return samp

def lhcsl2norm(n, m):
    """
    Latin hypercube sampling n samples of m-dimensional points.
    This function gurantees that 2-norm of each sample is greater
    than 0.0001.
    """
    k = n
    samp = []
    pass_ = 1
    while len(samp) < n:
        temp = lhcs(k, m)
        for v in temp:
            if vops.norm(v, 1) > 0.0001:
                samp.append(v)
        k = n - len(samp)
        # print("pass {0:d}, need {1:d} points.".format(pass_, k))
        pass_ = pass_ + 1
    return samp

if __name__ == "__main__":
    seed = 123456
    random.seed(seed)
    s = lhcs(100, 5)
    fmt.cat(s)
    s = lhcsl2norm(1000, 8)
    fmt.cat(s)

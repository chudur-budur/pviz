#!/usr/bin/python3

import math

"""
    This file contains different utility functions.
"""

def normalize(vec, lb, ub):
    """
        Normalize a vector vec = [x1, x2, ... xn] w.r.t. [lb, ub]
        The parameter vec can be a list of lists, 
        in that case lb and ub will be the same.
    """
    errmsg = "all parameters need to be of the same structure."
    assert (type(vec[0]) == type(lb) == type(ub)), errmsg
    if (type(vec[0]) is list):
        assert (len(vec[0]) == len(lb) == len(ub)), errmsg
        return [[(x - lb[i])/float(ub[i] - lb[i]) for i,x in enumerate(e)] for e in vec]
    else:
        assert (type(lb) is not list and type(ub) is not list), errmsg
        if lb == ub:
            return vec
        else:
            return [(e - lb)/float(ub - lb) for e in vec]

def get_bound(vec):
    """
        get the bound of the data, the data can be a list of lists. 
    """
    if type(vec[0]) is list:
        lb = list(vec[0]) # ideal point
        ub = list(vec[0]) # nadir point
        for v in vec[1:]:
            for i,e in enumerate(v):
                if e <= lb[i]:
                    lb[i] = e
                if e >= ub[i]:
                    ub[i] = e
        return [lb, ub]
    else:
        return [min(vec), max(vec)]

def dot(a, b):
    errmsg = "the parameters must be the vectors of same length."
    assert type(a) is list and type(b) is list, errmsg
    assert len(a) == len(b), errmsg
    return math.fsum(list(map(lambda x,y: (x * y), a, b)))

def norm(a, p = 1):
    errmsg = "the parameter must be a vector."
    assert type(a) is list, errmsg
    return math.pow(math.fsum(list(map(lambda x: math.fabs(x)**p, a))), 1.0/p)

def unit(a):
    errmsg = "the parameter must be a vector."
    assert type(a) is list, errmsg
    nrm = norm(a, 2)
    return [v/float(nrm) for v in a]

def distlp(a, b, p = 2):
    errmsg = "the parameters must be the vectors of same length."
    assert type(a) is list and type(b) is list, errmsg
    assert len(a) == len(b), errmsg
    return math.pow(math.fsum(list(map(lambda x,y: math.fabs(x - y)**p, a, b))), 1.0/p)

def cross(a, b):
    errmsg = "the parameters must be the vectors of maximum length 3."
    assert type(a) is list and type(b) is list, errmsg
    assert len(a) == len(b) == 3, errmsg
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]

def angle(a, b):
    return math.degrees(math.acos(dot(a,b)/float(norm(a, 2) * norm(b, 2))))

def mean(a):
    return math.fsum(a)/float(len(a))

def median(a):
    # this is slow, need to write the 'quickselect' algorithm
    a_ = sorted(a)
    l = len(a_)
    ret = (a_[int(l/2 - 1)] + a_[int(l/2)])/2.0 if l % 2 == 0 else a_[int((l-1)/2)]
    return ret

def tostr(a):
    string = "["
    for v in a:
        if type(v) is list:
            string = string + tostr(v) + ", "
        else:
            val = "{0:.3f}".format(v) if type(v) is float else str(v)
            string = string + val + ", "
    string = string[0:-2] + "]" if len(a) > 0 else "[]"
    return string

def tocsvstr(a):
    string = ""
    for v in a:
        if type(v) is list:
            string = string + tocsvstr(v) + ","
        else:
            val = "{0:.3f}".format(v) if type(v) is float else str(v)
            string = string + val + ","
    string = string[0:-1] if len(a) > 0 else ""
    return string



# tester main
if __name__ == "__main__":
    
    v1 = [1, 2, 3, 4]
    [lb, ub] = get_bound(v1)
    print("normalize([{0:s}], {1:d}, {2:d}) = [{3:s}]".format(
        ", ".join([str(v) for v in v1]), lb, ub, 
        ", ".join(["{0:.2f}".format(v) for v in normalize(v1, lb, ub)])))
    
    v2 = [[1, 5, 4], [6, 2, 3]]
    [lb, ub] = get_bound(v2)
    print("normalize([{0:s}], [{1:s}], [{2:s}]) = [{3:s}]".format(
        ", ".join([str(v) for v in v1]), 
        ", ".join([str(v) for v in lb]), 
        ", ".join([str(v) for v in ub]), 
        ", ".join([str(v) for v in normalize(v2, lb, ub)])))
    
    v3 = [2, 2, 2, 2]
    print("dot([{0:s}], [{1:s}]) = {2:.2f}".format(
        ", ".join([str(v) for v in v1]), 
        ", ".join([str(v) for v in v3]),
        dot(v1, v3)))
    
    print("norm([{0:s}]) = {1:.2f}".format(", ".join([str(v) for v in v1]), norm(v1)))
    print("norm([{0:s}], 1) = {1:.2f}".format(", ".join([str(v) for v in v1]), norm(v1, 1)))
    print("norm([{0:s}], 2) = {1:.2f}".format(", ".join([str(v) for v in v1]), norm(v1, 2)))
    print("norm([{0:s}], 3) = {1:.2f}".format(", ".join([str(v) for v in v1]), norm(v1, 3)))
    print("unit([{0:s}]) = [{1:s}]".format(
        ", ".join([str(v) for v in v1]), 
        ", ".join(["{0:.2f}".format(v) for v in unit(v1)])))
    v4 = [1, 2, 3]
    v5 = [2, 2, 2]
    print("cross([{0:s}], [{1:s}]) = [{2:s}]".format(
        ", ".join([str(v) for v in v4]), 
        ", ".join([str(v) for v in v5]), 
        ", ".join([str(v) for v in cross(v4, v5)])))

    v6 = [1, 0, 1, 0, 1, 0]
    v7 = [0, 1, 0, 1, 0, 1]
    print(angle(v6, v7))
    print(dot(v6, v6)/(norm(v6, 2) * norm(v6, 2)))
    # print(angle(v6, v6))

    v8 = [0, 0, 0, 0]
    v9 = [1, 1, 1, 1]
    print(distlp(v8, v9))

    v10 = [0.12345, 0.1213, [1, 0.1234, [[4, 10.0, 0.23490, 10], 0.9897], 10], 6, [5, 0.4637]]
    v11 = [1]
    v12 = []
    print(tostr(v10))
    print(tostr(v11))
    print(tostr(v12))


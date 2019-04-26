import sys
import os
import utils

"""
This script loads a data file, then dumps the first front (i.e. Pareto front)
from the input data to another file.
"""

def dominates(a, b):
    """
    This function is called by ndsort() function.
    Checks if an objective vector a dominates b or not.
    """
    m = len(a)
    f1 = 0 
    f2 = 0
    for i in range(m):
        if a[i] < b[i]:
            f1 = 1
        elif a[i] > b[i]:
            f2 = 1
    if f1 == 1 and f2 == 0:
        return 1
    elif f1 == 0 and f2 == 1:
        return -1
    else:
        return 0

def ndsort(vals):
    """
    The non-dominated sorting method, naive version.
    """
    P = [i for i in range(len(vals))]
    F = []
    for p in P:
        Sp = []
        np = 0
        for q in P:
            if dominates(vals[p], vals[q]) == 1:
                Sp.append(q)
            elif dominates(vals[q], vals[p]) == 1:
                np = np + 1
        if np == 0:
            F.append(p)
    # print(F)
    return F

if __name__ == "__main__":
    data_file = sys.argv[1].strip()
    path, filename = os.path.split(data_file)
    cv_all_file = None
    pf_file = os.path.join(path, filename.split('.')[0] + "-pf.out")
    vals = utils.load(data_file)
    print("Non-dominated sorting of {0:d} data points".format(len(vals)))
    idx = ndsort(vals)
    vals_ = [vals[i] for i in idx]
    print("Pareto-front contains {0:d} data points".format(len(vals_)))
    # print("Saving Pareto-front into {0:s} ...".format(pf_file))
    # utils.save(vals_, pf_file)
    utils.cat(vals_)

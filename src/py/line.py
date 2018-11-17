import sys
import os
import math
import random
import lhcs
import spherical
import vectorutils as vu
import utils
import ndsort

"""
This script generates a mix constrained line surface 
Pareto-optimal front on the first quadrant.
"""

def constraint_violation(xvals, fvals):
    """
    Computes the constraint violation values and returns
    only the feasible points.
    """
    fvals_, xvals_, cv = [], [], []
    m = len(fvals[0])
    for i,fv in enumerate(fvals):
        g = [1.0] * m
        for j,f in enumerate(fv[0:-1]):
            g[j] = -fv[-1] - (m * f) + 1
        g[-1] = -(2 * fv[-1]) - math.fsum(fv[0:-1]) + 1.0
        if all(v <= 0 for v in g):
            cv.append(math.fsum(g))
            fvals_.append(fv)
            xvals_.append(xvals[i])
    return (xvals_, fvals_, cv)

def surface(n, m, seed):
    """
    Generate surface points using LHC sampling,
    n samples of m dimensional points.
    """
    S = lhcs.lhcs(n, m, seed)
    fvals = []
    xvals = []
    for x in S:
        f = x
        if f is not None:
            fvals.append(f)
            xvals.append(x)
    print("Computing constraint violation ...")
    (xvals_, fvals_, cv) = constraint_violation(xvals, fvals)
    print("{0:d} points generated, doing non-domination sort ...".format(len(fvals_)))
    idx = ndsort.ndsort(fvals_)
    xvals__ = [xvals_[i] for i in idx]
    fvals__ = [fvals_[i] for i in idx]
    cv_ = [cv[i] for i in idx]
    print("{0:d} points found. Done.".format(len(fvals__)))
    return (xvals__, fvals__, cv_)

if __name__ == "__main__":
    seed = 123456
    n = {3: 3000, 4: 2000, 8: 4000}
    m = 3
    if len(sys.argv) > 1:
        m = int(sys.argv[1].strip())    
    (x, f, g) = surface(n[m], m, seed)
    path = "./data/isolated/"
    try:
        os.makedirs(path)
    except OSError:
        pass
    
    fg = [v + [g[i]] for i,v in enumerate(f)]
    utils.cat(fg)
    outfile = path + "isolated-{0:d}d.out".format(m)
    utils.save(f, outfile)
    outfile = path + "isolated-{0:d}d-cv.out".format(m)
    utils.save(g, outfile)

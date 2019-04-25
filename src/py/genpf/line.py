import sys
import os
import math
import random as rng

sys.path.insert(0, "./utils")
import lhcs
import utils
import ndsort

import spherical

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

def surface(n, m, mode):
    """
    Generate surface points using LHC sampling,
    n samples of m dimensional points.
    """
    S = lhcs.lhcs(n, m)
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
    rng.seed(seed)
    n = {3: 3000, 4: 2000, 8: 4000}
    m = 3
    mode = "uniform"
    if len(sys.argv) > 1:
        m = int(sys.argv[1].strip())
        if len(sys.argv) > 2:
            mode = sys.argv[2].strip()

    print("Using {0:s} mode over {1:d} dim ...".format(mode, m))
    (x, f, g) = surface(n[m], m, mode)

    print("{0:d} points generated, doing non-domination sort ...".format(len(f)))
    idx = ndsort.ndsort(f)
    xv = [x[i] for i in idx]
    fv = [f[i] for i in idx]
    gv = [g[i] for i in idx]
    print("{0:d} points found. Done.".format(len(fv)))

    path = "./data/line/"
    try:
        os.makedirs(path)
    except OSError:
        pass
    
    fg = [v + [gv[i]] for i,v in enumerate(f)]
    utils.cat(fg)
    outfile = path + "line-{0:d}d.out".format(m)
    utils.save(f, outfile)
    outfile = path + "line-{0:d}d-cv.out".format(m)
    utils.save(g, outfile)

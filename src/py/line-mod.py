import sys
import os
import math
import random as rng
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
            g[j] = -fv[-1] - (4 * fv[j]) + 1.0
        minval = float('inf')
        for k in range(m-1):
            for l in range(m-1):
                if k != l and (fv[k] + fv[l] <= minval):
                    minval = fv[k] + fv[l]
        g[-1] = -(2 * fv[-1]) - minval + 1.0
        if all((v <= 0) for v in g):
            cv.append(math.fsum(g))
            fvals_.append(fv)
            xvals_.append(xvals[i])
    return (xvals_, fvals_, cv)

def surface(n, m):
    """
    Generate surface points using LHC sampling,
    n samples of m dimensional points.
    """
    S = lhcs.lhcs(n, m)
    fv = []
    xv = []
    for x in S:
        f = x
        fv.append(f)
        xv.append(x)
    print("Computing constraint violation ...")
    (xv_, fv_, cv) = constraint_violation(xv, fv)
    return (xv_, fv_, cv)

if __name__ == "__main__":
    seed = 123456
    rng.seed(seed)
    n = {3: 12000, 4: 2000, 8: 4000}
    m = 3
    if len(sys.argv) > 1:
        m = int(sys.argv[1].strip())    
    
    (x, f, g) = surface(n[m], m)
    
    print("{0:d} points generated, doing non-domination sort ...".format(len(f)))
    idx = ndsort.ndsort(f)
    xv = [x[i] for i in idx]
    fv = [f[i] for i in idx]
    gv = [g[i] for i in idx]
    print("{0:d} points found. Done.".format(len(fv)))    
    
    path = "./data/line-mod/"
    try:
        os.makedirs(path)
    except OSError:
        pass
    
    fgv = [v + [g[i]] for i,v in enumerate(f)]
    utils.cat(fgv)
    outfile = path + "line-mod-{0:d}d.out".format(m)
    utils.save(fv, outfile)
    outfile = path + "line-mod-{0:d}d-cv.out".format(m)
    utils.save(gv, outfile)

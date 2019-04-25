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
This script generates a constrained c2dtlz2 surface 
on the first quadrant.
"""

def constraint_violation(xvals, fvals):
    """
    Computes the constraint violation values and returns
    only the feasible points.
    """
    fvals_, xvals_, cv = [], [], []
    m = len(fvals[0])
    r = 0.4 if m == 3 else 0.5
    r1 = 0.50 * r
    for k,fv in enumerate(fvals):
        lhs = min([((f - 1.0) ** 2) + math.fsum([(fv[j] ** 2) for j in range(m) if j != i]) - (r ** 2.0) \
                    for i,f in enumerate(fv)])
        rhs = math.fsum([((f - (1.0 / (m ** 0.5))) ** 2) for f in fv]) - (r1 ** 2)
        c = min(lhs, rhs)
        if c <= 0:
            cv.append(c)
            fvals_.append(fvals[k])
            xvals_.append(xvals[k])
    return (xvals_, fvals_, cv)

def surface(n, m, mode):
    """
    Generate surface points using LHC sampling,
    n samples of m dimensional points.
    """
    (xv, fv) = spherical.surface(n, m, mode)
    print("Computing constraint violation ...")
    (xv_, fv_, cv) = constraint_violation(xv, fv)
    return (xv_, fv_, cv)

if __name__ == "__main__":
    seed = 123456
    rng.seed(seed)
    n = {3: 8000, 4: 10000, 5: 50000, 8: 150000}
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
    
    path = "./data/c2dtlz2/"
    try:
        os.makedirs(path)
    except OSError:
        pass
    
    fgv = [v + [gv[i]] for i,v in enumerate(f)]
    utils.cat(fgv)
    outfile = os.path.join(path, "c2dtlz2-{0:d}d.out".format(m))
    utils.save(fv, outfile)
    outfile = os.path.join(path, "c2dtlz2-{0:d}d-cv.out".format(m))
    utils.save(gv, outfile)

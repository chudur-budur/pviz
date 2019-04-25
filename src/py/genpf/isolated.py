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
This script generates a constrained spherical surface 
with isolated cluster on the first quadrant.
"""

def constraint_violation(xvals, fvals):
    """
    Computes the constraint violation values and returns
    only the feasible points.
    """
    fvals_, xvals_, cv = [], [], []
    m = len(fvals[0])
    for i,f in enumerate(fvals):
        c = (0.98 - f[-1]) * (f[-1] - 0.75)
        if c <= 0:
            cv.append(c)
            fvals_.append(fvals[i])
            xvals_.append(xvals[i])
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
    # n = {3: 2000, 4: 4000, 8: 8000} # uniform
    n = {3: 5000, 4: 3000, 8: 6000} # random
    m = 3
    mode = "random"
    if len(sys.argv) > 1:
        m = int(sys.argv[1].strip())
        if len(sys.argv) > 2:
            mode = sys.argv[2].strip()

    print("Using {0:s} mode over {1:d} dim ...".format(mode, m))
    (x, f, g) = surface(n[m], m, mode = mode)

    print("{0:d} points generated, doing non-domination sort ...".format(len(f)))
    idx = ndsort.ndsort(f)
    xv = [x[i] for i in idx]
    fv = [f[i] for i in idx]
    gv = [g[i] for i in idx]
    print("{0:d} points found. Done.".format(len(fv)))
    
    path = "./data/isolated/"
    try:
        os.makedirs(path)
    except OSError:
        pass
    
    fgv = [v + [gv[i]] for i,v in enumerate(fv)]
    utils.cat(fgv)
    outfile = os.path.join(path, "isolated-{0:d}d.out".format(m))
    utils.save(fv, outfile)
    outfile = os.path.join(path, "isolated-{0:d}d-cv.out".format(m))
    utils.save(gv, outfile)

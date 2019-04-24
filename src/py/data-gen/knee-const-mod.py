import sys
import os
import math
import random as rng
import lhcs
import knee
import vectorutils as vu
import utils
import ndsort


"""
This script generates a constrained knee surface on the first quadrant
"""

def constraint_violation(xvals, fvals):
    """
    Computes the constraint violation values and returns
    only the feasible points.
    """
    fvals_, xvals_, cv = [], [], []
    epsilon = {3: 1.5, 4: 1.5, 8: 1.75} 
    m = len(fvals[0])
    for i,fv in enumerate(fvals):
        c = (math.fsum([(f - 3.0 / (1.414 ** (m-1))) ** 2 for f in fv[0:-1]]) \
                + ((fv[-1] - 3.0 / (1.414 ** (m-2))) ** 2)) - 1
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
    (xv, fv) = knee.surface(n, m, mode)
    print("Computing constraint violation ...")
    (xv_, fv_, cv) = constraint_violation(xv, fv)
    return (xv_, fv_, cv)

if __name__ == "__main__":
    seed = 123456
    rng.seed(seed)
    # 2262->986 3279->1232 3962->3789
    n = {3: 8000, 4: 15000, 8: 47500}
    m = 3
    mode = "uniform"
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

    path = "./data/knee-const-mod/"
    try:
        os.makedirs(path)
    except OSError:
        pass
    
    fgv = [v + [gv[i]] for i,v in enumerate(fv)]
    utils.cat(fgv)
    outfile = path + "knee-const-mod-{0:d}d.out".format(m)
    utils.save(fv, outfile)
    outfile = path + "knee-const-mod-{0:d}d-cv.out".format(m)
    utils.save(gv, outfile)

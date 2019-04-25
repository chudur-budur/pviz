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
This script generates a constrained knee surface on the first quadrant
"""

def rf(x, k):
    """
    The radius function for the knee problem.
    """
    return 5.0 + (10.0 * (x - 0.5) * (x - 0.5)) \
            + ((2.0 / k) * math.cos(2.0 * k * math.pi * x))    

def knee(x):
    """
    Transform a spherical surface into knee surface.
    """
    g = 1.0
    r = sum([rf(v, 1) for v in x])/len(x)
    f = [g * r * v for v in spherical.random(x)]
    return f

def surface(n, m, mode):
    """
    Generate surface points using LHC sampling,
    n samples of m dimensional points.
    """
    (xv_, fv_) = spherical.surface(n, m, mode)
    xv, fv = xv_, []
    for x in xv:
        f = knee(x)
        fv.append(f)
    return (xv, fv)

if __name__ == "__main__":
    seed = 123456
    rng.seed(seed)
    # 1005 1966 3980
    n = {3: 1500, 4:2500, 8:4000}
    m = 3
    mode = "uniform"
    if len(sys.argv) > 1:
        m = int(sys.argv[1].strip())
        if len(sys.argv) > 2:
            mode = sys.argv[2].strip()
   
    print("Using {0:s} mode over {1:d} dim ...".format(mode, m))
    (x_, f_) = surface(n[m], m, mode = mode)

    print("{0:d} points generated, doing non-domination sort ...".format(len(f_)))
    idx = ndsort.ndsort(f_)
    x = [x_[i] for i in idx]
    f = [f_[i] for i in idx]
    print("{0:d} points found. Done.".format(len(f)))    
    
    path = "./data/knee/"
    try:
        os.makedirs(path)
    except OSError:
        pass
    
    outfile = path + "knee-{0:d}d.out".format(m)
    utils.cat(f)
    utils.save(f, outfile)

import sys
import os
import math
import random as rng

sys.path.insert(0, "./utils")
import lhcs
import vectorops as vops
import utils
import ndsort

"""
This script generates a spherical surface on the first quadrant
"""

def random(x):
    """
    This function creates a spherical surface on the first quadrant.
    The independent variables (azimuth and polar angles) are sampled
    using LHS.
    """
    m = len(x) + 1
    f = [1] * m
    for i in range(m):
        fstr = ""
        for j in range(m - (i + 1)):
            f[i] = f[i] * math.sin(x[j] * 0.5 * math.pi)
            fstr = fstr + "sin({0:.2f} * pi/2) ".format(x[j])
        if(i != 0):
            aux = m - (i + 1)
            f[i] = f[i] * math.cos(x[aux] * 0.5 * math.pi)
            fstr = fstr + "cos({0:.2f} * pi/2) ".format(x[aux])
        # print("f({0:d}) = {1:s}".format(i, fstr))
    return f

def uniform(x):
    """
    This function creates a spherical surface on the first quadrant.
    This function tries to create uniformly distributed points on a 
    spherical surface using the method described in:
        http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    nrm = vops.norm(x, 2)
    f = [(v / nrm) for v in x]
    return f

def to_spherical(f):
    """
    Returns the spherical coordinate of a cartesian coordinate point
    in radians.
    """
    x = [(math.acos((v / vops.norm(f[i:], 2))) / (0.5 * math.pi)) \
            for i,v in enumerate(f[:-1])]
    return x

def surface(n, m, mode = "random"):
    """
    Generate surface points using LHC sampling,
    n samples of m dimensional points.
    """
    print("Generating {0:d} points ...".format(n))
    if mode == "random":
        S = lhcs.lhcs(n, m - 1)
    elif mode == "uniform":
        S = lhcs.lhcsl2norm(n, m)
    xv, fv = [], []
    if mode == "random":
        for x in S:
            f = random(x)
            if f is not None:
                fv.append(f)
                xv.append(x)
    elif mode == "uniform":
        for f_ in S:
            f = uniform(f_)[::-1]
            if f is not None:
                fv.append(f)
                xv.append(to_spherical(f))
    return (xv, fv)

def test():
    (x_, f_) = surface(10, 4, mode = "uniform")
    for i,x in enumerate(x_):
        f = random(x)
        print(f_[i], f)

if __name__ == "__main__":
    seed = 123456
    rng.seed(seed)
    n = {3: 1000, 4:2000, 8:4000}
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
    
    path = "./data/spherical/"
    try:
        os.makedirs(path)
    except OSError:
        pass
    
    utils.cat(f)
    outfile = path + "spherical-{0:d}d.out".format(m)
    utils.save(f, outfile)

import sys
import os
import math
import random

sys.path.append("./")
from paretoviz.utils import fmt
import ndsort

"""
This script generates a spherical surface on the first quadrant
"""

def fibonacci_sphere(samples = 1, randomize = True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples
    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))
        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r
        
        if x > 0 and y > 0 and z > 0:
            points.append([x,y,z])
    return points

def surface(N):
    points = fibonacci_sphere(samples = N, randomize = True)
    return (None, points)

if __name__ == "__main__":
    n = {3: 3000}
    m = 3
    mode = "equidist"
    if len(sys.argv) > 1:
        m = int(sys.argv[1].strip())
        if len(sys.argv) > 2:
            mode = sys.argv[2].strip()
   
    print("Using {0:s} mode over {1:d} dim ...".format(mode, m))
    (x_, f_) = surface(n[m])

    print("{0:d} points generated, doing non-domination sort ...".format(len(f_)))
    idx = ndsort.ndsort(f_)
    if x_ is not None:
        x = [x_[i] for i in idx]
    f = [f_[i] for i in idx]
    print("{0:d} points found. Done.".format(len(f)))
    
    path = "./data/spherical-equidist/"
    try:
        os.makedirs(path)
    except OSError:
        pass

    outfile = os.path.join(path, "spherical-equidist-{0:d}d.out".format(m))
    fmt.cat(f)
    fmt.save(f, outfile)

import sys
import os

from utils import vectorops as vops
from utils import fmt

"""
This script takes a data file and normalizes it.
Usage: py3 normalize input.out
"""

def normalize(vals):
    """
    Normalize the data.
    """
    [lb, ub] = vops.get_bound(vals)
    return vops.normalize(vals, lb, ub)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 normalize.py [input data file]")
        sys.exit(1)
    
    rawfpath = sys.argv[1].strip()
    vals = fmt.load(rawfpath)
    
    print("Normalizing {0:d} data points.".format(len(vals)))
    vals_ = normalize(vals)
    fmt.cat(vals_)
    
    path, rawfile = os.path.split(rawfpath)
    normfile = os.path.join(path, rawfile.split('.')[0] + "-norm.out")
    print("Saving normalized data into {0:s} ...".format(normfile))
    fmt.save(vals_, normfile)

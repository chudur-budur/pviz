import os
import sys

sys.path.insert(0, "./utils")
import vectorops as vops
import utils

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
    fullpath = sys.argv[1]
    path, rawfile = os.path.split(fullpath)
    normfile = os.path.join(path, rawfile.split('.')[0] + "-norm.out")
    vals = utils.load(fullpath)
    print("Normalizing {0:d} data points.".format(len(vals)))
    vals_ = normalize(vals)
    print("Saving normalized data into {0:s} ...".format(normfile))
    utils.save(vals_, normfile)

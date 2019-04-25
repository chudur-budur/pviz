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
    raw_file = sys.argv[1].strip()
    norm_file = raw_file.split('.')[0] + "-norm.out"
    vals = utils.load(raw_file)
    print("Normalizing {0:d} data points.".format(len(vals)))
    vals_ = normalize(vals)
    print("Saving normalized data into {0:s} ...".format(norm_file))
    utils.save(vals_, norm_file)

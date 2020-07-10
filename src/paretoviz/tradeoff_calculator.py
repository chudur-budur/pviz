import os
import sys
from scipy.spatial import cKDTree

from utils import fmt

"""
This script calculates the knee value from a data file. 
This script assumes that the data are already normalized.
"""

def compute_tradeoff(a, eps = 0.05, penalize_extremes = False):
    """
        Calculate the trade-off weight mu(xi,xj) described in this paper:

        Rachmawati, L. & Srinivasan, D. 
        Multiobjective Evolutionary Algorithm With Controllable Focus on the Knees of the Pareto Front 
        IEEE Transactions on Evolutionary Computation, 
        Institute of Electrical and Electronics Engineers (IEEE), 2009, 13, 810-824

        The neighbourhood will be considered as epsilon neighbour. 
        All the objective function vector needs to be normalized.
    """
    n, m = len(a), len(a[0])
    tree = cKDTree(a)
    mu = [0.0] * n
    for i in range(n):
        # First try to find neighbors within epsilon radius
        neighbors = tree.query_ball_point(a[i], eps)
        # If the neighborhood is empty then get m + 1 closest neighbors.
        # The m + 1 comes from the total number of vertices in a m-dim simplex.
        if len(neighbors) < m + 1:
            neighbors = tree.query(a[i], k = m + 1)[1]
        w = []
        for j in neighbors:
            gain, loss = 0.0, 0.0
            for k in range(m):
                gain = gain + max(0, a[j][k] - a[i][k])
                loss = loss + max(0, a[i][k] - a[j][k])
            # what if the denominator is 0? 
            # i.e. there is no loss from point i
            ratio = gain/float(loss) if loss > 0 else float('inf')
            if ratio < float('inf'):
                w.append(ratio)
        if len(w) > 0:
            if penalize_extremes:
                # The below is a modification to the original
                # knee point measurement where the extreme 
                # points will be penalized.
                denom = max(a[i]) - min(a[i])
                mu[i] = min(w)/denom if denom > 0.0 else mu[i]
            else:
                mu[i] = min(w)
    min_mu = min([m for m in mu if m > 0.0])
    mu = [m if m > 0.0 else min_mu for m in mu]
    return mu

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tradeoff.py [normalized data file] [epsilon]")
        sys.exit(1)

    epsilon = 0.125 if len(sys.argv) < 3 else float(sys.argv[2].strip())
    
    normfpath = sys.argv[1].strip()
    points = fmt.load(normfpath)
    
    mu = compute_tradeoff(points, epsilon, normalize = False)
    fmt.cat(mu)
    
    path, normfname = os.path.split(normfpath)
    mufpath = os.path.join(path, normfname.split('.')[0] + "-mu.out")
    print("Saving tradeoff values to {0:s} ...".format(mufpath))
    fmt.save(mu, mufpath)

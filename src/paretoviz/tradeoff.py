import os
import sys
from scipy.spatial import cKDTree

from utils import fmt

"""
This script calculates the knee value from a data file. 
This script assumes that the data are already normalized.
"""

def compute_tradeoff(points, epsilon = 0.05, normalize = False):
    """
        Calculate the trade-off weight mu(xi,xj) described in this paper:

        Rachmawati, L. & Srinivasan, D. 
        Multiobjective Evolutionary Algorithm With Controllable Focus on the Knees of the Pareto Front 
        IEEE Transactions on Evolutionary Computation, 
        Institute of Electrical and Electronics Engineers (IEEE), 2009, 13, 810-824

        The neighbourhood will be considered as epsilon neighbour. 
        All the objective function vector needs to be normalized.
    """
    m = len(points[0])
    n = len(points)
    print("Computing neighborhood ...")
    tree = cKDTree(points)
    print("Computing tradeoff values ...")
    mu, mu_ = [0.0] * n, [0.0] * n
    # for i in idx:
    for i in range(n):
        # First try to find neighbors within epsilon radius
        nbrs = tree.query_ball_point([points[i]], epsilon).tolist()[0]
        # If the neighborhood is empty then get m + 1 closest neighbors.
        # The m + 1 comes from the total number of vertices in a m-dim simplex.
        if len(nbrs) < m + 1:
            nbrs = tree.query([points[i]], k = m + 1)[1].tolist()[0]
        w = []
        for j in nbrs:
            gain, loss = 0.0, 0.0
            for m_ in range(m):
                gain = gain + max(0, points[j][m_] - points[i][m_])
                loss = loss + max(0, points[i][m_] - points[j][m_])
            # what if the denominator is 0? 
            # i.e. there is no loss from point i
            ratio = gain/float(loss) if loss > 0 else float('inf')
            if ratio < float('inf'):
                w.append(ratio)
        if len(w) > 0:
            mu[i] = min(w)
            # The below is a modification to the original
            # knee point measurement where the extreme 
            # points will be penalized.
            denom = max(points[i]) - min(points[i])
            mu_[i] = min(w)/denom if denom > 0.0 else mu[i]
    min_mu = min([m for m in mu if m > 0.0])
    mu = [m if m > 0.0 else min_mu for m in mu]
    min_mu_ = min([m for m in mu_ if m > 0.0])
    mu_ = [m if m > 0.0 else min_mu_ for m in mu_]
    return mu_ if normalize else mu

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tradeoff.py [normalized data file] [epsilon]")
        sys.exit(1)
    
    fullpath = sys.argv[1].strip()
    path, filename = os.path.split(fullpath)
    epsilon = 0.125
    if len(sys.argv) > 2:
        epsilon = float(sys.argv[2].strip())
    mufile = os.path.join(path, filename.split('.')[0] + "-mu.out")
    points = fmt.load(fullpath)
    mu = compute_tradeoff(points, epsilon, normalize = False)
    
    fmt.cat(mu)
    
    print("Saving tradeoff values to {0:s} ...".format(mufile))
    fmt.save(mu, mufile)

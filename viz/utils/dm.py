"""dm.py -- A Collection of Different Utility Functions for Decision Making (DM)
    
    This module provides different utility functions for decision making (DM) in
    multi-objective optimization (MOP) scenario, for example, finding knee 
    points and the relative trade-off of the points in the objective function space.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np
from scipy.spatial import cKDTree
import viz.utils.transform as tr

__all__ = ["epsilons", "nadir", "ideal", "knees", "tradeoff"]

# A set of good epsilon values to be used while computing 
# the tradeoff neighborhoods with respect to the dimension 
# of the space.
epsilons = {'3d': 0.125, '4d': 0.125, '5d': 0.25, \
            '6d': 0.375, '7d': 0.497, '8d': 0.497, \
            '9d': 0.6, '10d': 0.7}

def nadir(F):
    r"""Find the Nadir objective vector from 'F'.

    Parameters
    ----------
    F : ndarray
        A sample of `n` data points in `m` dimension, i.e. `|F| = n x m`.

    Returns
    -------
    Fmax : ndarray
        The 'numpy.max' values from each column.
    """

    Fmax = np.max(F, axis=0)
    return Fmax

def ideal(F):
    r"""Find the Ideal objective vector from 'F'.
    
    Parameters
    ----------
    F : ndarray
        A sample of `n` data points in `m` dimension, i.e. `|F| = n x m`.

    Returns
    -------
    Fmin : ndarray
        The 'numpy.min' values from each column.
    """

    Fmin = np.min(F, axis=0)
    return Fmin

def knees(Mu):
    r"""Find the best trade-off values.

    This function filters out the best trade-off values from a vector of
    trade-off values in 'Mu'. This function considers the values greater
    or equal to 'mean(Mu) + t * stdev(Mu)' are the `best` trade-off 
    values, where t = 2.75 found to be optimal. It was found empirically.

    Parameters
    ----------
    Mu : ndarray
        A vector of 'n' floats, i.e. '|Mu| = n x 1'.

    Returns
    -------
    Ik : ndarray (or None)
        An array of integers, each value is the index of best trade-off 
        values in 'Mu'. A 'None' will be returned if 'Ik' is empty.
    """

    # filter out nan and inf values    
    Mu_ = Mu[(~np.isnan(Mu)) & (~np.isinf(Mu))]

    
    mu, sigma = Mu_.mean(), Mu_.std()
    # We are considering 2*sigma are the "best" 
    # trade-off points, i.e. knee points.
    dev = (Mu_ - mu) / sigma
    
    # Ik = np.where(dev >= 2)[0]
    
    d = mu + 2 * sigma
    Ik = np.where(Mu_ >= d)[0]

    if Ik.shape[0] < 1 and dev.max() > 1:
        Ik = np.array([np.array(np.argmax(Mu_))])

    return Ik if Ik.shape[0] > 0 else None


def tradeoff(F, epsilon=0.125, k=None, penalize_extremes=False):
    r"""Calculate the trade-off values in the objective vectors 'F' to find the 'knee points'.

    Calculate the trade-off weight 'mu(.)' of each points in the objective vectors in 'F'. 
    The method is described in [1]_.

    By default, the neighbourhood will be considered as `epsilon = 0.125` neighbour, 
    therefore, all the objective function vector will be normalized within '[0.0, 1.0]' 
    before the calculation.
    
    Parameters
    ----------
    F : ndarray
        Input data points 'F' of 'm'-dimensional objective function vector, 
        i.e. '|F| = n x m'.
    epsilon : float, optional
        The 'epsilon' neighborhood radius. Default 0.125 when optional.
    k : int, optional
        The number of neighboring points. If 'k' is set, 'epsilon' will 
        not be used. Default 'None' when optional.
    penalize_extremes : boolean, optional
        The method discussed in [1]_ puts higher weights on extreme points. 
        If this variable is set to 'True', the extreme points will have lower 
        trade-off weights, leaving only the knee points in the non-extreme 
        (other/intermediary) part of the Pareto-optimal front. Default 'None' 
        when optional

    Returns
    -------
    Mu : ndarray
        An array of trade-off values of each corresponding vector in 'F'. 
        It's an array of floats, i.e. '|Mu| = n x 1'. `Mu` values are normalized 
        within [0.0,1.0].
    Ik : ndarray
        The indices of "knee" points in 'F'. Returns 'None' if no such point exists. 

    References
    ----------
    .. [1] Rachmawati, L. & Srinivasan, D., Multiobjective Evolutionary Algorithm 
        With Controllable Focus on the Knees of the Pareto Front. IEEE Transactions 
        on Evolutionary Computation, Institute of Electrical and Electronics 
        Engineers (IEEE), 2009, 13, 810-824
    """

    # use either epsilon or k
    if k:
        epsilon = None

    n,m = F.shape
    F_ = tr.normalize(F, lb=np.zeros(F.shape), ub=np.ones(F.shape))
    
    # Threshold on neighborhood size: 2m + 2
    s = 2 * m + 2
    # s = 2 * m + 1
    # s = 2 * m
    
    tree = cKDTree(F_)
    
    Mu = np.full(n, -np.inf)
    for i in range(n):
        # use epsilon
        if epsilon:
            S = np.array(tree.query_ball_point(F_[i], r = epsilon))
        # use k
        else:
            S = tree.query(F_[i], k = k)[1]
        
        # if neighborhood is sparse, get at least 2m + 2 number of points
        # The total number of vertices in a m-dim simplex is m+1
        if S.shape[0] < s:
            S = tree.query(F_[i], k = s + 1)[1]

        diff = F_[S] - F_[i]
        loss = np.maximum(0, diff).sum(axis = 1)
        gain = np.maximum(0, -diff).sum(axis = 1)

        np.warnings.filterwarnings('ignore')
        T = loss / gain
        
        Mu[i] = np.nanmin(T)
        
        if penalize_extremes:
            denom = max(F_[i]) - min(F_[i])
            Mu[i] = np.nanmin(T)/denom if denom > 0.0 else Mu[i]
        
    Ik = knees(Mu)
    Mu = tr.normalize(Mu, lb=np.array([0.0]), ub=np.array([1.0]))
    return Mu, Ik

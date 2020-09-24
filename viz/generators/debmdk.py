"""debmdk.py -- A Module to Generate Points on a 'DEBMDK' Problem
    
    This module provides functions to generate points on an m-dimensional 
    Knee problem ('DEBMDK') [1]_.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA


    References
    ----------
    .. [1] Branke J., Deb K., Dierolf H., Osswald M. (2004) Finding Knees in 
        Multi-objective Optimization. In: Yao X. et al. (eds) Parallel Problem 
        Solving from Nature - PPSN VIII. PPSN 2004. Lecture Notes in Computer 
        Science, vol 3242. Springer, Berlin, Heidelberg.

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np
from viz.generators import dtlz2

__all__ = ["surface"]

def rf(x, k=1):
    r"""The radius function for the knee problem.
    """
    return 5.0 + (10.0 * (x - 0.5) * (x - 0.5)) \
            + ((2.0 / k) * np.cos(2.0 * k * np.pi * x))

def surface(r=1, k=1, n=10, m=2, mode='lhc', **kwargs):
    r"""Generate `n` number of points on an `m`-dimensional knee ('DEBMDK') problem [1]_.

    The Pareto-optimal front for "DEBMDK" problem is basically a hypersurface of
    a distorted 'm'-sphere, such that there are 'K' number of bulges on it.

    The radius of the `m`-sphere is specified as 1. The total number of "knees"
    is specified in `k`. The point generation is currently done in two ways of 
    random sampling -- Latin Hypercube (LHC) and LHC with normalization. Other 
    ways will be added later.

    For non-random and equidistant point samplin,g we use Das-Dennis's method 
    [3]_ to generate points using NBI technique.

    Parameters
    ----------
    k : int, optional
        The total number of knees. Default 1 when optional.
    n : int, optional
        The total number of points. Default 10 when optional.
    m : int, optional
        The dimension of the sphere. Default 2 when optional.
    mode : str, {'lhc', 'lhcl2', 'dd'}, optional
        If `mode = `lhc``, then LHC sampling will be used and points will be generated
        using standard spherical coordinate systems. If `mode = `lhcl2``, then we will
        use a normalized LHC sampling to generate uniformly distributed points on the
        sphere using the method described in [2]_. If 'mode = 'dd', then we will generate
        points using the subproblem decomposition technique used in NBI method 
        (a.k.a. "Das-Dennis's Approach") discussed in [3]_. Default first when optional.

    Other Parameters
    ----------------
    delta : float, optional
        `delta` value for normalized LHC, this is used so that we only keep vectors
        `V` such that `np.linalg.norm(V, 1) > delta`. The default value is 0.0001 but
        you might want to change it according to your application.

    Returns
    -------
    F : ndarray
        `n` points on the `m`-sphere, i.e. `|F| = n x m`.
    X : ndarray
        `n` points of the `m-1` dimensional spherical coordinate values, i.e. `|X| = n x (m-1)`.

    References
    ----------
    .. [1] Branke J., Deb K., Dierolf H., Osswald M. (2004) Finding Knees in 
        Multi-objective Optimization. In: Yao X. et al. (eds) Parallel Problem 
        Solving from Nature - PPSN VIII. PPSN 2004. Lecture Notes in Computer 
        Science, vol 3242. Springer, Berlin, Heidelberg.

    .. [2] Simon C., "Generating uniformly distributed numbers on a sphere," online: 
        http://corysimon.github.io/articles/uniformdistn-on-sphere/ 

    .. [3] I. Das and J. E. Dennis, "Normal-Boundary Intersection: A New Method for 
        Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems," 
        SIAM Journal on Optimization, vol. 8, (3), pp. 631-27, 1998.
    """
    # This is needed for dtlz2
    delta = kwargs['delta'] if (len(kwargs) > 0 and 'delta' in kwargs) else 0.0001
    
    F, X = dtlz2.surface(r=r, n=n, m=m, mode=mode, delta=delta)

    g = 1.0
    vrf = np.vectorize(rf)
    R = (np.sum(vrf(X), axis = 1) / X.shape[1])
    F = (F.T * (g * R)).T

    return F, X

"""c2dtlz2.py -- A Module to Generate Points on a C2-DTLZ2 Pareto-optimal front.
    
    This module provides functions to generate points on 'm'-dimensional Pareto-front
    on a C2-DTLZ2 problem [1]_. 

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA

    References
    ----------
    .. [1] H. Jain and K. Deb, "An Evolutionary Many-Objective Optimization Algorithm Using 
        Reference-Point Based Nondominated Sorting Approach, Part II: Handling Constraints 
        and Extending to an Adaptive Approach," in IEEE Transactions on Evolutionary Computation, 
        vol. 18, no. 4, pp. 602-622, Aug. 2014, doi: 10.1109/TEVC.2013.2281534. 

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np
from viz.generators import dtlz2
from viz.utils import transform as tr

__all__ = ["surface"]

def cvf(F):
    r"""The constraint function.

    This procedure computes the constraint violation values on
    an objective vector 'F'.
    """
    m = F.shape[0]
    if m == 2:
        r = 0.25
    elif m == 3:
        r = 0.4
    else:
        r = 0.5
    r1 = 0.50 * r
    lhs = min([((f - 1.0) ** 2) + np.sum([(F[j] ** 2) for j in range(m) if j != i]) - (r ** 2.0) \
            for i,f in enumerate(F)])
    rhs = np.sum([((f - (1.0 / (m ** 0.5))) ** 2) for f in F]) - (r1 ** 2)
    c = min(lhs, rhs)
    return c

def get_feasible(F, X):
    r"""Filters out the feasible solutions from given data points 'F' (and 'X').

    This function computes the constraint violation values for each data points 
    in matrix 'F' and returns only the feasible points along with the corresponding
    design variable values in 'X'.
    """

    G = np.apply_along_axis(cvf, 1, F)
    If = np.where(G <= 0.0)[0]
    return F[If], X[If], G[If]

def surface(r=1, n=10, m=2, mode='lhc', **kwargs):
    r"""Generate `n` number of points on split `m`-shpere.

    The details of 'C2-DTLZ2' is described in [1]_.

    The radius of the `m`-sphere is specified in `r`. The point generation is 
    currently done in two ways of random sampling -- Latin Hypercube (LHC) and
    LHC with normalization. Other ways will be added later.

    Parameters
    ----------
    r : float, optional
        The radius of the sphere. Default 1 when optional.
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
    .. [1] H. Jain and K. Deb, "An Evolutionary Many-Objective Optimization Algorithm Using 
        Reference-Point Based Nondominated Sorting Approach, Part II: Handling Constraints 
        and Extending to an Adaptive Approach," in IEEE Transactions on Evolutionary Computation, 
        vol. 18, no. 4, pp. 602-622, Aug. 2014, doi: 10.1109/TEVC.2013.2281534.

    .. [2] Simon C., "Generating uniformly distributed numbers on a sphere," online: 
        http://corysimon.github.io/articles/uniformdistn-on-sphere/ 

    .. [3] I. Das and J. E. Dennis, "Normal-Boundary Intersection: A New Method for 
        Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems," 
        SIAM Journal on Optimization, vol. 8, (3), pp. 631-27, 1998.
    """
    # This is needed for dtlz2
    delta = kwargs['delta'] if (len(kwargs) > 0 and 'delta' in kwargs) else 0.0001

    F, X = dtlz2.surface(r=r, n=n, m=m, mode=mode, delta=delta) 
    F, X, G = get_feasible(F, X)
    CV = tr.normalize(G)
    return F, X, G, CV

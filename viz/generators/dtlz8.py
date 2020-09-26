"""dtlz8.py -- A Module to Generate Points on a 'DTLZ8' Problem
    
    This module provides functions to generate points on a 'DTLZ8' problem [1]_.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA

    References
    ----------
    .. [1] K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective 
        Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np
from viz.utils import sampling as smp
from viz.utils import transform as tr

__all__ = ["surface"]

# Suggested values for the parameters:
#    m, nl, ns = 2, 50, 750
#    m, nl, ns = 3, 100, 11000
#    m, nl, ns = 4, 400, 80000
#    m, nl, ns = 6, 600, 160000
#    m, nl, ns = 8, 800, 160000

def surface(m=2, nl=50, ns=750, mode='lhc', **kwargs):
    r"""Generate `nl + ns` number of points on 'DTLZ8' problem.
    
    This function generates points on the Pareto-optimal front of 'DTLZ8' 
    problem [1]_. The Pareto-optimal hypersurface is composed of a 3 dimensional 
    (2 dimensional in the case of 'm = 2') line and an 'm'-dimensional hypersurface.

    Since we are not solving the actual 'DTLZ8' problem, this function samples
    points directly on the Pareto-optimal hypersurface. As a result this function
    does not return the design variables.

    Parameters
    ----------
    m : int, optional
        The dimension of the Pareto-optimal front. Default 2 when optional.
    nl : int, optional
        The number of points on the 3 (or 2)-dimensional line. Default 50 when optional.
    ns : int, optional
        The number of points on the 'm'-dimensional hypersurface. Default 750 when optional.
    mode : str, {'lhc', 'lhcl2', 'grid'}, optional
        If `mode = `lhc``, then LHC sampling will be used and points will be generated
        using standard spherical coordinate systems. If `mode = `lhcl2``, then we will
        use a normalized LHC sampling to generate uniformly distributed points on the
        sphere using the method described in [2]_. If 'mode = 'grid', then we will 
        generate points using a grid. 'Default first when optional.

    Other Parameters
    ----------------
    delta : float, optional
        `delta` value for normalized LHC, this is used so that we only keep vectors
        `V` such that `np.linalg.norm(V, 1) > delta`. The default value is 0.0001 but
        you might want to change it according to your application.
    feasible_only : bool, optional
        When 'feasible_only = True', the function will return only the feasible solutions.
        Otherwise it will return all the points. Default 'True' when optional.

    Returns
    -------
    F : ndarray
        `n` number of points in the `m`-dimension, i.e. `|F| = n x m`.
    X : ndarray
        `n` number of points in `m-1`-dimension, i.e. `|X| = n x (m-1)`.
    G : ndarray
        'n' constraint violation values for each point. Where `|G| = n x m`.
        Since there are 'm' constraints. 'G[i,j] < 0' means infeasible and
        'G[i,j] >= 0' means feasible.
    CV : ndarray
        'n' cumulative constraint violation values. Where '|CV| = n x 1'. 'CV'
        is just the row wise sums of 'G'.

    References
    ----------
    .. [1] K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective 
        Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.

    .. [2] Simon C., "Generating uniformly distributed numbers on a sphere," online: 
        http://corysimon.github.io/articles/uniformdistn-on-sphere/ 
    """
    
    mkeys = {'lhc', 'lhcl2', 'grid'}
    if mode not in mkeys:
        raise KeyError("'mode' has to be any of {:s}.".format(str(mkeys)))    

    feasible_only = kwargs['feasible_only'] if (len(kwargs) > 0 and 'feasible_only' in kwargs) else True
    delta = kwargs['delta'] if (len(kwargs) > 0 and 'delta' in kwargs) else 0.0001

    t = {2: 0, 3: 0, 4: -0.15, 6: -0.7, 8: -1.5}
    d = {2: m + 1, 3: m + 1, 4: m, 6: m - 2.5, 8: m - 4.5}

    pt1 = np.zeros((1, m))
    pt1[0,-1] = 1.0
    pt2 = np.ones((1, m)) / d[m]

    delta = 1 / nl

    L = (np.ones((nl, 1)) * pt1) \
        + (np.arange(0, 1, delta) * (pt2 - pt1).T).T
    # print("L.shape", L.shape)

    if mode == 'lhcl2':
        S = (np.ones((ns, m - 1)) / (m + 1)) \
            + (smp.lhcl2(ns, m - 1, delta = delta) * 1 - (1.0 / (m + 1)))
    elif mode == 'lhc':
        S = (np.ones((ns, m - 1)) / (m + 1)) \
            + (smp.lhc(ns, m - 1) * 1 - (1.0 / (m + 1)))
        print(smp.lhc(ns, m-1).shape)
    elif mode == 'grid':
        Sg = smp.grid(ns, m - 1)
        ns = Sg.shape[0]
        S = (np.ones((ns, m - 1)) / (m + 1)) \
            + (Sg * 1 - (1.0 / (m + 1)))

    # print("S.shape", S.shape)

    Fs = np.append(S, np.zeros((S.shape[0], 1)), 1)
    # print("Fs.shape", Fs.shape)
    Fs[:,-1] = (1 - np.sum(S, axis = 1)) / 2.0
 
    # for i in range(ns):
    #     Fs[i,-1] = (1 - np.sum(S[i])) / 2.0
    #     for j in range(m-1):
    #         Gs[i,j] = Fs[i,-1] - (1.0 - (m * Fs[i,j])) # infeasible < 0, feasible >= 0
    #     Gs[i,-1] = Fs[i,-1] - t[m] # infeasible < 0, feasible >= 0

    Gs = np.zeros(Fs.shape)
    # print("Gs.shape", Gs.shape)
    Gs[:,0:m-1] = (Fs[:,-1] - (1.0 - m * Fs[:,0:m-1]).T).T
    Gs[:,-1] = Fs[:,-1] - t[m]
    
    F = np.concatenate((L, Fs), axis=0)
    
    # For X, we assume the design variables are the
    # projection of F onto m-1 dimensions where the 
    # m-th dimension being collapsed.
    X = F[:,:-1]

    # print("F.shape", F.shape)
    
    maxG = np.max(Gs, axis = 0)
    # print("maxG", maxG)
    Gl = np.tile(maxG, (L.shape[0], 1)) 
    # print("Gl", Gl)
    
    G = np.concatenate((Gl, Gs), axis=0)
    # print("G.shape", G.shape)

    if feasible_only:
       If = np.where(~np.any(G < 0, axis=1))
       F = F[If]
       G = G[If]
       X = X[If]
    
    CV = tr.normalize(np.sum(G, axis=1))
    # print("CV.shape", CV.shape)
    # print("CV", CV)

    # We don't have X for this, since we are sampling the surface itself.
    # X = None

    return F, X, G, CV

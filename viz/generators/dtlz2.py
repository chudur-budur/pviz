"""dtlz2.py -- A Module to Generate Points on a 'DTLZ2' Problem ('m'-Sphere)
    
    This module provides functions to generate points on a 'DTLZ2' problem [1]_.

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

__all__ = ["surface"]


def surface(r=1, n=10, m=2, mode='lhc', **kwargs):
    r"""Generate `n` number of points on 'DTLZ2' problem (`m`-shpere).
    
    This function generates points on a 'DTLZ2' problem [1]_, which is
    in fact an 'm'-sphere. Therefore, this function samples points on an
    'm'-sphere.

    The radius of the `m`-sphere is specified in `r`. The point generation is 
    currently done in two ways of random sampling -- Latin Hypercube (LHC) and
    LHC with normalization. Other ways will be added later.

    For non-random and equidistant point samplin,g we use Das-Dennis's method 
    [3]_ to generate points using NBI technique.

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
    .. [1] K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective 
        Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.

    .. [2] Simon C., "Generating uniformly distributed numbers on a sphere," online: 
        http://corysimon.github.io/articles/uniformdistn-on-sphere/ 

    .. [3] I. Das and J. E. Dennis, "Normal-Boundary Intersection: A New Method for 
        Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems," 
        SIAM Journal on Optimization, vol. 8, (3), pp. 631-27, 1998.
    """

    mkeys = {'lhc', 'lhcl2', 'dd'}
    if mode not in mkeys:
        raise KeyError("'mode' has to be any of {:s}.".format(str(mkeys)))
    if m < 2:
        raise ValueError("\'m\' must be greater than 2.")

    delta = kwargs['delta'] if (len(kwargs) > 0 and 'delta' in kwargs) else 0.0001

    F = np.ones((n, m))
    if mode == 'lhc':
        theta = smp.lhc(n=n, m=m - 1)
        X = np.copy(theta)
         
        # # print for sanity check
        # for i in range(m):
        #     if i < m-1:
        #         s1 = "".join(["sin(t_{:d} * pi/2)".format(v) for v in range(0,i)])
        #         s2 = "cos(t_{:d} * pi/2)".format(i)
        #         s = s1 + s2
        #     else:
        #         s1 = "".join(["sin(t_{:d} * pi/2)".format(v) for v in range(0,i-1)])
        #         s2 = "sin(t_{:d} * pi/2)".format(i-1)
        #         s = s1 + s2
        #     print("x_{:d} = ".format(i) + s)
        
        # This is what is described in wikipedia
        for i in range(m):
            if i < m-1:
                F[:,i] = np.prod(np.sin(theta[:,0:i] * (np.pi / 2)), axis = 1) \
                        * np.cos(theta[:,i] * (np.pi / 2))
            else:
                F[:,i] = np.prod(np.sin(theta[:,0:i-1] * (np.pi / 2)), axis = 1) \
                        * np.sin(theta[:,i-1] * (np.pi / 2))

        # This is reverse of what is described in wikipedia
        # # print for sanity check
        # for i in range(m):
        #     s = "".join(["sin(t_{:d} * pi/2)".format(v) for v in range(0,m - (i + 1))])
        #     if i > 0:
        #         s = s + "cos(t_{:d} * pi/2)".format(m - (i + 1)) 
        #     print("x_{:d} = ".format(i) + s)
        # for i in range(m):
        #     F[:,i] = np.prod(np.sin(theta[:,0:m - (i + 1)] * (np.pi / 2)), axis = 1)
        #     if i > 0:
        #         F[:,i] = F[:,i] * np.cos(theta[:,m - (i + 1)] * (np.pi / 2)) 

    elif mode == 'lhcl2':
        Y = smp.lhcl2(n=n, m=m, delta=delta)
        Y_ = np.linalg.norm(Y, 2, axis=1)
        F = (Y.T / Y_).T
        X = np.zeros((n, m-1))
        
        # # Print for sanity check
        # for i in range(m-2):
        #     s = "arccos(x_{:d}".format(i) + " / " + "√(" \
        #             + " + ".join(["x_{:d}²".format(v) for v in range(m-1,i-1,-1)]) \
        #             + "))"
        #     print("t_{:d} = ".format(i) + s)
        #
        # s1 = "arccos(x_{:d}".format(m-2) + " / " + "√(" \
        #         + " + ".join(["x_{:d}²".format(v) for v in range(m-1,m-3,-1)]) \
        #         + "))" + " if x_{:d} >= 0".format(m-1)
        # s2 = "(2 * pi - arccos(x_{:d}".format(m-2) + " / " + "√(" \
        #         + " + ".join(["x_{:d}²".format(v) for v in range(m-1,m-3,-1)]) \
        #         + ")))" + " if x_{:d} < 0".format(m-1)
        # print("t_{:d} = ".format(m-2) + s1 + " or " + s2)

        # This is what is explained in wikipedia
        for i in range(m-2):
            denom = np.linalg.norm(F[:,i:m], 2, axis=1)
            Inz = np.nonzero(denom)
            X[Inz,i] = np.arccos(F[Inz,i] / denom[Inz]) / (np.pi / 2) 
        denom = np.linalg.norm(F[:,m-2:m], 2, axis=1)
        Inz = np.nonzero(denom)
        X[Inz,m-2] = np.arccos(F[Inz,m-2] / denom[Inz]) / (np.pi / 2)
        
        # We don't need this since already all F[:,-1] > 0 
        # Ip = np.where(F[:,-1] >= 0)[0]
        # if len(Ip) > 0:
        #     X[Ip, m-2] = np.arccos(F[Ip,m-2] / denom[Ip]) / (np.pi / 2)
        # In = np.where(F[:,-1] < 0)[0]
        # if len(In) > 0:
        #     X[In, m-2] = (2 * np.pi) - (np.arccos(F[In,m-2] / denom[In]) / (np.pi / 2))

        # This is reverse of what is explained in wikipedia
        # for i in range(m-1):
        #     denom = np.linalg.norm(F[:,(m-1)-i::-1], 2, axis = 1)
        #     X[:,i] = np.arccos(F[:,(m-1)-i] / denom) / (np.pi / 2)

    elif mode == 'dd':
        F = smp.das_dennis(n = n, m = m)
        X = np.zeros((F.shape[0], m-1))
        for i in range(m-2):
            denom = np.linalg.norm(F[:,i:m], 2, axis=1)
            Inz = np.nonzero(denom)
            X[Inz,i] = np.arccos(F[Inz,i] / denom[Inz]) / (np.pi / 2)
        
        denom = np.linalg.norm(F[:,m-2:m], 2, axis=1)
        Inz = np.nonzero(denom)
        X[Inz,m-2] = np.arccos(F[Inz,m-2] / denom[Inz]) / (np.pi / 2)
    
    # We flip it since all MOPs are flipped and we
    # are using the definitions used in wikipedia.
    F = F[:,::-1] 
    return r * F, X

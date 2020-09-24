"""sampling.py -- A Collection of Different Utility Functions for DOE
    
    This module provides different utility functions for random sampling, 
    for example, Latin Hypercube Sampling (LHS) etc.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import random
import warnings
import numpy as np
from scipy.special import comb

__all__ = ["grid", "lhs", "lhcl2", "das_dennis", "risez"]

def grid(n=100, m=2):
    r""" A simple meshgrid generator for 'm'-dimensional coordinate.

    A very simple function to generate grid points in an 'm'-dimensional
    coordinate. It will generate `⌈n^(1/m)⌉^m' number of points.

    Parameters
    ----------
    n : int, optional
        The number of points. Default is 100 when optional.
    m : int
        The number of dimensions. Default is 2 when optional.

    Returns
    -------
    F : ndarray
        A sample of `⌈n^(1/m)⌉^m` data points in `m` dimension, 
        i.e. `|F| = ⌈n^(1/m)⌉^m x m`.

    """
    d = np.ceil(n ** (1/m)).astype(np.int64)
    if m > 1:
        M = np.meshgrid(*[np.linspace(0,1,d) for _ in range(m)])
        F = np.hstack(M).swapaxes(0,1).reshape(m,-1).T
    else:
        F = np.linspace(0,1,d)[:,None] # to make it (n,) to (n,1)
    if F.shape[0] != n:
        warnings.warn("Not possible to generate {:d} points in grid.".format(n))
        warnings.warn("Genenrated {:d} points instead.".format(F.shape[0]))
    return F


def lhc(n=10, m=2):
    r""" Latin Hyper-cube Sampling (LHS) of `n` points in `m` dimension.

    A very simple LHS code, every point is within `[0.0, 1.0]`.

    Parameters
    ----------
    n : int, optional
        The number of points. Default is 10 when optional.
    m : int
        The number of dimensions. Default is 2 when optional.

    Returns
    -------
    F : ndarray
        A sample of `n` data points in `m` dimension, i.e. `|F| = n x m`.

    """
    d = 1.0 / float(n) ;
    N = np.arange(n)
    F = np.zeros((n, m))
    for i in range(m):
        F[:,i] = (N * d) + (((N + 1.0) * d) - (N * d)) * np.random.random(n)
        np.random.shuffle(F[:,i])
    return F


def lhcl2(n=10, m=2, delta=0.0001):
    r""" Latin Hyper-cube Sampling of `n` points in `m` dimension with L2-norm constraint.

    Latin hypercube sampling n samples of m-dimensional points. This function guarantees 
    that 2-norm of each sample is greater than 0.0001. This function is slower than `lhc()`.
    Every point is within `[0.0, 1.0]`.

    Parameters
    ----------
    n : int
        The number of points.
    m : int
        The number of dimensions.

    Returns
    -------
    F : ndarray
        A sample of `n` data points in `m` dimension, i.e. `|F| = n x m`.

    """
    F = np.zeros((n, m))
    i = 0
    # skip_count = 0
    while i < n:
        k = n - i
        T = lhc(k, m)
        idx = np.argwhere(np.linalg.norm(T, 2, axis = 1) > delta)
        # skip_count += k - idx.shape[0]
        j = i + idx.shape[0]
        if j < n:
            F[i:j,:] = T[0:idx.shape[0],:]
            i = j
        else:
            F[i:,:] = T[0:k,:]
            i += k
    # print('skip_count =', skip_count)
    return F


def _das_dennis_inner(ref_dirs, ref_dir, p, beta, depth):
    r""" The inner function for '_das_dennis()'.

    The recursive inner function part for the _das_dennis() procedure.
    """
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * p)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * p)
            _das_dennis_inner(ref_dirs, np.copy(ref_dir), p, beta - i, depth + 1)


def _das_dennis(p, m):
    r"""Finding weights for the subproblem decomposition in NBI. 
    
    This function will be called by 'das_dennis()'. This function will
    recursively create the decomposition weights depending on the 'p'
    and 'm' values.
    """
    if p == 0:
        return np.full((1, m), 1 / m)
    else:
        ref_dirs = []
        ref_dir = np.full(m, np.nan)
        _das_dennis_inner(ref_dirs, ref_dir, p, p, 0)
        return np.concatenate(ref_dirs, axis = 0)


def das_dennis(n=100, m=2, manifold='sphere'):
    r""" Generating equidistant points on an 'm'-simplex (or 'm'-sphere).

    This function uses the recursive subdivision method described by 
    the NBI approach in [1]_ to generate equidistant points on an 
    'm'-dimensional simplex or on an 'm'-sphere.

    The method described in [1]_ is known as "Das-Dennis's" method.
    This function will approximate the partition parameter 'p' from given
    'n' and 'm' values. Therefore, if 'n' is not equal to '((m + p - 1)!)/(p!(m-1)!)',
    this function will try to generate points close to that number.

    Parameters
    ----------
    n : int, optional
        The number of points. Default is 100 when optional.
    m : int
        The number of dimensions. Default is 2 when optional.
    manifold : str, {'sphere', 'simplex'}, optional
        If the 'manifold' is 'simplex', the function will generate points 
        on an 'm'-simplex. If the 'manifold' is 'sphere', they will be on an 
        'm'-sphere. First when default.

    Returns
    -------
    F : ndarray
        A sample of `((m + p - 1)!)/(p!(m-1)!)` data points in `m` dimension, 
        i.e. `|F| = ((m + p - 1)!)/(p!(m-1)!) x m`.

    References
    ----------

    .. [1] I. Das and J. E. Dennis, "Normal-Boundary Intersection: A New Method for 
        Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems," 
        SIAM Journal on Optimization, vol. 8, (3), pp. 631-27, 1998.
    """
    mkeys = {'sphere', 'simplex'}
    if manifold not in mkeys:
        raise KeyError("Invalid 'manifold', use any of {:s}".format(str(mkeys)))

    if m == 1:
        d = np.ceil(n ** (1/m)).astype(np.int64)
        return np.linspace(0,1,d)[:,None]

    p = 1
    n_ = comb(m + p - 1, p, exact=False).astype(np.int64)
    while n_ <= n:
        p += 1
        n_ = comb(m + p - 1, p, exact=False).astype(np.int64)
    p = p - 1
    
    R = _das_dennis(m=m, p=p)
    if R.shape[0] != n:
        warnings.warn("Das-Dennis's method couldn't generate {:d} points.".format(n))
        warnings.warn("Genenrated {:d} points instead.".format(R.shape[0]))
    
    if manifold == 'sphere':
        F = R / np.linalg.norm(R, axis=1)[:,None]
    else:
        F = R
    return F

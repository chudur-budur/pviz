"""transform.py -- A collection of different utility functions for different data transformations
    
    This module provides different utility functions for 
    the transformation of data and other things.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np


__all__ = ["normalize", "pfindices"] 


def normalize(A, lb=None, ub=None):
    r"""Normalize a matrix within `[lb, ub]`.

    Just a simple normalization procedure that uses `numpy`.

    Parameters
    ----------
    A : array_like
        The input matrix.
    lb : array_like, optional
        The lower bound of each column. Default 'None' when optional.
    ub : array_like, optional
        The upper bound of each column. Default `None` when optional.

    Returns
    -------
    B : ndarray
        The result after normalizing the input array `A`.

    """
    if lb is None:
        lb = 0  if len(A.shape) == 1 else np.zeros(A.shape[1])
    if ub is None:
        ub = 1 if len(A.shape) == 1 else np.ones(A.shape[1])
    num = (A - A.min(axis=0)) * (ub - lb)
    denom = A.max(axis=0) - A.min(axis=0)
    if isinstance(denom, np.ndarray):
        denom[denom == 0] = 1
    else:
        denom = 1 if denom == 0.0 else denom
    B = lb + num / denom
    return B


def pfindices(A):
    r"""Find indices of non-dominated points in `A`.

    Find the indices of all non-dominated points (Pareto-optimal front)
    of the data points in matrix 'A'.

    Parameters
    ----------
    A : array_like
        The input matrix.

    Returns
    -------
    I : ndarray
        The indices of all non-dominated points the input array `A`.

    """

    # Testing code
    # G = np.array([[2, 1], [1, 4], [6, 2], [2, 6]])
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.scatter(G[:,0], G[:,1])
    # ax.set_xlim((0,7))
    # ax.set_ylim((0,7))
    # plt.show()
    # i = tr.pfindices(G)
    # print(i)

    # G = np.array([[2.0, 1.0], [1.0, 4.0], [4.0, 1.0], \
    #               [6.0, 0.5], [1.0, 5.0], [0.5, 6.0], \
    #               [2.0, 3.0], [1.5, 4.0], [3.0, 2.0], [4.0, 4.0]])
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.scatter(G[:,0], G[:,1])
    # ax.set_xlim((0,7))
    # ax.set_ylim((0,7))
    # i = tr.pfindices(G)
    # print(i)

    D = np.zeros((A.shape[0], A.shape[0]), dtype=np.bool)
    for i in range(A.shape[1]):
        x = A[:,i]
        if i > 0:
            D = D & (x[..., np.newaxis] > x[np.newaxis, ...])
        else:
            D = x[..., np.newaxis] > x[np.newaxis, ...]
    # print(D)
    I = np.where(~D.any(axis=1))[0]
    return I

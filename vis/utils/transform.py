"""`transform.py` -- A Collection of Different Utility Functions for Different Data Transformations
    
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
import matplotlib.cm as cm
import matplotlib.colors as mc

__all__ = ["normalize", "pfindices", "color_by_cv", "color_by_dist", \
            "enhance_color", "resize_by_tradeoff"]

def normalize(A, lb = None, ub = None):
    r"""Normalize a matrix within `[lb, ub]`.

    Just a simple normalization procedure that uses `numpy`.

    Parameters
    ----------
    'A' : array_like
        The input matrix.
    `lb` : array_like, optional
        The lower bound of each column. Default 'None' when optional.
    `ub` : array_like, optional
        The upper bound of each column. Default `None` when optional.

    Returns
    -------
    `B` : ndarray
        The result after normalizing the input array `A`.

    """
    if lb is None:
        lb = 0  if len(A.shape) == 1 else np.zeros(A.shape[1])
    if ub is None:
        ub = 1 if len(A.shape) == 1 else np.ones(A.shape[1])
    num = (A - A.min(axis = 0)) * (ub - lb)
    denom = A.max(axis = 0) - A.min(axis = 0)
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
    'A' : array_like
        The input matrix.

    Returns
    -------
    `I` : ndarray
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

    D = np.zeros((A.shape[0], A.shape[0]), dtype = np.bool)
    for i in range(A.shape[1]):
        x = A[:,i]
        if i > 0:
            D = D & (x[..., np.newaxis] > x[np.newaxis, ...])
        else:
            D = x[..., np.newaxis] > x[np.newaxis, ...]
    # print(D)
    I = np.where(~D.any(axis = 1))[0]
    return I

def color_by_cv(CV, factor = 0.8, alpha = 0.5):
    r"""Generate an array of color values from CV.

    Generate an array of RGBA color values from an array of 
    cumulative constraint violation values. This function
    uses the color gradient from `cm.cool`.

    Parameters
    ----------
    'CV': `ndarray`
        An nd-array of cumulative contstraint violation values.
    `factor`: float, optional
        If `factor = 1.0`, then we will use the actual gradient
        from `cm.cool`. When this value is smaller than 1.0, then
        the gradient will be shifted left (i.e. some portion from 
        the highest end of the color gradient will be skipped). 
        0.8 when default.
    `alpha`: float, optional
        Alpha transparency value. 0.5 when default.
        
    Returns
    -------
    `C` : ndarray
        An array of RGBA color values.
    """
    CV_ = normalize(CV, lb = 0.0, ub = 1.0)
    C = np.array([mc.to_rgba(cm.cool(v * factor), alpha) for v in CV_])
    return C

def color_by_dist(X, P, alpha = 0.5, factor = 1.75):
    r"""Generate an array of RGBA color values w.r.t distance of 'X' from 'P'

    Generate an array of RGBA color values for the corresponding points
    in `X` with respect to the their distances from a single point `P`.
    We generally use `P` as the center of mass of `X`, but can be used
    in other contexts.

    Parameters
    ----------
    'X': `ndarray`
        An set of `m`-dimensional data points, i.e. `|X| = n x m` 
    'P': 1D-`array`
        A point `P`, a 1-D array. This can be the center of mass of `X`.
    `alpha`: float, optional
        Alpha transparency value. 0.5 when default.
    `factor`: float, optional
        If `factor = 1.0`, then we will use the actual gradient
        from `cm.winter_r`. When this value is smaller than 1.0, then
        the gradient will be shifted right (i.e. some portion from 
        the lowest end of the color gradient will be skipped). 
        1.75 when default. A user might want to try with different 
        values for factor.
        
    Returns
    -------
    `C` : ndarray
        An array of RGBA color values.
    """
    D = normalize(np.linalg.norm(P - X, axis = 1), lb = 0.1, ub = 1.0)
    C = np.array([mc.to_rgba(cm.winter_r(v * 1.75), alpha) for v in D])
    return C

def enhance_color(C, Ik, alpha = 1.0, color = mc.TABLEAU_COLORS['tab:red']):
    r"""Enhance the color of selected data points.

    Given an array of RGBA color values `C`, this function will enhance
    all the points indexed by `Ik` by recoloring them with TABLEAU red
    color. Assuming that the color of the other points won't overlap with
    the enhanced points.
    
    Parameters
    ----------
    'C': `ndarray`
        An array of RGBA color values as input.
    'Ik': `array` of `int`
        An array of `int` indices.
    `alpha`: float, optional
        Alpha transparency value. 0.5 when default.
    `color`: RGB color value, optional
        The color to be used to enhance the points. 
        `mc.TABLEAU_COLORS['tab:red']` when default.
        
    Returns
    -------
    `C_` : ndarray
        An array of RGBA color values.
    """
    C_ = np.array(C, copy=True)
    C_[Ik] = np.array([mc.to_rgba(color, alpha) for _ in range(C[Ik].shape[0])])
    return C_

def default_color(n, alpha = 1.0):
    r"""Get an array of RGBA color values for default coloring.

    In any case, if we need to revert the point colorings to the 
    default matplotlib (`mc.TABLEAU_COLORS['tab:blue']`) coloring 
    scheme, we can use this function.

    Parameters
    ----------
    'n': int
        The length of the output array containing RGBA color values.
    `alpha`: float, optional
        Alpha transparency value. 0.5 when default.
        
    Returns
    -------
    `C` : ndarray
        An array of RGBA color values.
    """
    C = np.array([mc.to_rgba(mc.TABLEAU_COLORS['tab:blue'], alpha) for _ in range(n)])
    return C

def resize_by_tradeoff(Mu, k = None, minsize = 2.0, maxsize = 10.0, kminsize = 3.0, kmaxsize = 5.0):
    r"""Resize the points w.r.t. tradeoff values.

    If we need to resize the points with respect to the tradeoff values, 
    we can use this function. This function assumes `Mu` values are within [0,1]. 

    Since point size of 0 is not visible, we normalize them within `[minsize, maxsize]`.
    In order to capture the relative tradeoffs, we then scale the sizes by the power of 2.
    If the user wants to emphasize a specific set of points so that they 'stand out' 
    from all the rest of the points, they can be indexed with `k`. Those points will be 
    pronounced by normalizing them within `[kminsize, kmaxsize]` and then scale them by 
    the power of 2.
    
    Parameters
    ----------
    'Mu': 1-D `array`
        A 1D-array of tradeoff values, float. Also `0.0 <= Mu <= 1.0`.
    `k`: 1-D `array`, optional
        A 1D-array of `int` indices to be used to specify which points
        will be increased in size by `kfactor`. `None` when default.
    `minsize`: float, optional
        The minimum allowable size of each point before the exponential 
        scaling. 2.0 when default.
    `maxsize`: float, optional
        The maximum allowable size of each point before the exponential 
        scaling. 10.0 when default.
    `kminsize`: float, optional
        The minimum allowable size of the points indexed by `k`, before 
        the exponential scaling. 10.0 when default.
    `kmaxsize`: float, optional
        The maximum allowable size of the points indexed by `k`, before 
        the exponential scaling. 10.0 when default.
    """
    S = np.power(normalize(Mu, lb = np.array([minsize]), ub = np.array([maxsize])), 2)
    if k is not None:
        S[k] = np.power(normalize(S[k], lb = np.array([kminsize]), ub = np.array([kmaxsize])), 3)
    return S

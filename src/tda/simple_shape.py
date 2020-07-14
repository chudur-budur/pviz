"""`simple_shape.py` -- A simple convex-hull shape extraction algorithm

    This module provides different utility functions for a `simple
    convex-hull shape extraction` algorithm. In the case of simple
    convex-hull, we first collapse a dimension (drop a column) in 
    the data-set, then we project the data points on a unit simplex, 
    then we apply qhull algorithm to find the convex-hull boundary 
    points. Each depth contour found from the qhull will be considered
    as the `layers` for palette-viz representation of the data points.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np

def collapse(F, d = 0):
    r""" Function to collapse a dimension of the data points `F`.

    A very simple function that drops a column in the data points `F`.

    Parameters
    ----------
    'F' : ndarray
        A sample of `n` data points in `m` dimension, i.e. `|F| = n x m`.
    `d` : int, optional
        The dimension to be dropped. Default is 0 (i.e. first coordinate) 
        when optional.

    Returns
    -------
    `F` : ndarray
        Data points with one less column than that of `F`, i.e. `|F| = n x (m-1)` 

    """
    return F[:,np.delete(np.arange(0,F.shape[1]).astype(np.int64), d)]

def project(F):
    r""" Function to project data points `F` onto a sunit simplex.

    A very simple function that projects data points `F` onto a 
    unit simplex, where `|F| = n x m`.

    Parameters
    ----------
    'F' : ndarray
        A sample of `n` data points in `m` dimension, i.e. `|F| = n x m`.

    Returns
    -------
    `P` : ndarray
        Projected data points from `F`, i.e. `|P| = n x m`.

    """
    N,M = F.shape
    u = 1 / np.sqrt(np.ones(M) * M)
    uTF = np.sum(u.T * F, axis = 1)
    uTuTFT = (np.tile(u, (N,1)).T * uTF).T
    P = (F - uTuTFT) + (u / np.sqrt(M))
    return P

def shape():
    pass

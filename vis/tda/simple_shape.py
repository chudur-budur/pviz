"""simple_shape.py -- A simple convex-hull shape extraction algorithm

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
from scipy.spatial import ConvexHull

def collapse(F, d=0):
    r""" Function to collapse a dimension of the data points `F`.

    A very simple function that drops a column in the data points `F`.

    Parameters
    ----------
    F : ndarray
        A sample of `n` data points in `m` dimension, i.e. `|F| = n x m`.
    d : int, optional
        The dimension to be dropped. Default 0 (i.e. first coordinate) 
        when optional.

    Returns
    -------
    F : ndarray
        Data points with one less column than that of `F`, i.e. `|F| = n x (m-1)` 

    """
    return F[:,np.delete(np.arange(0,F.shape[1]).astype(np.int64), d)]

def project(F):
    r""" Function to project data points `F` onto a sunit simplex.

    A very simple function that projects data points `F` onto a 
    unit simplex, where `|F| = n x m`.

    Parameters
    ----------
    F : ndarray
        A sample of `n` data points in `m` dimension, i.e. `|F| = n x m`.

    Returns
    -------
    P : ndarray
        Projected data points from `F`, i.e. `|P| = n x m`.

    """
    n,m = F.shape
    u = 1 / np.sqrt(np.ones(m) * m)
    uTF = np.sum(u.T * F, axis=1)
    uTuTFT = (np.tile(u, (n,1)).T * uTF).T
    P = (F - uTuTFT) + (u / np.sqrt(m))
    return P

def depth_contours(F, project_collapse=True):
    r"""Function to find all the depth-contours of the data point in `F`.

    This function applies convex-hull (i.e. qhull, `scipy.spatial.ConvexHull`)
    algorithm to find the first depth-contour of `F`. Then removes them and flag
    them as the layer 1. Then applies qhull on the remaining data points and keep
    doing the same until more than `2 * m + 1` data points are left.

    Parameters
    ----------
    F : ndarray
        A sample of `n` data points in `m` dimension, i.e. `|F| = n x m`.
    project_collapse : bool, optional
        A switch to activate/deactivate projection and collapse of the last dimension
        (i.e. column) of the projected data points of `F`. If `project_collapse = False`, 
        there might be just one layer if the points are on a full convex surface. Also it 
        will be extremely slow if there are many high-dimensional points (m > 4). Therefore, 
        not recommended for a large number of high-dimensional data. Default `True` when 
        optional.

    Returns
    -------
    L : A jagged ndarray
        Projected data points from `F`, i.e. `|P| = n x m`.
    """
    n,m = F.shape
    if project_collapse:
        if m >= 3:
            F_ = project(F)
            P = collapse(F, d=m-1)
        else:
            P = project(F)
    else:
        P = np.array(F, copy=True)

    I = np.arange(0, n, 1).astype(int)
    Id = np.arange(0, n).astype(int)
    # print("Id.shape:", Id.shape)
    G = P[Id]

    L = []
    i = 0
    while Id.shape[0] >= (2 * m + 1):
        H = ConvexHull(G, qhull_options="Qa QJ Q12")
        Ih = H.vertices
        # print("i:", i, "Ih:", Ih, "Ih.shape:", Ih.shape)
        # print("i:", i, "Id[Ih]:", Id[Ih], "Id[Ih].shape:", Id[Ih].shape)
        L.append(Id[Ih])
        Id = np.delete(Id, Ih)
        # print("i:", i, "Id.shape:", Id.shape, "\n")
        G = P[Id]
        i = i + 1
    L.append(Id)

    return np.array(L, dtype=object)

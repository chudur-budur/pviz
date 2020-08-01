"""paletteviz.py -- A customized and more flexible Paletteviz plotting module. 

    This module provides a customized and more flexible function for PaletteViz [1]_ 
    plotting. This module also provides different relevant fucntions, parameters and tools.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA
    
    References
    ----------
    .. [1] A. K. A. Talukder and K. Deb, "PaletteViz: A Visualization Method for Functional 
        Understanding of High-Dimensional Pareto-Optimal Data-Sets to Aid Multi-Criteria 
        Decision Making," in IEEE Computational Intelligence Magazine, vol. 15, no. 2, 
        pp. 36-48, May 2020, doi: 10.1109/MCI.2020.2976184.

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""
import os
import numpy as np
import matplotlib.colors as mc
from mpl_toolkits.mplot3d import Axes3D
from vis.plotting.radviz import get_radviz_coordinates
from vis.plotting.star import get_star_coordinates
from vis.plotting.utils import set_polar_anchors, set_polar_anchor_labels
from vis.tda import simple_shape
from vis.utils import io


__all__ = ["get_palette_star_coordinates", "plot"]


def get_palette_star_coordinates(X, depth_contour_path=None, \
                                    n_partitions=float('inf'), \
                                    inverted=True, normalized=True, \
                                    kwargs=None):
    r"""Generate Star-coordinates from data points `X`.

    Maps all the data points in `X` (i.e. `|X| = n x m`) onto 
    star-coordinate [1]_ positions. 

    Parameters
    ----------
    X : ndarray 
        `n` number of `m` dimensiomal points as input.
    depth_contour_path : str or pathlib.Path object, optional
        The path to the depth contour indices. Default 'None' when optional.
    n_partitions : int, optional
        The total number of layers in the final PaletteViz plot. We recommend 4.
        Default `float('inf')` when optional. When default, the total number of 
        layers in the PaletteViz will be the same as the number of depth contours. 
        Also if `n_partitions` is bigger than the total number depth contours, 
        the total number of layers in the PaletteViz will be equal to the total
        number of depth contours.
    inverted : bool, optional
        See `vis.plotting.star` for more details.
    normalized : bool, optional
        See `vis.plotting.star` for details.

    Other Parameters
    ----------------
    project_collapse : bool, optional
        See `vis.tda.simple_shape` module for more details.
    verbose : bool, optional
        Verbose level. Default `False` when optional.

    Returns
    -------
    (P,K,B) : tuple of ndarray
        `P` is an ndarray of PaletteViz coordinates (i.e. `|P| = n x 3`), 
        `K` is the position of the anchor points (i.e. `|K| = m x 2`),
        and `B` is the lower bound and upper bound of `P`. `K` and `B`
        will be used to draw anchor points and the polygon.

    """

    project_collapse = kwargs['project_collapse'] if 'project_collapse' in kwargs else True
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    L = None
    if depth_contour_path is None:
        # compute layers
        if verbose:
            print("Computing depth contours since no file provided.")
        L = simple_shape.depth_contours(X, project_collapse=project_collapse, verbose=verbose) 
    elif depth_contour_path is not None and os.path.exists(depth_contour_path):
        # load depth contours
        if verbose:
            print("Loading depth contours from {0:s}.".format(depth_contour_path))
        L = io.loadtxt(depth_contour_path, dtype=int, delimiter=',') 

    if L is not None:
        (P,K,B) = get_star_coordinates(X, inverted=inverted, normalized=normalized)
        n,m,p = P.shape[0], P.shape[1], L.shape[0]
        n_partitions = p if n_partitions >= p else n_partitions
        # q = number of layers in each partition
        # r = number of layers left after dividing them into n_partition layers
        # dz = the gap in z-axis between each pair of consedutive layers
        q, r, dz = p // n_partitions, p % n_partitions, 1 / n_partitions
        P_ = np.zeros((n, 3))
        z = 1.0
        for i in range(0, p-r, q):
            L_ = L[i:i+q]
            for l in L_:
                Id = l.astype(int)
                P_[Id,0:m] = P[Id,:]
                P_[Id,m] = np.ones(Id.shape[0]) * z
            z = z - dz
        # if there is any remaining layer, merge them with the last one
        if r > 0:
            z = z + dz
            for i in range(L.shape[0]-1,(L.shape[0]-r)-1,-1):
                Id = L[i].astype(int)
                P_[Id,0:m] = P[Id,:]
                P_[Id,m] = np.ones(Id.shape[0]) * z
        return (P_, K, B)
    else:
        raise ValueError("No depth contours found.")


def plot(A, plt, depth_contour_path=None, mode='star', \
            n_partitions=float('inf'), s=1, c=mc.TABLEAU_COLORS['tab:blue'], \
            inverted=True, normalized=True, \
            draw_axes=False, draw_anchors=True, **kwargs):
    r"""A customized and more enhanced PaletteViz plot.

    This PaletteViz plot is customized for the experiments. It allows both
    Radviz and Star-coodinate based PaletteViz plots, depending on the options
    selected.
    
    Parameters
    ----------
    A : ndarray 
        `n` number of `m` dim. points to be plotted.
    plt : A `matplotlib.pyplot` object
        It needs to be passed.
    depth_contour_path : str or pathlib.Path object, optional
        See `get_palette_star_coordinates()` or `get_palette_radviz_coordinates()` 
        functions for detail. Default 'None' when optional.
    mode : str {'star', 'radviz'}, optional
        By which way each layer will be mapped. It can be Radviz, Star-coordinate etc.
        If this value is 'star', then Star-coordinate plot will be used. If the value
        is 'radviz', then Radviz plot will be used. Default 'star' when optional.
    n_partitions : int, optional
        See `get_palette_star_coordinates()` or `get_palette_radviz_coordinates()`
        functions for details. Default `float('inf')` when optional.
    s : int or 1-D array_like, optional
        Point size, or an array of point sizes. Default 1 when optional.
    c : A `matplotlib.colors` object, str or an array RGBA color values.
        Colors to be used. Default `mc.TABLEAU_COLORS['tab:blue']` when 
        optional.
    inverted : bool, optional
        See `get_star_coordinates()` function for details.
    normalized : bool, optional
        If needed, the data points in `A` can be normalized within `[0.0, 1.0]`. 
        Default `True` when optional.
    draw_axes: bool, optional
        If `True`, the radviz plot will show axes. Default `False` when optional.
    draw_anchors: bool, optional
        If `False`, the radviz plot will hide anchors. Default `True` when optional.

    Other Parameters
    ----------------
    label_prefix : str, optional
        See `set_anchor_labels()` function for details.
    label_fontsize : str or int, optional
        See `set_anchor_labels()` function for details.
    label_fontname : str, optional
        See `set_anchor_labels()` function for details.
    label_fontstyle : str, optional
        See `set_anchor_labels()` function for details.
    title : str, optional
        The plot title. Default `None` when optional.

    Returns
    -------
    (fig, ax) : tuple of `matplotlib.pyplot.figure` and `matplotlib.axes.Axes`
        A tuple of `matplotlib.pyplot.figure` and `matplotlib.axes.Axes` 
        (or `mpl_toolkits.mplot3d.axes.Axes`) objects.
    """
    # by default label_prefix is $f_n$
    label_prefix = kwargs['label_prefix'] if (kwargs is not None and 'label_prefix' in kwargs) \
                            else r"$f_{:d}$"
    # default label font size is 'large'
    label_fontsize = kwargs['label_fontsize'] if (kwargs is not None and 'label_fontsize' in kwargs) \
                            else 'large'
    # default label font is None
    label_fontname = kwargs['label_fontname'] if (kwargs is not None and 'label_fontname' in kwargs) \
                            else None
    # default label font style is 'normal'
    label_fontstyle = kwargs['label_fontstyle'] if (kwargs is not None and 'label_fontstyle' in kwargs) \
                            else 'normal'
    # default plot title is empty
    title = kwargs['title'] if (kwargs is not None and 'title' in kwargs) else None

    if plt is not None:
        P, K, [lb, ub] = get_palette_star_coordinates(A, depth_contour_path=depth_contour_path, \
                                                        n_partitions=n_partitions, \
                                                        inverted=inverted, normalized=normalized, \
                                                        kwargs=kwargs)
        fig = plt.figure()
        if title is not None:
            fig.suptitle(title)
        ax = Axes3D(fig)
        ax.scatter(P[:,0], P[:,1], P[:,2], s=s, c=c)
        
        # if draw_axes:
        #     # ax.set_xticklabels([])
        #     # ax.set_yticklabels([])
        #     ax.set_axis_on()
        # else:
        #     ax.set_axis_off()
        # if draw_anchors:
        #     set_polar_anchors(ax, K)
        #     set_polar_anchor_labels(ax, K, label_prefix, label_fontsize, label_fontname, label_fontstyle)
        # ax.set_xlim(lb[0] - 0.1 if lb[0] < -1 else -1.1, ub[0] + 0.1 if ub[0] > 1 else 1.1)
        # ax.set_ylim(lb[1] - 0.1 if lb[1] < -1 else -1.1, ub[1] + 0.1 if ub[1] > 1 else 1.1)
        # ax.set_aspect('equal')
        return (fig, ax)
    else:
        raise TypeError("A valid `matplotlib.pyplot` object must be provided.")

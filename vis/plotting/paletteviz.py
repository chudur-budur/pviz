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


__all__ = ["get_palette_star_coordinates", "get_palette_radviz_coordinates", "plot"]


# Some good camera angles for paletteviz plots.
camera_angles_star = {
    'dtlz2': {'3d': (-20,25), '4d':(-50,20), '8d': (-50,25)}, \
    'dtlz2-nbi': {'3d': (-60,30), '4d':(-65,20), '8d': (-60,30)}, \
    'debmdk': {'3d': (15,25), '4d': (35,15), '8d': (130,25)}, \
    'debmdk-nbi': {'3d': (135,25), '4d': (-60,20), '8d': (-50,20)}, \
    'debmdk-all': {'3d': (25,25), '4d': (80,20), '8d': (60,20)}, \
    'debmdk-all-nbi': {'3d': (135,25), '4d': (80,20), '8d': (-60,15)}, \
    'dtlz8': {'3d': (-60,30), '4d': (-20,20), '6d': (30,15), '8d': (55,25)}, \
    'dtlz8-nbi': {'3d': (-60,30), '4d': (-20,20), '6d': (30,15), '8d': (55,25)}, \
    'c2dtlz2': {'3d': (-20,45), '4d': (125,20), '5d': (60,30), '8d': (-20,20)}, \
    'c2dtlz2-nbi': {'3d': (20,45), '4d': (125,20), '5d': (60,30), '8d': (-20,20)}, \
    'cdebmdk': {'3d': (-165,25), '4d': (35,20), '8d': (-60,20)}, \
    'cdebmdk-nbi': {'3d': (90,20), '4d': (35,30), '8d': (-60,20)}, \
    'c0dtlz2': {'3d': (95,20), '4d': (20,20), '8d': (160,35)}, \
    'c0dtlz2-nbi': {'3d': (125,20), '4d': (165,20), '8d': (160,35)}, \
    'crash-nbi': {'3d': (-25,30)}, 'crash-c1-nbi': {'3d': (-115,15)}, 'crash-c2-nbi': {'3d': (-170,20)}, \
    'gaa': {'10d': (0,25)}, \
    'gaa-nbi': {'10d': (0,25)}
}

camera_angles_radviz = {
    'dtlz2': {'3d': (-50,30), '4d':(-55,25), '8d': (-50,15)}, \
    'dtlz2-nbi': {'3d': (-50,30), '4d':(-65,20), '8d': (-60,30)}, \
    'debmdk': {'3d': (-50,30), '4d': (-60,25), '8d': (-40,15)}, \
    'debmdk-nbi': {'3d': (-60,30), '4d': (-60,20), '8d': (-55,-20)}, \
    'debmdk-all': {'3d': (55,30), '4d': (-60,25), '8d': (-115,15)}, \
    'debmdk-all-nbi': {'3d': (60,30), '4d': (-60,25), '8d': (-145,20)}, \
    'dtlz8': {'3d': (-60,30), '4d': (-20,20), '6d': (30,15), '8d': (5,35)}, \
    'dtlz8-nbi': {'3d': (-60,30), '4d': (-20,20), '6d': (30,15), '8d': (55,25)}, \
    'c2dtlz2': {'3d': (-60,35), '4d': (125,20), '5d': (35,25), '8d': (-20,20)}, \
    'c2dtlz2-nbi': {'3d': (80,20), '4d': (160,20), '5d': (35,25), '8d': (-20,20)}, \
    'cdebmdk': {'3d': (-165,25), '4d': (35,20), '8d': (-60,20)}, \
    'cdebmdk-nbi': {'3d': (90,20), '4d': (35,30), '8d': (-60,20)}, \
    'c0dtlz2': {'3d': (180,25), '4d': (20,20), '8d': (-115,35)}, \
    'c0dtlz2-nbi': {'3d': (165,20), '4d': (20,20), '8d': (160,35)}, \
    'crash-nbi': {'3d': (45,20)}, 'crash-c1-nbi': {'3d': (-115,15)}, 'crash-c2-nbi': {'3d': (-170,20)}, \
    'gaa': {'10d': (0,25)}, \
    'gaa-nbi': {'10d': (0,25)}
}

def make_partitions(P, K, B, L, n_partitions):
    r"""Merge depth contours to make total of n_partitions number of layers.

    This function will be used by `get_palette_star_coordinates()` and 
    `get_palette_radviz_coordinates()` functions. Here we try to merge 
    the individual depth contours so that the final number of layers are 
    at most `n_partitions`. Let's say there are total 20 depth contours, 
    and if the user wants to visualise at most 4 layers, then we merge 
    every 5 depth contours from the start to make the final number of 
    layers to be 4. If `n_partitions` is bigger than the number of depth 
    contours, the total number of final layers will be the same as the 
    total number of depth contours.

    Parameters
    ----------
    P : ndarray 
        `n` number of 2D points found from radviz or star-coordinates.
    K : array_like, 2D
        The coordinate of the 2D anchor points found from radviz or star-coordinates.
    B : array_like, 2D
        The bounds of the anchor positions found from radviz or star-coordinates.
    L : ndarray, jagged
        The indices of each depth contours. The top row has the lowest
        depth and the bottom row has the highest depth. Each column is
        the index of the points in the original space.
    n_partitions : int
        The total number of layers in the final PaletteViz plot. We recommend 4.
        Default `float('inf')` when optional. When default, the total number of 
        layers in the PaletteViz will be the same as the number of depth contours. 
        Also if `n_partitions` is bigger than the total number depth contours, 
        the total number of layers in the PaletteViz will be equal to the total
        number of depth contours.

    Returns
    -------
    (P,K,B,Z) : tuple of ndarray
        `P` is an ndarray of PaletteViz coordinates (i.e. `|P| = n x 3`), 
        `K` is the position of the anchor points (i.e. `|K| = m x 2`),
        and `B` is the lower bound and upper bound of `P`. `Z` is an array
        of z-coordinate values of all the anchor-sets (i.e. `|Z| = n_partitions x 1`). 
        `K` and `Z` values will be used to draw anchor points and the polygon. 
    """
    n,m,p = P.shape[0], P.shape[1], L.shape[0]
    n_partitions = p if n_partitions >= p else n_partitions
    # q = number of layers in each partition
    # r = number of layers left after dividing them into n_partition layers
    # dz = the gap in z-axis between each pair of consedutive layers
    q, r, dz = p // n_partitions, p % n_partitions, 1 / n_partitions
    P_ = np.zeros((n, 3))
    z, Z = 1.0, np.zeros(n_partitions)
    for j,i in enumerate(range(0, p-r, q)):
        L_ = L[i:i+q]
        for l in L_:
            Id = l.astype(int)
            P_[Id,0:m] = P[Id,:]
            P_[Id,m] = np.ones(Id.shape[0]) * z
        Z[j] = z
        z = z - dz
    # if there is any remaining layer, merge them with the last one
    if r > 0:
        for i in range(L.shape[0]-1,(L.shape[0]-r)-1,-1):
            Id = L[i].astype(int)
            P_[Id,0:m] = P[Id,:]
            P_[Id,m] = np.ones(Id.shape[0]) * Z[-1]
    else:
        # if the last layer has very small number of points, smaller than
        # 1/10-th of the number of points in the previous layer, merge them 
        # with the previous layer. Therefore, the final number of layers 
        # will be n_partitions-1.
        if Z.shape[0] > 1:
            Iz = np.where(P_[:,-1] == Z[-1])[0]
            Iz_1 = np.where(P_[:,-1] == Z[-2])[0]
            if Iz.shape[0] < Iz_1.shape[0] / 10:
                P_[Iz,-1] = Z[-2]
                Z = Z[:-1]
    # update P
    P = P_
    # add z-bounds to B
    B[0] = np.append(B[0], np.min(Z))
    B[1] = np.append(B[1], np.max(Z))
    return (P,K,B,Z)


def get_palette_star_coordinates(X, depth_contours=None, n_partitions=float('inf'), kwargs=None):
    r"""Generate Star-coordinates from data points `X`.

    Maps all the data points in `X` (i.e. `|X| = n x m`) onto 
    star-coordinate [1]_ positions. 

    Parameters
    ----------
    X : ndarray 
        `n` number of `m` dimensiomal points as input.
    depth_contours : ndarray or str path, optional
        An `ndarray` containing the depth contours or a path to the file 
        containing depth contour indices. Default 'None' when optional.
    n_partitions : int, optional
        The total number of layers in the final PaletteViz plot. We recommend 4.
        Default `float('inf')` when optional. When default, the total number of 
        layers in the PaletteViz will be the same as the number of depth contours. 
        Also if `n_partitions` is bigger than the total number depth contours, 
        the total number of layers in the PaletteViz will be equal to the total
        number of depth contours.

    Other Parameters
    ----------------
    inverted : bool, optional
        See `vis.plotting.star` for more details.
    normalized : bool, optional
        See `vis.plotting.star` for details.
    project_collapse : bool, optional
        See `vis.tda.simple_shape` module for more details.
    verbose : bool, optional
        Verbose level. Default `False` when optional.

    Returns
    -------
    (P,K,B,Z) : tuple of ndarray
        `P` is an ndarray of PaletteViz coordinates (i.e. `|P| = n x 3`), 
        `K` is the position of the anchor points (i.e. `|K| = m x 2`),
        and `B` is the lower bound and upper bound of `P`. `Z` is an array
        of z-coordinate values of all the anchor-sets (i.e. `|Z| = n_partitions x 1`). 
        `K` and `Z` values will be used to draw anchor points and the polygon. 

    """

    inverted = kwargs['inverted'] if (kwargs is not None and 'inverted' in kwargs) else True
    normalized = kwargs['normalized'] if (kwargs is not None and 'normalized' in kwargs) else True
    project_collapse = kwargs['project_collapse'] if (kwargs is not None and 'project_collapse' in kwargs) \
            else True
    verbose = kwargs['verbose'] if (kwargs is not None and 'verbose' in kwargs) else False

    L = None
    if depth_contours is None:
        # compute layers
        if verbose:
            print("Computing depth contours since no file provided.")
        L = simple_shape.depth_contours(X, project_collapse=project_collapse, verbose=verbose) 
    else: 
        if isinstance(depth_contours, str):
            # load depth contours
            if verbose:
                print("Loading depth contours from {0:s}.".format(depth_contours))
            L = io.loadtxt(depth_contours, dtype=int, delimiter=',') 
        elif isinstance(depth_contours, np.ndarray):
            # use depth contours
            if verbose:
                print("Using depth contours ndarray.")
            L = depth_contours

    if L is not None:
        (P,K,B) = get_star_coordinates(X, inverted=inverted, normalized=normalized)
        (P,K,B,Z) = make_partitions(P,K,B,L,n_partitions)
        return (P, K, B, Z)
    else:
        raise ValueError("No depth contours found.")


def get_palette_radviz_coordinates(X, depth_contours=None, n_partitions=float('inf'), kwargs=None):
    r"""Generate Radviz coordinates from data points `X`.

    Maps all the data points in `X` (i.e. `|X| = n x m`) onto 
    Radviz coordinate [1]_ positions. 

    Parameters
    ----------
    X : ndarray 
        `n` number of `m` dimensiomal points as input.
    depth_contours : ndarray or str path, optional
        An `ndarray` containing the depth contours or a path to the file 
        containing depth contour indices. Default 'None' when optional.
    n_partitions : int, optional
        The total number of layers in the final PaletteViz plot. We recommend 4.
        Default `float('inf')` when optional. When default, the total number of 
        layers in the PaletteViz will be the same as the number of depth contours. 
        Also if `n_partitions` is bigger than the total number depth contours, 
        the total number of layers in the PaletteViz will be equal to the total
        number of depth contours.

    Other Parameters
    ----------------
    spread_factor : str {'auto'} or float, optional
        See `vis.plotting.radviz` for more details.
    normalized : bool, optional
        See `vis.plotting.radviz` for details.
    project_collapse : bool, optional
        See `vis.tda.simple_shape` module for more details.
    verbose : bool, optional
        Verbose level. Default `False` when optional.

    Returns
    -------
    (P,K,B,Z) : tuple of ndarray
        `P` is an ndarray of PaletteViz coordinates (i.e. `|P| = n x 3`), 
        `K` is the position of the anchor points (i.e. `|K| = m x 2`),
        and `B` is the lower bound and upper bound of `P`. `Z` is an array
        of z-coordinate values of all the anchor-sets (i.e. `|Z| = n_partitions x 1`). 
        `K` and `Z` values will be used to draw anchor points and the polygon. 

    """

    spread_factor = kwargs['spread_factor'] if (kwargs is not None and 'spread_factor' in kwargs) else 'auto'
    normalized = kwargs['normalized'] if (kwargs is not None and 'normalized' in kwargs) else True
    project_collapse = kwargs['project_collapse'] if (kwargs is not None and 'project_collapse' in kwargs) \
                            else True
    verbose = kwargs['verbose'] if (kwargs is not None and 'verbose' in kwargs) else False

    L = None
    if depth_contours is None:
        # compute layers
        if verbose:
            print("Computing depth contours since no file provided.")
        L = simple_shape.depth_contours(X, project_collapse=project_collapse, verbose=verbose) 
    else: 
        if isinstance(depth_contours, str):
            # load depth contours
            if verbose:
                print("Loading depth contours from {0:s}.".format(depth_contours))
            L = io.loadtxt(depth_contours, dtype=int, delimiter=',') 
        elif isinstance(depth_contours, np.ndarray):
            # use depth contours
            if verbose:
                print("Using depth contours ndarray.")
            L = depth_contours

    if L is not None:
        (P,K,B) = get_radviz_coordinates(X, spread_factor=spread_factor, normalized=normalized)
        (P,K,B,Z) = make_partitions(P,K,B,L,n_partitions)
        return (P, K, B, Z)
    else:
        raise ValueError("No depth contours found.")


def plot(A, plt, depth_contours=None, mode='star', \
            n_partitions=float('inf'), s=1, c=mc.TABLEAU_COLORS['tab:blue'], \
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
    depth_contours : ndarray or str path, optional
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
    draw_axes: bool, optional
        If `True`, the radviz plot will show axes. Default `False` when optional.
    draw_anchors: bool, optional
        If `False`, the radviz plot will hide anchors. Default `True` when optional.

    Other Parameters
    ----------------
    euler : tuple (i.e. a pair) of int, optional
        The azmiuth and elevation angle. Default `(-60,30)` when optional.
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
    verbose : bool, optional
        The verbosity. Default 'False' when optional.

    Returns
    -------
    (fig, ax) : tuple of `matplotlib.pyplot.figure` and `matplotlib.axes.Axes`
        A tuple of `matplotlib.pyplot.figure` and `matplotlib.axes.Axes` 
        (or `mpl_toolkits.mplot3d.axes.Axes`) objects.
    """
    # azimuth is -60 and elevation is 30 by default
    euler = kwargs['euler'] if (kwargs is not None and 'euler' in kwargs) else (-60, 30)
    # by default label_prefix is $f_n$
    label_prefix = kwargs['label_prefix'] if (kwargs is not None and 'label_prefix' in kwargs) \
                            else r"$f_{{{:d}}}$"
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
    # verbosity
    verbose = kwargs['verbose'] if (kwargs is not None and 'verbose' in kwargs) else False

    if plt is not None:
        if mode == 'star':
            if verbose:
                print("Plotting palette-star-viz.")
            P, K, _, Z = get_palette_star_coordinates(A, depth_contours=depth_contours, \
                                                        n_partitions=n_partitions, kwargs=kwargs)
        elif mode == 'radviz':
            if verbose:
                print("Plotting palette-radviz.")
            P, K, _, Z = get_palette_radviz_coordinates(A, depth_contours=depth_contours, \
                                                        n_partitions=n_partitions, kwargs=kwargs)
        else:
            raise ValueError("Unknown mode, it has to be one of {'star', 'radviz'}.")

        fig = plt.figure()
        if title is not None:
            fig.suptitle(title)
        ax = Axes3D(fig)
        ax.scatter(P[:,0], P[:,1], P[:,2], s=s, c=c)
        ax.view_init(euler[1], euler[0])
        
        if draw_axes:
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            ax.set_axis_on()
        else:
            ax.set_axis_off()
        if draw_anchors:
            for z in Z:
                set_polar_anchors(ax, K, z=z)
                if mode == 'radviz':
                    set_polar_anchor_labels(ax, K, z=z, label_prefix=label_prefix, \
                                    label_fontsize=label_fontsize, label_fontname=label_fontname, \
                                    label_fontstyle=label_fontstyle)
                elif mode == 'star':
                    set_polar_anchor_labels(ax, K, z=z, draw_circle=True, label_prefix=label_prefix, \
                                    label_fontsize=label_fontsize, label_fontname=label_fontname, \
                                    label_fontstyle=label_fontstyle)
                else:
                    raise ValueError("Unknown mode, it has to be one of {'star', 'radviz'}.")
        # ax.set_xlim(lb[0] - 0.1 if lb[0] < -1 else -1.1, ub[0] + 0.1 if ub[0] > 1 else 1.1)
        # ax.set_ylim(lb[1] - 0.1 if lb[1] < -1 else -1.1, ub[1] + 0.1 if ub[1] > 1 else 1.1)
        # ax.set_zlim(lb[2] - 0.1 if lb[2] < -1 else -1.1, ub[2] + 0.1 if ub[2] > 1 else 1.1)
        # ax.set_aspect('equal')
        return (fig, ax)
    else:
        raise TypeError("A valid `matplotlib.pyplot` object must be provided.")

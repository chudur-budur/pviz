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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from viz.plotting.radviz import get_radviz_coordinates
from viz.plotting.star import get_star_coordinates
from viz.plotting.utils import set_polar_anchors, set_polar_anchor_labels, pop, group_labels_by_appearance
from viz.tda import simple_shape
from viz.utils import io


__all__ = ["get_palette_star_coordinates", "get_palette_radviz_coordinates", "plot", \
        "camera_angles_star", "camera_angles_radviz"]


# Some good camera angles for paletteviz plots.
camera_angles_star = {
    'dtlz2': {'3d': (20,30), '4d':(-40,20), '8d': (-25,30)}, \
    'dtlz2-nbi': {'3d': (-60,30), '4d':(-65,20), '5d': (-60,30), '8d': (-60,30)}, \
    'debmdk': {'3d': (130,20), '4d': (140,20), '8d': (155,20)}, \
    'debmdk-nbi': {'3d': (100,25), '4d': (0,20), '8d': (-45,20)}, \
    'debmdk-all': {'3d': (135,20), '4d': (160,20), '8d': (165,25)}, \
    'debmdk-all-nbi': {'3d': (85,20), '4d': (10,20), '8d': (-35,20)}, \
    'dtlz8': {'3d': (0,15), '4d': (-5,10), '6d': (-180,20), '8d': (-180,25)}, \
    'dtlz8-nbi': {'3d': (0,15), '4d': (-5,10), '6d': (-180,20), '8d': (-180,25)}, \
    'c2dtlz2': {'3d': (-125,20), '4d': (125,20), '5d': (60,30), '8d': (-20,20)}, \
    'c2dtlz2-nbi': {'3d': (30,30), '4d': (-20,15), '5d': (95,35), '8d': (-20,20)}, \
    'cdebmdk': {'3d': (-60,35), '4d': (70,30), '8d': (80,30)}, \
    'cdebmdk-nbi': {'3d': (-70,45), '4d': (60,30), '8d': (80,30)}, \
    'c0dtlz2': {'3d': (120,30), '4d': (165,30), '8d': (-150,45)}, \
    'c0dtlz2-nbi': {'3d': (125,20), '4d': (165,20), '8d': (160,35)}, \
    'crash-nbi': {'3d': (0,10)}, 'crash-c1-nbi': {'3d': (-25,15)}, 'crash-c2-nbi': {'3d': (-15,15)}, \
    'gaa': {'10d': (-15,25)}, \
    'gaa-nbi': {'10d': (0,25)}
}

camera_angles_radviz = {
    'dtlz2': {'3d': (-50,30), '4d':(-55,25), '8d': (-50,15)}, \
    'dtlz2-nbi': {'3d': (-50,30), '4d':(-65,20), '5d': (-60,30), '8d': (-60,30)}, \
    'debmdk': {'3d': (-35,25), '4d': (-30,25), '8d': (-40,15)}, \
    'debmdk-nbi': {'3d': (-65,25), '4d': (-60,20), '8d': (-55,-20)}, \
    'debmdk-all': {'3d': (100,20), '4d': (-30,20), '8d': (-20,15)}, \
    'debmdk-all-nbi': {'3d': (-75,25), '4d': (-70,20), '8d': (-145,20)}, \
    'dtlz8': {'3d': (-60,30), '4d': (-20,20), '6d': (0,35), '8d': (5,35)}, \
    'dtlz8-nbi': {'3d': (-60,30), '4d': (-20,20), '6d': (30,15), '8d': (55,25)}, \
    'c2dtlz2': {'3d': (-60,35), '4d': (125,20), '5d': (35,25), '8d': (-20,20)}, \
    'c2dtlz2-nbi': {'3d': (85,25), '4d': (160,20), '5d': (35,25), '8d': (-20,20)}, \
    'cdebmdk': {'3d': (-165,25), '4d': (-10,30), '8d': (-120,25)}, \
    'cdebmdk-nbi': {'3d': (180,30), '4d': (-115,30), '8d': (-80,25)}, \
    'c0dtlz2': {'3d': (180,25), '4d': (20,20), '8d': (-115,35)}, \
    'c0dtlz2-nbi': {'3d': (175,25), '4d': (20,20), '8d': (160,35)}, \
    'crash-nbi': {'3d': (-175,30)}, 'crash-c1-nbi': {'3d': (100,25)}, 'crash-c2-nbi': {'3d': (-100,25)}, \
    'gaa': {'10d': (0,25)}, \
    'gaa-nbi': {'10d': (20,40)}
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
    P : ndarray of float
        An ndarray of PaletteViz coordinates (i.e. `|P| = n x 3`).
    K : ndarray of float
        The positions of the anchor points (i.e. `|K| = m x 2`),
    B : ndarray of float
        The lower bound and upper bound of `P`. 
    Z : ndarray of float 
        An ndarray of z-coordinate values of all the anchor-sets 
        (i.e. `|Z| = n_partitions x 1`). `K` and `Z` values will be used to draw 
        anchor points and the polygon. 
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


def get_palette_star_coordinates(X, depth_contours=None, n_partitions=float('inf'), \
                                    inverted=True, normalized=True, project_collapse=True, \
                                    verbose=False):
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
    inverted : bool, optional
        See `viz.plotting.star` for more details.
    normalized : bool, optional
        See `viz.plotting.star` for details.
    project_collapse : bool, optional
        See `viz.tda.simple_shape` module for more details.
    verbose : bool, optional
        Verbose level. Default `False` when optional.

    Returns
    -------
    P : ndarray of float
        An ndarray of PaletteViz coordinates (i.e. `|P| = n x 3`).
    K : ndarray of float
        The positions of the anchor points (i.e. `|K| = m x 2`),
    B : ndarray of float
        The lower bound and upper bound of `P`. 
    Z : ndarray of float 
        An ndarray of z-coordinate values of all the anchor-sets 
        (i.e. `|Z| = n_partitions x 1`). `K` and `Z` values will be used to draw 
        anchor points and the polygon.
    """

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


def get_palette_radviz_coordinates(X, depth_contours=None, n_partitions=float('inf'), \
                                    spread_factor='auto', normalized=True, project_collapse=True, \
                                    verbose=False):
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
    spread_factor : str {'auto'} or float, optional
        See `viz.plotting.radviz` for more details.
    normalized : bool, optional
        See `viz.plotting.radviz` for details.
    project_collapse : bool, optional
        See `viz.tda.simple_shape` module for more details.
    verbose : bool, optional
        Verbose level. Default `False` when optional.

    Returns
    -------
    P : ndarray of float
        An ndarray of PaletteViz coordinates (i.e. `|P| = n x 3`).
    K : ndarray of float
        The positions of the anchor points (i.e. `|K| = m x 2`),
    B : ndarray of float
        The lower bound and upper bound of `P`. 
    Z : ndarray of float 
        An ndarray of z-coordinate values of all the anchor-sets 
        (i.e. `|Z| = n_partitions x 1`). `K` and `Z` values will be used to draw 
        anchor points and the polygon.
    """

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


def plot(A, ax=None, depth_contours=None, mode='star', n_partitions=float('inf'), \
        s=1.0, c=mc.TABLEAU_COLORS['tab:blue'], draw_axes=False, \
        draw_anchors={'labels': [0,1,2,3], 'polygons': [0,1,2,3], 'circles': [0,1,2,3]}, **kwargs):
    r"""A customized and more enhanced PaletteViz plot.

    This PaletteViz plot is customized for the experiments. It allows both
    Radviz and Star-coodinate based PaletteViz plots, depending on the options
    selected.
    
    Parameters
    ----------
    A : ndarray 
        `n` number of `m` dim. points to be plotted.
    ax : An `mpl_toolkits.mplot3d.axes.Axes3D` object, optional
        Default `None` when optional.
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
    s : float or 1-D array_like, optional
        Point size, or an array of point sizes. Default 1.0 when optional.
    c : A `matplotlib.colors` object, str or an array RGBA color values.
        Colors to be used. Default `mc.TABLEAU_COLORS['tab:blue']` when 
        optional.
    draw_axes : bool, optional
        If `True`, the radviz plot will show axes. Default `False` when optional.
    draw_anchors : None or {'labels': list of int, 'polygons': list of int, 'circles': list of int}, optional
        If `None`, there will be no polygons, circles or the labels. Each `list` of `int` 
        denotes which layer will have polygon, labels or circles. The `labels`, `polygons` and/or 
        `circles` will be drawn depending on the `list` of `int` provided. Default `{'labels': [0,1,2,3], `
        `'polygons': [0,1,2,3], 'circles': [0,1,2,3]}` when optional. 

    Other Parameters
    ----------------
    euler : tuple (i.e. a pair) of int, optional
        The azmiuth and elevation angle. Default `(-60,30)` when optional.
    title : str, optional
        The plot title. Default `None` when optional.
    labels : str, array_like or list of str, optional
        A string or an array/list of strings for labeling each point. Which basically
        means the class label of each row. Default `None` when optional. This will be
        used to set the legend in the figure. If `None` there will be no legend.
    spread_factor : str {'auto'} or float, optional
        See `viz.plotting.radviz` for more details.
    inverted : bool, optional
        See `viz.plotting.star` for more details.
    normalized : bool, optional
        See `viz.plotting.star` for details.
    project_collapse : bool, optional
        See `viz.tda.simple_shape` module for more details.
    colorbar : (Cbc, Cbg, Cbl, Cbp) a tuple of (ndarray, ndarray, str, float), optional
        If a user wants to put a colorbar, a tuple `(Cbc, Cbg, Cbl)` tuple can be 
        provided. `Cbc` is an array of RGBA color values or an `matplotlib.colors` 
        object. The gradient of the colorbar is specified in `Cbg` which is an 1-D 
        array of float. Cbl is the label of the colorbar, a string. `Cbp` is the 
        colorbar padding width in `float`. `colorbar` is default `None` when optional. 
    lims : 3-tuple of pairs of float
        The data point limits on three coordinates, each tuple is the limits on the
        coordinates to be used by `matplotlib.axes.Axes.set_x/y/zlim. If the `lims`
        is `(xl, yl, zl)`, then `xl`, `yl` and `zl` will be used for x-limit, y-limit
        and z-limit, respectively. Default `(None, None, None)` when optional.
    hide_layers : tuple of int
        List of layer indices to hide. Where there are L layers, The top layer is 
        indexed at 0 which will have z-axis value of 1.0 and the bottom layer is 
        indexed at L-1 which will have z-axis value of 0.0, when there are L layers. 
        Default `None` when optional.
    anchor_linewidth : float, optional
        See `set_polar_anchor()` function for details.
    anchor_label_prefix : str, optional
        See `set_anchor_labels()` function for details.
    anchor_label_fontsize : str or int, optional
        See `set_anchor_labels()` function for details.
    anchor_label_fontname : str, optional
        See `set_anchor_labels()` function for details.
    anchor_label_fontstyle : str, optional
        See `set_anchor_labels()` function for details.
    verbose : bool, optional
        The verbosity. Default 'False' when optional.
    **kwargs : dict
        All other keyword args for matplotlib `scatter()` function.

    Returns
    -------
    ax : `mpl_toolkits.mplot3d.axes.Axes3D` object
        An `mpl_toolkits.mplot3d.axes.Axes3D` object.
    P : ndarray of float
        `P` is an ndarray of PaletteViz coordinates (i.e. `|P| = n x 3`), 
    """
    # decide on what kind of axes to use
    if not ax:
        ax = Axes3D(plt.figure())

    euler = kwargs['euler'] if kwargs and 'euler' in kwargs else (-60, 30)
    title = kwargs['title'] if kwargs and 'title' in kwargs else None
    labels = kwargs['labels'] if kwargs and 'labels' in kwargs else None

    spread_factor = kwargs['spread_factor'] if kwargs and 'spread_factor' in kwargs else 'auto'
    inverted = kwargs['inverted'] if kwargs and 'inverted' in kwargs else True
    normalized = kwargs['normalized'] if kwargs and 'normalized' in kwargs else True
    project_collapse = kwargs['project_collapse'] if kwargs and 'project_collapse' in kwargs else True

    colorbar = kwargs['colorbar'] if kwargs and 'colorbar' in kwargs else None
    lims = kwargs['lims'] if kwargs and 'lims' in kwargs else (None, None, None)
    hide_layers = kwargs['hide_layers'] if kwargs and 'hide_layers' in kwargs else None

    anchor_linewidth = kwargs['anchor_linewidth'] if kwargs and 'anchor_linewidth' in kwargs else 1.0
    anchor_label_prefix = kwargs['anchor_label_prefix'] \
            if kwargs and 'anchor_label_prefix' in kwargs else r"$f_{{{:d}}}$"
    anchor_label_fontsize = kwargs['anchor_label_fontsize'] \
            if kwargs and 'anchor_label_fontsize' in kwargs else 'large'
    anchor_label_fontname = kwargs['anchor_label_fontname'] \
            if kwargs and 'anchor_label_fontname' in kwargs else None
    anchor_label_fontstyle = kwargs['anchor_label_fontstyle'] \
            if kwargs and 'anchor_label_fontstyle' in kwargs else 'normal'

    verbose = kwargs['verbose'] if kwargs and 'verbose' in kwargs else False

    # remove once they are read
    kwargs = pop(kwargs, 'euler')
    kwargs = pop(kwargs, 'title')
    kwargs = pop(kwargs, 'labels')
    kwargs = pop(kwargs, 'spread_factor')
    kwargs = pop(kwargs, 'inverted')
    kwargs = pop(kwargs, 'normalized')
    kwargs = pop(kwargs, 'project_collapse')
    kwargs = pop(kwargs, 'colorbar')
    kwargs = pop(kwargs, 'lims')
    kwargs = pop(kwargs, 'hide_layers')
    kwargs = pop(kwargs, 'anchor_linewidth')
    kwargs = pop(kwargs, 'anchor_label_prefix')
    kwargs = pop(kwargs, 'anchor_label_fontsize')
    kwargs = pop(kwargs, 'anchor_label_fontname')
    kwargs = pop(kwargs, 'anchor_label_fontstyle')
    kwargs = pop(kwargs, 'verbose')

    # decide on what kind of axes to use
    if ax is None:
        ax = Axes3D(plt.figure())
 
    # paletteviz coordinates
    if ax:
        if mode == 'star':
            if verbose:
                print("Plotting palette-star-viz.")
            P, K, _, Z = get_palette_star_coordinates(A, depth_contours=depth_contours, \
                                                        n_partitions=n_partitions, \
                                                        inverted=inverted, \
                                                        normalized=normalized, \
                                                        project_collapse=project_collapse, \
                                                        verbose=verbose)
        elif mode == 'radviz':
            if verbose:
                print("Plotting palette-radviz.")
            P, K, _, Z = get_palette_radviz_coordinates(A, depth_contours=depth_contours, \
                                                        n_partitions=n_partitions, \
                                                        spread_factor=spread_factor, \
                                                        normalized=normalized, \
                                                        project_collapse=project_collapse, \
                                                        verbose=verbose)
        else:
            raise ValueError("Unknown mode, it has to be one of {'star', 'radviz'}.")

        # if there is any layer to hide
        P_ = None # we keep a copy of the original data since they are going to change here.
        if hide_layers is not None:
            I = np.ones(P.shape[0]).astype(bool)
            for l in hide_layers:
                I[np.argwhere((P[:,-1] > Z[l] - 0.01) & (P[:,-1] < Z[l] + 0.01))] = False
            P_ = np.array(P, copy=True) # copy
            P = P[I]
            if c is not None:
                c = c[I]
            if s is not None:
                s = s[I]
            if labels is not None:
                labels = labels[I]
            Z = Z[np.array(list(set(range(Z.shape[0])) - set(hide_layers)))]

        # do the plot
        if labels is not None:
            if isinstance(labels, str):
                ax.scatter(P[:,0], P[:,1], P[:,2], s=s, c=c, label=labels, **kwargs)
            else:
                if isinstance(labels, np.ndarray): 
                    labels = labels.tolist()
                label_groups = group_labels_by_appearance(labels)
                for i,v in enumerate(label_groups):
                    ax.scatter(P[v[0],0], P[v[0],1], P[v[0],2], s=s[v[0]], c=c[v[0]], label=v[1], \
                            zorder=label_groups.shape[0]-i, **kwargs)
        else:
            ax.scatter(P[:,0], P[:,1], P[:,2], s=s, c=c, **kwargs)

        # set lims
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])
        
        # set camera angle
        ax.view_init(euler[1], euler[0])

        # draw axes?
        if draw_axes:
            ax.set_axis_on()
        else:
            ax.set_axis_off()

        # draw anchors?
        if draw_anchors:
            polygon_layers = draw_anchors['polygons']
            label_layers = draw_anchors['labels']
            circle_layers = draw_anchors['circles']
            for i,z in enumerate(Z):
                if i in polygon_layers:
                    set_polar_anchors(ax, K, z=z, anchor_linewidth=anchor_linewidth)
                if i in label_layers:
                    if mode == 'radviz':
                        set_polar_anchor_labels(ax, K, z=z, \
                                        label_prefix=anchor_label_prefix, \
                                        label_fontsize=anchor_label_fontsize, \
                                        label_fontname=anchor_label_fontname, \
                                        label_fontstyle=anchor_label_fontstyle)
                    elif mode == 'star':
                        draw_circle = True if i in circle_layers else False
                        set_polar_anchor_labels(ax, K, z=z, draw_circle=draw_circle, \
                                        label_prefix=anchor_label_prefix, \
                                        label_fontsize=anchor_label_fontsize, \
                                        label_fontname=anchor_label_fontname, \
                                        label_fontstyle=anchor_label_fontstyle)
                    else:
                        raise ValueError("Unknown mode, it has to be one of {'star', 'radviz'}.")

        # colorbar?
        if colorbar and isinstance(colorbar, tuple) and len(colorbar) >= 2 \
                and isinstance(colorbar[0], np.ndarray) and isinstance(colorbar[1], np.ndarray):
            vmin,vmax = 0.0, 1.0
            cbc, cbg = colorbar[0], colorbar[1]
            cbl = colorbar[2] if len(colorbar) > 2 and colorbar[2] else None
            cbp = colorbar[3] if len(colorbar) > 3 and colorbar[3] else -0.05
            Id = np.column_stack((cbg,cbc)).astype(object)
            Id = Id[np.argsort(Id[:, 0])] 
            c, g = Id[:,1:].astype(float), Id[:,0].astype(float)
            vmin, vmax = np.min(g), np.max(g)
            norm = mc.Normalize(vmin=vmin, vmax=vmax)
            cmap = ListedColormap(c)
            if cbl:
                ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), \
                        orientation='vertical', label=cbl, pad=cbp, shrink=0.5)
            else:
                ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), \
                            orientation='vertical', pad=cbp, shrink=0.5) 

        # where to put the legend
        if labels:
            ax.legend(loc="best", ncol=2)
        
        # title?
        ax.set_title(title, pad=0.0)

        if P_ is not None:
            return (ax, P_)
        else:
            return (ax, P)
    else:
        raise TypeError("A valid `mpl_toolkits.mplot3d.axes.Axes3D` object is not found.")

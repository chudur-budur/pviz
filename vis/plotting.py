"""`plotting.py` -- Different plotting functions for high-dimensional Pareto-optimal front

    This module provides different plotting functions for high-dimensional
    Pareto-optimal fronts. For example, scatter, radviz, star-coordinate,
    parallel-coordinates, paletteviz, radar-chart, heatmap, t-sne etc.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mc
from matplotlib.patches import Circle
from vis.utils import transform as tr

__all__ = ["scatter", "camera_scatter", "radviz"]

# Some good camera angles for scatter plots.
camera_scatter = {
    'dtlz2': {'3d': (60,20), '4d':(-60,30), '8d': (22,21)}, \
    'dtlz2-nbi': {'3d': (60,20), '4d':(-60,30), '8d': (-60,30)}, \
    'debmdk': {'3d': (-30,15), '4d': (-20,32), '8d': (-60,30)}, \
    'debmdk-nbi': {'3d': (-60,30), '4d': (-60,30), '8d': (-60,30)}, \
    'debmdk-all': {'3d': (-60,30), '4d': (-60,30), '8d': (-60,30)}, \
    'debmdk-all-nbi': {'3d': (-60,30), '4d': (-60,30), '8d': (-60,30)}, \
    'dtlz8': {'3d': (-60,30), '4d': (-60,30), '6d': (-60,30), '8d': (-60,30)}, \
    'dtlz8-nbi': {'3d': (-60,30), '4d': (-60,30), '6d': (-60,30), '8d': (-60,30)}, \
    'c2dtlz2': {'3d': (45,15), '4d': (-20,40), '5d': (-25,30), '8d': (-25,30)}, \
    'c2dtlz2-nbi': {'3d': (45,15), '4d': (-20,40), '5d': (-25,30), '8d': (-25,30)}, \
    'cdebmdk': {'3d': (20,15), '4d': (-60,30), '8d': (-60,30)}, \
    'cdebmdk-nbi': {'3d': (20,15), '4d': (-60,30), '8d': (-60,30)}, \
    'c0dtlz2': {'3d': (20,25), '4d': (-60,30), '8d': (-60,30)}, \
    'c0dtlz2-nbi': {'3d': (20,25), '4d': (-60,30), '8d': (-60,30)}, \
    'crash-nbi': {'3d': (30,25)}, 'crash-c1-nbi': {'3d': (30,25)}, 'crash-c2-nbi': {'3d': (30,25)}, \
    'gaa': {'10d': (-60,30)}, \
    'gaa-nbi': {'10d': (-60,30)}
}

def scatter(A, plt, s = 1, c = mc.TABLEAU_COLORS['tab:blue'], **kwargs):
    r"""A scatter plot function.

    This uses `matplotlib.axes.Axes.scatter` function to do a scatter plot.

    Parameters
    ----------
    `A` : ndarray 
        `n` number of `m` dim. points to be plotted.
    `plt`: A `matplotlib.pyplot` object
        It needs to be passed.
    `s`: `int` or 1-D array, optional
        Point size, or an array of point sizes. 1 when default.
    `c`: A `matplotlib.colors` object, str or an array RGBA color values.
        Colors to be used. `mc.TABLEAU_COLORS['tab:blue']` when default.

    Other Parameters
    ----------------
    `**kwargs` : str, `label_prefix`
        The axis-label-prefix to be used, `r"$f_{:d}$"` when default.
    `**kwargs` : str or int, `label_fontsize`
        The fontsize for the axes labels. `'large'` when default.
    `**kwargs` : list of int, `axes`
        The list of columns of `A` to be plotted. `(0, 1, 2)` when default.
    `**kwargs` : pair of int, `euler`
        The azmiuth and elevation angle. `(-60,30)` when default.
    `**kwargs` : pair of int, `xbound`, `ybound` and `zbound`
        The bounds on the axes. `None` when default.
    `**kwargs` : str, `title`
        The plot title. `None` when default.

    Returns
    -------
    `(fig,ax)` : tuple, `(matplotlib.pyplot.figure, matplotlib.axes.Axes)`
        A tuple of `matplotlib.pyplot.figure` and `matplotlib.axes.Axes` 
        (or `mpl_toolkits.mplot3d.axes.Acxes`) objects.
    """
    # all other parameters
    # by default label_prefix is $f_n$
    label_prefix = kwargs['label_prefix'] if (kwargs is not None and 'label_prefix' in kwargs) \
                            else r"$f_{:d}$"
    # default label font size is 'large'
    label_fontsize = kwargs['label_fontsize'] if (kwargs is not None and 'label_fontsize' in kwargs) \
                            else 'large'
    # plot first 3 axes by default
    axes = kwargs['axes'] if (kwargs is not None and 'axes' in kwargs) else (0, 1, 2)
    # azimuth is -60 and elevation is 30 by default
    euler = kwargs['euler'] if (kwargs is not None and 'euler' in kwargs) else (-60, 30)
    # by default, take the entire range
    xbound = kwargs['xbound'] if (kwargs is not None and 'xbound' in kwargs) else None
    ybound = kwargs['ybound'] if (kwargs is not None and 'ybound' in kwargs) else None
    zbound = kwargs['zbound'] if (kwargs is not None and 'zbound' in kwargs) else None
    # by default, no title
    title = kwargs['title'] if (kwargs is not None and 'title' in kwargs) else None

    if plt is not None:
        fig = plt.figure()
        if title is not None:
            fig.suptitle(title)
        if A.shape[1] < 3:
            ax = fig.gca()
            ax.scatter(A[:,axes[0]], A[:,axes[1]], s = s, c = c)
            ax.set_xbound(ax.get_xbound() if xbound is None else xbound)
            ax.set_ybound(ax.get_ybound() if ybound is None else ybound)
            ax.set_xlabel(label_prefix.format(axes[0] + 1), fontsize = label_fontsize)
            ax.set_ylabel(label_prefix.format(axes[1] + 1), fontsize = label_fontsize)
        else:
            ax = Axes3D(fig)
            ax.scatter(A[:,axes[0]], A[:,axes[1]], A[:,axes[2]], s = s, c = c) 
            ax.set_xbound(ax.get_xbound() if xbound is None else xbound)
            ax.set_ybound(ax.get_ybound() if ybound is None else ybound)
            ax.set_zbound(ax.get_zbound() if zbound is None else zbound)
            ax.set_xlabel(label_prefix.format(axes[0] + 1), fontsize = label_fontsize)
            ax.set_ylabel(label_prefix.format(axes[1] + 1), fontsize = label_fontsize)
            ax.set_zlabel(label_prefix.format(axes[2] + 1), fontsize = label_fontsize)
            ax.xaxis.set_rotate_label(False)
            ax.yaxis.set_rotate_label(False)
            ax.zaxis.set_rotate_label(False)
            ax.view_init(euler[1], euler[0])
        return (fig, ax)
    else:
        raise TypeError("A valid `matplotlib.pyplot` object must be provided.")

def get_radviz_coordinates(X, factor = 1.0, normalized=True):
    r"""Generate Radviz coordinates from data points `X`.

    Maps all the data points in `X` (i.e. `|X| = n x m`) onto
    Radviz coordinate positions. The `factor` parameter can be
    used to "spread-out" the data points on Radviz. In higher
    dimension, the points on Radviz tend to be clustered in the
    center. We can increase this value to make the points more
    visible.

    Parameters
    ----------
    X : ndarray 
        `n` number of `m` dim. points as input.
    factor: float, optional
        The "spread-factor" to make the points less cluttered in the 
        center while dealing with high-dimensional data points. 
        1.0 when default.
    normalized: boolean, optional 
        If this value is True, then all the points in `X` will be 
        normalized within `[0.0, 1.0]`. True when default.

    Returns
    -------
    (`P`, `K`, `B`)  : a tuple of ndarrays
        `P` is an ndarray of radviz coordinates (i.e. `|P| = n x 2`), 
        `K` is the position of the anchor points (i.e. `|K| = m x 2`),
        and `B` is the lower bound and upper bound of `P`. `K` and `B`
        will be used to draw anchor points and the polygon.
        
    """
    m = X.shape[1]
    if normalized:
        X_ = tr.normalize(X, lb = np.zeros(m), ub = np.ones(m))
    else:
        X_ = X
    T = 2 * np.pi * (np.arange(0, m, 1).astype(int) / m)
    COS = np.cos(T)
    SIN = np.sin(T)
    
    Y = np.power(X_, factor)
    S = np.sum(Y, axis=1)
    U = np.sum(Y * COS, axis=1) / S
    V = np.sum(Y * SIN, axis=1) / S
    P = np.column_stack((U, V))

    B = [np.amin(P, axis=0), np.amax(P, axis=0)]
    K = np.column_stack((COS, SIN))
    return (P, K, B)

def set_anchors(ax, A):
    tgc = mc.TABLEAU_COLORS['tab:gray']
    for i in range(0, A.shape[0]-1):
        # draw one polygon line
        ax.plot([A[i,0], A[i + 1,0]], [A[i,1], A[i + 1,1]], c=tgc, alpha=1.0, \
                    linewidth=1.0, linestyle='dashdot')
        # draw a pair of polygon points
        ax.scatter(A[i,0], A[i,1], c=tgc, marker='o', s=20.0, alpha=1.0)
    # last polygon line
    ax.plot([A[-1,0], A[0,0]], [A[-1,1], A[0,1]], c=tgc, alpha=1.0, \
                linewidth=1.0, linestyle='dashdot')
    # last pair of polygon points
    ax.scatter(A[-1,0], A[-1,1], c=tgc, marker='o', s=20.0, alpha=1.0)

def set_anchor_labels(ax, A, label_prefix, label_fontsize, label_fontname, label_fontstyle):
    tgc = mc.TABLEAU_COLORS['tab:gray']
    # now put all the corner labels, like f1, f2, f3, ... etc.
    for xy, name in zip(A, [label_prefix.format(i+1) for i in range(A.shape[0])]):
        if xy[0] < 0.0 and xy[1] < 0.0:
            ax.text(xy[0] - 0.025, xy[1] - 0.025, s=name, ha='right', va='top', \
                    fontname=label_fontname, fontsize=label_fontsize, fontstyle=label_fontstyle)
        elif xy[0] < 0.0 and xy[1] >= 0.0: 
            ax.text(xy[0] - 0.025, xy[1] + 0.025, s=name, ha='right', va='bottom', \
                    fontname=label_fontname, fontsize=label_fontsize, fontstyle=label_fontstyle)
        elif xy[0] >= 0.0 and xy[1] < 0.0:
            ax.text(xy[0] + 0.025, xy[1] - 0.025, s=name, ha='left', va='top', \
                    fontname=label_fontname, fontsize=label_fontsize, fontstyle=label_fontstyle)
        elif xy[0] >= 0.0 and xy[1] >= 0.0:
            ax.text(xy[0] + 0.025, xy[1] + 0.025, s=name, ha='left', va='bottom', \
                    fontname=label_fontname, fontsize=label_fontsize, fontstyle=label_fontstyle)
    p = Circle((0, 0), 1, fill=False, linewidth=0.8, color=tgc)
    ax.add_patch(p)

def radviz(A, plt, s = 1, c = mc.TABLEAU_COLORS['tab:blue'], \
            normalized=True, draw_axes=False, draw_anchors=True, \
            spread_factor='auto', **kwargs):
    r"""
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

    # by default spread-factor is auto
    if type(spread_factor) == str and spread_factor == 'auto':
        factor = 2.0 if A.shape[1] > 3 else 1.0
    elif type(spread_factor) == int or type(spread_factor) == float:
        factor = spread_factor
    else:
        raise TypeError("`spread_factor` must be either 'auto' or a float/int.")

    P, K, [lb, ub] = get_radviz_coordinates(A, factor=factor, \
                                            normalized=normalized)
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(P[:,0], P[:,1], s=s, c=c)
    if draw_axes:
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        ax.set_axis_on()
    else:
        ax.set_axis_off()
    if draw_anchors:
        set_anchors(ax, K)
        set_anchor_labels(ax, K, label_prefix, label_fontsize, label_fontname, label_fontstyle)
    ax.set_xlim(lb[0] - 0.1 if lb[0] < -1 else -1.1, ub[0] + 0.01 if ub[0] > 1 else 1.1)
    ax.set_ylim(lb[1] - 0.1 if lb[1] < -1 else -1.1, ub[1] + 0.01 if ub[1] > 1 else 1.1)
    ax.set_aspect('equal')
    return (fig, ax)

"""star.py -- A customized and more flexible Star-coordinate plotting module. 

    This module provides a customized and more flexible function for Star-coordinate [1]_ 
    plotting. This module also provides different relevant fucntions, parameters and tools.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA
    
    References
    ----------
    .. [1] Eser Kandogan. 2001. Visualizing Multi-dimensional Clusters, Trends, 
        and Outliers Using Star Coordinates. In Proceedings of the Seventh ACM 
        SIGKDD International Conference on Knowledge Discovery and Data Mining 
        (KDD '01). ACM, New York, NY, USA, 107--116.

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc
from matplotlib.colors import ListedColormap
from viz.utils import transform as tr
from viz.plotting.utils import set_polar_anchors, set_polar_anchor_labels, pop, group_labels_by_appearance


__all__ = ["get_star_coordinates", "plot"]


def get_star_coordinates(X, inverted=True, normalized=True):
    r"""Generate Star-coordinates from data points `X`.

    Maps all the data points in `X` (i.e. `|X| = n x m`) onto 
    star-coordinate [1]_ positions. 

    Parameters
    ----------
    X : ndarray 
        `n` number of `m` dimensiomal points as input.
    inverted : bool, optional
        In the original star-coordinate plot, a point that is exactly on 
        an anchor means the point has the lowest value in that dimension.
        However, in our case, we need to make it reverted since we want 
        to place a point farther from an anchor point if the point has the
        lowest value in that dimension. Default 'True' when optional.
    normalized : bool, optional
        We need to normalize within `[0.0, 1.0]`. 
        However in any case, if someone is not opted.
        Default 'True' when optional.

    Returns
    -------
    P : ndarray of float
        An ndarray of PaletteViz coordinates (i.e. `|P| = n x 3`).
    K : ndarray of float
        The positions of the anchor points (i.e. `|K| = m x 2`),
    B : ndarray of float
        The lower bound and upper bound of `P`. 

    References
    ----------
    .. [1] Eser Kandogan. 2001. Visualizing Multi-dimensional Clusters, Trends, 
        and Outliers Using Star Coordinates. In Proceedings of the Seventh ACM 
        SIGKDD International Conference on Knowledge Discovery and Data Mining 
        (KDD '01). ACM, New York, NY, USA, 107--116.
    """

    m = X.shape[1]
    # check if the data needs to be normalized
    if normalized:
        X_ = tr.normalize(X, lb=np.zeros(m), ub=np.ones(m))
    else:
        X_ = X

    if inverted:
        X_ = 1.0 - X_
    
    T = 2 * np.pi * (np.arange(0, m, 1).astype(int) / m)
    COS,SIN = np.cos(T), np.sin(T)
    LB,UB = np.min(X_, axis=0), np.max(X_, axis=0)
    D = (UB - LB)
    U, V = (COS / D), (SIN / D)
    P = np.column_stack((np.sum(X_ * U, axis=1), np.sum(X_ * V, axis=1)))
    
    B = [np.min(P, axis=0), np.max(P, axis=0)]
    K = np.column_stack((COS, SIN))
    return (P, K, B)


def plot(A, ax=None, s=1, c=mc.TABLEAU_COLORS['tab:blue'], inverted=True, \
        normalized=True, draw_axes=False, draw_anchors=True, **kwargs):
    r"""A customized star-coordinate plot.

    This star-coordinate [1]_ plot is customized for the experiments. 
    
    Parameters
    ----------
    A : ndarray 
        `n` number of `m` dim. points to be plotted.
    ax : An `matplotlib.axes.Axes` object, optional
        Default `None` when optional.
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
    draw_axes : bool, optional
        If `True`, the radviz plot will show axes. Default `False` when optional.
    draw_anchors : bool, optional
        If `False`, the radviz plot will hide anchors. Default `True` when optional.

    Other Parameters
    ----------------
    title : str, optional
        The plot title. Default `None` when optional.
    labels : str, array_like or list of str, optional
        A string or an array/list of strings for labeling each point. Which basically
        means the class label of each row. Default `None` when optional. This will be
        used to set the legend in the figure. If `None` there will be no legend.
    colorbar : (Cbc, Cbg, Cbl, Cbp) a tuple of (ndarray, ndarray, str, float), optional
        If a user wants to put a colorbar, a tuple `(Cbc, Cbg, Cbl)` tuple can be 
        provided. `Cbc` is an array of RGBA color values or an `matplotlib.colors` 
        object. The gradient of the colorbar is specified in `Cbg` which is an 1-D 
        array of float. Cbl is the label of the colorbar, a string. `Cbp` is the 
        colorbar padding width in `float`. `colorbar` is default `None` when optional. 
    label_prefix : str, optional
        See `set_anchor_labels()` function for details.
    label_fontsize : str or int, optional
        See `set_anchor_labels()` function for details.
    label_fontname : str, optional
        See `set_anchor_labels()` function for details.
    label_fontstyle : str, optional
        See `set_anchor_labels()` function for details.
    **kwargs : dict
        All other keyword args for matplotlib `scatter()` function.

    Returns
    -------
    ax : `matplotlib.axes.Axes` object
        A `matplotlib.axes.Axes` object.
    P : ndarray of float
        `P` is an ndarray of radviz coordinates (i.e. `|P| = n x 2`), 

    References
    ----------
    .. [1] Eser Kandogan. 2001. Visualizing Multi-dimensional Clusters, Trends, 
        and Outliers Using Star Coordinates. In Proceedings of the Seventh ACM 
        SIGKDD International Conference on Knowledge Discovery and Data Mining 
        (KDD '01). ACM, New York, NY, USA, 107--116.
    """

    # decide on what kind of axes to use
    if not ax:
        ax = fig.gca(plt.figure())
    
    title = kwargs['title'] if kwargs and 'title' in kwargs else None
    labels = kwargs['labels'] if kwargs and 'labels' in kwargs else None
    colorbar = kwargs['colorbar'] if kwargs and 'colorbar' in kwargs else None

    label_prefix = kwargs['label_prefix'] if kwargs and 'label_prefix' in kwargs else r"$f_{{{:d}}}$"
    label_fontsize = kwargs['label_fontsize'] if kwargs and 'label_fontsize' in kwargs else 'large'
    label_fontname = kwargs['label_fontname'] if kwargs and 'label_fontname' in kwargs else None
    label_fontstyle = kwargs['label_fontstyle'] if kwargs and 'label_fontstyle' in kwargs else 'normal'

    # remove once they are read
    kwargs = pop(kwargs, 'title')
    kwargs = pop(kwargs, 'labels')
    kwargs = pop(kwargs, 'colorbar')
    kwargs = pop(kwargs, 'label_prefix')
    kwargs = pop(kwargs, 'label_fontsize')
    kwargs = pop(kwargs, 'label_fontname')
    kwargs = pop(kwargs, 'label_fontstyle')

    if ax:
        P, K, [lb, ub] = get_star_coordinates(A, inverted=inverted, normalized=normalized)
        
        # do the plot
        if labels is not None:
            if isinstance(labels, str):
                ax.scatter(P[:,0], P[:,1], s=s, c=c, label=labels, **kwargs)
            else:
                if isinstance(labels, np.ndarray): 
                    labels = labels.tolist()
                label_groups = group_labels_by_appearance(labels)
                for i,v in enumerate(label_groups):
                    ax.scatter(P[v[0],0], P[v[0],1], s=s[v[0]], c=c[v[0]], label=v[1], \
                            zorder=label_groups.shape[0]-i, **kwargs)
        else:
            ax.scatter(P[:,0], P[:,1], s=s, c=c, **kwargs)        

        if draw_axes:
            ax.set_axis_on()
        else:
            ax.set_axis_off()
        
        if draw_anchors:
            set_polar_anchors(ax, K)
            set_polar_anchor_labels(ax, K, draw_circle=True, \
                        label_prefix=label_prefix, label_fontsize=label_fontsize, \
                        label_fontname=label_fontname, label_fontstyle=label_fontstyle)
        ax.set_xlim(lb[0] - 0.1 if lb[0] < -1 else -1.1, ub[0] + 0.1 if ub[0] > 1 else 1.1)
        ax.set_ylim(lb[1] - 0.1 if lb[1] < -1 else -1.1, ub[1] + 0.1 if ub[1] > 1 else 1.1)
        ax.set_aspect('equal')
        
        # colorbar?
        if colorbar and isinstance(colorbar, tuple) and len(colorbar) >= 2 \
                and isinstance(colorbar[0], np.ndarray) and isinstance(colorbar[1], np.ndarray):
            vmin,vmax = 0.0, 1.0
            cbc, cbg = colorbar[0], colorbar[1]
            cbl = colorbar[2] if len(colorbar) > 2 and colorbar[2] else None
            cbp = colorbar[3] if len(colorbar) > 3 and colorbar[3] else 0.05
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
        if labels is not None:
            ax.legend(loc="best", ncol=1)
        
        # title?
        ax.set_title(title, y=ax.get_ylim()[-1]-0.05)
        
        return (ax, P)
    else:
        raise TypeError("A valid `matplotlib.axes.Axes` object is not found.")

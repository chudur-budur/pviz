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
import matplotlib.colors as mc
from vis.utils import transform as tr
from vis.plotting.utils import set_polar_anchors, set_polar_anchor_labels


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
    (P,K,B) : tuple of ndarray
        `P` is an ndarray of radviz coordinates (i.e. `|P| = n x 2`), 
        `K` is the position of the anchor points (i.e. `|K| = m x 2`),
        and `B` is the lower bound and upper bound of `P`. `K` and `B`
        will be used to draw anchor points and the polygon.

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


def plot(A, plt, s=1, c=mc.TABLEAU_COLORS['tab:blue'], \
            inverted=True, normalized=True, \
            draw_axes=False, draw_anchors=True, **kwargs):
    r"""A customized star-coordinate plot.

    This star-coordinate [1]_ plot is customized for the experiments. 
    
    Parameters
    ----------
    A : ndarray 
        `n` number of `m` dim. points to be plotted.
    plt : A `matplotlib.pyplot` object
        It needs to be passed.
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

    References
    ----------
    .. [1] Eser Kandogan. 2001. Visualizing Multi-dimensional Clusters, Trends, 
        and Outliers Using Star Coordinates. In Proceedings of the Seventh ACM 
        SIGKDD International Conference on Knowledge Discovery and Data Mining 
        (KDD '01). ACM, New York, NY, USA, 107--116.
    """
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

    if plt is not None:
        P, K, [lb, ub] = get_star_coordinates(A, inverted=inverted, normalized=normalized)
        fig = plt.figure()
        if title is not None:
            fig.suptitle(title)
        ax = fig.gca()
        ax.scatter(P[:,0], P[:,1], s=s, c=c)
        if draw_axes:
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
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
        return (fig, ax)
    else:
        raise TypeError("A valid `matplotlib.pyplot` object must be provided.")

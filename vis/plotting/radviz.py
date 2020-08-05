"""radviz.py -- A customized and more flexible Radviz plotting module. 

    This module provides a customized and more flexible function for Radviz [1]_ plotting.
    This module also provides different relevant fucntions, parameters and tools.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA
    
    References
    ----------
    .. [1] P. Hoffman, G. Grinstein and D. Pinkney, Dimensional anchors: A graphic primitive 
    for multidimensional multivariate information visualizations", Proc. 1999 Workshop on 
    New Paradigms in Information Visualization and Manipulation in Conjunction with the 
    Eighth ACM Int. Conf. Information and Knowledge Management (NPIVM ’99), pp. 9-16, 1999.

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from mpl_toolkits.mplot3d import Axes3D
from vis.utils import transform as tr
from vis.plotting.utils import set_polar_anchors, set_polar_anchor_labels 


__all__ = ["get_radviz_coordinates", "plot"]


def get_radviz_coordinates(X, spread_factor='auto', normalized=True):
    r"""Generate Radviz coordinates from data points `X`.

    Maps all the data points in `X` (i.e. `|X| = n x m`) onto radviz [1]_ coordinate positions. 
    The `factor` parameter can be used to "spread-out" the data points on Radviz. In higher
    dimension, the points on Radviz tend to be clustered in the center. We can increase this 
    value to make the points more visible.

    Parameters
    ----------
    X : ndarray 
        `n` number of `m` dimensional points as input.
    spread_factor : str {'auto'} or float, optional
        The `spread_factor` might be needed if the data points are not sparse enough
        in the original space. Setting this bigger than 1.0 will help the data points
        to "spread-out" even more, while maintaining the placement of the points inside
        the radviz anchor points. If `spread_factor = 'auto'`, the `factor` is 2.0 if 
        `m > 3`, otherwise 1.0. Default `'auto'` when default.
    normalized : bool, optional 
        If this value is True, then all the points in `X` will be 
        normalized within `[0.0, 1.0]`. Default `True` when optional.

    Returns
    -------
    (P,K,B) : tuple of ndarray
        `P` is an ndarray of radviz coordinates (i.e. `|P| = n x 2`), 
        `K` is the position of the anchor points (i.e. `|K| = m x 2`),
        and `B` is the lower bound and upper bound of `P`. `K` and `B`
        will be used to draw anchor points and the polygon.

    References
    ----------
    .. [1] P. Hoffman, G. Grinstein and D. Pinkney, Dimensional anchors: A graphic primitive 
    for multidimensional multivariate information visualizations", Proc. 1999 Workshop on 
    New Paradigms in Information Visualization and Manipulation in Conjunction with the 
    Eighth ACM Int. Conf. Information and Knowledge Management (NPIVM ’99), pp. 9-16, 1999.

    .. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

    """
    m = X.shape[1]

    # by default spread-factor is auto
    if type(spread_factor) == str and spread_factor == 'auto':
        factor = 2.0 if m > 3 else 1.0
    elif type(spread_factor) == int or type(spread_factor) == float:
        factor = spread_factor
    else:
        raise TypeError("`spread_factor` must be either 'auto' or a float/int.")
    
    # check if the data needs to be normalized
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

    B = [np.min(P, axis=0), np.max(P, axis=0)]
    K = np.column_stack((COS, SIN))
    return (P, K, B)


def plot(A, ax=None, s=1, c=mc.TABLEAU_COLORS['tab:blue'], \
            normalized=True, draw_axes=False, draw_anchors=True, \
            spread_factor='auto', **kwargs):
    r"""A customized radviz plot.

    This radviz [1]_ plot is customized for the experiments. 
    
    Parameters
    ----------
    A : ndarray 
        `n` number of `m` dim. points to be plotted.
    ax : A `matplotlib.axes.Axes` object, optional
        Default `None` when optional.
    s : int or 1-D array_like, optional
        Point size, or an array of point sizes. Default 1 when optional.
    c : A `matplotlib.colors` object, str or an array RGBA color values.
        Colors to be used. Default `mc.TABLEAU_COLORS['tab:blue']` when 
        optional.
    normalized : bool, optional
        If needed, the data points in `A` can be normalized within `[0.0, 1.0]`. 
        This helps to "spread-out" the data more on the radviz space. Default 
        `True` when optional.
    draw_axes : bool, optional
        If `True`, the radviz plot will show axes. Default `False` when optional.
    draw_anchors : bool, optional
        If `False`, the radviz plot will hide anchors. Default `True` when optional.
    spread_factor : str {'auto'} or float, optional
        See `get_radviz_coordinates()` function for details.

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
    ax : `matplotlib.axes.Axes` object
        A `matplotlib.axes.Axes` object.
    
    References
    ----------
    .. [1] P. Hoffman, G. Grinstein and D. Pinkney, Dimensional anchors: A graphic primitive 
    for multidimensional multivariate information visualizations", Proc. 1999 Workshop on 
    New Paradigms in Information Visualization and Manipulation in Conjunction with the 
    Eighth ACM Int. Conf. Information and Knowledge Management (NPIVM ’99), pp. 9-16, 1999.
    """
    
    # decide on what kind of axes to use
    if ax is None:
        ax = fig.gca(plt.figure())

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

    if ax is not None:
        P, K, [lb, ub] = get_radviz_coordinates(A, spread_factor=spread_factor, normalized=normalized)
        ax.scatter(P[:,0], P[:,1], s=s, c=c)
        if draw_axes:
            ax.set_axis_on()
        else:
            ax.set_axis_off()
        if draw_anchors:
            set_polar_anchors(ax, K)
            set_polar_anchor_labels(ax, K, label_prefix=label_prefix, label_fontsize=label_fontsize, \
                    label_fontname=label_fontname, label_fontstyle=label_fontstyle)
        ax.set_xlim(lb[0] - 0.1 if lb[0] < -1 else -1.1, ub[0] + 0.01 if ub[0] > 1 else 1.1)
        ax.set_ylim(lb[1] - 0.1 if lb[1] < -1 else -1.1, ub[1] + 0.01 if ub[1] > 1 else 1.1)
        ax.set_aspect('equal')
        ax.set_title(title, y=ax.get_ylim()[1]-0.05)
        return ax
    else:
        raise TypeError("Either a valid `matplotlib.axes.Axes` object is not found.")

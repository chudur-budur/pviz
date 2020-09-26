"""utils.py -- A collection of different utility functions for plotting.
    
    This module provides different utility functions for plotting. For example,
    changing/setting point and line colors, point and line size, setting up
    2D/3D patches etc.

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
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from viz.utils import transform as tr

__all__ = ["resize_by_tradeoff", "default_color", "color_by_cv", \
            "color_by_dist", "enhance_color", \
            "set_polar_anchors", "set_polar_anchor_labels", \
            "pop", "Arrow3D", "group_labels_by_appearance", "cfs"]


"""
    Different color factors for different problems with constraint functions.
    Users can use these for better viewing. These were found empirically.
"""
cfs = {'dtlz8': { 2: 2.0, 3: 6.0, 4: 8.0, 6: 14.0, 8: 16.0}}


def group_labels_by_appearance(labels):
    r""" A function to partition the data according to the labels.

    Let's say we have `n` data points, `F = {p1, p2, p3, ..., pn}`.
    Also each data point has a label, and there are 2 kind of labels, 
    i.e. `L = {l1, l2, l1, l1, ..., l2}` where each label `li` corresponds
    to point `pi`. This function partitions `F` with respect to `L` and
    returns a list of lists `ul`, where `ul = [[li, [i1, i2, i3, ..., ip]], 
    [lj, [j1, j2, j3, ..., jq]] ... ]`. This means all the data points 
    indexed by `ii` have labels `li` and all the data points indexed by 'jj' 
    have labels `lj` and so on.
    
    Parameters
    ----------
    labels : list of str
        A list of strings specifying label for each data point.

    Returns
    -------
    ul : ndarray of object
        A jagged array of arrays. Where each array has two items.
        The first item is the label and the next item is an array
        of int for indices of data points with that label.
    """
    ld = {}
    for i,label in enumerate(labels):
        if label in ld:
            ld[label].append(i)
        else:
            ld[label] = []
    ul = []
    for k in ld.keys():
        ul.append([ld[k], k])
    sorted(ul, key = lambda v: len(v[0]), reverse=True)
    return np.array(ul, dtype=object)


def pop(d, k):
    r"""A safe popping from a dict.
    """
    r = dict(d)
    if k in d:
        del r[k]
    return r

# we might need to draw arrows.
class Arrow3D(FancyArrowPatch):
    r"""A 3D arrow class.

    This class is derived from `matplotlib.patches.FancyArrowPath`.
    
    Usage: A 3D arrow from `(x1,y1,z1)` to `(x2,y2,z2)` will look like this.

    ```
    a = Arrow3D([x1, x2], [y1, y2], [z1, z2], mutation_scale = 10, lw = 1.0, \
                arrowstyle = "-|>", color = 'black')
    ax.add_artist(a)
    ```
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def resize_by_tradeoff(Mu, k=None, minsize=2.0, maxsize=10.0, kminsize=3.0, kmaxsize=5.0):
    r"""Resize the points w.r.t. tradeoff values.

    If we need to resize the points with respect to the tradeoff values, we can use this 
    function. This function assumes `Mu` values are within `[0.0,1.0]`. 

    Since point size of 0 is not visible, we normalize them within `[minsize, maxsize]`.
    In order to capture the relative tradeoffs, we then scale the sizes by the power of 2.
    If the user wants to emphasize a specific set of points so that they 'stand out' 
    from all the rest of the points, they can be indexed with `k`. Those points will be 
    pronounced by normalizing them within `[kminsize, kmaxsize]` and then scale them by 
    the power of 2.
    
    Parameters
    ----------
    Mu : 1-D array_like
        A 1D-array of tradeoff values, float. Also `0.0 <= Mu <= 1.0`.
    k : 1-D array_like, optional
        A 1D-array of integer indices to be used to specify which points
        will be increased in size by `kfactor`. Default `None` when optional.
    minsize : float, optional
        The minimum allowable size of each point before the exponential 
        scaling. Default 2.0 when optional.
    maxsize : float, optional
        The maximum allowable size of each point before the exponential 
        scaling. Default 10.0 when optional.
    kminsize : float, optional
        The minimum allowable size of the points indexed by `k`, before 
        the exponential scaling. Default 3.0 when optional.
    kmaxsize : float, optional
        The maximum allowable size of the points indexed by `k`, before 
        the exponential scaling. Default 5.0 when optional.
    """
    
    S = np.power(tr.normalize(Mu, lb=np.array([minsize]), ub=np.array([maxsize])), 2)
    if k is not None:
        S[k] = np.power(tr.normalize(Mu[k], lb=np.array([kminsize]), ub=np.array([kmaxsize])), 3)
    return S


def default_color(n, c=mc.TABLEAU_COLORS['tab:blue'], alpha=1.0):
    r"""Get an array of RGBA color values for default coloring.

    In any case, if we need to revert the point colorings to the 
    default matplotlib (`mc.TABLEAU_COLORS['tab:blue']`) coloring 
    scheme, we can use this function.

    Parameters
    ----------
    n : int
        The length of the output array containing RGBA color values.
    c : A `matplotlib.colors` object, str or an array RGBA color values.
        Colors to be used. Default `mc.TABLEAU_COLORS['tab:blue']` when 
        optional.
    alpha : float, optional
        Alpha transparency value. Default 0.5 when optional.
        
    Returns
    -------
    C : ndarray
        An array of RGBA color values.
    """
    C = np.array([mc.to_rgba(c, alpha) for _ in range(n)])
    return C


def color_by_cv(CV, factor=0.8, alpha=0.5):
    r"""Generate an array of color values from CV.

    Generate an array of RGBA color values from an array of 
    cumulative constraint violation values. This function
    uses the color gradient from `cm.cool`.

    Parameters
    ----------
    CV : ndarray
        An nd-array of cumulative contstraint violation values.
    factor : float, optional
        If `factor = 1.0`, then we will use the actual gradient
        from `cm.cool`. When this value is smaller than 1.0, then
        the gradient will be shifted left (i.e. some portion from 
        the highest end of the color gradient will be skipped). 
        Default 0.8 when optional.
    alpha : float, optional
        Alpha transparency value. Default 0.5 when optional.
        
    Returns
    -------
    `C` : ndarray
        An array of RGBA color values.
    """
    CV_ = tr.normalize(CV, lb=0.0, ub=1.0)
    C = np.array([mc.to_rgba(cm.cool(v * factor), alpha) for v in CV_])
    return C


def color_by_dist(X, P, alpha=0.5, factor=1.75):
    r"""Generate an array of RGBA color values w.r.t distance of 'X' from 'P'

    Generate an array of RGBA color values for the corresponding points
    in `X` with respect to the their distances from a single point `P`.
    We generally use `P` as the center of mass of `X`, but can be used
    in other contexts.

    Parameters
    ----------
    X : ndarray
        An set of `m`-dimensional data points, i.e. `|X| = n x m` 
    P : 1D-array
        A point `P`, a 1-D array. This can be the center of mass of `X`.
    alpha : float, optional
        Alpha transparency value. Default 0.5 when optional.
    factor : float, optional
        If `factor = 1.0`, then we will use the actual gradient
        from `cm.winter_r`. When this value is smaller than 1.0, then
        the gradient will be shifted right (i.e. some portion from 
        the lowest end of the color gradient will be skipped). 
        Default 1.75 when optional. A user might want to try with 
        different values for factor.
        
    Returns
    -------
    C : ndarray
        An array of RGBA color values.
    D : array_like
        An array of distance values.
    """
    D = tr.normalize(np.linalg.norm(P - X, axis=1), lb=0.1, ub=1.0)
    C = np.array([mc.to_rgba(cm.winter_r(v * factor), alpha) for v in D])
    return C, D


def enhance_color(C, k, alpha=1.0, c=mc.TABLEAU_COLORS['tab:red']):
    r"""Enhance the color of selected data points.

    Given an array of RGBA color values `C`, this function will enhance
    all the points indexed by `Ik` by recoloring them with TABLEAU red
    color. Assuming that the color of the other points won't overlap with
    the enhanced points.
    
    Parameters
    ----------
    C: ndarray
        An array of RGBA color values as input.
    Ik : array_like of int
        An array of integer indices.
    alpha : float, optional
        Alpha transparency value. Default 0.5 when optional.
    color : RGB color value, optional
        The color to be used to enhance the points. 
        Default `mc.TABLEAU_COLORS['tab:red']` when optional.
        
    Returns
    -------
    C_ : ndarray
        An array of RGBA color values.
    """
    C_ = np.array(C, copy=True)
    C_[k] = np.array([mc.to_rgba(c, alpha) for _ in range(C[k].shape[0])])
    return C_


def set_polar_anchors(ax, A, z=None, anchor_linewidth=1.0):
    r"""Function to draw anchor points for radviz (and other related plots).

    This function draws anchor points for radviz and other related plots
    like Star-coordinate plots.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes` object
        This object must be passed.
    A : ndarray 
        `m x 2` size of sin and cos values for the anchor coordinates.
    z : float, optional
        The z-coordinate values of the corresponding acnhor in 3D.
        Default 'None' when optional.
    anchor_linewidth : float, optional
        The width for anchor lines. Default 1.0 when optional.
    """
    tgc = mc.TABLEAU_COLORS['tab:gray']
    for i in range(0, A.shape[0]-1):
        if z is None:
            # draw one polygon line
            ax.plot([A[i,0], A[i + 1,0]], [A[i,1], A[i + 1,1]], c=tgc, alpha=1.0, \
                        linewidth=anchor_linewidth, linestyle='dashdot')
            # draw a pair of polygon points
            ax.scatter(A[i,0], A[i,1], c=tgc, marker='o', s=20.0, alpha=1.0)
        else:
            # draw one polygon line
            ax.plot([A[i,0], A[i + 1,0]], [A[i,1], A[i + 1,1]], zs=z, c=tgc, alpha=1.0, \
                        linewidth=anchor_linewidth, linestyle='dashdot')
            # draw a pair of polygon points
            ax.scatter(A[i,0], A[i,1], zs=z, c=tgc, marker='o', s=20.0, alpha=1.0)
    if z is None:
        # last polygon line
        ax.plot([A[-1,0], A[0,0]], [A[-1,1], A[0,1]], c=tgc, alpha=1.0, \
                    linewidth=anchor_linewidth, linestyle='dashdot')
        # last pair of polygon points
        ax.scatter(A[-1,0], A[-1,1], c=tgc, marker='o', s=20.0, alpha=1.0)
    else:
        # last polygon line
        ax.plot([A[-1,0], A[0,0]], [A[-1,1], A[0,1]], zs=z, c=tgc, alpha=1.0, \
                    linewidth=anchor_linewidth, linestyle='dashdot')
        # last pair of polygon points
        ax.scatter(A[-1,0], A[-1,1], zs=z, c=tgc, marker='o', s=20.0, alpha=1.0)


def set_polar_anchor_labels(ax, A, z=None, draw_circle=False, label_prefix=r"$f_{:d}$", \
                            label_fontsize='large', label_fontname=None, label_fontstyle=None):
    r"""Function to put anchor labels for radviz and other related plots.

    Function to draw anchor point labels for radivz and other related plots
    like Star-coordinate plots.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes` object
        This object must be passed.
    A : ndarray 
        `m x 2` size of sin and cos values for the anchor coordinates.
    z : float, optional
        The z-coordinate values of the corresponding acnhor in 3D.
        Default 'None' when optional.
    draw_circle : bool, optional
        Draws a circum-cricle around the anchor points. 
        Default `False` when optional.
    label_prefix : str, optional
        The axis-label-prefix to be used, default `r"$f_{:d}$"` when optional.
    label_fontsize : str or int, optional
        The fontsize for the axes labels. Default `'large'` when optional.
    label_fontname : str, optional
        The fontname for the axes labels. Default `None` when optional.
    label_fontstyle : str, optional
        The fontstyle for the axes labels. Default `None` when optional.
    """
    tgc = mc.TABLEAU_COLORS['tab:gray']
    # now put all the corner labels, like f1, f2, f3, ... etc.
    for xy, name in zip(A, [label_prefix.format(i+1) for i in range(A.shape[0])]):
        if z is None:
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
        else:
            if xy[0] < 0.0 and xy[1] < 0.0:
                ax.text(xy[0] - 0.025, xy[1] - 0.025, z=z, s=name, ha='right', va='bottom', \
                        fontname=label_fontname, fontsize=label_fontsize, fontstyle=label_fontstyle)
            elif xy[0] < 0.0 and xy[1] >= 0.0: 
                ax.text(xy[0] - 0.025, xy[1] + 0.025, z=z, s=name, ha='right', va='bottom', \
                        fontname=label_fontname, fontsize=label_fontsize, fontstyle=label_fontstyle)
            elif xy[0] >= 0.0 and xy[1] < 0.0:
                ax.text(xy[0] + 0.025, xy[1] - 0.025, z=z, s=name, ha='left', va='bottom', \
                        fontname=label_fontname, fontsize=label_fontsize, fontstyle=label_fontstyle)
            elif xy[0] >= 0.0 and xy[1] >= 0.0:
                ax.text(xy[0] + 0.025, xy[1] + 0.025, z=z, s=name, ha='left', va='bottom', \
                        fontname=label_fontname, fontsize=label_fontsize, fontstyle=label_fontstyle)
    if draw_circle:
        p = Circle((0, 0), 1, fill=False, linewidth=0.8, color=tgc)
        ax.add_patch(p)
        if z:
            art3d.pathpatch_2d_to_3d(p, z=z)

"""scatter.py -- A customized scatter plotting module. 

    This module provides a customized scatter plotting functions for
    high-dimensional Pareto-optimal fronts. It also provides different
    relevant parameters, tools and utilities.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from viz.plotting.utils import pop, group_labels_by_appearance


__all__ = ["camera_angles", "plot", "camera_angles"]


# Some good camera angles for scatter plots.
camera_angles = {
    'dtlz2': {'3d': (60,10), '4d':(105,15), '8d': (15,15)}, \
    'dtlz2-nbi': {'3d': (60,10), '4d':(105,15), '5d': (105,15), '8d': (110,15)}, \
    'debmdk': {'3d': (115,5), '4d': (105,15), '8d': (110,15)}, \
    'debmdk-nbi': {'3d': (115,5), '4d': (105,15), '8d': (110,15)}, \
    'debmdk-all': {'3d': (115,5), '4d': (105,15), '8d': (110,15)}, \
    'debmdk-all-nbi': {'3d': (115,5), '4d': (105,15), '8d': (110,15)}, \
    'dtlz8': {'3d': (110,15), '4d': (-65,15), '6d': (-65,15), '8d': (-65,15)}, \
    'dtlz8-nbi': {'3d': (110,15), '4d': (-65,15), '6d': (-65,15), '8d': (-65,15)}, \
    'c2dtlz2': {'3d': (30,10), '4d': (-75,15), '5d': (-65,15), '8d': (110,15)}, \
    'c2dtlz2-nbi': {'3d': (30,10), '4d': (-75,15), '5d': (-65,15), '8d': (110,15)}, \
    'cdebmdk': {'3d': (30,10), '4d': (-75,15), '8d': (110,15)}, \
    'cdebmdk-nbi': {'3d': (30,10), '4d': (-75,15), '8d': (110,15)}, \
    'c0dtlz2': {'3d': (30,10), '4d': (-75,15), '8d': (110,15)}, \
    'c0dtlz2-nbi': {'3d': (30,10), '4d': (-75,15), '8d': (110,15)}, \
    'crash-nbi': {'3d': (30,15)}, 'crash-c1-nbi': {'3d': (30,15)}, 'crash-c2-nbi': {'3d': (30,15)}, \
    'gaa': {'10d': (-65,15)}, \
    'gaa-nbi': {'10d': (-65,15)}
}


def plot(A, ax=None, s=1, c=mc.TABLEAU_COLORS['tab:blue'], **kwargs):
    r"""A scatter plot function.

    This uses `matplotlib.axes.Axes.scatter` function to do a scatter plot.

    Parameters
    ----------
    A : ndarray 
        `n` number of `m` dim. points to be plotted.
    ax : An `mpl_toolkits.mplot3d.axes.Axes3D` or an `matplotlib.axes.Axes` object, optional
        Axes to be used to plotting. Default `None` when optional.
    s : int or 1-D array_like, optional
        Point size, or an array of point sizes. Default 1 when optional.
    c : A `matplotlib.colors` object, str or an array RGBA color values.
        Colors to be used. Default `mc.TABLEAU_COLORS['tab:blue']` when optional.

    Other Parameters
    ----------------
    euler : tuple (i.e. a pair) of int, optional
        The azmiuth and elevation angle. Default `(-60,30)` when optional.
    title : str, optional
        The plot title. Default `None` when optional.
    axes : tuple of int, optional
        The list of columns of `A` to be plotted. Default `(0, 1, 2)` when optional.
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
    xlim : tuple (i.e. a pair) of int, optional
        The limits on the X-axis. Default `None` when optional.
    ylim : tuple (i.e. a pair) of int, optional 
        The limits on the Y-axis. Default `None` when optional.
    zlim : tuple (i.e. a pair) of int, optional 
        The limits on the Z-axis. Default `None` when optional.
    label_prefix : str, optional
        The axis-label-prefix to be used, default `r"$f_{{{:d}}}$"` when optional.
    label_fontsize : str or int, optional
        The fontsize for the axes labels. Default `'large'` when optional.
    **kwargs : dict
        All other keyword args for matplotlib's `scatter()` function.

    Returns
    -------
    ax : An `mpl_toolkits.mplot3d.axes.Axes3D` or `matplotlib.axes.Axes` object
        An `mpl_toolkits.mplot3d.axes.Axes3D` or an `matplotlib.axes.Axes` object.
    """

    # all other parameters
    euler = kwargs['euler'] if kwargs and 'euler' in kwargs else (-60, 30)
    title = kwargs['title'] if kwargs and 'title' in kwargs else None
    axes = kwargs['axes'] if kwargs and 'axes' in kwargs else (0, 1, 2)
    labels = kwargs['labels'] if kwargs and 'labels' in kwargs else None
    colorbar = kwargs['colorbar'] if kwargs and 'colorbar' in kwargs else None
    xlim = kwargs['xlim'] if kwargs and 'xlim' in kwargs else None
    ylim = kwargs['ylim'] if kwargs and 'ylim' in kwargs else None
    zlim = kwargs['zlim'] if kwargs and 'zlim' in kwargs else None
    label_prefix = kwargs['label_prefix'] if kwargs and 'label_prefix' in kwargs else r"$f_{{{:d}}}$"
    label_fontsize = kwargs['label_fontsize'] if kwargs and 'label_fontsize' in kwargs else 'large'
            
    # remove once they are read
    kwargs = pop(kwargs, 'euler')
    kwargs = pop(kwargs, 'title')
    kwargs = pop(kwargs, 'axes')
    kwargs = pop(kwargs, 'labels')
    kwargs = pop(kwargs, 'colorbar')
    kwargs = pop(kwargs, 'xlim')
    kwargs = pop(kwargs, 'ylim')
    kwargs = pop(kwargs, 'zlim')
    kwargs = pop(kwargs, 'label_prefix')
    kwargs = pop(kwargs, 'label_fontsize')

    # decide on what kind of axes to use
    if not ax:
        ax = Axes3D(plt.figure()) if A.shape[1] > 2 else plt.figure().gca()

    if ax:
        # do the plot
        if labels is not None:
            if isinstance(labels, str):
                if A.shape[1] > 2:
                    ax.scatter(A[:,axes[0]], A[:,axes[1]], A[:,axes[2]], s=s, c=c, label=labels, **kwargs)
                else:
                    ax.scatter(A[:,axes[0]], A[:,axes[1]], s=s, c=c, label=labels, **kwargs)
            else:
                if isinstance(labels, np.ndarray): 
                    labels = labels.tolist()
                label_groups = group_labels_by_appearance(labels)
                for i,v in enumerate(label_groups):
                    if A.shape[1] > 2:
                        ax.scatter(A[v[0],axes[0]], A[v[0],axes[1]], A[v[0],axes[2]], \
                                s=s[v[0]], c=c[v[0]], label=v[1], zorder=label_groups.shape[0]-i, **kwargs)
                    else:
                        ax.scatter(A[v[0],axes[0]], A[v[0],axes[1]], s=s[v[0]], c=c[v[0]], \
                                label=v[1], zorder=label_groups.shape[0]-i, **kwargs)
        else:
            if A.shape[1] > 2:
                ax.scatter(A[:,axes[0]], A[:,axes[1]], A[:,axes[2]], s=s, c=c, **kwargs)
            else:
                ax.scatter(A[:,axes[0]], A[:,axes[1]], s=s, c=c, **kwargs)

        # set limits, put labels, fix labels, view and title
        if A.shape[1] > 2:
            ax.set_xlim(ax.get_xlim() if xlim is None else xlim)
            ax.set_ylim(ax.get_ylim() if ylim is None else ylim)
            ax.set_zlim(ax.get_zlim() if zlim is None else zlim)
            if len(axes) > 0:
                ax.set_xlabel(label_prefix.format(axes[0] + 1), fontsize=label_fontsize)
            if len(axes) > 1:
                ax.set_ylabel(label_prefix.format(axes[1] + 1), fontsize=label_fontsize)
            if len(axes) > 2:
                ax.set_zlabel(label_prefix.format(axes[2] + 1), fontsize=label_fontsize)
            ax.xaxis.set_rotate_label(False)
            ax.yaxis.set_rotate_label(False)
            ax.zaxis.set_rotate_label(False)
            ax.view_init(euler[1], euler[0])
            ax.set_title(title, pad=0.1)
        else:
            ax.set_xlim(ax.get_xlim() if xlim is None else xlim)
            ax.set_ylim(ax.get_ylim() if ylim is None else ylim)
            if len(axes) > 0:
                ax.set_xlabel(label_prefix.format(axes[0] + 1), fontsize=label_fontsize)
            if len(axes) > 1:
                ax.set_ylabel(label_prefix.format(axes[1] + 1), fontsize=label_fontsize)
            ax.set_title(title, y=ax.get_ylim()[-1]-0.05)
        
        # colorbar?
        if colorbar and isinstance(colorbar, tuple) and len(colorbar) >= 2 \
                and isinstance(colorbar[0], np.ndarray) and isinstance(colorbar[1], np.ndarray):
            vmin,vmax = 0.0, 1.0
            cbc, cbg = colorbar[0], colorbar[1]
            cbl = colorbar[2] if len(colorbar) > 2 and colorbar[2] else None
            cbp = colorbar[3] if len(colorbar) > 3 and colorbar[3] else 0.01
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
            ax.legend(loc="best", ncol=1)

        return ax
    else:
        raise TypeError("A valid `mpl_toolkits.mplot3d.axes.Axes3D`/`matplotlib.axes.Axes` " 
                + "object is not found.")

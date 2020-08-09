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
from vis.plotting.utils import pop


__all__ = ["camera_angles", "plot"]


# Some good camera angles for scatter plots.
camera_angles = {
    'dtlz2': {'3d': (60,20), '4d':(-60,30), '8d': (22,21)}, \
    'dtlz2-nbi': {'3d': (60,20), '4d':(-60,30), '5d': (-60,30), '8d': (-60,30)}, \
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


def plot(A, ax=None, s=1, c=mc.TABLEAU_COLORS['tab:blue'], draw_colorbar=False, **kwargs):
    r"""A scatter plot function.

    This uses `matplotlib.axes.Axes.scatter` function to do a scatter plot.

    Parameters
    ----------
    A : ndarray 
        `n` number of `m` dim. points to be plotted.
    ax : An `mpl_toolkits.mplot3d.axes.Axes3D` or an `matplotlib.axes.Axes` object, optional
        Default `None` when optional.
    s : int or 1-D array_like, optional
        Point size, or an array of point sizes. Default 1 when optional.
    c : A `matplotlib.colors` object, str or an array RGBA color values.
        Colors to be used. Default `mc.TABLEAU_COLORS['tab:blue']` when optional.
    draw_colorbar : bool, optional
        Decide whether we are showing any colorbar. The plot supports only vertical
        colorbars at the outside of the right side of the y-axis. Default `False`
        when optional. Also, points indexed by Ik will not be used for colorbar.

    Other Parameters
    ----------------
    euler : tuple (i.e. a pair) of int, optional
        The azmiuth and elevation angle. Default `(-60,30)` when optional.
    title : str, optional
        The plot title. Default `None` when optional.
    axes : tuple of int, optional
        The list of columns of `A` to be plotted. Default `(0, 1, 2)` when optional.
    xlim : tuple (i.e. a pair) of int, optional 
        The limits on the X-axis. Default `None` when optional.
    ylim : tuple (i.e. a pair) of int, optional 
        The limits on the Y-axis. Default `None` when optional.
    zlim : tuple (i.e. a pair) of int, optional 
        The limits on the Z-axis. Default `None` when optional.
    label_prefix : str, optional
        The axis-label-prefix to be used, default `r"$f_{:d}$"` when optional.
    label_fontsize : str or int, optional
        The fontsize for the axes labels. Default `'large'` when optional.
    Ik : array_like of int, optional
        The indices of knee points or any other points of interest. Default
        `None` when optional. If `Ik` is provided, the data points will be
        divided into two groups, one indexed by `Ik` and others not. Then the
        points indexed by `Ik` will be plotted at the end.
    point_labels : str, array_like or list of str, optional
        A string or an array/list of strings for labeling each point. Which basically
        means the class label of each row. Default `None` when optional. This will be
        used to set the legend in the figure. If `None` there will be no legend.
    cbar_grad : array_like of float, optional
        The gradient of the colorbar. A 1-D array of floats.
        Default `None` when optional.
    cbar_label : str, optional
        The label of the colorbar. Default `None` when optional.
    **kwargs : dict
        All other keyword args for matplotlib's `scatter()` function.

    Returns
    -------
    ax : An `mpl_toolkits.mplot3d.axes.Axes3D` or `matplotlib.axes.Axes` object
        An `mpl_toolkits.mplot3d.axes.Axes3D` or an `matplotlib.axes.Axes` object.
    """

    # all other parameters
    euler = kwargs['euler'] if (kwargs is not None and 'euler' in kwargs) else (-60, 30)
    title = kwargs['title'] if (kwargs is not None and 'title' in kwargs) else None
    axes = kwargs['axes'] if (kwargs is not None and 'axes' in kwargs) else (0, 1, 2)
    xlim = kwargs['xlim'] if (kwargs is not None and 'xlim' in kwargs) else None
    ylim = kwargs['ylim'] if (kwargs is not None and 'ylim' in kwargs) else None
    zlim = kwargs['zlim'] if (kwargs is not None and 'zlim' in kwargs) else None
    label_prefix = kwargs['label_prefix'] \
            if (kwargs is not None and 'label_prefix' in kwargs) else r"$f_{{{:d}}}$"
    label_fontsize = kwargs['label_fontsize'] \
            if (kwargs is not None and 'label_fontsize' in kwargs) else 'large'
    Ik = kwargs['Ik'] if (kwargs is not None and 'Ik' in kwargs) else None
    point_labels = kwargs['point_labels'] if (kwargs is not None and 'point_labels' in kwargs) else None
    cbar_grad = kwargs['cbar_grad'] if (kwargs is not None and 'cbar_grad' in kwargs) else None
    cbar_label = kwargs['cbar_label'] if (kwargs is not None and 'cbar_label' in kwargs) else None
            
    # remove once they are read
    kwargs = pop(kwargs, 'euler')
    kwargs = pop(kwargs, 'title')
    kwargs = pop(kwargs, 'axes')
    kwargs = pop(kwargs, 'xlim')
    kwargs = pop(kwargs, 'ylim')
    kwargs = pop(kwargs, 'zlim')
    kwargs = pop(kwargs, 'label_prefix')
    kwargs = pop(kwargs, 'label_fontsize')
    kwargs = pop(kwargs, 'Ik')
    kwargs = pop(kwargs, 'point_labels')
    kwargs = pop(kwargs, 'cbar_grad')
    kwargs = pop(kwargs, 'cbar_label')

    # if Ik is provided, partition the data into two groups
    Ip = None
    if Ik is not None:
        I = np.zeros(A.shape[0]).astype(bool)
        I[Ik] = True
        Ik,Ip = I, ~I

    # get a list of point labels, i.e. class labels
    if point_labels is not None and isinstance(point_labels, str):
        pl = point_labels
        point_labels = np.array([pl for _ in range(A.shape[0])])

    # decide on what kind of axes to use
    if ax is None:
        ax = Axes3D(plt.figure()) if A.shape[1] > 2 else plt.figure().gca()

    if ax is not None:
        # do the plot
        if point_labels is not None:
            if Ik is not None:
                lp = np.unique(point_labels[Ip])
                lk = np.unique(point_labels[Ik])
                if A.shape[1] < 3:
                    ax.scatter(A[Ip,axes[0]], A[Ip,axes[1]], s=s[Ip], c=c[Ip], label=lp[0], **kwargs)
                    ax.scatter(A[Ik,axes[0]], A[Ik,axes[1]], s=s[Ik], c=c[Ik], label=lk[0], **kwargs)
                else:
                    ax.scatter(A[Ip,axes[0]], A[Ip,axes[1]], A[Ip,axes[2]], \
                            s=s[Ip], c=c[Ip], label=lp[0], **kwargs)
                    ax.scatter(A[Ik,axes[0]], A[Ik,axes[1]], A[Ik,axes[2]], \
                            s=s[Ik], c=c[Ik], label=lk[0], **kwargs)
            else:
                lp = np.unique(point_labels)
                if A.shape[1] < 3:
                    ax.scatter(A[:,axes[0]], A[:,axes[1]], s=s, c=c, label=lp[0], **kwargs)
                else:
                    ax.scatter(A[:,axes[0]], A[:,axes[1]], A[:,axes[2]], s=s, c=c, label=lp[0], **kwargs)
        else:
            if Ik is not None:
                if A.shape[1] < 3:
                    ax.scatter(A[Ip,axes[0]], A[Ip,axes[1]], s=s[Ip], c=c[Ip], **kwargs)
                    ax.scatter(A[Ik,axes[0]], A[Ik,axes[1]], s=s[Ik], c=c[Ik], **kwargs)
                else:
                    ax.scatter(A[Ip,axes[0]], A[Ip,axes[1]], A[Ip,axes[2]], \
                            s=s[Ip], c=c[Ip], label=lp[0], **kwargs)
                    ax.scatter(A[Ik,axes[0]], A[Ik,axes[1]], A[Ik,axes[2]], \
                            s=s[Ik], c=c[Ik], label=lk[0], **kwargs)
            elif A.shape[1] < 3:
                ax.scatter(A[:,axes[0]], A[:,axes[1]], s=s, c=c, **kwargs)
            else:
                ax.scatter(A[:,axes[0]], A[:,axes[1]], A[:,axes[2]], s=s, c=c, **kwargs)

        if A.shape[1] < 3:
            ax.set_xlim(ax.get_xlim() if xlim is None else xlim)
            ax.set_ylim(ax.get_ylim() if ylim is None else ylim)
            if len(axes) > 0:
                ax.set_xlabel(label_prefix.format(axes[0] + 1), fontsize=label_fontsize)
            if len(axes) > 1:
                ax.set_ylabel(label_prefix.format(axes[1] + 1), fontsize=label_fontsize)
            if title is not None:
                ax.set_title(title, y=ax.get_ylim()[-1]-0.05)
        else:
            ax.scatter(A[:,axes[0]], A[:,axes[1]], A[:,axes[2]], s=s, c=c, **kwargs) 
            
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
            if title is not None:
                ax.set_title(title, pad=0.1)
        # colorbar?
        if draw_colorbar:
            vmin,vmax = 0.0, 1.0
            if cbar_grad is not None:
                if Ik is not None:
                    c, cbar_grad = c[Ip], cbar_grad[Ip]
                Id = np.column_stack((cbar_grad,c)).astype(object)
                Id = Id[np.argsort(Id[:, 0])] 
                c, cbar_grad = Id[:,1:].astype(float), Id[:,0].astype(float)
                vmin, vmax = np.min(cbar_grad), np.max(cbar_grad)
            norm = mc.Normalize(vmin=vmin, vmax=vmax)
            cmap = ListedColormap(c)
            if cbar_label is not None:
                ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), \
                            orientation='vertical', label=cbar_label, pad=0.03, shrink=0.75)
            else:
                ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), \
                            orientation='vertical', pad=0.03, shrink=0.75)
        
        # where to put the legend
        if point_labels is not None:
            ax.legend(loc="upper right", ncol=1)

        return ax
    else:
        raise TypeError("A valid `mpl_toolkits.mplot3d.axes.Axes3D`/`matplotlib.axes.Axes` " 
                + "object is not found.")

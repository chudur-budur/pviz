"""pcp.py -- A customized and more flexible Parallel-coordinate plotting module. 

    This module provides a customized and more flexible function for Parallel-coordinate 
    Plot (PCP) [1]_ visualization. This module also provides different relevant fucntions, 
    parameters and tools.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA
    
    References
    ----------
    .. [1] A. Inselberg and T. Avidan, "Classification and visualization for high-dimensional 
        data", Proc. 6th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD ‘00), 
        pp. 370-374, 2000.

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mc
from matplotlib.colors import ListedColormap
from viz.plotting.utils import pop 
from viz.utils import transform as tr
from viz.utils import dm

__all__ = ["plot"]

xmargins = {10: [0.7, 0.3], 9:[0.6, 0.3], 8:[0.5, 0.3], 7:[0.45, 0.3], 6:[0.3, 0.2]}
ymargins = {10: [-0.1, 0.08], 9:[-0.09, 0.075], 8:[-0.09, 0.075], 7:[-0.08, 0.075], 6:[-0.08, 0.075], 5:[-0.8, 0.08], 4:[-0.08, 0.08], 3:[-0.08, 0.08]}

def is_xticklabels_off(ax):
    r"""Checks if axes has already xtick labels

    Checks if an `matplotlib.axes.Axes` object has already
    xtick labels. 

    Parameters
    ----------
    ax : matplotlib axes object
        An `matplotlib.axes.Axes` object.
    
    Returns
    -------
    True/False : bool
    """

    xtl = ax.get_xticklabels()
    for s in xtl:
        if str(s) != "Text(0, 0, \'\')":
            return False
    return True

def get_yaxis_bounds(A):
    r"""
    """
    ub = dm.nadir(A)
    lb = dm.ideal(A)
    ubs = ["{:1.1e}".format(v) for v in ub]
    lbs = ["{:1.1e}".format(v) for v in lb]
    return [lbs, ubs]

def plot(A, ax=None, show_bounds=True, c=mc.TABLEAU_COLORS['tab:blue'], lw=1.0, labels=None, \
        xtick_labels=None, draw_vertical_lines=True, draw_grid=False, **kwargs):
    r"""A customized and more enhanced Parallel-coordinate plot.

    This Parallel-coordinate plot (PCP) [1]_ is customized for the experiments. 
    A lot of settings are customizable and configurable. Also it gives more 
    flexibility to the user compared to similar functions implemented in other 
    libraries like Pandas and seaborn. 
    
    Parameters
    ----------
    A : ndarray 
        `n` number of `m` dim. points to be plotted.
    ax : An `mpl_toolkits.mplot3d.axes.Axes3D` object, optional
        Default `None` when optional.
    show_bounds : bool, optional
        If `True` then the plot will show the lower and upper bounds of each data
        point (i.e. lines). Default `False` when optional.
    c : A `matplotlib.colors` object, str or an array RGBA color values.
        Colors to be used. Default `mc.TABLEAU_COLORS['tab:blue']` when 
        optional.
    lw : float, optional
        The line-width of each line in PCP. Default 1.0 when optional.
    labels : str, array_like or list of str, optional
        A string or an array/list of strings for labeling each line. Which basically
        means the class label of each row. Default `None` when optional. This will be
        used to set the legend in the figure. If `None` there will be no legend.
    xtick_labels : str, array_like or list of str, optional
        A string or an array/list of strings for xtick labels, for each column.
        Default `None` when optional. In that case, the labels will be `f_0`, `f_1` etc.
    draw_vertical_lines : bool, optional
        Decide whether we are going to put vertical y-axis lines in the plot for each
        column/feature. Default `True` when optional.
    draw_grid : bool, optional
        Decide whether we are going to put x-axis grid-lines in the plot. Default
        `False` when optional.

    Other Parameters
    ----------------
    title : str, optional
        The title of the figure. Default `None` when optional.
    column_indices : array_like or list of int, optional
        The indices of the columns of `A` to be plotted. Default `None` when optional.
    colorbar : (Cbc, Cbg, Cbl) a tuple of two ndarray and a str, optional
        If a user wants to put a colorbar, a tuple `(Cbc, Cbg, Cbl)` can be provided. 
        `Cbc` is an array of RGBA color values or an `matplotlib.colors` object. The 
        gradient of the colorbar is specified in `Cbg` which is an 1-D array of float. 
        Cbl is the label of the colorbar, a string. Default `None` when optional.
    axvline_width : float, optional
        The width of the vertical lines. Default 1.0 when optional.
    axvline_color : A `matplotlib.colors` object, str or an array RGBA color values.
        The color of the vertical lines. Default `black` when optional.
    **kwargs : dict
        All other keyword args for matplotlib `plot()` function.

    Returns
    -------
    ax : `mpl_toolkits.mplot3d.axes.Axes3D` object
        An `mpl_toolkits.mplot3d.axes.Axes3D` object.

    References
    ----------
    .. [1] A. Inselberg and T. Avidan, "Classification and visualization for high-dimensional 
        data", Proc. 6th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD ‘00), 
        pp. 370-374, 2000.
    """
    
    # collect extra kwargs
    title = kwargs['title'] if kwargs and 'title' in kwargs else None    
    column_indices = kwargs['column_indices'] if kwargs and 'column_indices' in kwargs else None   
    colorbar = kwargs['colorbar'] if kwargs and 'colorbar' in kwargs else None
    axvline_width = kwargs['axvline_width'] if kwargs and 'axvline_width' in kwargs else 1.0
    axvline_color = kwargs['axvline_color'] if kwargs and 'axvline_color' in kwargs else 'black'
    
    # remove once they are read
    kwargs = pop(kwargs, 'title')
    kwargs = pop(kwargs, 'column_indices')
    kwargs = pop(kwargs, 'colorbar')
    kwargs = pop(kwargs, 'axvline_width')
    kwargs = pop(kwargs, 'axvline_color')
    
    if not ax:
        ax = plt.figure().gca()        

    lbs, ubs = get_yaxis_bounds(A)
    F = tr.normalize(A, lb=np.zeros(A.shape[1]), ub=np.ones(A.shape[1]))

    # build color list for each data point
    if (not isinstance(c, list)) and (not isinstance(c, np.ndarray)):
        c_ = c
        c = np.array([c_ for _ in range(F.shape[0])])
    elif (isinstance(c, list) and len(c) != F.shape[0]) \
        or (isinstance(c, np.ndarray) and c.shape[0] != F.shape[0]):
            raise ValueError("The length of c needs to be same as F.shape[0].")
        
    # build linewidth list for each data point
    if (not isinstance(lw, list)) and (not isinstance(lw, np.ndarray)):
        lw_ = lw
        lw = np.array([lw_ for _ in range(F.shape[0])])
    elif (isinstance(lw, list) and len(lw) != F.shape[0]) \
        or (isinstance(lw, np.ndarray) and lw.shape[0] != F.shape[0]):
            raise ValueError("The length of lw needs to be same as F.shape[0].")

    # get a list of column indices
    if column_indices:
        x = np.array(column_indices)
    else:
        x = np.arange(0,F.shape[1],1).astype(int)
    if len(x) < 2:
        raise ValueError("column_indices must be of length > 1.")
            
    # get a list of xtick_labels
    if xtick_labels is None:
        xtick_labels = ["$f_{:d}$".format(i) for i in range(F.shape[1])]
        
    # get a list of line labels, i.e. class labels
    if labels is not None and isinstance(labels, str):
        label = labels
        labels = np.array([label for _ in range(F.shape[0])])
        
    # draw the actual plot
    used_legends = set()
    for i in range(F.shape[0]):
        y = F[i,x]
        if labels is not None:
            label = labels[i]
            if label not in used_legends:
                used_legends.add(label)
                ax.plot(x, y, color=c[i], label=label, linewidth=lw[i], **kwargs)
            else:
                ax.plot(x, y, color=c[i], linewidth=lw[i], **kwargs)
        else:
            ax.plot(x, y, color=c[i], linewidth=lw[i], **kwargs)
    
    # decide on vertical axes
    if draw_vertical_lines:
        for i in x:
            ax.axvline(i, linewidth=axvline_width, color=axvline_color)

    # draw grid?
    if draw_grid:
        ax.grid()
    else:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
 
    # decide on xtick_labels
    if xtick_labels is not None:
        if is_xticklabels_off(ax):
            ax.set_xticks(x)
            ax.set_xticklabels(xtick_labels)
            # Now completely change the axis ticks and labels
            # if there are bounds to be shown
            if show_bounds:
                ax.set_yticks([])
                ax.set_yticklabels([])
                plt.setp(ax.get_xticklabels(), fontsize=11, 
                        rotation=-45, ha="left", rotation_mode="anchor")
                ax.set_ylim([-0.1, 1.1])
                bottom, top = -0.1 + ymargins[F.shape[1]][0], 1.1 + ymargins[F.shape[1]][1]
                for i in range(A.shape[1]):
                    ax.text(i + ((0.68/10) * A.shape[1]), bottom, lbs[i], fontsize=11, \
                            ha='center', va='center', rotation=-45)
                    ax.text(i + ((0.3/10) * A.shape[1]), top, ubs[i], fontsize=11, \
                            ha='center', va='center', rotation=45)
            else:
                ax.set_xlim(x[0], x[-1])
        else:
            if len(ax.get_xticklabels()) < len(x):
                ax.set_xticks(x)
                ax.set_xticklabels(xtick_labels)
            if not show_bounds:
                xl, xr = ax.get_xlim()
                xl = x[0] if x[0] <= xl else xl
                xr = x[-1] if x[-1] >= xr else xr
                ax.set_xlim(xl, xr)

    if not show_bounds or is_xticklabels_off(ax):
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12) 
    
    # where to put the legend
    if labels is not None:
        ax.legend(loc="upper right") 
       
    # colorbar?
    if colorbar and isinstance(colorbar, tuple) and len(colorbar) >= 2 \
            and isinstance(colorbar[0], np.ndarray) and isinstance(colorbar[1], np.ndarray):
        vmin,vmax = 0.0, 1.0
        cbc, cbg = colorbar[0], colorbar[1]
        cbl = colorbar[2] if len(colorbar) > 2 and colorbar[2] else None
        Id = np.column_stack((cbg,cbc)).astype(object)
        Id = Id[np.argsort(Id[:, 0])] 
        c, g = Id[:,1:].astype(float), Id[:,0].astype(float)
        vmin, vmax = np.min(g), np.max(g)
        norm = mc.Normalize(vmin=vmin, vmax=vmax)
        cmap = ListedColormap(c)
        if cbl:
            ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), \
                    orientation='vertical', label=cbl, pad=0.01, shrink=0.99)
        else:
            ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), \
                        orientation='vertical', pad=0.01, shrink=0.99)

    # title?
    ax.set_title(title)

    return ax

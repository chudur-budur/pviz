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
import matplotlib.cm as cm
import matplotlib.colors as mc
from matplotlib.colors import ListedColormap
from vis.plotting.utils import pop 
from vis.utils import transform as tr

__all__ = ["plot"]

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

def plot(A, ax=None, normalized=False, c=mc.TABLEAU_COLORS['tab:blue'], lw=1.0, \
        xtick_labels=None, line_labels=None, draw_vertical_lines=True, \
        draw_grid=False, draw_colorbar=False, **kwargs):
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
    normalized : bool, optional
        Decide whether the points will be normalized before plotting.
        Default `False` when optional.
    c : A `matplotlib.colors` object, str or an array RGBA color values.
        Colors to be used. Default `mc.TABLEAU_COLORS['tab:blue']` when 
        optional.
    lw : float, optional
        The line-width of each line in PCP. Default 1.0 when optional.
    xtick_labels : str, array_like or list of str, optional
        A string or an array/list of strings for xtick labels, for each column.
        Default `None` when optional. In that case, the labels will be `f_0`, `f_1` etc.
    line_labels : str, array_like or list of str, optional
        A string or an array/list of strings for labeling each line. Which basically
        means the class label of each row. Default `None` when optional. This will be
        used to set the legend in the figure. If `None` there will be no legend.
    draw_vertical_lines : bool, optional
        Decide whether we are going to put vertical y-axis lines in the plot for each
        column/feature. Default `True` when optional.
    draw_grid : bool, optional
        Decide whether we are going to put x-axis grid-lines in the plot. Default
        `False` when optional.
    draw_colorbar : bool, optional
        Decide whether we are showing any colorbar. The plot supports only vertical
        colorbars at the outside of the right side of the y-axis. Default `False`
        when optional.

    Other Parameters
    ----------------
    title : str, optional
        The title of the figure. Default `None` when optional.
    column_indices : array_like or list of int, optional
        The indices of the columns of `A` to be plotted. Default `None` when optional.
    cbar_grad : array_like of float, optional
        The gradient of the colorbar. A 1-D array of floats.
        Default `None` when optional.
    cbar_label : str, optional
        The label of the colorbar. Default `None` when optional.
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
    title = kwargs['title'] if (kwargs is not None and 'title' in kwargs) else None    
    column_indices = kwargs['column_indices'] \
            if (kwargs is not None and 'column_indices' in kwargs) else None    
    cbar_grad = kwargs['cbar_grad'] if (kwargs is not None and 'cbar_grad' in kwargs) else None
    cbar_label = kwargs['cbar_label'] if (kwargs is not None and 'cbar_label' in kwargs) else None
    axvline_width = kwargs['axvline_width'] if (kwargs is not None and 'axvline_width' in kwargs) else 1.0
    axvline_color = kwargs['axvline_color'] if (kwargs is not None and 'axvline_color' in kwargs) else 'black'
    
    # remove once they are read
    kwargs = pop(kwargs, 'title')
    kwargs = pop(kwargs, 'column_indices')
    kwargs = pop(kwargs, 'cbar_grad')
    kwargs = pop(kwargs, 'cbar_label')
    kwargs = pop(kwargs, 'axvline_width')
    kwargs = pop(kwargs, 'axvline_color')
    
    if ax is None:
        ax = plt.figure().gca()        

    if normalized:
        F = tr.normalize(A, lb=np.zeros(A.shape[1]), ub=np.ones(A.shape[1]))
    else:
        F = np.array(A, copy=True)

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
    if column_indices is not None:
        x = np.array(column_indices)
    else:
        x = np.arange(0,F.shape[1],1).astype(int)
    if len(x) < 2:
        raise ValueError("column_indices must be of length > 1.")
            
    # get a list of xtick_labels
    if xtick_labels is None:
        xtick_labels = ["$f_{:d}$".format(i) for i in range(F.shape[1])]
        
    # get a list of line labels, i.e. class labels
    if line_labels is not None and isinstance(line_labels, str):
        ll = line_labels
        line_labels = np.array([ll for _ in range(F.shape[0])])
        
    # draw the actual plot
    used_legends = set()
    for i in range(F.shape[0]):
        y = F[i,x]
        if line_labels is not None:
            label = line_labels[i]
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

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=12)
    # decide on xtick_labels
    if xtick_labels is not None:
        if is_xticklabels_off(ax):
            ax.set_xticks(x)
            ax.set_xticklabels(xtick_labels[x])
            ax.set_xlim(x[0], x[-1])
        else:
            if len(ax.get_xticklabels()) < len(x):
                ax.set_xticks(x)
                ax.set_xticklabels(xtick_labels[x])
            xl, xr = ax.get_xlim()
            xl = x[0] if x[0] <= xl else xl
            xr = x[-1] if x[-1] >= xr else xr
            ax.set_xlim(xl, xr)        
    
    # where to put the legend
    if line_labels is not None:
        ax.legend(loc="upper right")
        
    # draw grid?
    if draw_grid:
        ax.grid()
        
    # title?
    if title is not None:
        ax.set_title(title)

    # colorbar?
    if draw_colorbar:
        vmin, vmax = 0.0, 1.0
        if cbar_grad is not None:
            Id = np.column_stack((cbar_grad,c)).astype(object)
            Id = Id[np.argsort(Id[:, 0])] 
            c, cbar_grad = Id[:,1:].astype(float), Id[:,0].astype(float)
            vmin, vmax = np.min(cbar_grad), np.max(cbar_grad)
        norm = mc.Normalize(vmin=vmin, vmax=vmax)
        cmap = ListedColormap(c)
        if cbar_label is not None:
            ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), \
                        orientation='vertical', label=cbar_label, pad=0.015)
        else:
            ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), \
                        orientation='vertical', pad=0.015) 
    return ax

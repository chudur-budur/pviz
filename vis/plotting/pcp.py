"""pcp.py -- A customized and more flexible Parallel-coordinate plotting module. 

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
import matplotlib.cm as cm
import matplotlib.colors as mc
from matplotlib.colors import ListedColormap
from vis.plotting.utils import pop 
from vis.utils import transform as tr

__all__ = ["plot"]

def is_xticklabels_off(ax):
    xtl = ax.get_xticklabels()
    for s in xtl:
        if str(s) != "Text(0, 0, \'\')":
            return False
    return True

def plot(F, ax=None, normalized=False, c=mc.TABLEAU_COLORS['tab:blue'], lw=1, \
        column_indices=None, xtick_labels=None, line_labels=None, draw_vertical_lines=True, \
        draw_grid=False, sort_columns=False, title=None, show_colorbar=False, **kwargs):
    
    # collect extra kwargs
    axvline_width = kwargs['axvline_width'] if (kwargs is not None and 'axvline_width' in kwargs) else 1
    axvline_color = kwargs['axvline_color'] if (kwargs is not None and 'axvline_color' in kwargs) else 'black'
    cbar_grad = kwargs['cbar_grad'] if (kwargs is not None and 'cbar_grad' in kwargs) else None
    cbar_label = kwargs['cbar_label'] if (kwargs is not None and 'cbar_label' in kwargs) else None
    
    # remove once they are read
    kwargs = pop(kwargs, 'axvline_width')
    kwargs = pop(kwargs, 'axvline_color')
    kwargs = pop(kwargs, 'cbar_grad')
    kwargs = pop(kwargs, 'cbar_label')
    
    if ax is None:
        ax = plt.figure().gca()        

    if normalized:
        F = tr.normalize(F, lb=np.zeros(F.shape[1]), ub=np.ones(F.shape[1]))

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
    if isinstance(line_labels, str):
        ll = line_labels
        line_labels = np.array([ll for _ in range(F.shape[0])])
        
    # draw the actual plot
    used_legends = set()
    for i in range(F.shape[0]):
        # y = df.iloc[i].values
        y = F[i,x]
        # kls = class_col.iat[i]
        # label = pprint_thing(kls)
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
    if show_colorbar:
        norm = None
        if cbar_grad is not None:
            vmin = np.min(cbar_grad)
            vmax = np.max(cbar_grad)
            norm = mc.Normalize(vmin=vmin, vmax=vmax)
        cmap = None if c is None else ListedColormap(c)
        ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), \
                       orientation='vertical', label=cbar_label, pad=0.015)
    return ax

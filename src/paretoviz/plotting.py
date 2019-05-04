import sys
import os
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

from utils import fmt
from utils import vectorops as vops
import decorator as dcor

"""
This script contains different plotting functions including PaletteViz
"""

def scatter(points, s = 1.0, c = 'black', alpha = [1.0, 1.0], \
                axes = [0, 1, 2], lims = None, camera = [None, None], \
                knee_idx = None, label = 'f{:d}', title = ""):
    """
    This function plots a set of points. If dimension is more than 3 then
    this function will use the axes specified in the axes paramter. By default,
    they are now x, y, and z axes.
    """
    fig, ax = None, None
    dim = len(points[0])
    if knee_idx is not None:
        knee_points = [points[i] for i in knee_idx]
        color = [c[i] for i in knee_idx]
        size = [s[i] for i in knee_idx]
        other_idx = list(set([i for i in range(len(points))]) - set(knee_idx))
        other_points = [points[i] for i in other_idx]
        color_ = [c[i] for i in other_idx]
        size_ = [s[i] for i in other_idx]
        p = list(zip(*knee_points))
        p_ = list(zip(*other_points))
    else:
        p, color, size = list(zip(*points)), c, s
        p_, color_, size_ = None, None, None

    if dim < 3:
        fig = plt.figure()
        ax = plt.gca()
        fig.suptitle(title)
        ax.set_xlabel(label.format(axes[0] + 1))
        ax.set_ylabel(label.format(axes[1] + 1))
        if lims is not None and len(lims) > 1:
            ax.set_xlim(lims[0][0], lims[0][1])
            ax.set_ylim(lims[1][0], lims[1][1])
        # plot others first
        if p_ is not None and color_ is not None and size_ is not None:
            ax.scatter(p_[axes[0]], p_[axes[1]], \
                    color = color_, marker = 'o', s = size_, alpha = alpha[0])
        # then plot knee
        ax.scatter(p[axes[0]], p[axes[1]], \
                   color = color, marker = 'o', s = size, alpha = alpha[1])
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(title)
        ax.set_xlabel(label.format(axes[0] + 1))
        ax.set_ylabel(label.format(axes[1] + 1))
        ax.set_zlabel(label.format(axes[2] + 1))
        if lims is not None and len(lims) > 2:
            ax.set_xlim(lims[0][0], lims[0][1])
            ax.set_ylim(lims[1][0], lims[1][1])
            ax.set_zlim(lims[2][0], lims[2][1])
        ax.view_init(elev = camera[0], azim = camera[1])
        # plot others first
        if p_ is not None and color_ is not None and size_ is not None:
            ax.scatter(p_[axes[0]], p_[axes[1]], p_[axes[2]], \
                       color = color_, marker = 'o', s = size_, alpha = alpha[0])
        # then plot knee
        ax.scatter(p[axes[0]], p[axes[1]], p[axes[2]], \
                   color = color, marker = 'o', s = size, alpha = alpha[1])
    return (fig, ax)

def make_scaffold_rv(m, layers, ax, label = "f{:d}"):
    """
    If the Palette visualization needs to show the scaffolding for the RadVis, this
    function will do all the necessary stuffs to show the scaffold.
    """
    # calculate the coordinates of all polygon corners.
    S = [[math.cos(t), math.sin(t)] for t in [2.0 * math.pi * (i/float(m)) for i in range(m)]]
    for z in layers:
        # draw polygons
        for i in range(0, len(S)-1):
            # draw one polygon line
            ax.plot([S[i][0], S[i + 1][0]], [S[i][1], S[i + 1][1]], zs = [z, z], \
                    c = 'gray', alpha = 0.15 * len(S), linewidth = 1.0)
            # draw a pair of polygon points
            ax.scatter(S[i][0], S[i][1], zs = z, color = 'gray', marker = 'o', \
                       s = 20.0, alpha = 1.0)
        # last polygon line
        ax.plot([S[len(S) - 1][0], S[0][0]], [S[len(S) - 1][1], S[0][1]], zs = [z, z], \
                c = 'gray', alpha = 0.15 * len(S), linewidth = 1.0)
        # last pair of polygon points
        ax.scatter(S[len(S) - 1][0], S[len(S) - 1][1], zs = z, \
                   c = 'gray', marker = 'o', s = 20.0, alpha = 1.0)
        # now put all the corner labels, like f1, f2, f3, ... etc.
        for xy, name in zip(S, [label.format(i+1) for i in range(m)]):
            if xy[0] < 0.0 and xy[1] < 0.0:
                ax.text(xy[0] - 0.025, xy[1] - 0.025, z = z, s = name, ha = 'right', \
                        va = 'top', size = 'small')
            elif xy[0] < 0.0 and xy[1] >= 0.0: 
                ax.text(xy[0] - 0.025, xy[1] + 0.025, z = z, s = name, ha = 'right', \
                        va = 'bottom', size = 'small')
            elif xy[0] >= 0.0 and xy[1] < 0.0:
                ax.text(xy[0] + 0.025, xy[1] - 0.025, z = z, s = name, ha = 'left', \
                        va = 'top', size = 'small')
            elif xy[0] >= 0.0 and xy[1] >= 0.0:
                ax.text(xy[0] + 0.025, xy[1] + 0.025, z = z, s = name, ha = 'left', \
                        va = 'bottom', size = 'small')

                
def make_scaffold_sc(m, layers, ax, label = "f{:d}"):
    """
    If the Palette visualization needs to show the scaffolding for the RadVis, this
    function will do all the necessary stuffs to show the scaffold.
    """
    # calculate the coordinates of all polygon corners.
    S = [[math.cos(t), math.sin(t)] for t in [2.0 * math.pi * (i/float(m)) for i in range(m)]]
    for z in layers:
        # draw polygons
        for i in range(0, len(S)-1):
            # draw one polygon line
            ax.plot([S[i][0], S[i + 1][0]], [S[i][1], S[i + 1][1]], zs = [z, z], \
                    c = 'gray', alpha = 0.15 * len(S), linewidth = 0.75, linestyle = 'dashdot')
            # draw a pair of polygon points
            ax.scatter(S[i][0], S[i][1], zs = z, color = 'gray', marker = 'o', \
                       s = 20.0, alpha = 1.0)
        # last polygon line
        ax.plot([S[len(S) - 1][0], S[0][0]], [S[len(S) - 1][1], S[0][1]], zs = [z, z], \
                c = 'gray', alpha = 0.15 * len(S), linewidth = 0.75, linestyle = 'dashdot')
        # last pair of polygon points
        ax.scatter(S[len(S) - 1][0], S[len(S) - 1][1], zs = z, \
                   c = 'gray', marker = 'o', s = 20.0, alpha = 1.0)
        # draw a circle on each layer
        p = Circle((0, 0), 1, fill = False, linewidth = 0.8, color = 'gray')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z = z)
        # now put all the corner labels, like f1, f2, f3, ... etc.
        for xy, name in zip(S, [label.format(i+1) for i in range(m)]):
            if xy[0] < 0.0 and xy[1] < 0.0:
                ax.text(xy[0] - 0.025, xy[1] - 0.025, z = z, s = name, ha = 'right', \
                        va = 'top', size = 'small')
            elif xy[0] < 0.0 and xy[1] >= 0.0: 
                ax.text(xy[0] - 0.025, xy[1] + 0.025, z = z, s = name, ha = 'right', \
                        va = 'bottom', size = 'small')
            elif xy[0] >= 0.0 and xy[1] < 0.0:
                ax.text(xy[0] + 0.025, xy[1] - 0.025, z = z, s = name, ha = 'left', \
                        va = 'top', size = 'small')
            elif xy[0] >= 0.0 and xy[1] >= 0.0:
                ax.text(xy[0] + 0.025, xy[1] + 0.025, z = z, s = name, ha = 'left', \
                        va = 'bottom', size = 'small')
                
def paletteviz(coords, m = 3, s = 1.0, c = None, alpha = [1.0, 1.0], \
               camera = [None, None], knee_idx = None, \
               scaffold = True, label = "f{:d}", title = "", \
               mode = "rv"):
    """
    The Palette visualization method. This function assumes
    the palette coordinate values are already computed.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev = camera[0], azim = camera[1])
    fig.suptitle(title)
    if knee_idx is not None:
        knee_coords = [coords[i] for i in knee_idx]
        knee_color = [c[i] for i in knee_idx]
        knee_size = [s[i] for i in knee_idx]
        other_idx = list(set([i for i in range(len(coords))]) - set(knee_idx))
        other_coords = [coords[i] for i in other_idx]
        other_color = [c[i] for i in other_idx]
        other_size = [s[i] for i in other_idx]
        # plot others first
        [u, v, w_] = list(zip(*other_coords))
        ax.scatter(u, v, w_, marker = 'o', \
                s = other_size, color = other_color, alpha = alpha[0])
        # then knee points
        [u, v, w] = list(zip(*knee_coords))
        ax.scatter(u, v, w, marker = 'o', \
                s = knee_size, color = knee_color, alpha = alpha[1])
        layers = list(set(w).union(set(w_)))
    else:
        [u, v, w] = list(zip(*coords))
        layers = list(set(w))
        ax.scatter(u, v, w, marker = 'o', s = size, color = color)        
        
    if scaffold:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_axis_off()
        if mode == "rv":
            make_scaffold_rv(m, layers, ax, label)
        if mode == "sc":
            make_scaffold_sc(m, layers, ax, label)
    return (fig, ax)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plotting.py [raw data file]")
        sys.exit(1)
    
    rawfpath = sys.argv[1]
    # rawfpath = "./data/yeast/yeast-8d.out"
    
    # do visualization based on distance from the centroid?
    docentroid = False

    # get the path and filename from the rawfpath
    path, rawfile = os.path.split(rawfpath)
    # get the prefix
    prefix = rawfile.split('.')[0]

    # load the normalized points
    points = fmt.load(os.path.join(path, prefix + "-norm.out"))
    # load the normalized trade-off values
    mu = [v[0] if len(v) == 1 else v for v in \
            fmt.load(os.path.join(path, prefix + "-norm-mu.out"))]

    # load the CV values
    cvfpath = os.path.join(path, prefix + "-cv.out")
    if not docentroid and os.path.exists(cvfpath):
        cv = [v[0] if len(v) == 1 else v for v in fmt.load(cvfpath)]
        [low, up] = vops.get_bound(cv)
        cv = vops.normalize(cv, low, up)
        color = dcor.recolor_by_cv(cv)
    else:
        color = dcor.recolor_by_centroid(points)

    # resize the points w.r.t. trade-offs
    size = dcor.rescale_by_tradeoff(mu)
    (color, knee_idx) = dcor.recolor_by_tradeoff(size, color)

    # dtypes for class labels
    dtypes = {"yeast-8d": "str"}
    
    # load the class labels
    classfpath = os.path.join(path, prefix + "-class.out")
    if os.path.exists(classfpath):
        labels = [v[0] if len(v) == 1 else v for v in \
                fmt.load(classfpath, dtype = dtypes[prefix])]
        color = dcor.recolor_by_labels(labels, dtype = dtypes[prefix])
        size = [5.0 for _ in range(len(points))]

    # alpha values
    alpha = [0.2, 0.8] # alpha for plots with knee
    # alpha = [1.0, 1.0] # alpha for general case
    
    # use the original obj values for scatter plot.
    rawpoints = fmt.load(rawfpath)
    # do the scatter plot
    (fig, ax) = scatter(rawpoints, s = size, c = color, alpha = alpha, \
                    camera = dcor.cam_scatter[prefix], knee_idx = knee_idx, \
                    title = "Scatter plot (frist 3 dim.)")
    # save the scatter plot
    scatterfpath = os.path.join(path, prefix + "-scatter.pdf")
    plt.savefig(scatterfpath, transparent = False)

    palette_coords = fmt.load(os.path.join(path, prefix + "-norm-palette.out"))
    # do the paletteviz plot
    (fig, ax) = paletteviz(palette_coords, m = len(points[0]), \
                s = size, c = color, alpha = alpha, \
                camera = dcor.cam_palette[prefix], knee_idx = knee_idx, \
                title = "PaletteViz (with RadViz)")
    # save the paletteviz plot
    fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    palettefpath = os.path.join(path, prefix + "-norm-palette.pdf")
    plt.savefig(palettefpath, transparent = False, bbox_inches = 'tight', pad_inches = 0)

    palette_coords = fmt.load(os.path.join(path, prefix + "-norm-palette-star.out"))
    # do the paletteviz plot with star-coordinate
    (fig, ax) = paletteviz(palette_coords, m = len(points[0]), \
                s = size, c = color, alpha = alpha, \
                camera = dcor.cam_palette[prefix], knee_idx = knee_idx, \
                title = "PaletteViz (with Star Coordinate)")
    # save the paletteviz plot
    fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    palettefpath = os.path.join(path, prefix + "-norm-palette-star.pdf")
    plt.savefig(palettefpath, transparent = False, bbox_inches = 'tight', pad_inches = 0)
    
    # show all plots
    plt.show()                                                  

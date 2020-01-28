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
        ax.set_xlabel(label.format(axes[0] + 1), fontsize = 'large')
        ax.set_ylabel(label.format(axes[1] + 1), fontsize = 'large')
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
        ax.set_xlabel(label.format(axes[0] + 1), fontsize = 'large')
        ax.set_ylabel(label.format(axes[1] + 1), fontsize = 'large')
        ax.set_zlabel(label.format(axes[2] + 1), fontsize = 'large')
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

def radviz(points, s = 1.0, c = None, alpha = [1.0, 1.0], \
            knee_idx = None, label = "f{:d}", title = "", \
            show_axes = False, fontname = None, fontsize = None, fontstyle = None):
    """
    This function does a generic radviz plot.
    """
    dim = len(points[0])
    S = [[math.cos(t), math.sin(t)] for t in \
            [2.0 * math.pi * (i/float(dim)) for i in range(dim)]]
    rvpts = []
    factor = 2.0 if dim > 3 else 1.0
    for f in points:
        fsum = math.fsum([(v ** factor) for v in f])
        u,v = 0.0, 0.0
        if fsum > 0.0:
            u = math.fsum([(f[i] ** factor) * t[0] for i,t in enumerate(S)]) / fsum
            v = math.fsum([(f[i] ** factor) * t[1] for i,t in enumerate(S)]) / fsum
        rvpts.append([u,v])

    [lb, ub] = vops.get_bound(rvpts)

    fig, ax = None, None
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim(lb[0] - 0.1 if lb[0] < -1 else -1.1, ub[0] + 0.01 if ub[0] > 1 else 1.1)
    ax.set_ylim(lb[1] - 0.1 if lb[1] < -1 else -1.1, ub[1] + 0.01 if ub[1] > 1 else 1.1)
    ax.set_aspect('equal')
    if not show_axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_axis_off()
    fig.suptitle(title)
    if knee_idx is not None:
        knee_points = [rvpts[i] for i in knee_idx]
        color = [c[i] for i in knee_idx]
        size = [s[i] for i in knee_idx]
        other_idx = list(set([i for i in range(len(rvpts))]) - set(knee_idx))
        other_points = [rvpts[i] for i in other_idx]
        color_ = [c[i] for i in other_idx]
        size_ = [s[i] for i in other_idx]
        [u, v] = list(zip(*knee_points))
        [u_, v_] = list(zip(*other_points))
        # plot others first
        ax.scatter(u_, v_, marker = 'o', s = size_, \
                color = color_, alpha = alpha[0])
        # then knee points
        ax.scatter(u, v, marker = 'o', s = size, \
                color = color, alpha = alpha[1])
    else:
        [u, v] = list(zip(*rvpts))
        ax.scatter(u, v, marker = 'o', s = s, color = c, alpha = alpha[0])
    
    for i in range(0, len(S)-1):
        # draw one polygon line
        ax.plot([S[i][0], S[i + 1][0]], [S[i][1], S[i + 1][1]], \
                c = 'gray', alpha = 0.15 * len(S), linewidth = 0.75, linestyle = 'dashdot')
        # draw a pair of polygon points
        ax.scatter(S[i][0], S[i][1], color = 'gray', marker = 'o', \
                s = 20.0, alpha = 1.0)
    # last polygon line
    ax.plot([S[len(S) - 1][0], S[0][0]], [S[len(S) - 1][1], S[0][1]], \
            c = 'gray', alpha = 0.15 * len(S), linewidth = 0.75, linestyle = 'dashdot')
    # last pair of polygon points
    ax.scatter(S[len(S) - 1][0], S[len(S) - 1][1], \
            c = 'gray', marker = 'o', s = 20.0, alpha = 1.0)
    # now put all the corner labels, like f1, f2, f3, ... etc.
    for xy, name in zip(S, [label.format(i+1) for i in range(dim)]):
        if xy[0] < 0.0 and xy[1] < 0.0:
            ax.text(xy[0] - 0.025, xy[1] - 0.025, s = name, ha = 'right', \
                    va = 'top', size = 'large', \
                    fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
        elif xy[0] < 0.0 and xy[1] >= 0.0: 
            ax.text(xy[0] - 0.025, xy[1] + 0.025, s = name, ha = 'right', \
                    va = 'bottom', size = 'large', \
                    fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
        elif xy[0] >= 0.0 and xy[1] < 0.0:
            ax.text(xy[0] + 0.025, xy[1] - 0.025, s = name, ha = 'left', \
                    va = 'top', size = 'large', \
                    fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
        elif xy[0] >= 0.0 and xy[1] >= 0.0:
            ax.text(xy[0] + 0.025, xy[1] + 0.025, s = name, ha = 'left', \
                    va = 'bottom', size = 'large', \
                    fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
    p = Circle((0, 0), 1, fill = False, linewidth = 0.8, color = 'gray')
    ax.add_patch(p)
    return (fig, ax)

def star(points, s = 1.0, c = None, alpha = [1.0, 1.0], \
            knee_idx = None, label = "f{:d}", title = "", \
            show_axes = False, fontname = None, fontsize = None, fontstyle = None):
    """
    This function does a generic radviz plot.
    """
    dim = len(points[0])
    C = [[math.cos(t), math.sin(t)] for t in \
            [2.0 * math.pi * (i/float(dim)) for i in range(dim)]]
    b = vops.get_bound(points)
    U = [[x / (b[1][i] - b[0][i]), y / (b[1][i] - b[0][i])] for i, [x, y] in enumerate(C)]

    rvpts = []
    for f in points:
        u = math.fsum([f[i] * u[0] for i,u in enumerate(U)])
        v = math.fsum([f[i] * u[1] for i,u in enumerate(U)])
        rvpts.append([u,v])

    [lb, ub] = vops.get_bound(rvpts)
    
    fig, ax = None, None
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim(lb[0] - 0.1 if lb[0] < -1 else -1.1, ub[0] + 0.1 if ub[0] > 1 else 1.1)
    ax.set_ylim(lb[1] - 0.1 if lb[1] < -1 else -1.1, ub[1] + 0.1 if ub[1] > 1 else 1.1)
    ax.set_aspect('equal')
    if not show_axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_axis_off()
    fig.suptitle(title)
    if knee_idx is not None:
        knee_points = [rvpts[i] for i in knee_idx]
        color = [c[i] for i in knee_idx]
        size = [s[i] for i in knee_idx]
        other_idx = list(set([i for i in range(len(rvpts))]) - set(knee_idx))
        other_points = [rvpts[i] for i in other_idx]
        color_ = [c[i] for i in other_idx]
        size_ = [s[i] for i in other_idx]
        [u, v] = list(zip(*knee_points))
        [u_, v_] = list(zip(*other_points))
        # plot others first
        ax.scatter(u_, v_, marker = 'o', s = size_, \
                color = color_, alpha = alpha[0])
        # then knee points
        ax.scatter(u, v, marker = 'o', s = size, \
                color = color, alpha = alpha[1])
    else:
        [u, v] = list(zip(*rvpts))
        ax.scatter(u, v, marker = 'o', s = s, color = c, alpha = alpha[0])
    
    for i in range(0, len(C)-1):
        # draw one polygon line
        ax.plot([C[i][0], C[i + 1][0]], [C[i][1], C[i + 1][1]], \
                c = 'gray', alpha = 0.15 * len(C), linewidth = 0.75, linestyle = 'dashdot')
        # draw a pair of polygon points
        ax.scatter(C[i][0], C[i][1], color = 'gray', marker = 'o', \
                s = 20.0, alpha = 1.0)
    # last polygon line
    ax.plot([C[len(C) - 1][0], C[0][0]], [C[len(C) - 1][1], C[0][1]], \
            c = 'gray', alpha = 0.15 * len(C), linewidth = 0.75, linestyle = 'dashdot')
    # last pair of polygon points
    ax.scatter(C[len(C) - 1][0], C[len(C) - 1][1], \
            c = 'gray', marker = 'o', s = 20.0, alpha = 1.0)
    # now put all the corner labels, like f1, f2, f3, ... etc.
    for xy, name in zip(C, [label.format(i+1) for i in range(dim)]):
        if xy[0] < 0.0 and xy[1] < 0.0:
            ax.text(xy[0] - 0.025, xy[1] - 0.025, s = name, ha = 'right', \
                    va = 'top', size = 'large', \
                    fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
        elif xy[0] < 0.0 and xy[1] >= 0.0: 
            ax.text(xy[0] - 0.025, xy[1] + 0.025, s = name, ha = 'right', \
                    va = 'bottom', size = 'large', \
                    fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
        elif xy[0] >= 0.0 and xy[1] < 0.0:
            ax.text(xy[0] + 0.025, xy[1] - 0.025, s = name, ha = 'left', \
                    va = 'top', size = 'large', \
                    fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
        elif xy[0] >= 0.0 and xy[1] >= 0.0:
            ax.text(xy[0] + 0.025, xy[1] + 0.025, s = name, ha = 'left', \
                    va = 'bottom', size = 'large', \
                    fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
    p = Circle((0, 0), 1, fill = False, linewidth = 0.8, color = 'gray')
    ax.add_patch(p)
    return (fig, ax)

def make_scaffold_rv(dim, layers, ax, label = "f{:d}", label_layers = None, \
        fontname = None, fontsize = None, fontstyle = None):
    """
    If the Palette visualization needs to show the scaffolding for the RadVis, this
    function will do all the necessary stuffs to show the scaffold.
    """
    # calculate the coordinates of all polygon corners.
    S = [[math.cos(t), math.sin(t)] for t in [2.0 * math.pi * (i/float(dim)) for i in range(dim)]]
    if label_layers is not None and -1 in label_layers:
        label_layers[len(layers) - 1] = label_layers[-1]
        del label_layers[-1]
    for li, z in enumerate(layers):
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

        # if label_layers is None, then label all layers (default)
        if label_layers is None:
            label_layers = {}
            for i in range(len(layers)):
                    label_layers[i] = None
        if li in label_layers:
            # now put all the corner labels, like f1, f2, f3, ... etc.
            # if label_layers[li] is None, label all anchors
            if label_layers[li] is None:
                label_layers[li] = [i+1 for i in range(dim)]
            for j, (xy, name) in enumerate(zip(S, [label.format(i+1) for i in range(dim)])):
                if j+1 in label_layers[li]:
                    if xy[0] < 0.0 and xy[1] < 0.0:
                        ax.text(xy[0] - 0.025, xy[1] - 0.025, z = z, s = name, ha = 'right', \
                                va = 'bottom', size = 'large', \
                                fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
                    elif xy[0] < 0.0 and xy[1] >= 0.0: 
                        ax.text(xy[0] - 0.025, xy[1] + 0.025, z = z, s = name, ha = 'right', \
                                va = 'bottom', size = 'large', \
                                fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
                    elif xy[0] >= 0.0 and xy[1] < 0.0:
                        ax.text(xy[0] + 0.025, xy[1] - 0.025, z = z, s = name, ha = 'left', \
                                va = 'bottom', size = 'large', \
                                fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
                    elif xy[0] >= 0.0 and xy[1] >= 0.0:
                        ax.text(xy[0] + 0.025, xy[1] + 0.025, z = z, s = name, ha = 'left', \
                                va = 'bottom', size = 'large', \
                                fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)

def make_scaffold_sc(dim, layers, ax, label = "f{:d}", label_layers = None, \
        fontname = None, fontsize = None, fontstyle = None):
    """
    If the Palette visualization needs to show the scaffolding for the RadVis, this
    function will do all the necessary stuffs to show the scaffold.
    """
    # calculate the coordinates of all polygon corners.
    S = [[math.cos(t), math.sin(t)] for t in [2.0 * math.pi * (i/float(dim)) for i in range(dim)]]
    if label_layers is not None and -1 in label_layers:
        label_layers[len(layers) - 1] = label_layers[-1]
        del label_layers[-1]
    for li, z in enumerate(layers):
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

        # if label_layers is None, then label all layers (default)
        if label_layers is None:
            label_layers = {}
            for i in range(len(layers)):
                    label_layers[i] = None
        if li in label_layers:
            # now put all the corner labels, like f1, f2, f3, ... etc.
            # if label_layers[li] is None, label all anchors
            if label_layers[li] is None:
                label_layers[li] = [i+1 for i in range(dim)]
            for j, (xy, name) in enumerate(zip(S, [label.format(i+1) for i in range(dim)])):
                if j+1 in label_layers[li]:
                    if xy[0] < 0.0 and xy[1] < 0.0:
                        ax.text(xy[0] - 0.025, xy[1] - 0.025, z = z, s = name, ha = 'right', \
                                va = 'bottom', size = 'large', \
                                fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
                    elif xy[0] < 0.0 and xy[1] >= 0.0: 
                        ax.text(xy[0] - 0.025, xy[1] + 0.025, z = z, s = name, ha = 'right', \
                                va = 'bottom', size = 'large', \
                                fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
                    elif xy[0] >= 0.0 and xy[1] < 0.0:
                        ax.text(xy[0] + 0.025, xy[1] - 0.025, z = z, s = name, ha = 'left', \
                                va = 'bottom', size = 'large', \
                                fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
                    elif xy[0] >= 0.0 and xy[1] >= 0.0:
                        ax.text(xy[0] + 0.025, xy[1] + 0.025, z = z, s = name, ha = 'left', \
                                va = 'bottom', size = 'large', \
                                fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
                
def paletteviz(points, dim = 3, s = 1.0, c = 'black', alpha = [1.0, 1.0], \
               camera = [None, None], knee_idx = None, \
               scaffold = True, label = "f{:d}", title = "", \
               mode = "rv", show_axes = False, label_layers = None,
               fontname = None, fontsize = None, fontstyle = None):
    """
    The Palette visualization method. This function assumes
    the palette coordinate values are already computed.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev = camera[0], azim = camera[1])
    fig.suptitle(title)
    if knee_idx is not None:
        knee_points = [points[i] for i in knee_idx]
        knee_color = [c[i] for i in knee_idx]
        knee_size = [s[i] for i in knee_idx]
        other_idx = list(set([i for i in range(len(points))]) - set(knee_idx))
        other_points = [points[i] for i in other_idx]
        other_color = [c[i] for i in other_idx]
        other_size = [s[i] for i in other_idx]
        # plot others first
        [u, v, w_] = list(zip(*other_points))
        ax.scatter(u, v, w_, marker = 'o', \
                s = other_size, color = other_color, alpha = alpha[0])
        # then knee points
        [u, v, w] = list(zip(*knee_points))
        ax.scatter(u, v, w, marker = 'o', \
                s = knee_size, color = knee_color, alpha = alpha[1])
        layers = sorted(list(set(w).union(set(w_))), reverse = True)
    else:
        [u, v, w] = list(zip(*points))
        layers = sorted(list(set(w)), reverse = True)
        ax.scatter(u, v, w, marker = 'o', s = s, color = c, alpha = alpha[0])        
        
    if scaffold:
        if not show_axes:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_axis_off()
        if mode == "rv":
            make_scaffold_rv(dim, layers, ax, label, label_layers = label_layers, \
                    fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
        if mode == "sc":
            make_scaffold_sc(dim, layers, ax, label, label_layers = label_layers, \
                    fontname = fontname, fontsize = fontsize, fontstyle = fontstyle)
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

    # load the CV values
    cvfpath = os.path.join(path, prefix + "-cv.out")
    if not docentroid and os.path.exists(cvfpath):
        cv = [v[0] if len(v) == 1 else v for v in fmt.load(cvfpath)]
        [low, up] = vops.get_bound(cv)
        cv = vops.normalize(cv, low, up)
        color = dcor.recolor_by_cv(cv)
    else:
        color = dcor.recolor_by_centroid(points)

    knee_idx = None
    alpha = [1.0, 1.0]
    
    # now use the trade-off values to recolor
    mufpath = os.path.join(path, prefix + "-norm-mu.out")
    if os.path.exists(mufpath):
        # load the normalized trade-off values
        mu = [v[0] if len(v) == 1 else v for v in fmt.load(mufpath)]
            # resize the points w.r.t. trade-offs
        size = dcor.rescale_by_tradeoff(mu)
        (color, knee_idx) = dcor.recolor_by_tradeoff(size, color)
        # alpha values
        alpha = [0.2, 0.8] # alpha for plots with knee
    
    # load the class labels
    classfpath = os.path.join(path, prefix + "-class.out")
    if os.path.exists(classfpath):
        labels = [v[0] if len(v) == 1 else v for v in \
                fmt.load(classfpath, dtype = dcor.dtypes[prefix])]
        color = dcor.recolor_by_labels(labels, dtype = dcor.dtypes[prefix])
        size = [5.0 for _ in range(len(points))]
 
    # use the original obj values for scatter plot.
    if os.path.exists(rawfpath):
        rawpoints = fmt.load(rawfpath)
        # do the scatter plot
        print("Doing a scatter plot")
        (fig, ax) = scatter(rawpoints, camera = dcor.cam_scatter[prefix], \
                title = "Scatter plot (frist 3 dim.)")
        # save the scatter plot
        scatterfpath = os.path.join(path, prefix + "-scatter-mono.pdf")
        plt.savefig(scatterfpath, transparent = False)

        print("Doing a scatter plot with colors")
        (fig, ax) = scatter(rawpoints, s = size, c = color, alpha = alpha, \
                        camera = dcor.cam_scatter[prefix], knee_idx = knee_idx, \
                        title = "Scatter plot w/color (frist 3 dim.)")
        # save the scatter plot
        scatterfpath = os.path.join(path, prefix + "-scatter.pdf")
        plt.savefig(scatterfpath, transparent = False)

    # Now use the normalized data points for these plots
    normfpath = os.path.join(path, prefix + "-norm.out")
    if os.path.exists(normfpath):
        normpoints = fmt.load(normfpath)
        # do a radviz plot
        print("Doing a radviz plot")
        (fig, ax) = radviz(normpoints, s = size, c = color, alpha = alpha, \
                knee_idx = knee_idx, title = "Radviz plot")
        # save the radviz plot
        radvizfpath = os.path.join(path, prefix + "-radviz.pdf")
        plt.savefig(radvizfpath, transparent = False)
        
        # do a star-coordinate plot
        print("Doing a star-coordinate plot")
        (fig, ax) = star(normpoints, s = size, c = color, alpha = alpha, \
                knee_idx = knee_idx, title = "Star-Coordinate plot")
        # save the radviz plot
        radvizfpath = os.path.join(path, prefix + "-star.pdf")
        plt.savefig(radvizfpath, transparent = False)

    datapath = os.path.join(path, prefix + "-norm-palette-invsc.out")
    if os.path.exists(datapath):
        palette_coords = fmt.load(datapath)
        # do the paletteviz plot with inverted star-coordinate
        print("Doing a paletteviz with inverted star-coordinate plot")
        (fig, ax) = paletteviz(palette_coords, dim = len(points[0]), \
                    s = size, c = color, alpha = alpha, \
                    camera = dcor.cam_palette[prefix], knee_idx = knee_idx, \
                    title = "PaletteViz (with Inv. Star-Coordinate)", mode = "sc")
        # save the paletteviz plot
        fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
        palettefpath = os.path.join(path, prefix + "-norm-palette-invsc.pdf")
        plt.savefig(palettefpath, transparent = False, bbox_inches = 'tight', pad_inches = 0)
    
    datapath = os.path.join(path, prefix + "-norm-palette-invrv.out")
    if os.path.exists(datapath):
        palette_coords = fmt.load(datapath)
        # do the paletteviz plot with inverted radviz
        print("Doing a paletteviz with inverted radviz plot")
        (fig, ax) = paletteviz(palette_coords, dim = len(points[0]), \
                    s = size, c = color, alpha = alpha, \
                    camera = dcor.cam_palette[prefix], knee_idx = knee_idx, \
                    title = "PaletteViz (with Inv. RadViz)", mode = "rv")
        # save the paletteviz plot
        fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
        palettefpath = os.path.join(path, prefix + "-norm-palette-invrv.pdf")
        plt.savefig(palettefpath, transparent = False, bbox_inches = 'tight', pad_inches = 0)
    
    datapath = os.path.join(path, prefix + "-norm-palette-sc.out")
    if os.path.exists(datapath):
        palette_coords = fmt.load(datapath)
        # do the paletteviz plot with star-coordinate
        print("Doing a paletteviz with star-coordinate plot")
        (fig, ax) = paletteviz(palette_coords, dim = len(points[0]), \
                    s = size, c = color, alpha = alpha, \
                    camera = dcor.cam_palette[prefix], knee_idx = knee_idx, \
                    title = "PaletteViz (with Star-Coordinate)", mode = "sc")
        # save the paletteviz plot
        fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
        palettefpath = os.path.join(path, prefix + "-norm-palette-sc.pdf")
        plt.savefig(palettefpath, transparent = False, bbox_inches = 'tight', pad_inches = 0)
    
    datapath = os.path.join(path, prefix + "-norm-palette-rv.out")
    if os.path.exists(datapath):
        palette_coords = fmt.load(datapath)
        # do the paletteviz plot with radviz
        print("Doing a paletteviz with radviz plot")
        (fig, ax) = paletteviz(palette_coords, dim = len(points[0]), \
                    s = size, c = color, alpha = alpha, \
                    camera = dcor.cam_palette[prefix], knee_idx = knee_idx, \
                    title = "PaletteViz (with RadViz)", mode = "rv")
        # save the paletteviz plot
        fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
        palettefpath = os.path.join(path, prefix + "-norm-palette-rv.pdf")
        plt.savefig(palettefpath, transparent = False, bbox_inches = 'tight', pad_inches = 0)
    
    # show all plots
    plt.show()                                                  

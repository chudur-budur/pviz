import sys
import os
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import matplotlib.cm as cm

from utils import utils
from utils import vectorops as vops

"""
This script contains different plotting functions including PaletteViz
"""

def scatter(points, s = 1.0, c = 'blue', alpha = [1.0, 1.0], \
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
        p = list(zip(*points))
        p_, colors_, sizes_ = None, None, None

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

def make_scaffold(m, layers, ax, label = "f{:d}"):
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

def paletteviz(coords, m = 3, s = 1.0, c = None, alpha = [1.0, 1.0], \
                camera = [None, None], knee_idx = None, \
                scaffold = True, label = "f{:d}", title = ""):
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
        knee_color = [color[i] for i in knee_idx]
        knee_size = [size[i] for i in knee_idx]
        other_idx = list(set([i for i in range(len(coords))]) - set(knee_idx))
        other_coords = [coords[i] for i in other_idx]
        other_color = [color[i] for i in other_idx]
        other_size = [size[i] for i in other_idx]
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
        make_scaffold(m, layers, ax, label)
    return (fig, ax)

# Different point attribute modifiers

def get_centroid(points):
    """
    Get the center of the mass of the data points.
    """
    cols = list(zip(*points))
    centroid = []
    for col in cols:
        centroid.append(sum(col)/len(col))
    return centroid

def recolor_by_centroid(points, factor = 1.5):
    """
    Color the points according to the distance from the centroid.
    """
    c = get_centroid(points)
    cdist = []
    maxcdist, mincdist = float('-inf'), float('inf')
    for point in points:
        d = math.sqrt(sum([(p - c[i])*(p - c[i]) \
                           for i, p in enumerate(point)]))
        maxcdist = d if d >= maxcdist else maxcdist
        mincdist = d if d <= mincdist else mincdist
        cdist.append(d)
    color = [cm.winter_r(((v - mincdist)/(maxcdist - mincdist)) * factor) for v in cdist]
    return color
 
def rescale_by_knee(mu):
    """
    Change the sizes of the points according to the trade-off values.
    """
    nonzero_mu = [v for v in mu if v > 0.0]
    min_mu = min(nonzero_mu)
    max_mu = max(nonzero_mu)
    mu_ = vops.normalize(mu, min_mu, max_mu)
    mean_mu = math.fsum(mu_)/len(mu_)
    sd_mu = math.sqrt(math.fsum([(m - mean_mu)**2 for m in mu_])/(len(mu_) - 1))
    # sizes = [15.0 for _ in range(len(mu))]
    sizes = [((m + 0.01) * 100.0) for m in mu_]
    return sizes

def recolor_by_knee(size, color):
    """
    Recolor the points according to the trade-off values. This function
    will change the color of the points with higher trade-offs to 
    dark red.
    """
    max_sz = max(size)
    min_sz = min(size)
    mean_sz = math.fsum(size)/len(size)
    sd_sz = math.sqrt(math.fsum([((s - mean_sz) ** 2) for s in size])/(len(size) - 1))
    knee_idx = [i for i,s in enumerate(size) if s > (mean_sz + 3.0 * sd_sz)]
    knee_sizes = [size[i] for i in knee_idx]
    min_knee = min(knee_sizes)
    max_knee = max(knee_sizes)
    # print("max_sz:", max_sz, "min_sz:", min_sz, "mean_sz:", mean_sz, "sd_sz:", sd_sz)
    # print("knee_idx:", knee_idx)
    # print("min_knee:", min_knee, "max_knee:", max_knee)
    for i in range(len(color)):
        if i in knee_idx:
            knee_range = max_knee - min_knee
            knee_range = 0.0 # we are not doing this anymore.
            if knee_range > 0.0:
                color[i] = cm.Reds(((size[i] - min_knee) / knee_range) * 2.5)
            else:
                color[i] = cm.Reds(2.5)
    return (color, knee_idx)

def recolor_by_cv(cv):
    """
    This function will change the color of the points
    according to the total constraint violation value
    of each point.
    """
    color = [cm.cool(v * 1.0) for v in cv]
    return color

def recolor_by_layers(points, layers, seq = ['red', 'green']):
    """
    This function will change the color of the points
    according to the sequence of the layers. The color
    sequence is given in the seq variable.
    """
    colors = [''] * len(points)
    for i,layer in enumerate(layers):
        color = seq[i % len(seq)]
        for idx in layer:
            colors[idx] = color
    return colors

def recolor_by_labels(labels, dtype = 'float'):
    """
    If your data file has a class labels, then it will color
    the points according to class labels.
    """
    colors = [''] * len(labels)
    if dtype == 'float':
        lb = min(labels)
        ub = max(labels)
        vals = vops.normalize(labels, lb, ub)
        for i in range(len(vals)):
            colors[i] = cm.Reds(vals[i] * 2.5)
    elif dtype == 'str' or dtype == 'int':
        seq = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'brown', 'pink', 'grey']
        unique_labels = list(set(labels))
        for i,name in enumerate(labels):
            colors[i] = seq[unique_labels.index(name) % len(seq)]      
    return colors


# some camera angles for better visualization
cam_scatter = {\
   "spherical-3d": [32, 20], "spherical-4d": [32, 20], "spherical-8d": [32, 20], \
   "knee-3d": [19, -46], "knee-4d": [19, -46], "knee-8d": [19, -106], \
   "knee-const-3d": [25, 9], "knee-const-4d": [17, -24], "knee-const-8d": [11, -31], \
   "knee-const-mod-3d": [25, 9], "knee-const-mod-4d": [17, -24], "knee-const-mod-8d": [11, -31], \
   "isolated-3d": [32, 20], "isolated-4d": [8, -64], "isolated-8d": [14, -112], \
   "line-3d": [None, None], "line-4d": [None, None], "line-6d": [None, None], "line-8d": [None, None], \
   "c2dtlz2-3d": [32, 20], "c2dtlz2-4d": [37, -39], "c2dtlz2-5d": [37, -39], "c2dtlz2-8d": [25, -39], \
   "c2dtlz2-c1-3d": [32, 20], "c2dtlz2-c2-3d": [32, 20], "c2dtlz2-c3-3d": [32, 20], \
   "c2dtlz2-c4-3d": [32, 20], \
   "carcrash-3d": [46, 41], "carcrash-c1-3d": [30, 30,], "carcrash-c2-3d": [24, 64], \
   "gaa-das-10d": [22, -40], "gaa-lhs-10d": [22, -40], \
   "gaa-lhs-c1-10d": [7, -51], "gaa-lhs-c2-10d": [7, -51], "gaa-lhs-c3-10d": [7, -51], \
   "iris-4d": [None, None], "cccp-4d": [None, None], "airofoil-5d": [None, None], \
   "wil-7d": [None, None], "yeast-8d": [None, None], "concrete-8d": [None, None], \
   "banknote-4d": [None, None], "mammogram-5d": [None, None], "blood-4d": [None, None]}

# some camera angles for better visualization
cam_palette = {\
    "spherical-3d": [20, -98], "spherical-4d": [21, -67], "spherical-8d": [18, 158], \
    "knee-3d": [26, -52], "knee-4d": [18, -21], "knee-8d": [16, -168], \
    "knee-const-3d": [44, -50], "knee-const-4d": [23, 10], "knee-const-8d": [23, -178], \
    "knee-const-mod-3d": [44, -50], "knee-const-mod-4d": [23, 10], "knee-const-mod-8d": [23, -178], \
    "isolated-3d": [29, -76], "isolated-4d": [24, -161], "isolated-8d": [26, -109], \
    "line-3d": [None, None], "line-4d": [37, -61], "line-6d": [25, -7], "line-8d": [30, 13],\
    "c2dtlz2-3d": [36, -45], "c2dtlz2-4d": [23, -67], "c2dtlz2-5d": [24, -29], "c2dtlz2-8d": [26, -69], \
    "c2dtlz2-c1-3d": [36, -45], "c2dtlz2-c2-3d": [36, -45], "c2dtlz2-c3-3d": [36, -45], \
    "c2dtlz2-c4-3d": [36, -45], \
    "carcrash-3d": [24, -58], "carcrash-c1-3d": [25, 41], "carcrash-c2-3d": [19, -38], \
    "gaa-das-10d": [27, -105], "gaa-lhs-10d": [27, -105], \
    "gaa-lhs-c1-10d": [27, -105], "gaa-lhs-c2-10d": [27, -105], "gaa-lhs-c3-10d": [27, -105], \
    "iris-4d": [None, None], "cccp-4d": [None, None], "airofoil-5d": [None, None], \
    "wil-7d": [None, None], "yeast-8d": [None, None], "concrete-8d": [None, None], \
    "banknote-4d": [None, None], "mammogram-5d": [None, None], "blood-4d": [None, None]}

if __name__ == "__main__":
    rawfpath = "./data/line/line-4d.out"
    # rawfpath = "./data/yeast/yeast-8d.out"
    
    # do visualization based on distance from the centroid?
    docentroid = False

    # get the path and filename from the rawfpath
    path, rawfile = os.path.split(rawfpath)
    # get the prefix
    prefix = rawfile.split('.')[0]

    # load the normalized points
    points = utils.load(os.path.join(path, prefix + "-norm.out"))
    # load the normalized trade-off values
    mu = [v[0] if len(v) == 1 else v for v in \
            utils.load(os.path.join(path, prefix + "-norm-mu.out"))]

    # load the CV values
    cvfpath = os.path.join(path, prefix + "-cv.out")
    if not docentroid and os.path.exists(cvfpath):
        cv = [v[0] if len(v) == 1 else v for v in utils.load(cvfpath)]
        [low, up] = vops.get_bound(cv)
        cv = vops.normalize(cv, low, up)
        color = recolor_by_cv(cv)
    else:
        color = recolor_by_centroid(points)

    # resize the points w.r.t. trade-offs
    size = rescale_by_knee(mu)
    (color, knee_idx) = recolor_by_knee(size, color)

    # dtypes for class labels
    dtypes = {"yeast-8d": "str"}
    
    # load the class labels
    classfpath = os.path.join(path, prefix + "-class.out")
    if os.path.exists(classfpath):
        labels = [v[0] if len(v) == 1 else v for v in \
                utils.load(classfpath, dtype = dtypes[prefix])]
        color = recolor_by_labels(labels, dtype = dtypes[prefix])
        size = [5.0 for _ in range(len(points))]

    # alpha values
    alpha = [0.2, 0.8] # alpha for plots with knee
    # alpha = [1.0, 1.0] # alpha for general case
    
    # use the original obj values for scatter plot.
    rawpoints = utils.load(rawfpath)
    # do the scatter plot
    (fig, ax) = scatter(rawpoints, s = size, c = color, alpha = alpha, \
                    camera = cam_scatter[prefix], knee_idx = knee_idx, \
                    title = "Scatter plot (frist 3 dim.)")
    # save the scatter plot
    scatterfpath = os.path.join(path, prefix + "-scatter.pdf")
    plt.savefig(scatterfpath, transparent = False)

    palette_coords = utils.load(os.path.join(path, prefix + "-norm-palette.out"))
    # do the paletteviz plot
    (fig, ax) = paletteviz(palette_coords, m = len(points[0]), \
                s = size, c = color, alpha = alpha, \
                camera = cam_palette[prefix], knee_idx = knee_idx, \
                title = "PaletteViz (with RadViz)")
    # save the paletteviz plot
    fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    palettefpath = os.path.join(path, prefix + "-norm-palette.pdf")
    plt.savefig(palettefpath, transparent = False, bbox_inches = 'tight', pad_inches = 0)

    palette_coords = utils.load(os.path.join(path, prefix + "-norm-palette-star.out"))
    # do the paletteviz plot with star-coordinate
    (fig, ax) = paletteviz(palette_coords, m = len(points[0]), \
                s = size, c = color, alpha = alpha, \
                camera = cam_palette[prefix], knee_idx = knee_idx, \
                title = "PaletteViz (with Star Coordinate)")
    # save the paletteviz plot
    fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    palettefpath = os.path.join(path, prefix + "-norm-palette-star.pdf")
    plt.savefig(palettefpath, transparent = False, bbox_inches = 'tight', pad_inches = 0)
    
    # show all plots
    plt.show()                                                         
                                                                       
                                                                       
                                                                       
                                                                       
                                                                       
                                                                       
                                                                       
                                                                       
                                                                       
                                                                       
                                                                       

import math
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from paretoviz.utils import vectorops as vops

"""
This file contains different functions for color and size modifiers
of the points in the plots.
"""

# we might need to draw arrows.
class Arrow3D(FancyArrowPatch):
    """
    The 3d arrow class.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def recolor_by_centroid(points, factor = 1.5):
    """
    Color the points according to the distance from the centroid.
    """
    c = vops.mean(points)
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
 
def rescale_by_tradeoff(mu):
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

def recolor_by_tradeoff(size, color):
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
   "knee-3d": [15, -41], "knee-4d": [19, -46], "knee-8d": [19, -106], \
   "knee-const-3d": [25, 9], "knee-const-4d": [17, -24], "knee-const-8d": [11, -31], \
   "knee-const-mod-3d": [25, 9], "knee-const-mod-4d": [17, -24], "knee-const-mod-8d": [11, -31], \
   "isolated-3d": [32, 20], "isolated-4d": [8, -64], "isolated-8d": [14, -112], \
   "line-3d": [None, None], "line-4d": [None, None], "line-6d": [None, None], "line-8d": [None, None], \
   "c2dtlz2-3d": [32, 20], "c2dtlz2-4d": [37, -39], "c2dtlz2-5d": [37, -39], "c2dtlz2-8d": [25, -39], \
   "c2dtlz2-c1-3d": [32, 20], "c2dtlz2-c2-3d": [32, 20], "c2dtlz2-c3-3d": [32, 20], \
   "c2dtlz2-c4-3d": [32, 20], "osy-3d": [42, -32], "osy-4d": [None, None], \
   "carcrash-3d": [46, 41], "carcrash-c1-3d": [30, 30,], "carcrash-c2-3d": [24, 64], \
   "gaa-das-10d": [22, -40], "gaa-lhs-10d": [22, -40], \
   "gaa-lhs-c1-10d": [7, -51], "gaa-lhs-c2-10d": [7, -51], "gaa-lhs-c3-10d": [7, -51], \
   "iris-4d": [None, None], "cccp-4d": [None, None], "airofoil-5d": [None, None], \
   "wil-7d": [None, None], "yeast-8d": [None, None], "concrete-8d": [None, None], \
   "banknote-4d": [None, None], "mammogram-5d": [None, None], "blood-4d": [None, None]}

# some camera angles for better visualization
cam_palette = {\
    "spherical-3d": [20, -98], "spherical-4d": [21, -67], "spherical-8d": [18, 158], \
    "knee-3d": [14, -77], "knee-4d": [20, 155], "knee-8d": [25, 98], \
    "knee-const-3d": [42, -68], "knee-const-4d": [16, 119], "knee-const-8d": [24, 43], \
    "knee-const-mod-3d": [44, -50], "knee-const-mod-4d": [23, 10], "knee-const-mod-8d": [23, -178], \
    "isolated-3d": [29, -76], "isolated-4d": [19, 178], "isolated-8d": [25, -60], \
    "line-3d": [31, 95], "line-4d": [37, -61], "line-6d": [25, -7], "line-8d": [30, 13],\
    "c2dtlz2-3d": [33, 0], "c2dtlz2-4d": [19, -60], "c2dtlz2-5d": [24, -29], "c2dtlz2-8d": [26, -69], \
    "c2dtlz2-c1-3d": [36, -45], "c2dtlz2-c2-3d": [36, -45], "c2dtlz2-c3-3d": [36, -45], \
    "c2dtlz2-c4-3d": [36, -45], "osy-3d": [29, -78], "osy-4d": [21, 65], \
    "carcrash-3d": [24, -58], "carcrash-c1-3d": [25, 41], "carcrash-c2-3d": [19, -38], \
    "gaa-das-10d": [27, -105], "gaa-lhs-10d": [27, -105], \
    "gaa-lhs-c1-10d": [27, -105], "gaa-lhs-c2-10d": [27, -105], "gaa-lhs-c3-10d": [27, -105], \
    "iris-4d": [None, None], "cccp-4d": [None, None], "airofoil-5d": [None, None], \
    "wil-7d": [None, None], "yeast-8d": [None, None], "concrete-8d": [None, None], \
    "banknote-4d": [None, None], "mammogram-5d": [None, None], "blood-4d": [None, None]}

# class label data types for different UCI data
dtypes = {"airofoil-5d": "float", "banknote-4d": "int", "blood-4d": "int", "cccp-4d": "float", \
          "concrete-8d": "float", "iris-4d": "str", "wil-7d": "int", "yeast-8d": "str"}

if __name__ == "__main__":
    print("There is nothing in the __main__ yet.")

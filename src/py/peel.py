import sys
import math
import numpy as np
import scipy as sp
from scipy.spatial.qhull import QhullError
import pyhull.convex_hull as cvhull
import utils

"""
This script applies alpha-shape or simple convex hull
to find the shape of the high-dimensional data points.
Then it saves the layer in a file.
"""

def collapse(points, dim = 0):
    """
    This function collapse one of the n-dimension of the
    data points. By default it collapses the first dimension.
    But in most cases we collapse the last dimension, i.e.
    z-axis in case of 3D points. The parameter dim denotes
    which dimension it's going to collapse.
    """
    points_ = []
    for p in points:
        p_ = [x for i,x in enumerate(p) if i != dim]
        points_.append(p_)
    return points_

def project(points):
    """
    This function projects data point on a simplex hyperplane.
    """
    dim = len(points[0])
    u = [1.0 / math.sqrt(dim) for _ in range(dim)]
    points_ = []
    for p in points:
        uTp = sum([x * u[i] for i, x in enumerate(p)])
        uTpu = [x * uTp for x in u]
        p_ = [(x - uTpu[i]) + u[i]/math.sqrt(dim) for i, x in enumerate(p)]
        points_.append(p_)
    return points_

def get_convex_hull(points, indices, module = "pyhull"):
    """
    For a given set of points and a list of their indices,
    this function returns the list of indices of those points
    that form the convex hull.
    """
    print("Computing alpha-shape ...")
    vmap = {}
    p = []
    for i,idx in enumerate(indices):
        p.append(points[idx])
        vmap[i] = idx
    
    # Okay this was from the pyhull library, but for some strange reason
    # after the update, it became extremely slow.
    hull_indices = set()
    if module == "pyhull":
        hull = cvhull.ConvexHull(p)
        for simplex in hull.vertices:
            for vertex in simplex:
                hull_indices.add(indices[vertex])
    # Therefore, now I am using the ConvexHull() function from scipy.spatial
    elif module == "scipy":
        p = np.array(p)
        try:
            hull = sp.spatial.ConvexHull(p)
        except QhullError: 
            pass
        else:
            for vertex in hull.vertices.tolist():
                hull_indices.add(vmap[vertex])
    
    print("alpha-shape done.")
    return hull_indices

def peel(points, module = "pyhull"):
    """
    This function calls the get_convex_hull() function in a 
    recursive manner to find each layer of the boundary points
    and returns the boundary indices as list of lists.
    """
    m = len(points[0])
    boundaries = []
    all_indices = range(len(points))
    print("Total points:", len(all_indices))
    l = 0
    while len(all_indices) > m + 1:
        hull_indices = get_convex_hull(points, all_indices, module)
        # If the points are not in "general position", 
        # there might be no hull.
        if len(hull_indices) > 0:
            print("Layer {:d}:".format(l), end = " ")
            boundaries.append(list(hull_indices))
            print("points added {:d}".format(len(hull_indices)), end = ", ")
            all_indices = [x for x in all_indices if x not in hull_indices]
            print("points left {:d}.".format(len(all_indices)))
        else:
            # No hull found. May be the rest of the points are not in
            # general position, now lump them together as one layer.
            break;
        l += 1
    if len(all_indices) > 0:
        print("Layer {:d}:".format(l), end = " ")
        boundaries.append(all_indices)
        print("points added {:d}.".format(len(all_indices)), "\n")
    return boundaries

if __name__ == "__main__":
    data_file = sys.argv[1].strip()
    layer_file = data_file.split('.')[0] + "-layers.out"
    points = utils.load(data_file)
    m = len(points[0])
    print("Peeling data point cloud ...")
    ppoints = project(points)
    cpoints = collapse(ppoints, dim = m - 1)
    boundaries = peel(cpoints, module = "pyhull")
    print("Saving layers into {0:s} ...".format(layer_file))
    utils.save(boundaries, layer_file, dtype = 'int')    

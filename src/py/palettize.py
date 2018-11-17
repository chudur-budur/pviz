import math
import sys
import copy
from itertools import product
import vectorutils as vu
import utils

"""
This script takes the data points and their corresponding
boundary indices and generate data for palette visualization.
"""

def save_palette(palette_coords, filename):
    """
    Save the data.
    """
    fp = open(filename, 'w')
    idx = sorted(palette_coords.keys())
    for i in idx:
        fp.write("\t".join(["{0:.4f}".format(v) for v in palette_coords[i]]) + "\n")
    fp.close()

def reverse_normalize(points):
    """
    This function reverses all the vectors to minimization
    from maximization, where the later is followed in the
    original RadViz.
    """
    points_ = []
    for point in points:
        points_.append([(1.0 - f) for f in point])
    return points_

def scale(points, factor):
    """
    This function scales up the original objective
    vector, i.e. f is replaced with f^factor.
    """
    points_ = []
    for point in points:
        points_.append([f**factor for f in point])
    return points_

def swap_columns(points, cols):
    points_ = copy.copy(points)
    for point in points_:
        swap = point[cols[0]]
        point[cols[0]] = point[cols[1]]
        point[cols[1]] = swap
    return points_

def palettize(points, layers, n_layers = 0, zgap = 1.0):
    """
    This function takes the data points and their corresponding 
    layer wise assignment indices. Then it transforms the coordinates 
    into layer wise radvis coordinates.

    If local is True, then the (u,v) values of radviz coordinates 
    are computed according to the layer grouping. If local is False,
    then computes the (u,v) values of the radviz plot globally and
    then partitions the points according to the number of layers.
    """
    points_ = copy.copy(points)
    # points_ = swap_columns(points, 4, 7) # copy.copy(points) # original radviz
    # points_ = reverse_normalize(points_) # minimization radviz
    m = len(points_[0])
    if m > 3:
        points_ = scale(points_, 2.0)
    else:
        points_ = scale(points_, 1.0) # original with scaling of 2.0
    S = [[math.cos(t), math.sin(t)] for t in \
            [2.0 * math.pi * (i/float(m)) for i in range(m)]]
    n_layers_orig = len(layers)
    points_per_layer = len(points_)/n_layers if n_layers > 0 else float('inf')
    palette_coords = {}
    wl, wc, count = 0.0, 0.0, 0
    for layer in layers:
        for idx in layer:
            u,v = 0.0, 0.0
            f = points_[idx]
            fsum = math.fsum(f)
            if fsum > 0.0:
                u = math.fsum([f[i] * t[0] for i,t in enumerate(S)]) / fsum
                v = math.fsum([f[i] * t[1] for i,t in enumerate(S)]) / fsum
            # If the original number of layers < number of layers specified,
            # then use wl else use wc.
            palette_coords[idx] = [u, v, wl] \
                    if n_layers == 0 or n_layers_orig <= n_layers \
                    else [u, v, wc]
            count = count + 1
            if count >= points_per_layer:
                count = 0
                wc = wc - zgap
        wl = wl - zgap
    return palette_coords

def cosine_map(points, A):
    points_ = []
    p = (1.0 / (1.0 + math.exp(A/2)))
    q = (1.0 / (1.0 + math.exp(-A/2)))
    for vec in points:
        # points_.append([(1 - math.cos(math.pi * v)) * 0.5 for v in vec])
        points_.append([(((1.0 / (1 + math.exp(-A * (v - 0.5)))) - p) / (q - p)) for v in vec])
    return points_

def palettize_reverse(points, layers, n_layers = 0, zgap = 1.0):
    """
    This function takes the data points and their corresponding 
    layer wise assignment indices. Then it transforms the coordinates 
    into layer wise radvis coordinates.

    If local is True, then the (u,v) values of radviz coordinates 
    are computed according to the layer grouping. If local is False,
    then computes the (u,v) values of the radviz plot globally and
    then partitions the points according to the number of layers.
    """
    points_ = copy.copy(points)
    points_ = cosine_map(points_, 20.0)
    # points_ = reverse_normalize(points_) # reverse radviz
    # points_ = scale(points_, 1.0) # scaling of 2.0
    m = len(points_[0])
    S = [[math.cos(t), math.sin(t)] for t in \
            [2.0 * math.pi * (i/float(m)) for i in range(m)]]
    n_layers_orig = len(layers)
    points_per_layer = len(points_)/n_layers if n_layers > 0 else float('inf')
    palette_coords = {}
    wl, wc, count = 0.0, 0.0, 0
    for layer in layers:
        for idx in layer:
            u,v = 0.0, 0.0
            f = points_[idx]
            fsum = math.fsum(f)
            if fsum > 0.0:
                u = math.fsum([f[i] * t[0] for i,t in enumerate(S)]) / fsum
                v = math.fsum([f[i] * t[1] for i,t in enumerate(S)]) / fsum
            # If the original number of layers < number of layers specified,
            # then use wl else use wc.
            palette_coords[idx] = [u, v, wl] \
                    if n_layers == 0 or n_layers_orig <= n_layers \
                    else [u, v, wc]
            count = count + 1
            if count >= points_per_layer:
                count = 0
                wc = wc - zgap
        wl = wl - zgap
    return palette_coords

def palettize_polar(points, layers, n_layers = 0, zgap = 1.0):
    """
    This function takes the data points and their corresponding 
    layer wise assignment indices. Then it transforms the coordinates 
    into layer wise radvis coordinates.

    If local is True, then the (u,v) values of radviz coordinates 
    are computed according to the layer grouping. If local is False,
    then computes the (u,v) values of the radviz plot globally and
    then partitions the points according to the number of layers.
    """
    m = len(points[0])
    points_ = copy.copy(points)
    points_ = reverse_normalize(points_) # reverse radviz
    factor = 3.0 if m > 3 else 2.0 
    points_ = scale(points_, factor) # scaling
    S = [[math.cos(t), math.sin(t)] for t in \
            [2.0 * math.pi * (i/float(m)) for i in range(m)]]
    n_layers_orig = len(layers)
    points_per_layer = len(points_)/n_layers if n_layers > 0 else float('inf')
    palette_coords = {}
    wl, wc, count = 0.0, 0.0, 0
    for layer in layers:
        for idx in layer:
            u,v = 0.0, 0.0
            f = points_[idx]
            if fsum > 0.0:
                u = math.fsum([f[i] * t[0] for i,t in enumerate(S)])
                v = math.fsum([f[i] * t[1] for i,t in enumerate(S)])
            # If the original number of layers < number of layers specified,
            # then use wl else use wc.
            palette_coords[idx] = [u, v, wl] \
                    if n_layers == 0 or n_layers_orig <= n_layers \
                    else [u, v, wc]
            count = count + 1
            if count >= points_per_layer:
                count = 0
                wc = wc - zgap
        wl = wl - zgap
    return palette_coords

def tester():
    points_ = list(product([0.0, 0.5, 1.0]), repeat = 3)
    layers_ = range(len(points_))
    palette_coords = palettize(points_, [layers_], zgap = 0.0)
    for idx in palette_coords:
        print("\t".join(["{0:.4f}".format(v) for v in palette_coords[idx]]))

if __name__ == "__main__":
    # tester()
    data_file = sys.argv[1].strip()
    n_layers = 0
    if len(sys.argv) == 3:
        n_layers = int(sys.argv[2].strip())
    
    layer_file = data_file.split('.')[0] + "-layers.out"
    palette_file = data_file.split('.')[0] + "-palette.out"
    
    points = utils.load(data_file)
    layers = utils.load(layer_file, dtype = 'int')
    palette_coords = palettize(points, layers, n_layers = n_layers)
    # palette_coords = palettize_reverse(points, layers, n_layers = n_layers)
    # palette_coords = palettize_polar(points, layers, n_layers = n_layers)

    print("Saving palette coordinates into {0:s} ...".format(palette_file))
    save_palette(palette_coords, palette_file)

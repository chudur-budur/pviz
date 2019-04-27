import os
import sys
import math
import copy
from itertools import product

from utils import vectorops as vops
from utils import fmt

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

def palettize(points, layers, n_layers = 0, zgap = 1.0):
    """
    This function takes the data points and their corresponding 
    layer wise assignment indices. Then it transforms the coordinates 
    into layer wise radvis coordinates. Also each data point is scaled
    using f_i = f_i^p where p = 2.0 if m > 3 else 1.0.
    """
    points_ = copy.copy(points)
    m = len(points_[0])
    factor = 2.0 if m > 3 else 1.0
    points_ = scale(points_, factor) 
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

    But this function works opposite of what the original radviz does.
    The normalization is done in the reverse way, i.e. f_i = (1 - f_i)^p.
    Also the radviz points are not normalized, i.e. u = sum(f_i cos(a_j)) 
    instead of u = sum(f_i cos(a_j))/sum(f_i) and same with 
    v = sum(f_i cos(a_j)). Also in this case, p = 3.0 if m > 3 else p = 2.0
    """
    m = len(points[0])
    points_ = copy.copy(points)
    points_ = reverse_normalize(points_) # reverse radviz, f_i = (1 - f_i)^p
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

def logistic_map(points, A):
    points_ = []
    p = (1.0 / (1.0 + math.exp(A/2)))
    q = (1.0 / (1.0 + math.exp(-A/2)))
    for vec in points:
        # points_.append([(1 - math.cos(math.pi * v)) * 0.5 for v in vec])
        points_.append([(((1.0 / (1 + math.exp(-A * (v - 0.5)))) - p) / (q - p)) for v in vec])
    return points_

def palettize_logistic(points, layers, n_layers = 0, zgap = 1.0):
    """
    This function takes the data points and their corresponding 
    layer wise assignment indices. Then it transforms the coordinates 
    into layer wise radvis coordinates.

    This function scales the points using a sigmoid function to increase
    the spread and reduce the overlap in the final radviz plot. The rest 
    of the procedure is identical to the original radviz.
    """
    points_ = copy.copy(points)
    points_ = logistic_map(points_, 15.0)
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

def palettize_star(points, layers, n_layers = 0, zgap = 1.0):
    """
    This function maps the data points using the star-coordinate
    (SC) plot, instead of Radviz.
    """
    points_ = copy.copy(points)
    m = len(points_[0])
    C = [[math.cos(t), math.sin(t)] for t in \
            [2.0 * math.pi * (i/float(m)) for i in range(m)]]
    b = vops.get_bound(points)
    U = [[x / (b[1][i] - b[0][i]), y / (b[1][i] - b[0][i])] for i, [x, y] in enumerate(C)]
    n_layers_orig = len(layers)
    points_per_layer = len(points_)/n_layers if n_layers > 0 else float('inf')
    palette_coords = {}
    wl, wc, count = 0.0, 0.0, 0
    for layer in layers:
        for idx in layer:
            p,q = 0.0, 0.0
            f = points_[idx]
            p = math.fsum([f[i] * u[0] for i,u in enumerate(U)])
            q = math.fsum([f[i] * u[1] for i,u in enumerate(U)])
            # If the original number of layers < number of layers specified,
            # then use wl else use wc.
            palette_coords[idx] = [p, q, wl] \
                    if n_layers == 0 or n_layers_orig <= n_layers \
                    else [p, q, wc]
            count = count + 1
            if count >= points_per_layer:
                count = 0
                wc = wc - zgap
        wl = wl - zgap
    return palette_coords

def palettize_stardecay(points, layers, n_layers = 0, zgap = 1.0):
    """
    This function maps the data points using the star-coordinate
    (SC) plot, instead of Radviz.
    """
    points_ = copy.copy(points)
    m = len(points_[0])
    C = [[math.cos(t), math.sin(t)] for t in \
            [2.0 * math.pi * (i/float(m)) for i in range(m)]]
    b = vops.get_bound(points)
    U = [[x / (b[1][i] - b[0][i]), y / (b[1][i] - b[0][i])] for i, [x, y] in enumerate(C)]
    n_layers_orig = len(layers)
    points_per_layer = len(points_)/n_layers if n_layers > 0 else float('inf')
    palette_coords = {}
    wl, wc, count = 0.0, 0.0, 0
    for layer in layers:
        for idx in layer:
            p,q = 0.0, 0.0
            # f = points_[idx]
            # reverse 1 - f
            f = [(1.0 - v) for v in points_[idx]]
            # sort the index by the decreasing order of values
            sid = sorted([[i, v] for i,v in enumerate(f)], key = lambda x: x[1], reverse = True)
            # for each value f_i, replace with f_i * e^(-4 * 0.1 * i), i.e. apply decay function e^(-4x)
            # m = 3
            # 0 --> 0
            # 1 --> 0.50
            # 2 --> 1.0
            # m = 4
            # 0 --> 0
            # 1 --> 0.33
            # 2 --> 0.66
            # 3 --> 0.99
            # m = 5
            # 0 --> 0
            # 1 --> 0.25
            # 2 --> 0.50
            # 3 --> 0.75
            # 4 --> 1.0
            delta = 1.0 / (len(sid)-1)
            for i,p in enumerate(sid):
                f[p[0]] = f[p[0]] * math.exp(-4 * i * delta) 
            f = [(1.0 - v) for v in points_[idx]]
            p = math.fsum([f[i] * u[0] for i,u in enumerate(U)])
            q = math.fsum([f[i] * u[1] for i,u in enumerate(U)])
            # If the original number of layers < number of layers specified,
            # then use wl else use wc.
            palette_coords[idx] = [p, q, wl] \
                    if n_layers == 0 or n_layers_orig <= n_layers \
                    else [p, q, wc]
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
    if len(sys.argv) < 2:
        print("Usage: python3 palettize.py [normalized data file] [no. of layers] [mode]")
        sys.exit(1)
    
    # tester()
    normfpath = sys.argv[1].strip()
    path, normfile = os.path.split(normfpath)
    n_layers = 0
    mode = "default"
    if len(sys.argv) >= 3:
        n_layers = int(sys.argv[2].strip())
        if len(sys.argv) == 4:
            mode = sys.argv[3].strip()
        elif len(sys.argv) == 5:
            mode = sys.argv[3].strip()
            n = float(sys.argv[4].strip())
    
    layerfile = normfile.split('.')[0] + "-layers.out"
    
    points = fmt.load(normfpath)
    layers = fmt.load(os.path.join(path, layerfile), dtype = 'int')
    if mode == "default":
        palette_coords = palettize(points, layers, n_layers = n_layers)
        palettefpath = os.path.join(path, normfile.split('.')[0] + "-palette.out")
    elif mode == "polar":
        palette_coords = palettize_polar(points, layers, n_layers = n_layers)
        palettefpath = os.path.join(path, normfile.split('.')[0] + "-palette-polar.out")
    elif mode == "logistic":
        palette_coords = palettize_logistic(points, layers, n_layers = n_layers)
        palettefpath = os.path.join(path, normfile.split('.')[0] + "-palette-logistic.out")
    elif mode == "star":
        palette_coords = palettize_star(points, layers, n_layers = n_layers)
        palettefpath = os.path.join(path, normfile.split('.')[0] + "-palette-star.out")
    elif mode == "stardecay":
        palette_coords = palettize_stardecay(points, layers, n_layers = n_layers)
        palettefpath = os.path.join(path, normfile.split('.')[0] + "-palette-stardecay.out")
    else:
        print("Error: unknown mode \'{0:s}\'\n".format(mode))
        sys.exit(1)

    print("Saving palette coordinates into {0:s} ...".format(palettefpath))
    save_palette(palette_coords, palettefpath)

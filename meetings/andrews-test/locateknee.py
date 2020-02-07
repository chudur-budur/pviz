"""
This is the latest code.
This code implements the trade-off metric proposed in this paper:
    Lily Rachmawati and Dipti Srinivasan, 
    "Multiobjective Evolutionary Algorithm with Controllable Focus on the Knees of the Pareto Front", 
    IEEE Transactions on Evolutionary Computation, Vol 13, No 4, August 2009. 
    (http://ieeexplore.ieee.org/document/5208606/)
"""

import sys
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.polynomial import Polynomial, Legendre
from sympy.geometry import *

# list of all valid colors
clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# euclidean distance from vector x to vector y
def euclid_dist(x,y):
    return math.sqrt(sum([(y[i] - x_)**2 for i,x_ in enumerate(x)]))

# normalize a vector v = [x1, x2, ... xn] w.r.t. lb, ub
def normalize(v, lb, ub):
    return [[(x - lb[i])/(ub[i] - lb[i]) for i,x in enumerate(e)] for e in v]

# find mean of a real vector
def mean(v):
    return sum(v)/len(v)

# find std. deviation of a real vector
def stddev(v):
    mu = mean(v)
    return sum([(x-mu)*(x-mu) for x in v])/len(v)

# read csv file
def load_csv(filename):
    lst = []
    try:
        fp = open(filename, 'r')
        for line in fp:
            lst.append([float(v) for v in line.strip().split(',')])
        fp.close()
        return lst
    except FileNotFoundError:
        print("file not found")
        sys.exit

# save csv file
def save_csv(data, filename):
    fp = open(filename, 'w')
    for d in data:
        for i,v in enumerate(d):
            if i < len(d)-1:
                fp.write("{0:.2f},".format(v))
            else:
                fp.write("{0:.2f}".format(v))
        fp.write("\n")
    fp.close()

# get the convex hull of individual minima (chim) from PF
def get_chim(data):
    chim = {}
    for j in range(len(data[0])):
        chim[j] = Point(data[0])
        for v in data[1:]:
            if v[j] >= chim[j][j]:
                chim[j] = Point(v)
    chim_ = []
    for k,v in chim.items():
        chim_.append(chim[k])
    return chim_

# get the bound of the PF
def get_bound(data):
    global ideal, nadir
    ideal = list(data[0])
    nadir = list(data[0])
    for v in data[1:]:
        for i,e in enumerate(v):
            if e <= ideal[i]:
                ideal[i] = e
            if e >= nadir[i]:
                nadir[i] = e
    return [ideal, nadir]

# get a subset of data points in epsilon neighbourhod cetered at x
def get_epsilon_neighbours(x, data, epsilon):
    neighbours = []
    for y in data:
        if euclid_dist(x,y) < epsilon and euclid_dist(x,y) > 0:
            neighbours.append(y)
    return neighbours

# calculate the trade-off weight mu(xi,xj) described in the paper
def get_tradeoff_weights(data_, epsilon):
    tradeoff_weights = []
    data = sorted(data_, key = lambda x:x[0])
    m = len(data[0])
    for fi in data:
        neighbours = get_epsilon_neighbours(fi, data, epsilon)
        num = 0
        denom = 0
        weights = []
        for fj in neighbours:
            num = 0
            denom = 0
            for m_ in range(0,m):
                num = num + max(0, fj[m_] - fi[m_])
                denom = denom + max(0, fi[m_] - fj[m_])         
            ratio = num/denom if denom > 0 else float('inf')
            weights.append(ratio)
        if len(weights) > 0:
            tradeoff_weights.append(min(weights))
            # print("tradeoff_weights: ", tradeoff_weights)
        else:
            raise Exception("the \'weights\' list is empty!! too small neighbourhood?") 
    return [data, tradeoff_weights]

# compute the tradeoff weights using the mu/sigma approach
def get_tradeoff_weights_musigma(data_, epsilon):
    tradeoff_weights = []
    data = sorted(data_, key = lambda x:x[0])
    m = len(data[0])
    origin = [0 for v in range(m)]
    for fi in data:
        neighbours = get_epsilon_neighbours(fi, data, epsilon)
        ndists = [euclid_dist(f,origin) for f in neighbours]
        tradeoff_weights.append(mean(ndists)/stddev(ndists))
    return [data, tradeoff_weights]

def get_tradeoff_weights_projection(data_, epsilon):
    tradeoff_weights = []
    data = sorted(data_, key = lambda x:x[0])
    m = len(data[0])
    chim = get_chim(data)
    segment = Segment(chim[0], chim[1])
    for fi in data:
        d = segment.distance(Point(fi)).evalf()
        p = segment.projection(Point(fi))
        # print(d, p)
        neighbours = get_epsilon_neighbours(fi, data, epsilon)
        num = 0
        denom = 0
        weights = []
        for fj in neighbours:
            num = 0
            denom = 0
            for m_ in range(0,m):
                num = num + max(0, fj[m_] - fi[m_])
                denom = denom + max(0, fi[m_] - fj[m_])         
            ratio = num/denom if denom > 0 else float('inf')
            weights.append(ratio)
        if len(weights) > 0:
            tradeoff_weights.append(min(weights))
            # print("tradeoff_weights: ", tradeoff_weights)
        else:
            raise Exception("the \'weights\' list is empty!! too small neighbourhood?") 
    return [data, tradeoff_weights]

# compute the tradeoff weights using the projection approach
def get_tradeoff_weights_projection_(data_, epsilon):
    tradeoff_weights = []
    dim = len(data_[0])
    data = sorted(data_, key = lambda x:x[0])
    chim = get_chim(data)
    segment = Segment(chim[0], chim[1])
    if dim == 2:
        for v in data:
            neighbours = get_epsilon_neighbours(v, data, espsilon)
            for u in neighbours:
                d = segment.distance(Point(v)).evalf()
                p = segment.projection(Point(v))
                # print(d, p)
    elif dim == 3:
        print("plane")
    return [data, tradeoff_weights]


# now plot the data points colored according to the mu() value
# knee points will have darker shade
def plot_front_with_single_weights(title, weights, data):
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    theta = np.linspace(min(weights), max(weights), num = 100)
    x = []
    y = []
    z = []
    for i in range(0, len(data)):
        x.append(data[i][1])
        y.append(data[i][2])
        if len(data[i]) > 2:
            z.append(data[i][3])
    norm = mpl.colors.Normalize(vmin = 0, vmax = 1.0)
    cmap = cm.Greys
    # cmap = cm.RdBu
    # cmap = cm.jet_r
    # cmap = cm.rainbow_r
    m = cm.ScalarMappable(norm = norm, cmap = cmap)
    maxclr = max(weights)
    rgbs = [m.to_rgba(v/maxclr) for v in weights]
    # print("x: ", x)
    # print("y: ", y)
    # print("z: ", z)
    if len(z) > 0:
        ax = Axes3D(fig)
        ax.scatter(x, y, z, s = 30, marker = 'o', linewidth = 1, edgecolor = 'grey', facecolor = rgbs)
    else:
        plt.scatter(x, y, s = 30, marker = 'o', linewidth = 1, edgecolor = 'grey', facecolor = rgbs)

# data normalization function, some files in the 
# data folder are not normalized.
def normalize_file():
        # data = load_csv("../../data/do2dk.csv")
        # data = load_csv("../../data/do2dk-k4.csv")
        # data = load_csv("../../data/dtlz1_pof_3_190.csv")
        # data = load_csv("../../data/dtlz1_pof_10_190.csv")
        # data = load_csv("../../data/dtlz2_inv_pof_3_190.csv")
        # data = load_csv("../../data/dtlz2_mod_pof_3_190.csv")
        # data = load_csv("../../data/dtlz2_mod_inv_pof_3_190.csv")
        # data = load_csv("../../data/dtlz7_pof_3_190.csv")
        # data = load_csv("../../data/dtlz7_pof_10_190.csv")
        data = load_csv("knee5d.csv")
        # data = load_csv("knee4d.csv")
        # data = load_csv("knee3d.csv")
        # data = load_csv("knee2d.csv")
        [ideal, nadir] = get_bound(data)
        print("ideal: ", ideal)
        print("nadir: ", nadir)
        norm_data = normalize(data, ideal, nadir)
        chim = get_chim(norm_data)
        print("chim: ", chim)
        # save_csv(norm_data, "knee2d-norm.csv")
        # save_csv(norm_data, "knee3d-norm.csv")
        # save_csv(norm_data, "knee4d-norm.csv")
        save_csv(norm_data, "knee5d-norm.csv")
        # save_csv(norm_data, "../../data/dtlz7_pof_10_190-norm.csv")
        # save_csv(norm_data, "../../data/dtlz7_pof_3_190-norm.csv")
        # save_csv(norm_data, "../../data/dtlz2_mod_inv_pof_3_190-norm.csv")
        # save_csv(norm_data, "../../data/dtlz2_mod_pof_3_190-norm.csv")
        # save_csv(norm_data, "../../data/dtlz2_inv_pof_3_190-norm.csv")
        # save_csv(norm_data, "../../data/dtlz1_pof_10_190-norm.csv")
        # save_csv(norm_data, "../../data/dtlz1_pof_3_190-norm.csv")
        # save_csv(norm_data, "../../data/do2dk-k4-norm.csv")
        # save_csv(norm_data, "../../data/do2dk-norm.csv")

        # some testing for knee points only
        # knees = load_csv("../../data/debMd_3_190-knees.csv")
        # norm_knees = normalize(knees, ideal, nadir)
        # save_csv(norm_knees, "../../data/debMd_3_190-norm-knees.csv")

def main():
    try:
        # load the data
        # data = load_csv("../../data/deb2dk-norm.csv")
        # data = load_csv("../../data/do2dk-norm.csv")
        # data = load_csv("../../data/do2dk-k4-norm.csv")
        # data = load_csv("../../data/debMd_3_190-norm.csv")
        
        # data = load_csv("../../data/zdt2-norm.csv")
        # data = load_csv("../../data/zdt3-norm.csv")
        # data = load_csv("../../data/zdt4-norm.csv")
        
        # data = load_csv("../../data/dtlz1_pof_3_190-norm.csv")
        # data = load_csv("../../data/dtlz2_inv_pof_3_190-norm.csv")
        # data = load_csv("../../data/dtlz2_mod_pof_3_190-norm.csv")
        # data = load_csv("../../data/dtlz2_mod_inv_pof_3_190-norm.csv")
        # data = load_csv("../../data/dtlz7_pof_3_190-norm.csv")
        
        # data = load_csv("../../data/dtlz1_pof_10_190-norm.csv")
        # data = load_csv("../../data/dtlz7_pof_10_190-norm.csv")
        
        # knees = load_csv("../../data/debMd_3_190-norm-knees.csv")
        
        data = load_csv("knee5d-norm.csv")
        # data = load_csv("knee4d-norm.csv")
        # data = load_csv("knee3d-norm.csv")
        data_o = load_csv("knee5d.csv")
        
        # calculate the weights w.r.t. to epsilon radius
        epsilon = 0.4 # deb2dk, do2dk, zdt3
        # epsilon = 0.4 # debmd3
        [data_, weights] = get_tradeoff_weights(data, epsilon)
        # [data_, weights] = get_tradeoff_weights_musigma(data, epsilon)
        # [data_, weights] = get_tradeoff_weights_projection(data, epsilon)

        # just for debugging
        vals = []
        for idx,dat in enumerate(data_):
            vals.append([dat, data_o[idx], weights[idx]])
        vals = sorted(vals, key = lambda x: x[-1])
        print("[f] mu:")
        for v in vals:
            print(v)
        plot_front_with_single_weights('knee-viz', weights, data_)

        # show plot
        plt.show()
    except Exception as exp:
        print("Error: ", exp)

# entry points

main()
# normalize_file()

"""
Extensions:
    1. change the problem to M obj, use r() to find the location of the knee, set K = 1
    2. verify if the metric in the paper can actually locate it.
    3. do PCP and see if the knee lines have any detectable features, and find a way to
        locate those lines without computing knees.
"""

"""
No idea what it is actually doing.
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

# list of all valid colors
clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# euclidean distance from vector x to vector y
def euclid_dist(x,y):
    return math.sqrt(sum([(y[i] - x_)**2 for i,x_ in enumerate(x)]))

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
                fp.write("{0:s},".format(str(v)))
            else:
                fp.write("{0:s}".format(str(v)))
        fp.write("\n")
    fp.close()

# get the bound of the PF
ideal = []
nadir = []
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

# generate sinusoid coeff.
def sinusoid(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    return (coeff[0] * (1/math.sqrt(2))) + \
            sum([c * math.sin((i/2 + 1) * t) \
                if i % 2 == 0 else c * math.cos(((i + 1)/2) * t) \
                    for i,c in enumerate(coeff[1:])])

# generate coeff of sinusoid derivative.
def sinusoid_deriv(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    return (coeff[0] * (1/math.sqrt(2))) + \
            sum([c * math.sin((i/2 + 1) * t) \
                if i % 2 == 0 else c * math.cos(((i + 1)/2) * t) \
                    for i,c in enumerate(coeff[1:])])

# generate coeff. of a square wave function.
def square(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    return (4/math.pi) * sum([c * math.sin((2 + (4 * i)) * t) \
                    for i,c in enumerate(coeff)])

# generate coeff. of a triangle wave function.
def triangle(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    const = lambda x: 1/((2 * x + 1)**2) if x % 2 == 0 else -1/((2 * x + 1)**2)
    return (8/(math.pi**2)) * sum([const(i) * c * math.sin((2 + (4 * i)) * t) \
                    for i,c in enumerate(coeff)])

# generate coeff. of derivative of a triangle wave function.
def triangle_deriv(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    const = lambda x: 1/((2 * x + 1)**2) if x % 2 == 0 else -1/((2 * x + 1)**2)
    return (8/(math.pi**2)) * sum([const(i) * c * math.cos((2 + (4 * i)) * t) \
                    for i,c in enumerate(coeff)])

# generate coeff. of a triangle1 wave function.
def triangle1(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    const = lambda x: 1/((2 * x + 1)**2) if x % 2 == 0 else -1/((2 * x + 1)**2)
    return (8/(math.pi**2)) * sum([const(i) * c * math.sin((2 + (4 * i)) * t) * math.sin((2 + (4 * i)) * t)\
                    for i,c in enumerate(coeff)])

# generate coeff. of a triangle2 wave function.
def triangle2(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    random.shuffle(coeff)
    csum = sum(coeff)
    const = lambda x: 1/((2 * x + 1)**2) if x % 2 == 0 else -1/((2 * x + 1)**2)
    return (8/(math.pi**2)) * sum([const(i) * (c/(csum-c)) * math.sin((2 + (4 * i)) * t) \
                    for i,c in enumerate(coeff)])
    
# generate coeff. of a sawtooth wave function.
def sawtooth(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    return 1/2 - ((1/math.pi) * sum([c * math.sin(2 * (i+1) * t)/(i+1) \
                    for i,c in enumerate(coeff)]))

# generate coeff. of legendre ploynomial for n-th term
def legendre_bases(n):
    b = [[0 for j in range(n+1)] for i in range(n+1)]
    b[0][0] = 1.0
    
    if n <= 0:
        return c

    b[1][1] = 1.0
    for i in range(2, n+1):
        for j in range(0, i - 1):
            b[i][j] = (-i + 1) * (b[i-2][j] / i)
        for j in range(1, i + 1):
            b[i][j] = b[i][j] + (i + i - 1) * (b[i-1][j-1] / i)
    return b[1:]
   
# generate all coeff. of legendre ploynomial
b = []
def legendre(coeff, t):
    global b
    n = len(coeff)
    if len(b) == 0:
        print("computing legendre bases ...")
        b = legendre_bases(n)
        print("total of {0:d} sets computed.".format(len(b)))
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    return sum([(coeff[i] * v) 
                    for i,v in enumerate([sum([(t**i) * b__ 
                        for i,b__ in enumerate(b_)]) 
                            for b_ in b])])

# plots one andrews curve line
def andrews_curve(coeff, gen_bases, bound, clr):
    theta = np.linspace(bound[0], bound[1], num = 100)
    r = [gen_bases(coeff, th) for th in theta]
    plt.plot(theta, r, color = clr)

# plots a whole andrews curve
def plot_andrews(lst, gen_bases, bound, clrfunc):
    for i,dataset in enumerate(lst):
        for v in dataset:
                andrews_curve(v, gen_bases, bound, clrfunc(v))

# color function, color by file index
def clr_by_file(index):
    return clrs[index]

def clr_by_dist(vector):
    maxdist = euclid_dist(ideal,nadir)
    dist = euclid_dist(ideal, vector)
    norm = mpl.colors.Normalize(vmin = 0, vmax = 0.25)
    cmap = cm.RdBu
    # cmap = cm.jet_r
    # cmap = cm.rainbow_r
    m = cm.ScalarMappable(norm = norm, cmap = cmap)
    return m.to_rgba(dist/maxdist)

# increase knee neighbourhood
def augment_knees(data, knees, delta):
    augk = []
    for k in knees:
        augk.append(k)
        for d in data:
            if euclid_dist(k,d) < delta:
                augk.append(d)
    return augk

def plot_knee_distance3d(data):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    maxdist = euclid_dist(ideal, nadir)
    for i in range(0, len(data)):
        coeff = data[i]
        dist = euclid_dist(ideal, coeff)
        # dist = euclid_dist(nadir, coeff)
        bound = [0, 2 * math.pi]
        theta = np.linspace(bound[0], bound[1], num = 100)
        # r = [sinusoid(coeff, th) for th in theta]
        # r = [legendre(coeff, th) for th in theta]
        # r = [triangle(coeff, th) for th in theta]
        # r = [triangle1(coeff, th) for th in theta]
        # r = [triangle2(coeff, th) for th in theta]
        # r = [square(coeff, th) for th in theta]
        r = [sawtooth(coeff, th) for th in theta]
        x = [r[i] * math.cos(th) for i,th in enumerate(theta)]
        y = [r[i] * math.sin(th) for i,th in enumerate(theta)]
        ax.plot(x, y, ((dist/maxdist) * 100), color = clr_by_dist(coeff))

def normalize(data, ideal, nadir):
    return [[(x - ideal[i])/(nadir[i] - ideal[i]) for i,x in enumerate(e)] for e in data]

def knn_cluster(data, k):
    sorted_index = [e[0] for e in sorted(
                    [(i,d[0]) for i,d in enumerate(data)], 
                            key = lambda x: x[0])]
    cluster = {}
    for i in sorted_index:
        dists = []
        for j in sorted_index:
            if(i != j):
                dists.append([j, euclid_dist(data[i], data[j])])
        dists.sort(key = lambda x: x[1])
        cluster[i] = [d[0] for d in dists][0:k]
    return cluster

def obj_to_tradeoffs(data, clust_map):
    tradeoffs = {}
    # for p in clust_map[0]:
    print([(x - y) for x,y in zip(data[0], data[1])])
    return tradeoffs


def main():

    # trade-off andrews test
    global ideal, nadir
    data = load_csv("data/debMd_10_190-norm.csv")
    [ideal, nadir] = get_bound(data)
    norm_data = normalize(data, ideal, nadir)
    clust_map = knn_cluster(norm_data, 10)
    print(clust_map)
    tradeoffs = obj_to_tradeoffs(norm_data, clust_map)
    # print(tradeoffs)
    # ideal = [0 for x in range(len(data[0]))]
    # nadir = [1 for x in range(len(data[0]))]
    # plot_andrews([[tradeoffs[key] for key in tradeoffs]], sinusoid, [0, 2 * math.pi], clr_by_dist)
    
    # data normalization test
    # data = load_csv("data/debMd_3_190.csv")
    # [ideal, nadir] = get_bound(data)
    # norm_data = normalize(data, ideal, nadir)
    # save_csv(norm_data, "data/debMd_3_190-norm.csv")
    # knees = load_csv("data/debMd_3_190-knees.csv")
    # norm_knees = normalize(knees, ideal, nadir)
    # save_csv(norm_knees, "data/debMd_3_190-norm-knees.csv")

    # knee examples
    # data = load_csv("data/debMd_10_190.csv")
    # knees = load_csv("data/debMd_10_190-knees.csv")
    # [ideal, nadir] = get_bound(data)
    # plot_andrews([data, knees], sinusoid, [0, 2 * math.pi], clr_by_dist)
    # plot_knee_distance3d(data)

    # andrews plot examples
    # plot_andrews([data, knees], triangle, [-math.pi, -math.pi/2])
    # plot_andrews([data, knees], triangle2, [-math.pi, -math.pi/2])
    # plot_andrews([data, knees, aug_knees], sawtooth, [-math.pi, -math.pi/2])
    # plot_andrews([data, knees], legendre, [-1.0, 1.0])
    
    # test with clustering data
    # iris_sa = load_csv("data/iris-sa.csv")
    # iris_vc = load_csv("data/iris-vc.csv")
    # iris_va = load_csv("data/iris-va.csv")
    # plot_andrews([iris_sa, iris_vc, iris_va], sinusoid, [-math.pi, math.pi])
    # plot_andrews([iris_sa, iris_vc, iris_va], legendre, [-1.0, 1.0])
    # plot_andrews([iris_sa, iris_vc, iris_va], triangle, [-math.pi, math.pi])
    
    # show plot
    plt.show()

main()

"""
Some old testing code
"""

import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial, Legendre

# list of all valid colors
clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

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

def sinusoid(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    return (coeff[0] * (1/math.sqrt(2))) + \
            sum([c * math.sin((i/2 + 1) * t) \
                if i % 2 == 0 else c * math.cos(((i + 1)/2) * t) \
                    for i,c in enumerate(coeff[1:])])

def sinusoid_deriv(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    return (coeff[0] * (1/math.sqrt(2))) + \
            sum([c * math.sin((i/2 + 1) * t) \
                if i % 2 == 0 else c * math.cos(((i + 1)/2) * t) \
                    for i,c in enumerate(coeff[1:])])

def square(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    return (4/math.pi) * sum([c * math.sin((2 + (4 * i)) * t) \
                    for i,c in enumerate(coeff)])

def triangle(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    const = lambda x: 1/((2 * x + 1)**2) if x % 2 == 0 else -1/((2 * x + 1)**2)
    return (8/(math.pi**2)) * sum([const(i) * c * math.sin((2 + (4 * i)) * t) \
                    for i,c in enumerate(coeff)])

def triangle_deriv(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    const = lambda x: 1/((2 * x + 1)**2) if x % 2 == 0 else -1/((2 * x + 1)**2)
    return (8/(math.pi**2)) * sum([const(i) * c * math.cos((2 + (4 * i)) * t) \
                    for i,c in enumerate(coeff)])

def triangle1(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    const = lambda x: 1/((2 * x + 1)**2) if x % 2 == 0 else -1/((2 * x + 1)**2)
    return (8/(math.pi**2)) * sum([const(i) * c * math.sin((2 + (4 * i)) * t) * math.sin((2 + (4 * i)) * t)\
                    for i,c in enumerate(coeff)])

def triangle2(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    random.shuffle(coeff)
    csum = sum(coeff)
    const = lambda x: 1/((2 * x + 1)**2) if x % 2 == 0 else -1/((2 * x + 1)**2)
    return (8/(math.pi**2)) * sum([const(i) * (c/(csum-c)) * math.sin((2 + (4 * i)) * t) \
                    for i,c in enumerate(coeff)])
    
def sawtooth(coeff, t):
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    return 1/2 - ((1/math.pi) * sum([c * math.sin(2 * (i+1) * t)/(i+1) \
                    for i,c in enumerate(coeff)]))

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
   
b = []
def legendre(coeff, t):
    global b
    n = len(coeff)
    if len(b) == 0:
        print("making legendre bases")
        b = legendre_bases(n)
    # random.shuffle(coeff, lambda: 0.5)
    # random.shuffle(coeff)
    return sum([(coeff[i] * v) 
                    for i,v in enumerate([sum([(t**i) * b__ 
                        for i,b__ in enumerate(b_)]) 
                            for b_ in b])])

def andrews_curve(coeff, gen_bases, bound, clr):
    x = np.linspace(bound[0], bound[1], num = 100)
    y = [gen_bases(coeff, t) for t in x]
    plt.plot(x, y, clr)

def plot_andrews(lst, gen_bases, bound):
    for i,dataset in enumerate(lst):
        for v in dataset:
                andrews_curve(v, gen_bases, bound, clrs[i])

def euclid_dist(x,y):
    return math.sqrt(sum([(y[i] - x_)**2 for i,x_ in enumerate(x)]))

def augment_knees(data, knees, delta):
    augk = []
    for k in knees:
        augk.append(k)
        for d in data:
            if euclid_dist(k,d) < delta:
                augk.append(d)
    return augk

def main():
    data = load_csv("data/debMd_3_190.csv")
    knees = load_csv("data/debMd_3_190-knees.csv")
    aug_knees = augment_knees(data, knees, 1.0)
    save_csv(aug_knees, "data/debMd_3_190-aug-knees.csv")
    # plot_andrews([data, knees], triangle, [-math.pi, -math.pi/2])
    plot_andrews([data, knees, aug_knees], triangle2, [-math.pi, -math.pi/2])
    # plot_andrews([data, knees, aug_knees], legendre, [-1.0, 1.0])
    
    data = load_csv("data/debMd_10_190.csv")
    knees = load_csv("data/debMd_10_190-knees.csv")
    aug_knees = augment_knees(data, knees, 3.0)
    save_csv(aug_knees, "data/debMd_10_190-aug-knees.csv")
    # plot_andrews([data, knees], triangle, [-math.pi, -math.pi/2])
    # plot_andrews([data, knees], triangle2, [-math.pi, -math.pi/2])
    # plot_andrews([data, knees, aug_knees], sawtooth, [-math.pi, -math.pi/2])
    # plot_andrews([data, knees], legendre, [-1.0, 1.0])
    
    # iris_sa = load_csv("data/iris-sa.csv")
    # iris_vc = load_csv("data/iris-vc.csv")
    # iris_va = load_csv("data/iris-va.csv")
    # plot_andrews([iris_sa, iris_vc, iris_va], sinusoid, [-math.pi, math.pi])
    # plot_andrews([iris_sa, iris_vc, iris_va], legendre, [-1.0, 1.0])
    # plot_andrews([iris_sa, iris_vc, iris_va], triangle, [-math.pi, math.pi])
    
    plt.show()

main()

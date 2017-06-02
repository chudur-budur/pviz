import math
import random
import os
import sys
import vectorutils as vu

"""
    This file contains all the procedures to generate a many-objective
    Pareto-surface. The benchmark function includes:
        a. DEBMDK (n-dim knee problem)
        b. DTLZ6  (n-dim Surface with Pareto-optimal patches)
        c. DTLZ7  (n-dim Pareto-surface with line and a hyperplane)
        d. DEBDO  (n-dim Pareto-surface with disconnected outliers)
"""
    
def radius(u, K):
    """
        The radius function for the DEBMDK problem
        r(u) = 5.0 + (10.0 * (u - 0.5)^2) + ((2.0/K) * cos(2.0 * K * pi * u)) 
    """
    return 5.0 + (10.0 * (u - 0.5) * (u - 0.5)) \
            + ((2.0/K) * math.cos(2.0 * K * math.pi * u))

def debmdk(u, params = None):
    """
        Generate surface for the DEBMDK problem, where m = len(u) + 1
        fx(u,v) = r(u,v) * sin(u * (pi/2.0)) * sin(v * (pi/2.0))
        fy(u,v) = r(u,v) * sin(u * (pi/2.0)) * cos(v * (pi/2.0))
        fz(u,v) = r(u,v) * cos(u * (pi/2.0))
    """
    K = params[0] if params else 1.0
    M = len(u) + 1
    f = [1 for v in range(M)]
    for i in range(M):
        fstr = ""
        for j in range(M - (i+1)):
            f[i] = f[i] * math.sin(u[j] * 0.5 * math.pi)
            fstr = fstr + "sin({0:.2f} * pi/2) ".format(u[j])
        if(i != 0):
            aux = M - (i + 1)
            f[i] = f[i] * math.cos(u[aux] * 0.5 * math.pi)
            fstr = fstr + "cos({0:.2f} * pi/2) ".format(u[aux])
        # print("f({0:d}) = {1:s}".format(i, fstr))
    r = sum([radius(v, K) for v in u])/len(u) 
    f = [r * v for v in f]
    return f

def dtlz6(u, params = None):
    """
        Generate a surface for slightly modified dtlz6 problem
        where irrespective of dimensions, there will always be
        4 Pareto-optimal patches.
    """
    M = len(u) + 1
    f = u
    f.append(0.0)
    h = 0.0
    for i in range(M-1):
        if i <= 2:
            k = 3
        else:
            k = 1.5
        h = h + f[i] * (1.0 + math.sin(k * math.pi * f[i]))
    f[M-1] = M - h
    return f

def dtlz7(u, params = None):
    """
        This is the function with a line and a hyperplane
        connected.
    """
    N = len(u)
    M = N + 1
    f = [None] * M
    for j in range(M-1):
        f.append(u[M-1] + 4.0 * u[j] - 1.0)
    f[M-1] = 2.0 * u[M-1] + \
            min([u[i] + u[j] for i in range(M) for j in range(M) if i != j]) - 1.0
    return f

def debiso(u, params = None):
    """
        Similar to dtlz6 but it has some outlier patches.
    """
    M = len(u) + 1
    f = u
    f.append(0.0)
    h = 0.0
    for i in range(M-1):
        if i <= 1:
            k = 4
        else:
            k = 2
        h = h + (100.0 + math.exp(5.0 * f[i]) * math.sin(k * math.pi * f[i]))
    h = (1.0/(100.0 * (M-1))) * h
    f[M-1] = h
    return f

def slantgauss(u, param = None):
    """
        A Pareto-surface created from slanted gaussian function.
        variant 1:
            y1 = (1 - x)
            y2 = -0.18 * exp(-(x - 0.5)^2/0.025)
            f = y2 + y2
        variant 2:
            y1 = 3 - 2x
            y2 = -0.75 * exp(-(x - 0.8)^2/0.1)
            f = y2 + y1
    """
    N = len(u)
    M = N + 1
    f = [None] * M
    a = -1.0
    mu = 0.5
    s = 0.05
    f[0:M-1] = u[0:M-1]
    y1 = 4.0 * (1.0 - math.fsum(u))
    y2 = a * math.exp(-1.0 * math.fsum([(x - mu)**2/s for x in u]))
    f[M-1] = y1 + y2
    return f

def lhc_samples(n, M):
    """
    Latin hypercube sampling n samples of m-dimensional points
    """
    d = 1.0 / float(n) ;
    samp = []
    for i in range(M):
        temp = []
        for j in range(n):
            val = (j * d) + ((((j + 1.0) * d) - (j * d)) * random.random())
            temp.append(val)
        random.shuffle(temp)
        if len(samp) == 0:
            for item in temp:
                samp.append([item])
        else:
            for idx,item in enumerate(temp):
                samp[idx].append(item)
    # samp[0] = [0.5 for v in range(m)] # this might be the knee ?
    return samp

def generate_surface(n, M, objfunc, params = None):
    """
    Generate surface points using LHC sampling,
    n samples of m dimensional points.
    """
    U = lhc_samples(n, M-1)
    fv = []
    xv = []
    for x in U:
        f = eval(objfunc)(x, params)
        if f is not None:
            fv.append(f)
            xv.append(x)
    return xv, fv


def augment_knees(xknee, xvals, fvals, objfunc, params, n = 10):
    """
        Make a cloud of n points near the known knee point xknee
    """
    delta = 0.1
    idx = [v for v in range(len(fvals))]
    random.shuffle(idx)
    xvals[idx[0]] = xknee
    fvals[idx[0]] = eval(objfunc)(xknee, params)
    for i in range(1,n):
        xv = [random.uniform(x - delta, x + delta) for x in xknee]
        xvals[idx[i]] = xv
        fvals[idx[i]] = eval(objfunc)(xv, params)

def dominates(a, b):
    M = len(a)
    f1 = 0 
    f2 = 0
    for i in range(M):
        if a[i] < b[i]:
            f1 = 1
        elif a[i] > b[i]:
            f2 = 1
    if f1 == 1 and f2 == 0:
        return 1
    elif f1 == 0 and f2 == 1:
        return -1
    else:
        return 0

def ndsort(fvals):
    P = [i for i in range(len(fvals))]
    F = []
    for p in P:
        Sp = []
        np = 0
        for q in P:
            if dominates(fvals[p], fvals[q]) == 1:
                Sp.append(q)
            elif dominates(fvals[q], fvals[p]) == 1:
                np = np + 1
        if np == 0:
            F.append(p)
    # print(F)
    return F

def save_csv(xvals, fvals, fvals_, filename):
    """
        Save data into file
    """
    fp = open(filename, 'w')
    M = len(fvals[0])
    s1 = 0
    e1 = (s1 + len(fvals[0])) - 1
    s2 = e1 + 1
    e2 = (s2 + len(fvals_[0])) - 1
    s3 = e2 + 1
    e3 = (s3 + len(xvals[0])) - 1
    header = "# {" \
            + "\'n\':{0:d}".format(len(fvals)) + ", " \
            + "\'M\':{0:d}".format(len(fvals[0])) + ", " \
            + "\'N\':{0:d}".format(len(xvals[0])) + ", " \
            + "\'f\':[{0:d},{1:d}]".format(s1, e1) + ", " \
            + "\'f_\':[{0:d},{1:d}]".format(s2, e2) + ", " \
            + "\'x\':[{0:d},{1:d}]".format(s3, e3) + "}\n"
    fp.write(header)
    index = 0
    for i,fv in enumerate(fvals):
        line = str(index) + "," + ",".join(["{0:.3f}".format(f) for f in fv]) + "," \
                + ",".join(["{0:.3f}".format(f) for f in fvals_[i]]) + "," \
                + ",".join(["{0:.3f}".format(x) for x in xvals[i]]) + "\n"
        fp.write(line)
        index = index + 1
    fp.close()


# testing routine, nothing to do with the original code
def tester():
    u = [0.5, 0.5, 0.5, 0.5]
    f = debmdk(u, [1.0])
    print(f)

# main function
def main():
    random.seed(123457)
    n = 1000
    M = 3
    K = 1.0
    rootdir = "data"
    # objfunc = "debmdk"
    # objfunc = "dtlz6"
    # objfunc = "dtlz7"
    # objfunc = "debiso"
    objfunc = "slantgauss"
    params = [K]
    filename = objfunc + "{0:d}m{1:d}n.csv".format(M, n)
    path = os.path.join(rootdir, filename)
    xvals, fvals = generate_surface(n, M, objfunc, params)
    if objfunc == "dtlz6" or objfunc == "dtlz7" or objfunc == "debiso" or objfunc == "slantgauss":
        idx = ndsort(fvals)
        fvals = [fvals[i] for i in idx]
        xvals = [xvals[i] for i in idx]
    if objfunc == "debmdk":
        xknee = [0.5 for v in range(M-1)]
        augment_knees(xknee, xvals, fvals, objfunc, params)
    [ideal, nadir] = vu.get_bound(fvals) 
    fvals_ = vu.normalize(fvals, ideal, nadir)
    save_csv(xvals, fvals, fvals_, path)

# entry points
main()
# tester()

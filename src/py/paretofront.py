#!/usr/bin/python3

import sys
import math
import numpy as np
import vectorutils as vu

"""
    This class encapsulates all necessary stuffs for objective
    vectors in a pareto-front.
"""

class point(object):
    """
        This class models one pareto optimal point
            f:  the raw objective vector
            f_: the normalized objective vector
            x:  the domain of the objective vector
            mu: the knee indicator value found from wsn
            y:  the y coordinate values found from tsne
            y_: the normalized y
    """
    def __init__(self):
        # an empty object
        self.f = []
        self.f_ = []
        self.x = []
        self.mu = 0.0
        self.mu_ = 0.0
        self.y = []
        self.y_ = []

    def __init__(self, header, vector):
        # object with data parsed from a line and the file header
        if 'f' in header:
            chunk = header['f']
            self.f = vector[chunk[0]:chunk[1]+1]
        else:
            self.f = []
        if 'f_' in header:
            chunk = header['f_']
            self.f_ = vector[chunk[0]:chunk[1]+1]
        else:
            self.f_ = []
        if 'x' in header:
            chunk = header['x']
            self.x = vector[chunk[0]:chunk[1]+1]
        else:
            self.x = []
        if 'mu' in header:
            chunk = header['mu']
            self.mu = vector[chunk[0]:chunk[1]+1][0]
        else:
            self.mu = 0.0
        if 'mu_' in header:
            chunk = header['mu_']
            self.mu_ = vector[chunk[0]:chunk[1]+1][0]
        else:
            self.mu_ = 0.0
        if 'y' in header:
            chunk = header['y']
            self.y = vector[chunk[0]:chunk[1]+1]
        else:
            self.y = []
        if 'y_' in header:
            chunk = header['y_']
            self.y_ = vector[chunk[0]:chunk[1]+1]
        else:
            self.y_ = []

    def __str__(self):
        # stringize
        return vu.tostr(self.f) + vu.tostr(self.f_) + vu.tostr(self.x) \
                + "[{0:.3f}]".format(self.mu) + "[{0:.3f}]".format(self.mu_) \
                + vu.tostr(self.y) + vu.tostr(self.y_)

    def __repr__(self):
        # repr expression
        return self.__str__()

class frontdata(object):
    """
        This class manages all data points in a Pareto-front.
    """
    def __init__(self):
        # an empty init function
        self.data = {}
        self.header = {}

    def __str__(self):
        # string representation of this object
        string = str(self.header) + "\n"
        for k,v in self.data.items():
            string = string + "{0:5d}: {1:s}".format(k, str(v)) + '\n'
        return string[:-1]

    def __repr__(self):
        # repr expression
        return self.__str__()

    def load_csv(self, filepath):
        # load pareto front from a csv file.
        try:
            fp = open(filepath, 'r')
            self.header = eval((fp.readline().split('#')[1]).strip())
            for line in fp:
                vals = [float(v) if i > 0 else int(v) \
                        for i,v in enumerate(line.strip().split(','))]
                index = vals[0]
                vals = vals[1:]
                pt = point(self.header, vals)
                self.data[index] = pt
        except FileNotFoundError:
            print("file {0:s} not found.".format(filepath))
            sys.exit
   
    def sorted_indices(self, comparator = None, reverse = False):
        # get the sorted indices w.r.t a comparator.
        keys = self.data.keys()
        if not comparator:
            comparator = lambda i: self.data[i].f[0]
        return sorted(keys, key = comparator, reverse = reverse)

    def nbr_normeps(self, center, attr, epsilon = 0.05):
        # get the list of indices where all points is in the epsilon
        # neighbourhood of the index at center.
        idx = []
        for k,v in self.data.items():
            if k != center:
                if attr == 'f':
                    delta = math.fabs(vu.distlp(self.data[center].f, self.data[k].f))
                elif attr == 'f_':
                    delta = math.fabs(vu.distlp(self.data[center].f_, self.data[k].f_))
                elif attr == 'y':
                    delta = math.fabs(vu.distlp(self.data[center].y, self.data[k].y))
                elif attr == 'y_':
                    delta = math.fabs(vu.distlp(self.data[center].y_, self.data[k].y_))
                else:
                    raise Exception("invalid attribute.")
                if delta <= epsilon:
                    idx.append(k)
        return idx
    
    def nbr_knn(self, center, attr, k = 3):
        # same as above but uses knn, it used sorting method, so it's slow.
        if attr == 'f':
            idx = self.sorted_indices(\
                    lambda i: math.fabs(vu.distlp(self.data[center].f, self.data[i].f)))
        elif attr == 'f_':
            idx = self.sorted_indices(\
                    lambda i: math.fabs(vu.distlp(self.data[center].f_, self.data[i].f_)))
        elif attr == 'y_':
            idx = self.sorted_indices(\
                    lambda i: math.fabs(vu.distlp(self.data[center].y_, self.data[i].y_)))
        elif attr == 'y':
            idx = self.sorted_indices(\
                    lambda i: math.fabs(vu.distlp(self.data[center].y, self.data[i].y)))
        else:
            raise Exception("invalid attribute.")
        return idx[1:k+1] if len(idx[1:]) >= k else []

    def get_list(self, attr):
        # get all the attr values as a list
        if attr == 'f':
            return [self.data[i].f for i in range(len(self.data))]
        elif attr == 'f_':
            return [self.data[i].f_ for i in range(len(self.data))]
        elif attr == 'mu':
            return [self.data[i].mu for i in range(len(self.data))]
        elif attr == 'mu_':
            return [self.data[i].mu_ for i in range(len(self.data))]
        elif attr == 'y':
            return [self.data[i].y for i in range(len(self.data))]
        elif attr == 'y_':
            return [self.data[i].y_ for i in range(len(self.data))]
        else:
            raise Exception("the parameter attr is invalid.")

if __name__ == "__main__":
    """
        The tester function
    """
    pf = paretofront()
    # pf.load_csv("data/debmdk2m250n.csv")
    pf.load_csv("data/debmdk3m500n.csv")
    
    print("pf:")
    print(pf)

    print("sorted pf:")
    print(pf.sorted_indices())
    
    print("eps neighbourhood:")
    nbr = pf.nbr_normeps(66)
    for idx in nbr:
        dist = math.fabs(vu.distlp(pf.data[66].f_, pf.data[idx].f_))
        print("{0:d} - {1:d} = {2:.3f}".format(66, idx, dist))
    
    print("knn neighbourhood:")
    nbr = pf.nbr_knn(66, 5)
    for idx in nbr:
        dist = math.fabs(vu.distlp(pf.data[66].f_, pf.data[idx].f_))
        print("{0:d} - {1:d} = {2:.3f}".format(66, idx, dist))

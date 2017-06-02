#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10,
# and it requires a working installation of NumPy. The implementation comes with an
# example on the MNIST dataset. In order to plot the results of this example, a working
# installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import sys
import os
import numpy as np
# import pylab as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import vectorutils as vu
import paretofront


def Hbeta(D = np.array([]), beta = 1.0):
    """
        Compute the perplexity and the P-row for a specific value of the 
        precision of a Gaussian distribution.
    """
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
    """
        Performs a binary search to get P-values in such a way that each conditional 
        Gaussian has the same perplexity.
    """

    # Initialize some variables
    print ("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # Print progress
        if i % 500 == 0:
            print ("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print ("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return P


def pca(X = np.array([]), no_dims = 50):
    """
        Runs PCA on the NxD array X in order to reduce its 
        dimensionality to no_dims dimensions.
    """
    print ("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X = np.array([]), no_dims = 2, initial_dims = None, perplexity = 30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its 
        dimensionality to no_dims dimensions. The syntaxis of the 
        function is Y = tsne.tsne(X, no_dims, perplexity), where X 
        is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print ("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print ("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    if initial_dims is not None:
        print("applying pca to reduce the dimension to", initial_dims)
        X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    # max_iter = 250
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4		    # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / \
            (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(
                np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + \
            (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print ("Iteration ", (iter + 1), ": error is ", C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4

    # Return solution
    return Y

def set_tsne_values(pf, y):
    """
        Set the computed tsne coordinate values to the pf
    """
    bound = vu.get_bound(y)
    y_ = vu.normalize(y, bound[0], bound[1])
    for i in range(len(pf.data)):
        pf.data[i].y = y[i]
        pf.data[i].y_ = y_[i]

def save_csv(pf, filename):
    """
        Save data into file
    """
    fp = open(filename, 'w')
    s = pf.header['mu_'][1] + 1 
    e = (s + len(pf.data[0].y)) - 1 
    pf.header['y'] = [s, e]
    s = pf.header['y'][1] + 1
    e = (s + len(pf.data[0].y_)) - 1
    pf.header['y_'] = [s, e]
    fp.write("# {0:s}\n".format(str(pf.header)))
    idx = pf.sorted_indices(lambda i: pf.data[i].mu, reverse = True)
    for i in idx:
        line = str(i) + ',' \
                + vu.tocsvstr(pf.data[i].f) + ',' \
                + vu.tocsvstr(pf.data[i].f_) + ',' \
                + vu.tocsvstr(pf.data[i].x) + ',' \
                + "{0:.3f}".format(pf.data[i].mu) + ',' \
                + "{0:.3f}".format(pf.data[i].mu_) + ',' \
                + vu.tocsvstr(pf.data[i].y) + ',' \
                + vu.tocsvstr(pf.data[i].y_) + '\n'
        fp.write(line)
    fp.close()

def show_plot(Y, labels, outfile):
    maxl = max(labels)
    minl = min(labels)
    norml = vu.normalize(labels, minl, maxl)
    
    normc = mpl.colors.Normalize(vmin = 0, vmax = 1.0)
    # cmap = mpl.cm.Greys
    cmap = mpl.cm.jet
    clrmap = mpl.cm.ScalarMappable(norm = normc, cmap = cmap)
    rgbs = [clrmap.to_rgba(v) for v in norml]
    ps = [int(10 + (v * 100)) for v in norml]
    
    fig = plt.figure()
    fig.canvas.set_window_title(outfile)
    plt.scatter(Y[:, 0], Y[:, 1], s = ps, marker = 'o', facecolor = rgbs, alpha = 0.4, linewidth = 0.5)
    figfile = outfile.split('.')[0] + ".png"
    plt.savefig(figfile, bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":
    rootdir = "data"

    filelist = [\
            "debmdk2m250n-mu.csv", "debmdk3m500n-mu.csv", "debmdk4m1000n-mu.csv", "debmdk5m2000n-mu.csv", \
            "dtlz62m500n-mu.csv", "dtlz63m2000n-mu.csv", "dtlz64m4000n-mu.csv", "dtlz65m5000n-mu.csv", \
            "debiso2m1000n-mu.csv", "debiso3m3000n-mu.csv", "debiso4m4000n-mu.csv", "debiso5m5000n-mu.csv", \
            "slantgauss2m500n-mu.csv", "slantgauss3m1000n-mu.csv", "slantgauss4m2000n-mu.csv"\
            ]
    for filename in filelist:
        infile = os.path.join(rootdir, filename)
        
        pf = paretofront.frontdata()
        pf.load_csv(infile)
        
        print ("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
        print ("Running example on", infile)
        
        # X = np.loadtxt("mnist2500_X.txt")
        # labels = np.loadtxt("mnist2500_labels.txt")
        X = np.array(pf.get_list('f'))
        labels = pf.get_list('mu')
        
        Y = tsne(X, 2, None, 20.0)
        set_tsne_values(pf, Y.tolist())
   
        outfile = os.path.join(rootdir, filename.split('-')[0] + "-tsne.csv")
        save_csv(pf, outfile)

        # show_plot(Y, labels, outfile)


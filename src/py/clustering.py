import os
import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import vectorutils as vu
import paretofront

def test_dbscan():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples = 750, centers = centers, cluster_std = 0.4, random_state = 0)

    print(X)
    print(labels_true)

    db = DBSCAN(eps = 0.3, min_samples = 10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(labels)
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

def apply_dbscan(pf, path):
    f = np.array(pf.get_list('f_'))
    db = DBSCAN(eps = 0.3, min_samples = 100).fit(f)
    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(labels)
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(f, labels))
    return labels

def plot_clusters(pf, labels, filepath, axes = [0, 1, 2]):
    """
        Now plot the data points colored according to the mu() value
        knee points will have darker shade
    """
    # markers = ['o', 'v', '^', '*', '<', 's', '>', 'p', '8', 'h', 'H', 'D', 'd', 'P', 'X']
    markers = ['o', 'v', '*', 's', 'p', '8', 'h', 'H', 'D', 'x']
    muvals = pf.get_list('mu_')
    minmu = min(muvals)
    maxmu = max(muvals)
    normmu = vu.normalize(muvals, minmu, maxmu)
    
    x = []
    y = []
    z = []
    for i in range(len(pf.data)):
        x.append(pf.data[i].f[axes[0]])
        y.append(pf.data[i].f[axes[1]])
        if len(pf.data[i].f) > 2:
            z.append(pf.data[i].f[axes[2]])

    normc = mpl.colors.Normalize(vmin = 0, vmax = 1.0)
    # cmap = mpl.cm.Greys
    # cmap = cm.RdBu
    cmap = mpl.cm.jet
    # cmap = cm.rainbow_r
    clrmap = mpl.cm.ScalarMappable(norm = normc, cmap = cmap)
    maxclr = max(muvals)
    rgbs = [clrmap.to_rgba(v) for v in normmu]
    ps = [int(10 + (v * 100)) for v in normmu]
    ms = [markers[l] for l in labels]
    
    fig = plt.figure()
    fig.canvas.set_window_title(filepath)
    if len(z) > 0:
        ax = Axes3D(fig)
        for i in range(len(x)):
            ax.scatter(x[i], y[i], z[i], s = ps[i], marker = ms[i], \
                    facecolor = rgbs[i], alpha = 0.4, linewidth = 0.5)
    else:
        for i in range(len(x)):
            plt.scatter(x[i], y[i], z[i], s = ps[i], marker = ms[i], \
                    facecolor = rgbs[i], alpha = 0.4, linewidth = 0.5)
    
    figfile = filepath.split('-')[0] + "-clst.png"
    print("saving figure:", figfile)
    plt.savefig(figfile, bbox_inches = 'tight')
    print("showing plot")
    plt.show()

def plot_clusters_tsne(pf, labels, filepath):
    """
        Displays the t-SNE scatter plot from the saved csv data file.
    """
    markers = ['o', 'v', '*', 's', 'p', '8', 'h', 'H', 'D', 'x']
    muvals = np.array(pf.get_list('mu_'))
    mumax = max(muvals)
    mumin = min(muvals)
    munorm = vu.normalize(muvals, mumin, mumax)
    Y = np.array(pf.get_list('y_'))
    
    normc = mpl.colors.Normalize(vmin = 0.0, vmax = 1.0)
    # cmap = mpl.cm.Greys
    # cmap = mpl.cm.RdBu_r
    cmap = mpl.cm.jet
    clrmap = mpl.cm.ScalarMappable(norm = normc, cmap = cmap)
    rgbs = [clrmap.to_rgba(v) for v in munorm]
    ps = [int(10 + (v * 100)) for v in munorm]
    ms = [markers[l] for l in labels] 

    fig = plt.figure()
    fig.canvas.set_window_title(filepath)
    for i in range(pf.header['n']):
        plt.scatter(Y[i, 0], Y[i, 1], s = ps[i], marker = ms[i], \
                facecolor = rgbs[i], alpha = 0.4, linewidth = 0.5)
    
    figfile = filepath.split('-')[0] + "-clstsne.png"
    print("saving figure:", figfile)
    plt.savefig(figfile, bbox_inches = 'tight')
    print("showing plot")
    plt.show()

if __name__ == "__main__":
    # test_dbscan()

    rootdir = "data"
    filename = "dtlz63m2000n-tsne.csv"
    # filename = "dtlz64m4000n-tsne.csv"
    # filename = "dtlz65m5000n-tsne.csv"
    path = os.path.join(rootdir, filename)

    pf = paretofront.frontdata()
    pf.load_csv(path)

    labels = apply_dbscan(pf, path)

    plot_clusters(pf, labels, path)
    plot_clusters_tsne(pf, labels, path)

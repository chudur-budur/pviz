import sys
import os
import math
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import spectral_clustering

import utils.vectorops as vops
import utils.fmt as fmt

"""
This script takes a data file, applies DBSCAN algorithm
to find clusters and then save each cluster into separate
files.
"""

def apply_dbscan(points, params):
    """
    Apply DBSCAN algorithm to find the clusters.
    """
    f = np.array(points)
    algorithm = DBSCAN(eps = params[0], min_samples = params[1])
    clusters = algorithm.fit(f)
    labels = clusters.labels_
    
    cluster_map = {}
    for index, label in enumerate(clusters.labels_.tolist()):
        if label not in cluster_map:
            cluster_map[label] = [index]
        else:
            cluster_map[label].append(index)
    print("clusters:", [(k, len(cluster_map[k])) \
            for k in list(cluster_map.keys())])
    return cluster_map

def apply_kmeans(points, params):
    """
    Apply K-means algorithm to find the clusters.
    """
    f = np.array(points)
    algorithm = KMeans(n_clusters = params[0], random_state = 170, max_iter = params[1])
    clusters = algorithm.fit(f)
    labels = clusters.labels_
    
    cluster_map = {}
    for index, label in enumerate(clusters.labels_.tolist()):
        if label not in cluster_map:
            cluster_map[label] = [index]
        else:
            cluster_map[label].append(index)
    print("clusters:", [(k, len(cluster_map[k])) \
            for k in list(cluster_map.keys())])
    return cluster_map

def apply_affprop(points, params):
    """
    Apply Affinity Propagation algorithm to find the clusters.
    """
    f = np.array(points)
    algorithm = AffinityPropagation()
    clusters = algorithm.fit(f)
    labels = clusters.labels_
    
    cluster_map = {}
    for index, label in enumerate(clusters.labels_.tolist()):
        if label not in cluster_map:
            cluster_map[label] = [index]
        else:
            cluster_map[label].append(index)
    print("clusters:", [(k, len(cluster_map[k])) \
            for k in list(cluster_map.keys())])
    return cluster_map

def apply_meanshift(points, params):

    """
    Apply Mean-shift algorithm to find the clusters.
    """
    f = np.array(points)
    bandwidth = estimate_bandwidth(f, quantile = params[0], n_samples = params[1])
    algorithm = MeanShift(bandwidth = bandwidth, bin_seeding = True)
    clusters = algorithm.fit(f)
    labels = clusters.labels_
    
    cluster_map = {}
    for index, label in enumerate(clusters.labels_.tolist()):
        if label not in cluster_map:
            cluster_map[label] = [index]
        else:
            cluster_map[label].append(index)
    print("clusters:", [(k, len(cluster_map[k])) \
            for k in list(cluster_map.keys())])
    return cluster_map

def apply_spectral(points, params):

    """
    Apply Spectral Clustering algorithm to find the clusters.
    """
    f = np.array(points)
    algorithm = spectral_clustering(n_clusters = params[0], eigen_solver = params[1])
    clusters = algorithm.fit(f)
    labels = clusters.labels_
    
    cluster_map = {}
    for index, label in enumerate(clusters.labels_.tolist()):
        if label not in cluster_map:
            cluster_map[label] = [index]
        else:
            cluster_map[label].append(index)
    print("clusters:", [(k, len(cluster_map[k])) \
            for k in list(cluster_map.keys())])
    return cluster_map

def save_clusters(points, clusters, path, mode = "original"):
    """
    Saves different clusters into separate files.
    """
    base, tail = os.path.split(path)
    parts = tail.split('-')
    keys = [k for (n,k) in sorted([(len(clusters[k]), k) \
                    for k in list(clusters.keys())], reverse = True)]
    if mode == "original":
        cv_file = tail.split('.')[0] + "-cv.out"
        cv = None
        cv_path = base + "/" + cv_file
        if os.path.exists(cv_path):
            cv = fmt.load(cv_path)
        for label in keys:
            # If there are any noise, then group them as the "last" cluster.
            cluster_id = len(keys) if label == -1 else label + 1
            newpath = base + "/" + "-".join(parts[:-1]) + "-c{0:d}".format(cluster_id)
            try:
                os.makedirs(newpath)
            except OSError:
                pass

            # Save the original data point cluster
            filename = "-".join(parts[:-1]) + "-c{0:d}-".format(cluster_id) + parts[-1]
            filepath = newpath + "/" + filename
            fmt.save([points[i] for i in clusters[label]], filepath)
            print("Cluster {0:d} saved in {1:s}".format(cluster_id, filepath))
            
            # Save the precomputed cluster for cv
            if cv is not None:
                filename = "-".join(parts[:-1]) + "-c{0:d}-".format(cluster_id) \
                        + parts[-1].split('.')[0] + "-cv.out"
                filepath = newpath + "/" + filename
                fmt.save([points[i] for i in clusters[label]], filepath)
                print("Cluster {0:d} for cv saved in {1:s}".format(cluster_id, filepath))
    elif mode == "precomputed":
        norm_file = tail.split('.')[0] + "-norm.out"
        mu_file = tail.split('.')[0] + "-norm-mu.out"
        cv_file = tail.split('.')[0] + "-cv.out"
        plt_file = tail.split('.')[0] + "-norm-palette.out"
        plr_file = tail.split('.')[0] + "-norm-palette-polar.out"
        lgs_file = tail.split('.')[0] + "-norm-palette-logistic.out"

        norm_path = base + "/" + norm_file
        mu_path = base + "/" + mu_file
        cv_path = base + "/" + cv_file
        plt_path = base + "/" + plt_file
        plr_path = base + "/" + plr_file
        lgs_path = base + "/" + lgs_file

        norm_pts, mu, cv, plt, plr, lgs = None, None, None, None, None, None
        if os.path.exists(norm_path):
            norm_pts = fmt.load(norm_path)
        if os.path.exists(mu_path):
            mu = fmt.load(mu_path)
        if os.path.exists(cv_path):
            cv = fmt.load(cv_path)
        if os.path.exists(plt_path):
            plt = fmt.load(plt_path)
        if os.path.exists(plr_path):
            plr = fmt.load(plr_path)
        if os.path.exists(lgs_path):
            lgs = fmt.load(lgs_path)

        for label in keys:
            # If there are any noise, then group them as the "last" cluster.
            cluster_id = len(keys) if label == -1 else label + 1
            newpath = base + "/" + "-".join(parts[:-1]) + "-c{0:d}".format(cluster_id)
            try:
                os.makedirs(newpath)
            except OSError:
                pass

            # Save the original data point cluster
            filename = "-".join(parts[:-1]) + "-c{0:d}-".format(cluster_id) + parts[-1]
            filepath = newpath + "/" + filename
            fmt.save([points[i] for i in clusters[label]], filepath)
            print("Cluster {0:d} saved in {1:s}".format(cluster_id, filepath))
            
            # Save the precomputed cv cluster
            if cv is not None:
                filename = "-".join(parts[:-1]) + "-c{0:d}-".format(cluster_id) \
                        + parts[-1].split('.')[0] + "-cv.out"
                filepath = newpath + "/" + filename
                fmt.save([cv[i] for i in clusters[label]], filepath)
                print("Cluster {0:d} for cv saved in {1:s}".format(cluster_id, filepath))
            
            # Save the normalized data point cluster
            if norm_pts is not None:
                filename = "-".join(parts[:-1]) + "-c{0:d}-".format(cluster_id) \
                        + parts[-1].split('.')[0] + "-norm.out"
                filepath = newpath + "/" + filename
                fmt.save([norm_pts[i] for i in clusters[label]], filepath)
                print("Cluster {0:d} for normalized points saved in {1:s}"\
                        .format(cluster_id, filepath))
            
            # Save the precomputed mu value cluster
            if mu is not None:
                filename = "-".join(parts[:-1]) + "-c{0:d}-".format(cluster_id) \
                        + parts[-1].split('.')[0] + "-norm-mu.out"
                filepath = newpath + "/" + filename
                fmt.save([mu[i] for i in clusters[label]], filepath)
                print("Cluster {0:d} for mu saved in {1:s}"\
                        .format(cluster_id, filepath))
            
            # Save the precomputed palette coordinate cluster
            if plt is not None:
                filename = "-".join(parts[:-1]) + "-c{0:d}-".format(cluster_id) \
                        + parts[-1].split('.')[0] + "-norm-palette.out"
                filepath = newpath + "/" + filename
                fmt.save([plt[i] for i in clusters[label]], filepath)
                print("Cluster {0:d} for palette-coords saved in {1:s}"\
                        .format(cluster_id, filepath))
            
            # Save the precomputed polar palette coordinate cluster
            if plr is not None:
                filename = "-".join(parts[:-1]) + "-c{0:d}-".format(cluster_id) \
                        + parts[-1].split('.')[0] + "-norm-palette-polar.out"
                filepath = newpath + "/" + filename
                fmt.save([plr[i] for i in clusters[label]], filepath)
                print("Cluster {0:d} for polar-palette-corrds saved in {1:s}"\
                        .format(cluster_id, filepath))
            
            # Save the precomputed logistic palette coordinate cluster
            if lgs is not None:
                filename = "-".join(parts[:-1]) + "-c{0:d}-".format(cluster_id) \
                        + parts[-1].split('.')[0] + "-norm-palette-logistic.out"
                filepath = newpath + "/" + filename
                fmt.save([lgs[i] for i in clusters[label]], filepath)
                print("Cluster {0:d} for logistic-palette-corrds saved in {1:s}"\
                        .format(cluster_id, filepath))

if __name__ == "__main__":
    # Algorithm parameters
    dbscan = {"carcrash": (0.14, 10), "gaa": (0.125, 100), "c2dtlz2": (0.14, 10)}
    kmeans = {"gaa": (2, 1000)}
    affprop = {"gaa": (-50, 10)}
    meanshift = {"gaa": (0.3, 156)}
    spectral = {"gaa": (2, 'arpack')}
    
    path = sys.argv[1].strip()
    points = fmt.load(path)
    
    # First normalize the data into [0.0, 1.0] so that eps can be 
    # correctly utilized.
    print("Normalizing data points in {0:s} ...".format(path))
    [lb, ub] = vops.get_bound(points)
    points_ = vops.normalize(points, lb, ub)
    print("Normalization done. Now clustering ...")
    
    # Find clusters
    clusters = {}
    if "carcrash" in path:
        clusters = apply_dbscan(points_, dbscan["carcrash"])
    elif "gaa" in path:
        # clusters = apply_kmeans(points_, kmeans["gaa"])
        # clusters = apply_affprop(points_, affprop["gaa"])
        clusters = apply_meanshift(points_, meanshift["gaa"])
        # clusters = apply_spectral(points_, spectral["gaa"])
    elif "c2dtlz2" in path:
        clusters = apply_dbscan(points_, dbscan["c2dtlz2"])
    
    # Now save them.
    save_clusters(points, clusters, path, mode = "precomputed")

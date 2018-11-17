import sys
import math
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import mixture
import utils

"""
This script takes a data file, applies DBSCAN algorithm
to find clusters and then save each cluster into separate
files.
"""

def apply_dbscan(points):
    """
    Apply DBSCAN algorithm to find the clusters.
    """
    f = np.array(points)
    # eps = {3: 0.14, 4: 0.125, 8: 0.255}
    eps = 0.25
    algorithm = DBSCAN(eps = eps)
    # algorithm = mixture.GaussianMixture(n_components = 4, covariance_type='full')
    clusters = algorithm.fit(f)
    # db = DBSCAN(eps = 0.1, min_samples = 100).fit(f)
    # core_samples_mask = np.zeros_like(clusters.labels_, dtype = bool)
    # core_samples_mask[clusters.core_sample_indices_] = True
    labels = clusters.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    # print('Estimated number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(f, labels))
    
    clusters_map = {}
    for index, label in enumerate(clusters.labels_.tolist()):
        if label not in clusters_map:
            clusters_map[label] = [index]
        else:
            clusters_map[label].append(index)
    print("clusters:", list(clusters_map.keys()))
    # print(clusters_map)
    return clusters_map

def save_clusters(clusters, path):
    filepath = path + "-clusters.out"
    fp = open(filepath, 'w')
    for label in list(clusters.keys()):
        fp.write("\t".join(["{0:d}".format(v) for v in clusters[label]]) + "\n")
    fp.close()

if __name__ == "__main__":
    path = sys.argv[1].strip()
    points = utils.load(path)
    clusters = apply_dbscan(points)
    # print(clusters)
    path = path.split('.')[0].strip()
    save_clusters(clusters, path)


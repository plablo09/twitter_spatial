# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import ogr



class ClusterTree:

    def __init__(self,points):
        self.key = points
        self.children = []

    def append_child(self,child):
        self.children.append(ClusterTree(child))


def compute_dbscan(points,eps):
    #Standarize sample
    points = StandardScaler().fit_transform(points)
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=10).fit(points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #Obtain arrays for each cluster except for noise
    unique_labels = set(labels)
    clusters = []
    for l in unique_labels:
        if l != -1:
            class_member_mask = (labels == l)
            clusters.append(points[class_member_mask])
            print "cluster size: " + str(len(points[class_member_mask]))

    print "estoy mandando estos clusters: " + str(len(clusters))
    return clusters

cluster_tree = {}

def hierarchical_dbscan(cluster_list,eps):

    this_level = []
    processed = 0
    tmp = None
    for cluster in cluster_list:
        if len(cluster) > 100:
            tmp = compute_dbscan(cluster,eps)
            this_level.append(tmp)
            cluster_tree[cluster.tostring()] = tmp
            processed += 1

    if processed > 0:
        for group in this_level:
            hierarchical_dbscan(group,eps/2.0)

    else:
        print 'done'
        return 'done'




# Read data from shape file
driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.Open("data/dia/d_0_7.shp", 0)
l = ds.GetLayer()
points = []
for f in l:
    g = f.GetGeometryRef()
    points.append((g.GetX(),g.GetY()))

points = [np.array(points)]
hierarchical_dbscan(points,0.8)
#print cluster_tree


#
# clusters = compute_dbscan(points)
# for i, c in enumerate(clusters):
#     xy = clusters[i]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='k',
#              markeredgecolor='k', markersize=14)
#
# plt.show()

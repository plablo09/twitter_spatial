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


def compute_dbscan(points,eps, inverse = None):
    #Standarize sample
    print "compute_dbscan: recibí %s puntos" % len(points)
    print "compute_dbscan: el eps es de %s" % eps
    scaler = StandardScaler()
    points = scaler.fit_transform(points)
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=10).fit(points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #Obtain arrays for each cluster except for noise
    unique_labels = set(labels)
    points = scaler.inverse_transform(points)
    clusters = []
    for l in unique_labels:
        if l != -1:
            class_member_mask = (labels == l)
            clusters.append(points[class_member_mask])
            #print "cluster size: " + str(len(points[class_member_mask]))

    print "estoy mandando estos clusters: " + str(len(clusters))
    return clusters

cluster_tree = {}
flat_list = []

def hierarchical_dbscan(cluster_list,eps):

    this_level = []
    processed = 0
    tmp = None
    for cluster in cluster_list:
        if len(cluster) > 100:
            tmp = compute_dbscan(cluster,eps)
            this_level.append(tmp)
            cluster_tree[cluster.tostring()] = tmp
            flat_list.append(tmp)
            processed += 1
            print "hierarchical_dbscan: estoy guardando %s clusters en el dic" % len(tmp)

    print "hierarchical_dbscan: procesé %s clusters en este nivel" % processed
    if processed > 0:
        for group in this_level:
            hierarchical_dbscan(group,eps/2.0)

    else:
        print 'done'
        return 'done'




# Read data from shape file
driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.Open("data/dia/t_7.shp", 0)
l = ds.GetLayer()
points = []
for f in l:
    g = f.GetGeometryRef()
    points.append((g.GetX(),g.GetY()))

points = [np.array(points)]
hierarchical_dbscan(points,0.8)
la_ultima = flat_list[len(flat_list)-1]
colors = plt.cm.Spectral(np.linspace(0, 1, len(flat_list)))
for k, col in enumerate(colors):

    level = np.concatenate(flat_list[k])
    label = 'level ' + str(k)
    plt.plot(level[:, 0], level[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6, label=label)

plt.legend(loc='lower right',numpoints=1)
plt.show()




# points = np.array(points)
# scaler = StandardScaler()
# points = scaler.fit_transform(points)
# points = scaler.inverse_transform(points)
# plt.plot(points[:, 0], points[:, 1], 'o', markerfacecolor='k',
#               markeredgecolor='k', markersize=6)
# plt.show()
#

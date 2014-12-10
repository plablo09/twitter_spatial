import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import ogr

# Read data from shape file
driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.Open("data/dia/t_8.shp", 0)
l = ds.GetLayer()
points = []
for f in l:
    g = f.GetGeometryRef()
    points.append((g.GetX(),g.GetY()))

points = np.array(points)


def compute_dbscan(points):
    #Standarize sample
    points = StandardScaler().fit_transform(points)
    # Compute DBSCAN
    db = DBSCAN(eps=0.8, min_samples=10).fit(points)
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
            print len(points[class_member_mask])

    return clusters

def process_clusters(lista):
    if max([len(l) for l in lista]) < 100
        return 'done'
    else:
        for l in lista:
            clusters = compute_dbscan(l)
            process_clusters(clusters)

clusters = compute_dbscan(points)
for i, c in enumerate(clusters):
    xy = clusters[i]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='k',
             markeredgecolor='k', markersize=14)

plt.show()

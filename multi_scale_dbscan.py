# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import ogr


def compute_dbscan(points,eps, min_samples):
   """Sklearn DBSCAN wrapper.

   :param points: a (N,2) numpy array containing the obsevations to cluster
   :param eps: float to be passed as eps to sklearn.cluster.DBSCAN
   :param min_samples: int to be passed as min_samples to sklearn.cluster.DBSCAN
   :returns list with numpy arrays for all the clusters obtained

   """
   #Standarize sample
   scaler = StandardScaler()
   points = scaler.fit_transform(points)
   # Compute DBSCAN
   db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
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

   return clusters

cluster_tree = {}
flat_list = []

def hierarchical_dbscan(cluster_list,eps,min_samples,level=None):
   """
   Calls compute_dbscan with the specified parameters for each point array
   in cluster_list

   :param cluster_list: list of (N,2) numpy arrays
   :param eps: float to be passed as eps to sklearn.cluster.DBSCAN
   :param min_samples: int to be passed as min_samples to sklearn.cluster.DBSCAN
   :param level(optional): the depth in the cluster hierarchy
                        if None then 0-depth is implied
   """

   if level == None:
      level = 0

   this_level = []
   processed = 0
   tmp = None
   cluster_counter = 0
   for cluster in cluster_list:
     if len(cluster) > 100:
         tmp = compute_dbscan(cluster,eps,min_samples)
         this_level.append(tmp)
         name = 'level_' + str(level) + '_cluster_' + str(cluster_counter)
         cluster_tree[name] = tmp
         flat_list.append(tmp)
         processed += 1
         cluster_counter += 1

   print ("hierarchical_dbscan: %s clusters clusters processed at level %s"
         % (processed, level))
   level += 1
   if processed > 0:
      for group in this_level:
         hierarchical_dbscan(group,eps/2.0,min_samples,level)

   else:
     print 'done'
     return




# Read data from shape file and build points array
driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.Open("data/dia/d_0_9.shp", 0)
l = ds.GetLayer()
points = []
for f in l:
    g = f.GetGeometryRef()
    points.append((g.GetX(),g.GetY()))

points = [np.array(points)]

# Call hierarchical_dbscan
hierarchical_dbscan(points,0.8,10)
print cluster_tree.keys()

# Plot results
colors = plt.cm.Spectral(np.linspace(0, 1, len(flat_list)))
for k, col in enumerate(colors):
    level = np.concatenate(flat_list[k])
    label = 'level ' + str(k)
    plt.plot(level[:, 0], level[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6, label=label)

plt.legend(loc='lower right',numpoints=1)
plt.show()

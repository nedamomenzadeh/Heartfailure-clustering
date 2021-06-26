#Import Data and Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

D=pd.read_csv('heart_failure_clinical_records_dataset.csv',dtype='unicode')
Data=D.iloc[:, 0:11]
X_scaled=Data

"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(Data))
"""

#plot for estimating eps (elbow)
from sklearn.neighbors import NearestNeighbors
ns = 3
nbrs = NearestNeighbors(n_neighbors=ns).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)
distanceDec = sorted(distances[:,ns-1], reverse=False)
plt.plot(list(range(1,300)), distanceDec)

# DBSCAN
from sklearn.cluster import DBSCAN
"""
#for 1 cluster (more silhouette in case of scaling)
eps=5.8
min_samples=200
#for 2 clusters (less silhouette in case of scaling)
eps=2.991
min_samples=2
"""

#for 2 clusters with no scaling
eps=22000
min_samples=6
"""
#for 1 cluster with no scaling
eps=37000
min_samples=6
"""
db = DBSCAN(eps=eps, min_samples =min_samples)

clusters = db.fit_predict(X_scaled)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_




#information
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
from sklearn import metrics
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X_scaled, labels))



#Fowlkes-Mallows score (External Index)
from sklearn import metrics
fm=metrics.fowlkes_mallows_score(D['DEATH_EVENT'].astype(int),
                            labels.astype(int))
print("Fowlkes-Mallows Score:",fm)

#Counting the number of clusters
count0 = list(labels).count(0)
count1 = list(labels).count(1)
print("The number of 1s (High Death risk):" ,count1)
print("The number of 0s (Low Death Risk):" ,count0)


#3D and 2D plot for clusters

LABEL_COLOR_MAP = {0 : 'r',1 : 'b', -1:'k',2: 'y'}
label_color = [LABEL_COLOR_MAP[l] for l in labels]

#age and platelets vs. smoking 3D
fig = plt.figure() 
ax = Axes3D(fig) 
ax.scatter(Data.iloc[:, 0].astype(float), Data.iloc[:, 6].astype(float), Data.iloc[:,10].astype(float),c=label_color) 

plt.show()

#ejection_fraction vs. platelets 2D
plt.scatter(Data.iloc[:,4].astype(float),Data.iloc[:,6].astype(float),c=label_color) 
plt.show()
#creatinine_phosphokinase vs. platelets 2D
plt.scatter(Data.iloc[:,2].astype(float),Data.iloc[:,6].astype(float),c=label_color) 
plt.show()




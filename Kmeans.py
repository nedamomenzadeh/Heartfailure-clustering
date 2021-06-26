# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:05:27 2020

@author: 
"""
#Import Data and Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

D=pd.read_csv('heart_failure_clinical_records_dataset.csv',dtype='unicode')
Data=D.iloc[:, 0:11]
"""
#Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Data = pd.DataFrame(scaler.fit_transform(Data))

"""
#K-means
from sklearn.cluster import KMeans

kmeans = KMeans(2)
kfit = kmeans.fit(Data)
identified_clusters = kfit.predict(Data)
clustered_data = D.copy()
clustered_data['Cluster'] = identified_clusters

#
#silhouette score (Internal Index)
from sklearn.metrics import silhouette_score
score = silhouette_score(Data,clustered_data['Cluster'] , metric='euclidean')
print("Silhouette Score:",score)

from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2, 3, 4, 5]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(Data)


#Fowlkes-Mallows score (External Index)
from sklearn import metrics
fm=metrics.fowlkes_mallows_score(D['DEATH_EVENT'].astype(int),
                            clustered_data['Cluster'].astype(int))
print("Fowlkes-Mallows Score:",fm)

#Counting the number of clusters
count0 = list(clustered_data['Cluster']).count(0)
count1 = list(clustered_data['Cluster']).count(1)
print("The number of 1s (High Death risk):" ,count1)
print("The number of 0s (Low Death Risk):" ,count0)

#3D and 2D plot for clusters
labels=kfit.labels_
LABEL_COLOR_MAP = {0 : 'k',1 : 'r'}
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

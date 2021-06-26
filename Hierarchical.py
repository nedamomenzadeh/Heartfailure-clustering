# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 19:02:29 2020

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
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
Data = pd.DataFrame(scaler.fit_transform(Data))
"""
#Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
labels=cluster.fit_predict(Data)

from sklearn.metrics import silhouette_score
score = silhouette_score(Data,cluster.labels_ , metric='euclidean')
print("Silhouette Score:",score)

# Dendogram for Heirarchical Clustering
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot
pyplot.figure(figsize=(10, 7))  
pyplot.title("Dendrograms")  
z=shc.linkage(Data, method='ward')
dend = shc.dendrogram(z)


#Fowlkes-Mallows score (External Index)
from sklearn import metrics
fm=metrics.fowlkes_mallows_score(D['DEATH_EVENT'].astype(int),
                            cluster.labels_.astype(int))
print("Fowlkes-Mallows Score:",fm)

#Counting the number of clusters
c=pd.DataFrame(cluster.labels_)
count0 = np.count_nonzero(cluster.labels_ == 0)
count1 = np.count_nonzero(cluster.labels_ == 1)
print("The number of 1s (High Death risk):" ,count1)
print("The number of 0s (Low Death Risk):" ,count0)


#3D and 2D plot for clusters
labels=cluster.labels_
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


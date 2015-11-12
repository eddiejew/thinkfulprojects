"""kmeans1.py"""

#use scipy.cluster.vq.kmeans to calculate clusters on un.csv, using
# kmeans, for k in 0, ..., 10. Show the elbow plot (k vs average 
# within cluster SSQ) to help find the optimum k for clustering.

import numpy as np
import scipy.cluster.vq as scv
import pandas as pd
import matplotlib.pyplot as plt
import collections
from scipy.spatial.distance import cdist

# read in the data
df = pd.read_csv("un.csv")

# Data Analyis
# determine the raw size of the UN data
data_size = len(df)
print '\nSize of raw UN data is %s. \n' % data_size

# determine the number of rows that are not null in each column
for i in df:
	print 'Size of column %s without NaN is %s.' % (i, len(df[i].dropna()))
print

# determine data type of each column
for i in df:
	print 'Data type of column %s is %s.' % (i, df[i].dtype)
print

# determine number of countries in the dataset
print 'The number of countries in the dataset = %s.' % len(collections.Counter(df['country']))
print

# make a list of the names of the columns we're interested in
colnames = ['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']
# strip off these columns
df2 = pd.DataFrame(df[colnames])
# drop NAs in place
df2.dropna(inplace=True)
# range of k values
k_list = range(1,11)
# calculate the kmeans for each k in the list
KM = [ scv.kmeans(df2.values, k) for k in k_list ]
# now KM contains a list of centroids and distortions (variances) for each k
# let's remove the distortions
ave_wc_SSQ = [ var for (cent, var) in KM ]
# plot the elbow plot
plt.figure()
plt.scatter(k_list, ave_wc_SSQ)
plt.xlabel('Number of Clusters')
plt.ylabel('Average Within-Cluster SSQ')
plt.title('Elbow Plot for kMeans Clustering')
plt.show()
plt.clf()
# the plot starts to level out near k=3, biggest drops occur from k=1 to k=2, 
# and k=2 to k=3 -> we'll use k=3 for clustering
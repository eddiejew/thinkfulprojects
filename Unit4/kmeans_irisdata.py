"""kmeans_irisdata.py"""

import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

# load the data
iris = datasets.load_iris()
x = iris.data
y = iris.target

# create the iris dataframe
df = pd.DataFrame(data, columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], index = target)
df['Species'] = target
# print df to view all comumns have been added correctly

# plot different scatterplot graphs
# plot 'sepal_length and 'sepal_width' by species
plt.figure()
plt.scatter(irisdf['sepal_length'],irisdf['sepal_width'],c=y,cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

# plot 'petal_length' and 'petal_width' by species
plt.figure()
plt.scatter(irisdf['petal_length'],irisdf['petal_width'],c=y,cmap=plt.cm.Paired)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

# plot 'petal_length' and 'sepal_length' by speies
plt.figure()
plt.scatter(irisdf['petal_length'],irisdf['sepal_length'],c=y,cmap=plt.cm.Paired)
plt.xlabel('Petal length')
plt.ylabel('Sepal length')
plt.show()
"""iris.py"""

# plot sepal length vs sepal width (iris)

import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
irisdf = pd.DataFrame(iris.data,columns=["sepal_length","sepal_width","petal_length","petal_width"])
Y = iris.target

plt.figure()
plt.scatter(irisdf['sepal_length'],irisdf['sepal_width'],c=Y,cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()
"""reduce.py"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target
features = iris.feature_names
targets = iris.target_names

# shape of iris data
print "Iris dataset shape: ", X.shape

# plot the iris data: sepal length versus sepal width
plt.figure()
for c, i, target in zip("rbg", [0,1,2], targets):
	plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=target)
plt.legend()
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('Iris Data')
plt.show()
plt.clf()
# good separation between setosa and the others, but 
# poor separation between versicolor + virginica

pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)
# new shape
print "PCA reduced shape: ", X_pca.shape
print "PCA components (2): ", pca.components_
print "PCA explained variance ratio (first two components): ", pca.explained_variance_ratio_
for component in pca.components_:
	print "+".join("%.3f x %s" % (value, name)
		    for value, name in zip(component, features))

# plot the PCA
plt.figure()
for c, i, target in zip("rgb", [0,1,2], targets):
	plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=c, label=target)
plt.legend()
plt.title('PCA of Iris Data')
plt.show()
plt.clf()
# significantly better separation between versicolor + virginica

# perform KNN on the data
neighbors = KNeighborsClassifier(n_neighbors=3)
y_pred = neighbors.fit(X_pca, y).predict(X_pca)
plt.scatter(X_pca[:,0], X_pca[:, 1], c=y_pred)
plt.show()
plt.clf()
# appears similar to the PCA plot

# now perform LDA
lda = LDA(n_components=2)
X_lda = lda.fit(X,y).transform(X)
# a new shape
print "LDA reduced shape: ", X_lda.shape

# plot the LDA
plt.figure()
for c, i, target in zip("rgb", [0,1,2], targets):
	plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], c=c, label=target)
plt.legend()
plt.title('LDA of Iris Data')
plt.show()
plt.clf()
# good separation between the three species

# do KNN again on the LDA
neighbors = KNeighborsClassifier(n_neighbors=3)
y_pred = neighbors.fit(X_lda, y).predict(X_lda)
plt.scatter(X_lda[:,0], X_lda[:, 1], c=y_pred)
plt.show()
plt.clf()
# appears similar to the LDA plot
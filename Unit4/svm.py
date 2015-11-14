"""svm.py"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from matplotlib.colors import ListedColormap
import numpy as np

# load the iris data set
from sklearn import datasets
iris = datasets.load_iris()

# first build a scatter matrix of all possible pairings of features,
# using the three flower types
plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()
# this scatter plot shows us that, in general, it is easy to see
# how to separate setosa from the other two species regardless
# of features being used. However, it is not easy to see how to 
# separate virginica from versicolor

# plot sepal width versus petal length for setosa and versicolor:
plt.scatter(iris.data[0:100, 1], iris.data[0:100, 2], c=iris.target[0:100])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.savefig('setosa_versicolor_sepwid_v_petlen.png')
plt.clf()
# Very clear separation between sertosa and versicolor
# We could easily draw a line in the plane that separates the two
# species into different clusters

# Apply SVC module to this pairing:
svc = svm.SVC(kernel='linear')
X = iris.data[0:100, 1:3]
y = iris.target[0:100]

# plot_estimator function adapted from https://github.com/jakevdp/sklearn_scipy2013
# listedColormap is used for generating a custom colormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y, num):
	estimator.fit(X, y)
	x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
	y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, num),
		                 np.linspace(y_min, y_max, num))
	Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

	# put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

	# also plot the training points
	plt.scatter(X[:, 0], X[:, 1], cmap=cmap_bold)
	plt.axis('tight')
	plt.axis('off')
	plt.tight_layout()

# Now let's apply the plot_estimator function:
plot_estimator(svc, X, y, 100)
plt.savefig('svm_setosa_versicolor_sepwid_v_petlen.png')
plt.clf()
# As expected we get a nice separation of setosa and versicolor 
# across a line drawn in the plane

# Let's see how SVM works with versicolor and virginica,
# again using sepal width versus petal length
svc = svm.SVC(kernel='linear')
X = iris.data[50:150, 1:3]
y = iris.target[50:150]
plot_estimator(svc, X, y, 100)
plt.savefig('svm_versicolor_virginica_sepwid_v_petlen.png')
plt.clf()
# We previously thought it might be challenging to perfectly 
# separate versicolor from virginica, and this is reflected in the result
# A few plants from both species are misidentified. However, we have 
# properly classified the vast majority of plants
# Note: changing C to get a wider soft margin did nothing for this 
# SVM example, so I've omitted this code

# setosa versus versicolor / sepal length versus sepal width
svc = svm.SVC(kernel='linear')
X = iris.data[0:100, 0:2
y = iris.target[0:100]
plot_estimator(svc, X, y, 100)
plt.savefig('svm_setosa_versicolor_seplen_v_sepwid.png')
# we have a pretty clear separation between setosa and versicolor

# setosa versus versicolor / sepal length versus petal length
svc = svm.SVC(kernel='linear')
X = iris.data[0:100, [0,2]]
y = iris.target[0:100]
plot_estimator(svc, X, y, 100)
plt.savefig('svm_setosa_versicolor_seplen_v_petlen.png')
# again, pretty good separation

# setosa versus virginica / sepal length versus sepal width
svc = svm.SVC(kernel='linear')
X = np.concatenate((iris.data[0:50, [0,1]], iris.data[100:150, [0,1]]), axis=0)
y = np.concatenate((iris.target[0:50], iris.target[100:150]), axis=0)
plot_estimator(svc, X, y, 100)
plt.savefig('svm_setosa_virginica_seplen_v_sepwid.png')
# good separation, but on misclassification
# varying C does not change this
"""reduce.py"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
import random

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

# attempting ANN

BIAS = -1

"""
To view the structure of the Neural Network, type
print network_name
"""

class Neuron:
    def __init__(self, n_inputs ):
        self.n_inputs = n_inputs
        self.set_weights( [random.uniform(0,1) for x in range(0,n_inputs+1)] ) # +1 for bias weight

    def sum(self, inputs ):
        # Does not include the bias
        return sum(val*self.weights[i] for i,val in enumerate(inputs))

    def set_weights(self, weights ):
        self.weights = weights

    def __str__(self):
        return 'Weights: %s, Bias: %s' % ( str(self.weights[:-1]),str(self.weights[-1]) )

class NeuronLayer:
    def __init__(self, n_neurons, n_inputs):
        self.n_neurons = n_neurons
        self.neurons = [Neuron( n_inputs ) for _ in range(0,self.n_neurons)]

    def __str__(self):
        return 'Layer:\n\t'+'\n\t'.join([str(neuron) for neuron in self.neurons])+''

class NeuralNetwork:
    def __init__(self, n_inputs, n_outputs, n_neurons_to_hl, n_hidden_layers):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_to_hl = n_neurons_to_hl

            # Do not touch
        self._create_network()
        self._n_weights = None
            # end

    def _create_network(self):
        if self.n_hidden_layers>0:
                # create the first layer
            self.layers = [NeuronLayer( self.n_neurons_to_hl,self.n_inputs )]

                # create hidden layers
            self.layers += [NeuronLayer( self.n_neurons_to_hl,self.n_neurons_to_hl ) for _ in range(0,self.n_hidden_layers)]

                # hidden-to-output layer
            self.layers += [NeuronLayer( self.n_outputs,self.n_neurons_to_hl )]
        else:
                # If we don't require hidden layers
            self.layers = [NeuronLayer( self.n_outputs,self.n_inputs )]

    def get_weights(self):
        weights = []

        for layer in self.layers:
            for neuron in layer.neurons:
                weights += neuron.weights

        return weights

    @property
    def n_weights(self):
        if not self._n_weights:
            self._n_weights = 0
            for layer in self.layers:
                for neuron in layer.neurons:
                    self._n_weights += neuron.n_inputs+1 # +1 for bias weight
        return self._n_weights

    def set_weights(self, weights ):
        assert len(weights)==self.n_weights, "Incorrect amount of weights."

        stop = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                start, stop = stop, stop+(neuron.n_inputs+1)
                neuron.set_weights( weights[start:stop] )
        return self

    def update(self, inputs ):
        assert len(inputs)==self.n_inputs, "Incorrect amount of inputs."

        for layer in self.layers:
            outputs = []
            for neuron in layer.neurons:
                tot = neuron.sum(inputs) + neuron.weights[-1]*BIAS
                outputs.append( self.sigmoid(tot) )
            inputs = outputs   
        return outputs

    def sigmoid(self, activation,response=1 ):
            # the activation function
        try:
            return 1/(1+math.e**(-activation/response))
        except OverflowError:
            return float("inf")

    def __str__(self):
        return '\n'.join([str(i+1)+' '+str(layer) for i,layer in enumerate(self.layers)])
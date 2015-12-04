"""capstone.py"""

# Datasets obtained from Kaggle (https://www.kaggle.com/c/digit-recognizer/data)
# Which classifier is the "best" at predicting/recognizing hand-written digits?
# Classifiers to test: SVM, Random Forest, KNN

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

# Read data and transform into an array
df = pd.read_csv('/Users/eddiejew/thinkful/thinkfulprojects/Unit5/train.csv')
array = df.as_matrix()
print array.shape
# (42000, 785)

# Determine the best number of reduced dimensions using PCA
pca = PCA()
pca.fit(array[:, 1:])
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of dimensions')
plt.ylabel('cumulative explained variance');
plt.show()
plt.clf()

# Reduce dimensions using PCA
# Based on the graph, reducing data dimensions from 784 (excluding label)
# to 100 seems to be reasonable

pca = PCA(n_components = 100)
reduced_arr = pca.fit_transform(array[:, 1:]) # this would be X
print reduced_arr.shape
# (42000, 100)

# Split data into test and training data
# Split complete sample
c_train_data, c_test_data, c_train_label, c_test_label = train_test_split(reduced_arr, array[:,0],train_size = 0.5)
print c_train_data.shape
print c_train_label.shape
print c_train_label

# Split c_train_data and c_train_label into train_data and train_label to
# decrease number of samples to speed up process
train_data, test_data, train_label, test_label = train_test_split(c_train_data, c_train_label, train_size = 0.5)
print train_data.shape
print train_label.shape
print train_label
# c_train_data.shape (21000, 100)
# c_train_label.shape (2100,)
# c_train_label [1 5 3 ... 7 7 0]
# train_data.shape (10500, 100)
# train_label.shape (10500,)
# train_label [7 5 3 ... 5 3 2]

################################################################################################################################################

# Comparing Different Classifiers
# First, SVM (Support Vector Machine)
clf = SVC(kernel = 'poly')
print clf.fit(train_data, train_label)
pred_label = clf.predict(test_data)
# SVM (C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#     kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)

# Validation using 'accuracy score' and 'cross validation'
# Accuracy score with default SVM parameters and 'poly' kernel
print 'With default SVC parameters and "poly" kernel, SVM model accuracy score is {0}' .format(accuracy_score(test_label, pred_label))
# With default SVM parameters and "poly" kernel, SVM model accuracy is 0.9742

# Cross validation using the entire sample (42,000)
# cv1 = cross_val_score(SVC(kernel = 'poly'), reduced_arr, array[:,0], cv = 3)
# print cv1.mean()

# Cross validation using smaller sample (21,000)
cv2 = cross_val_score(SVC(kernel = 'poly'), train_data, train_label, cv = 5)
print cv2.mean()
# cv2 = 0.9680

# View where the errors occur using a confusion matrix
print confusion_matrix(test_label, pred_label)

#[[ 940    0    2    2    1    3    5    0    1    1]
# [   0 1231    6    2    3    0    0    2    0    0]
# [   2    0  995    4    4    2    0    9    9    1]
# [   0    0   14 1033    0   10    0    5   15    4]
# [   3    4    1    0 1054    0    2    1    2   21]
# [   0    1    4   11    2  869    6    1    6    3]
# [   1    2    3    0    2    3 1022    0    2    0]
# [   0    4    9    0    5    1    0 1121    0   15]
# [   0    0    5    7    2   11    0    2  959    7]
# [   4    3    1    7   19    5    0    7    5  969]]

plt.imshow(np.log(confusion_matrix(test_label, pred_label)),
	       cmap='coolwarm', interpolation='nearest')
plt.grid(False)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

################################################################################################################################################

# Random Forest + Extra Trees Classifiers
# Fit training data and predict test data using default parameters

model1 = RandomForestClassifier().fit(train_data, train_label)
test_pred1 = model1.predict(test_data)
print 'With default parameters, random forest gives {0} accuracy' .format(accuracy_score(test_label, test_pred1))
# Random Forest accuracy = 0.8530

xtra1 = ExtraTreesClassifier().fit(train_data, train_label)
xtra_pred1 = xtra1.predict(test_data)
print 'With default parameters, extra trees give {0} accuracy' .format(accuracy_score(test_label, xtra_pred1))
# Extra Trees accuracy = 0.8224

# Adjusting parameters to find optimal accuracy score
# Accuracy score with increasing number of max_depth
print 'Random Forest Classifier : \n'
for depth in [3, 5, 10, 15, 20, 25, 30]:
	forest = RandomForestClassifier(max_depth = depth)
	forest.fit(train_data, train_label)
	forest_pred = forest.predict(test_data)
	print 'Max depth of {0}, accuracy score = {1}' .format(depth, (accuracy_score(test_label, forest_pred)))
# Max depth of 3, accuracy score = 0.559428571429
# Max depth of 5, accuracy score = 0.746857142857
# Max depth of 10, accuracy score = 0.838380952381
# Max depth of 15, accuracy score = 0.852761904762
# Max depth of 20, accuracy score = 0.850571428571
# Max depth of 25, accuracy score = 0.851523809524
# Max depth of 30, accuracy score = 0.851714285714

print '\nExtra Trees Classifier : \n'
for depth in [3, 5, 10, 15, 20, 25, 30]:
	forest = ExtraTreesClassifier(max_depth = depth)
	forest.fit(train_data, train_label)
	forest_pred = forest.predict(test_data)
	print 'Max depth of {0}, accuracy score = {1}' .format(depth, (accuracy_score(test_label, forest_pred)))
# Max depth of 3, accuracy score = 0.503333333333
# Max depth of 5, accuracy score = 0.638666666667
# Max depth of 10, accuracy score = 0.796
# Max depth of 15, accuracy score = 0.827619047619
# Max depth of 20, accuracy score = 0.832952380952
# Max depth of 25, accuracy score = 0.821047619048
# Max depth of 30, accuracy score = 0.819238095238

# Random Forest CLF highest accuracy score in the adjusted max_depth is about the same 
# as the unadjusted max_depth, which is about 85%. 
# Extra Trees CLF highest accuracy score in the adjusted max_depth waivers around the 
# same as the unadjusted max_depth, which is about 82%

# Accuracy score using max_depth = 15 and increasing number of trees using 
# Random Forest Classifier
t0 = time.time()

x = range(10, 130, 10)
y = []
for tree in x:
	forest = RandomForestClassifier(max_depth=15, n_estimators=tree)
	forest.fit(train_data, train_label)
	forest_pred = forest.predict(test_data)
	print '{0} trees, accuracy is {1}' .format(tree, (accuracy_score(test_label, forest_pred)))
	y.append((accuracy_score(test_label, forest_pred)))

plt.plot(x, y, color='green', linestyle='solid', marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy Score (%)')
plt.show()

t1 = time.time()
print 'Speed for Random Forest Classifier is {0} seconds' .format(t1 - t0)
# Speed for Random Forest Classifier is 120.76 seconds

model2 = RandomForestClassifier(max_depth=15, n_estimators=120)
model2.fit(train_data, train_label)
test_pred2 = model2.predict(test_data)
print 'With adjusted parameters, Random Forest accuracy is {0}' .format(accuracy_score(test_label, test_pred2))
# With adjusted parameters, Random Forest accuracy is 0.9252

# Accuracy score using max_depth = 20 and increasing number of trees using 
# Extra Trees Classifier
# Unlike Random Forest, accuracy of Extra Trees appears to peak around max_depth=20
t0 = time.time()

x = range(10, 130, 10)
y = []
for tree in x:
	forest = ExtraTreesClassifier(max_depth=20, n_estimators=tree)
	forest.fit(train_data, train_label)
	forest_pred = forest.predict(test_data)
	print '{0} trees, accuracy is {1}' .format(tree, (accuracy_score(test_label, forest_pred)))
	y.append((accuracy_score(test_label, forest_pred)))

plt.plot(x, y, color='green', linestyle='solid', marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy score (%)')
plt.show()

t1 = time.time()
print 'Speed for Extra Trees Classifier is {0} seconds' .format(t1-t0)
# Speed for Extra Trees Classifier is 41.24 seconds

xtra2 = ExtraTreesClassifier(max_depth=20, n_estimators=120)
xtra2.fit(train_data, train_label)
xtra_pred2 = xtra2.predict(test_data)
print 'With adjusted parameters, Extra Trees accuracy is {0}' .format(accuracy_score(test_label, xtra_pred2))
# With adjusted parameters, Extra Trees accuracy is 0.9361

# Cross Validation
# Perform cross validation using complete/entire sample (42,000)
cv1 = cross_val_score(RandomForestClassifier(max_depth=15, n_estimators=120), reduced_arr, array[:,0], cv = 5)
print 'Mean of cross validation for Random Forest Classifier is {0}' .format(cv1.mean())
# Mean of cross validation for Random Forest Classifier is 0.9391

cv2 = cross_val_score(ExtraTreesClassifier(max_depth=20, n_estimators=120), reduced_arr, array[:,0], cv = 5)
print 'Mean of cross validation for Extra Trees Classifier is {0}' .format(cv2.mean())
# Mean of cross vlidation for Extra Trees Classifier is 0.9495

# Perform cross validation using half of the sample (21,000)
cv1 = cross_val_score(RandomForestClassifier(max_depth=15, n_estimators=120), train_data, train_label, cv = 5)
print 'Mean of cross validation for Random Forest Classifier is {0}' .format(cv1.mean())
# Mean of cross validation for Random Forest Classifier is 0.9182

cv2 = cross_val_score(ExtraTreesClassifier(max_depth=20, n_estimators=120), train_data, train_label, cv = 5)
print 'Mean of cross validation for Extra Trees Classifier is {0}' .format(cv2.mean())
# Mean of cross validation for Extra Trees Classifier is 0.9287

# View where the erros are using a confusion matrix
print 'Confusion matrix of default Random Forest \n{0} \n\n \
Confusion matrix of adjusted Random Forest \n{1} \n\n \
Confusion matrix of Extra Trees Classifier \n{2}' .format(confusion_matrix(test_label, test_pred1), \
	                                                      confusion_matrix(test_label, test_pred2), \
	                                                      confusion_matrix(test_label, xtra_pred2))
# Confusion matrix of default Random Forest 
# [[ 979    1    7   12    6   11   21    3   12    0]
#  [   5 1104    7   12    0    9    4    1    2    0]
#  [  25   11  911   28   12    6   12   13   20    7]
#  [  20    5   47  911    7   38    6    9   29   10]
#  [   9   10   25   18  876    4   17   14   13   81]
#  [  34    2   20   83   27  716   21    5   29   19]
#  [  18    3   39    4    7   13  907    0    7    1]
#  [   9   18   26   10   32    9    4  945    3   32]
#  [  24    9   48   77   16   47   17   12  732   19]
#  [   8    7   16   21   92   13    3   61    9  836]] 

# Confusion matrix of adjusted Random Forest 
# [[1015    0    6    6    3    3    9    0   10    0]
#  [   1 1113    5   12    0    3    2    0    6    2]
#  [   5    3  954   24   12    1    6   12   28    0]
#  [   2    2   20  990    2   21    5    8   23    9]
#  [   3   11    6    2  976    0   13    4    8   44]
#  [   7    1    5   47    8  849   22    1   10    6]
#  [   9    1    5    0    2   11  968    0    3    0]
#  [   4   10   20    1   12    3    1 1009    4   24]
#  [   2    7   11   49    7   27    3    6  874   15]
#  [   5    5    6   17   37    6    1   34   11  944]] 

# Confusion matrix of Extra Trees Classifier 
# [[1024    0    4    3    1    2   10    0    7    1]
#  [   0 1122    2    5    0    2    3    4    5    1]
#  [   9    6  968   16   13    1    4   14   14    0]
#  [   1    3   16 1001    2   15    3   10   24    7]
#  [   1    6    7    0  975    1   21    4    5   47]
#  [   9    2    2   51    6  849   22    0   11    4]
#  [   9    2    5    0    3    8  968    0    4    0]
#  [   3   10   13    1    8    0    0 1032    2   19]
#  [   4    6    9   44    8   28    6    5  879   12]
#  [   7    6    4   15   30    3    1   38    7  955]]


################################################################################################################################################

# K-Nearest Neighbors (KNN)
model3 = KNeighborsClassifier(n_neighbors=1)
model3.fit(train_data, train_label)
model3_pred = model3.predict(test_data)
accuracy_score(test_label, model3_pred)
print 'Accuracy of KNN is {0}' .format(accuracy_score(test_label, model3_pred))
# Accuracy of KNN is 0.9557

neighbors = range(1, 10)

for n in neighbors:
	clf = KNeighborsClassifier(n_neighbors=n)
	clf.fit(train_data, train_label)
	knn_pred = clf.predict(test_data)
	plt.plot(n, accuracy_score(test_label, knn_pred), color = 'purple', marker = 'o')

plt.xlabel('Neighbors')
plt.ylabel('Accuracy Score')
plt.show()

# Neighbors of 1 seems to have the best accuracy for this model

# Cross Validation
cv3 = cross_val_score(KNeighborsClassifier(n_neighbors=1), train_data, train_label, cv = 5)
cv3.mean()
print 'Mean of cross validation for KNN is {0}' .format(cv3.mean())
# Mean of cross validation for KNN is 0.9503

################################################################################################################################################

# Discussion
# From the results it seems that SVM and KNN are the most accurate classifiers in determining 
# handwritten digits from images. Initially I had assumed Random Forest and Extra Trees would 
# be similar in accuracy compared to SVM and KNN, although the results indicated this is certaily 
# incorrect at worst, and a stretch at best.
# It isn't too surprising that SVM and KNN are better at classifying images compared to the decision 
# tree calssifiers - image recognition programs relies on SVM to identify key features.
# Thinking about the code, we perhaps could have used K-Means as another method of classifying 
# handwritten digits. 
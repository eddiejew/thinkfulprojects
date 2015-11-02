"""random_forest.py"""

import pandas as pd
import sklearn.ensemble as ske
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt

# load the feature names (as a dataframe) and convert to a list
feature_df = pd.read_csv("samsungdata.csv",header=0)
feature_list = list(feature_df.values.flatten())

# load the x_train set and use feature_list to label its columns
xtrain = pd.read_csv("x_train.csv",header=None,names=feature_list)
# load the y_train set
ytrain = pd.read_csv("y_train.csv",header=None,names=["activity"])

# set up a 500 estimator random forest
clf = ske.RandomForestClassifier(n_estimators=500, oob_score=True)
# fit the random forest to the x_train and y_train data
clf = clf.fit(xtrain,ytrain.activity)
# get the oob_score
# oob_score: score of the training dataset obtained using an out-of-bag estimate
print clf.oob_score
# we get 0.9919

# get the importance scores
importances = np.array(clf.feature_importances_)
# get the top 10 indices of features
top10_indices = importances.argsort()[-10:][::-1]
print top10_indices
# we get 56, 40, 52, 58, 49, 558, 53, 42, 50, 41: higher importance scores indicate more important features
# which features are these?
top10_features = [feature_list[x] for x in top10_indices]
print top10_features
# tGravityAcc_energy_X, tGravityAcc_Mean_X, tGravityAcc_min_X, tGravityAcc_energy_Z,
# tGravityAcc_max_X, angle_X_gravityMean, tGravityAcc_min_Y, tGravityAcc_Mean_Z,
# tGravityAcc_max_Y, tGravityAcc_Mean_Y
# what are these scores?
top10_scores = [importances[x] for x in top10_indices]
print top10_scores
# 0.03275m 0.03063, 0.03034, 0.02846, 0.02786, 0.02531. 0.01754, 0.01653, 0.01589

# find accuracy scores:
# first we need to load the training and validation sets
xtest = pd.read_csv("x_test.csv",header=None,names=feature_list)
xval = pd.read_csv("x_validate.csv",header=None,names=feature_list)
ytest = pd.read_csv("y_test.csv",header=None,names=["activity"])
yval = pd.read_csv("y_validate.csv",header=None,names=["activity"])
# load the predictions (as arrays)
test_pred = clf.preict(xtest)
val_pred = clf.predict(xval)
# now we can get the scores
print skm.accuracy_score(ytest, test_pred)
# accuracy score for test set: 0.90923
print skm.accuracy_score(yval, val_pred)
# accuracy score for validation set: 0.81780

# find precision, recall, and F1 scores on the test set
print skm.precision_score(ytest, test_pred)
# precision score 0.81887
print skm.recall_score(ytest, test_pred)
# 0.80923
print skm.f1_score(ytest, test_pred)
# 0.80710

# plot confusion matrix
cm = skm.confusion_matrix(ytest, test_pred)
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

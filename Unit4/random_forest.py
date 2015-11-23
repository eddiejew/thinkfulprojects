"""random_forest.py"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import sklearn.ensemble as ske
import sklearn.metrics as skm
import matplotlib.pyplot as plt

# read feature names
feature_names = open("./features_clean.txt").read().splitlines()
col_names = [f.split(" ")[1] for f in feature_names]

# create data frames from combined data
df_X = pd.read_csv("./X_all.txt", sep=" *", engine="python", header=None, names=col_names)
df_y = pd.read_csv("./y_all.txt", header=None, names=["activity"])
df_sub = pd.read_csv("./subject_all.txt", header=None, names=["subject"])

# choose training, testing, and validation data - Scikit-learn doesn't 
# natively support pandas. So we'll need to convert training and testing
# sets into matrices for feeding to classifier algorithms

training_set = df_sub["subject"] >= 27
test_set = df_sub["subject"] <= 6
cv_set = (df_sub["subject"] >= 21) & (df_sub["subject"] < 27)

X_train = df_X[training_set].as_matrix()
y_train = df_y[training_set].as_matrix().squeeze()

X_test = df_X[test_set].as_matrix()
y_test = df_y[test_set].as_matrix().squeeze()

X_cv = df_X[cv_set].as_matrix()
y_cv = df_y[cv_set].as_matrix().squeeze()

# set up a 500 estimator random random_forest
clf = RandomForestClassifier(n_estimators=500, oob_score=True)
clf.fit(X_train, y_train) # fit estimator to the X_train + y_train data
print clf.oob_score_
# oob_score is 0.992592592593

# get the importance scores
importances = np.array(clf.feature_importances_)
# get the top 10 indices of features
top10_indices = importances.argsort()[-10:][::-1]
print top10_indices
# we get 56, 40, 59, 52, 58, 558, 41, 50, 51, 42
# which features are those?
top10_features = [feature_names[x] for x in top10_indices]
print top10_features
# we get '50 tGravityAccmaxX', '57 tGravityAccenergyX', '41 tGravityAccmeanX', 
# '53 tGravityAccminX', '559 angleX_gravityMean', '59 tGravityAccenergyZ', 
# '52 tGravityAccmaxZ', '42 tGravityAccmeanY', '54 tGravityAccminY', 
# '43 tGravityAccmeanZ'
# what are their scores?
top10_scores = [importances[x] for x in top10_indices]
print top10_scores
# [0.034478784847599558, 0.032912251649066554, 0.032443059888468948, 0.030203796386248014, 
# 0.029520767611458224, 0.029366880635201313, 0.017558907834240297, 0.016996435883561273, 
# 0.01480349368966152, 0.014500245838005512]

# find the accuracy scores
# first get the predictions (as arrays)
test_pred = clf.predict(X_test)
val_pred = clf.predict(X_cv)
# now we can get the scores
print skm.accuracy_score(y_test, test_pred)
# accuracy score for test set: 0.8056
print skm.accuracy_score(y_cv, val_pred)
# accuary score for validation set: 0.8165

# find precision, recall, and F1 scores on the test set
print skm.precision_score(y_test, test_pred)
# precision score: 0.8189
print skm.recall_score(y_test, test_pred)
# recall score: 0.8092
print skm.f1_score(y_test, test_pred)
# F1 score: 0.8071

# plot confusion matrix
cm = skm.confusion_matrix(y_test, test_pred)
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
"""cv.py"""

from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics

iris = datasets.load_iris()
Xtrain, Xtest, ytrain, ytest = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# number of points in the test set
print 'Xtest number of points: ', len(Xtest)
# number of points in the train set
print 'Xtrain number of points: ', len(Xtrain)

# fit linear svm
print 'Linear SVM'
clf = svm.SVC(kernel='linear', C=1).fit(Xtrain, ytrain)
# score
print clf.score(Xtest, ytest)

# 5-fold cross-validation
print 'Cross-validation'
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
print 'Cross val scores: ', scores
print 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() / 2)

print 'Using F1 scores'
f1scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5, score_func=metrics.f1_score)
print "F1 scores: ", f1scores
print "Mean: ", f1scores.mean()
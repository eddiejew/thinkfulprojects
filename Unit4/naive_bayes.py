"""naive_bayes.py"""

# using ideal_weight.csv

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.naive_bayes as skn

# read in the data
wt = pd.read_csv("ideal_weight.csv",header=0,names=["id","sex","actual","ideal","diff"])

#remove single quotes from the sex column - use a map function to remove the single quotes
wt['sex'] = wt['sex'].map(lambda x: x.replace("'",""))

# plot histograms of actual and ideal weights
plt.figure()
plt.hist([wt['actual'], wt['ideal']], bins = 30, histtype='bar', stacked=True)
plt.title('Actual + Ideal Weights')
plt.xlabel('Weight (lbs)')
plt.ylabel('Frequency')
plt.show()

# plot the histogram of difference in weight
plt.figure()
plt.hist(wt['diff'], bins=30, histtype='bar')
plt.title('Difference in Weights')
plt.xlabel('Difference (lbs)')
plt.ylabel('Frequency')
plt.show()

# map sex to a categorical variable
wt['sex'] = pd.Categorical(wt['sex'])

# Are there more women or men in the dataset?
# print the length of the Female and Mal columns to find our answer
print len(wt['sex'][wt['sex']=='Female'])
print len(wt['sex'][wt['sex']=='Male'])
# There are 119 women and 63 men
# Curious as to why there are more women than men in the dataset

# fit a Naive Bayes classifier - actual, ideal, and diff to sex
# using a Gaussian Naive Bayes classifier
# assuming the likelihood of the features is normally distributed (Gaussian)
x = wt[['actual', 'ideal', 'diff']]
y = wt['sex']
nb = skn.GaussianNB()
nb = nb.fit(x, y)
prediction = nb.predict(x) # find out how many points were mislabeled, figure out how many data points were not used in the prediction
print "Out of %d data points, %d were mislabeled." %(x.shape[0], (wt['sex'] != prediction).sum())
# Out of 182 data points, 14 were mislabeled

# predict the sex for an actual weight of 145, an ideal weight of 160, and a diff of -15
data = { 'actual' : 145, 'ideal' : 160, 'diff' : -15}
test = pd.DataFrame(data=data, index=[1])
prediction = nb.predict(test)
print prediction
# parameters are predicted to be male

# predict the sex for an actual weight of 160, an ideal weight of 145, and a diff of +15
data = { 'actual' : 160, 'idea' : 145, 'diff' : 15}
test = pd.DataFrame(data=data, index=[1])
prediction = nb.predict(test)
print prediction
# parameters are predicted to be male
# why did both predictions return male?
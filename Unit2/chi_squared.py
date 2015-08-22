# chi_squared.py

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import collections

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

#remove NAs
loansData.dropna(inplace=True)

freq = collections.Counter(loansData['Open.CREDIT.Lines'])

k = 0
for i in freq:
		k = k+1

print "The number of unique elements in freq is", k
print "The most frequent number of open credit lines is", freq.most_common(1)[0][0]

plt.figure()
plt.bar(freq.keys(), freq.values(), width=1)
plt.show()

#perform chi-squared test
chi, p = stats.chisquare(freq.values())

print "The chi-squared test and p-values are", chi, p
print "We can reject the null hypothesis that all open credit lines are uniformly distributed"
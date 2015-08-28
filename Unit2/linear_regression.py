"""linear_regression.py"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# remove '%' from Interest.Rate column
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))

# remove ' months' from Loan.Length column
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))

# convert FICO scores into a numerical value, and save it in a new column titled 'FICO.Score'
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: (float(x.split('-')[0])))

# plot a histogram of FICO scores
plt.figure()
p = loansData['FICO.Score'].hist()
plt.show()

# create a scatterplot matrix

a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# dependent variable
y = np.matrix(intrate).transpose()
# independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()
x = np.column_stack([x1,x2])

X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

f.summary()

print 'Coefficients: ', f.params[0:2]
print 'Intercept: ', f.params[2]
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared

# P-values are <= 0.05
# R-squared = 0.6566, a good fit
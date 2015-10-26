"""crossvalidation.py"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cross_validation import KFold

# read in the data
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
# drop the null rows
loansData.dropna(inplace=True)

# remove '%' from Interest.Rate column
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))

# remove ' months' from Loan.Length column
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))

# convert FICO scores into a numerical value, and save it in a new column titled 'FICO.Score'
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: (float(x.split('-')[0])))

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# break data into 10 segments using KFold
kf = KFold(len(loansData),n_folds=10, shuffle=True)

output = []
for train, test in kf:
	    y = np.matrix(intrate[test]).transpose()
	    x1 = np.matrix(fico[train]).transpose()
	    x2 = np.matrix(loanamt[train]).transpose()
	    x = np.column_stack([x1,x2])
	    X = sm.add_constant(x)
	    model = sm.OLS(y,X)
	    f = model.fit()

print output
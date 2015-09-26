"""multivariate.py"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# import loans data
loansData = pd.read_csv('/Users/eddiejew/thinkful/LoanStats3c.csv', low_memory=False)

# remove rows with null values
loansData.dropna(inplace=True)

# remove '%' from InterestRate column
loansData['int_rate'].map(lambda x: round(float(x.rstrip('%')), 4))

# add intercept in data
loansData['StatsModels.Intercept'] = loansData['int_rate'].map(lambda x: float(1.0)) # = loansData.Intercept 1

# add home_ownership to data
loansData['home_ownership'] = pd.Categorical(loansData.home_ownershihp).labels

# fit model
X = sm.add_constant(loansData['annual_inc']) # X is a constant - uses annual_inc as a constant, regression is on annual income
est = sm.OLS(loansData['int_rate'], X).fit() # fitting data: telling the data how to use the model

est.summary()

est = smf.OLS(loansData['int_rate'], X + loansData['home_ownership'].fit() # x ~ a + b + c => variable x is int_rate ~ annual_inc + home_ownership => mod.fit() *formula method

est.summary()
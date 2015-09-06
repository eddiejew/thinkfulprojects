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
loansData['StatsModels.Intercept'] = loansData['int_rate'].map(lambda x: float(1.0))

# add home_ownership to data
loansData['home_ownership'] = pd.Categorical(loansData.home_ownershihp).labels

# fit model
X = sm.add_constant(loansData['annual_inc'])
est = sm.OLS(loansData['int_rate'], X).fit()

est.summary()

est = smf.OLS(loansData['int_rate'], X + loansData['home_ownership'].fit()

est.summary()
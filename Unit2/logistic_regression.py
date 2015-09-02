"""logistic_regression.py"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import math as math

import clean_data
loansData = clean_data.clean_up_data()

# create a column for <12% interest rate
loansData['IR_TF'] = loansData['Interest.Rate'].map(lambda x: True if x < 0.12 else False)

# create a column for statsmodels intercept == 1.0
loansData['StatsModels.Intercept'] = loansData['Interest.Rate'].map(lambda x: 1.0)

# list of the column names of independent variables, including the intercept
ind_vars = ['Amount.Requested', 'FICO.Score', 'StatsModels.Intercept']

# logistic regression model:
logit = sm.Logit(loansData['IR_TF'], loansData[ind_vars])
# fit the model:
result = logit.fit()
# return the fitted coefficients from the results:
coeff = result.params
print coeff[0] # Amount.Requested
print coeff[1] # FICO.Score
print coeff[2] # StatsModels.Intercept

# logistic_function
def logistic_function(FICO_Score, Amount_Requested):
	logit = sm.Logit(loansData['IR_TF'], loansData[ind_vars])
	result = logit.fit()
	coeff = result.params
	p = 1/(1 + math.e**(coeff[2] + coeff[1]*FICO_Score + coeff[0]*Amount_Requested))
	return p

# determine probability we can obtain a loan at <=12% interest for $10K with a FICO score of 720
print logistic_function(720, 10000)


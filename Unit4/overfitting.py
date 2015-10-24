"""overfitting.py"""

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt

# set seed for reproducible results
np.random.seed(414)

# generate toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Linear Fit
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()

# Quadratic Fit
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()

# Cubic fit
poly_3 = smf.ols(formula='y ~ 1 + X + I(X**2) + I(X**3)', data=train_df).fit

# plot toy data
plt.plot(train_X, train_y, 'o')

# find MSE of liner fit
predicted_y = poly_1.predict(test_df['X'])[700:]
plt.plot(test_df['X'], test_df['y'], 'o')
plt.plot(test_df['X'], predicted_y, 'r')
mse = sum((predicted_y - test_df['y'])**2) / (len(predicted_y))
print "MSE = %s" %mse
# MSE = 6.548

resid = predicted_y - test_df['y']
plt.plot(test_df['X'], resid)

# find MSE of quadratic fit
predicted_y = poly_2.predict(test_df['X'])[700:]
plt.plot(test_df['X'], test_df['y'], 'o')
plt.plot(test_df['X'], predicted_y, 'r')
mse = sum((predicted_y - test_df['y'])**2) / (len(predicted_y))
print "MSE = %s" %mse
# MSE = 7.987

resid = predicted_y - test_df['y']
plt.plot(test_df['X'], resid)

# find MSE of cubic fit
predicted_y = poly_3.predict(test_df['X'])[700:]
plt.plot(test_df['X'], test_df['y'], 'o')
plt.plot(test_df['X'], predicted_y, 'r')
mse = sum((predicted_y - test_df['y'])**2) / (len(predicted_y))
print "MSE = %s" %mse
# MSE = 

resid = predicted_y - test_df['y']
plt.plot(test_df['X'], resid)
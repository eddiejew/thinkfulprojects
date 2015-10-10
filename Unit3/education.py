"""education.py"""

import sqlite3 as lite
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# connect to education.db
con = lite.connect('education.db')

# use data we need: year from education_pop and _1999 - _2010 from gdp_pop
# some country names don't match up, use inner join
df = pd.read_sql_query("SELECT year, _1999, _2000, _2001, _2002, _2003, _2004, _2005, _2006, _2007, _2008, _2009, _2010 FROM education INNER JOIN gdp WHERE country=country",con)
# some rows contain all NaN
df = df.dropna()

# separate the year and gdp data so we can average gdp data accross columns
edu = df['year']
gdp = df[['_1999','_2000','_2001','_2002','_2003','_2004','_2005','_2006','_2007','_2008','_2009','_2010']]
gdp = gdp.mean(axis=1)

# build the OLS model
y = np.matrix(edu.astype(float)).transpose()
x = np.matrix(gdp.astype(float)).transpose()
# take the log of the x variable to scale to y
x = np.log(x)

# correlation
corr = stats.pearsonr(y,x)
print corr
# corr = 

# add constant to x for OLS
X = sm.add_constant(x)
# make the model
model = sm.OLS(y,X)
f = model.fit()

print f.summary()
# r-squared = 
# slope = 
# some relationship between gdp and length of education - not very strong, but they are related
# scatter plot also shows an increasing linear trend
"""cleandecisiontrees.py"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections

# read the data file
samsungdata = pd.read_csv('/Users/eddiejew/thinkful/thinkfulprojects/Unit4/UCI_HAR/samsungdata.csv')

# clean column names
header = list(samsungdata.columns.values)
print 'Unclean column names: '
print header
print

clean_header = map(lambda x: x.replace('()', '').replace('-', '') \
	                    .replace(',', '_').replace('Body', '') \
	                    .replace('Mag', '').replace('mean', 'Mean') \
	                    .replace('std', 'STD'), header)
print 'Clean column names'
print header
print

samsungdata = pd.read_csv('/Users/eddiejew/thinkful/thinkfulprojects/Unit4/UCI_HAR/samsungdata.csv', names = clean_header, header = 0)
print samsungdata.columns.values

# make activity a categorical variable + find out what the categorical variables are
activity = samsungdata['activity'].dropna()
categories = collections.Counter(samsungdata['activity'])
print 'Activity categories: '
print categories
print

# plot a histogram of 'Body Acceleration Magnitue'
body_acc_mag = samsungdata['fAccMean'].dropna()
# print boddy_acc_mag
plt.hist(body_acc_mag, bins = 20)
plt.title('Body Acceleration Magnitue')
plt.show()
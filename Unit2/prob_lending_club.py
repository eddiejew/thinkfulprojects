"""prob_lending_club.py"""

import matplotlib.pyplot as plt
import pandas as pd

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# remove rows with null values
loansData.dropna(inplace=True)

# generate a box plot of the loan amounts
loansData.boxplot(column='Amount.Requested')
plt.show()
plt.savefig("loansData_boxplot.png")

# generate a historgram of the loan amounts
loansData.hist(column='Amount.Requested')
plt.show()
plt.savefig("loansData_histogram.png")

import scipy.stats as stats

# genereate a QQ-plot
plt.figure()
graph = stats.probplot(loansData['Amount.Requested'], dist="norm", plot=plt)
plt.show()
plt.savefig("loansData_QQplot")

# Boxplot: median is 10,000; first quartile is around 6000, third quartile is around 17,000
# Histogram: graph is skewed right, most amounts requested are les than median, more easily visible here than in boxplot
# QQ-plot: graph shows what is indicated in histogram: there is a higher probability of a loan requested amount being lower than 10,000 rather than higher than 10,000
# Comparison to 'Amount.Funded.By.Investors':
#	investors seem more willing to fund amounts below the median of 10,000 (or only provide amounts below median of 10,000) rather than fund higher amounts
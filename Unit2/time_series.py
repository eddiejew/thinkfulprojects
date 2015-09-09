"""time_series.py"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('/Users/eddiejew/thinkful/LoanStats3c.csv', header=1, low_memory=False)

# converts string to datetime object in pandas:
df['issue_d_format'] = pd.to_datetime(df['issue_d'])
dfts = df.set_index('issue_d_format')
year_month_summary = dfts.groupby(lambda x: x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

# reindex loan by month by year as dataframe
lcdf = pd.DataFrame(loan_count_summary)
lcdf = lcdf.rename(columns={'issue_d': 'month_count'})

lcdf.plot()

sm.graphics.tsa.plot_acf(loan_count_summary)
sm.graphics.tsa.plot_pacf(loan_count_summary)

plt.figure()
p = sm.graphics.tsa.plot_acf(df['loan_amt'])
plt.set_size_inches = (18.5,10.5)
plt.show()
plt.savefig('loan_amnt_acf.png', dpi=100)

plt.figure2()
p2 = sm.graphics.tsa.plot_pacf(df['loan_amt'])
plt.set_size_inches = (18.5,10.5)
plt.show()
plt.savefig('loan_amnt_pacf.png', dpi=100)
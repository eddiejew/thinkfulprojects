"""issue_d.py"""

# creates a monthly time series of the loan counts by the issue date

import pandas as pd
import numpy as np

df = pd.read_csv('/Users/eddiejew/thinkful/LoanStats3c.csv', header=1, low_memory=False)

# converts string to datetime object in pandas:
df['issue_d_format'] = pd.to_datetime(df['issue_d'])
dfts = df.set_index('issue_d_format')
year_month_summary = dfts.groupby(lambda x: x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

return df
"""clean_data.py"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

def clean_up_data():
	loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
	loansData.dropna(inplace=True)

	# remove '%' from Interest.Rate and Debt.To.Income.Ratio column
	loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
	loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))

	# remove ' months' from Loan.Length column
	loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))

	# convert FICO scores into a numerical value, and save it in a new column titled 'FICO.Score'
	loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: (float(x.split('-')[0])))

	# confirm values are in float or int format instead of strings
	loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: float(x))
	loansData['Amount.Requested'] = loansData['Amount.Requested'].map(lambda x: int(x))

	return loansData
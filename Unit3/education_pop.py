"""education_pop.py"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
import sqlite3 as lite
import matplotlib.pyplot as plt

# retrieve webpage
url = "http://web.archive.org/web/20110514112442/http://unstats.un.org/unsd/demographic/products/socind/education.htm"
r = requests.get(url)

# pass the webpage to BeautifulSoup
soup = BeautifulSoup(r.content)

# drill down to relevant data
data = soup('table')[6].find_all('tr')
data = data[3]
data = data('table')[0]
# data rows
data = data('tr')

# get data values from each row
datarows = []
for i in range(4, len(data)):
	tags = data[i]
	values = [tag.string for tag in tags('td')]
	datarows.append(list(values[i] for i in [0, 1, 7, 10]))

# convert the list of lists to a list of tuples
datarows = [tuple(element) for element in datarows]

# store data in SQLite
# connect to database
con = lite.connect('education.db')
cur = con.cursor()

# create and fill table
with con:
	# clear table if it exists
	cur.execute("DROP TABLE IF EXISTS undata")
	# create table
	cur.execute("CREATE TABLE undata (country TEXT, year INT, men INT, women INT);")
	# fill values
	cur.executemany('INSERT INTO undata (country, year, men, women) VALUES (?, ?, ?, ?)', datarows)

# put data into a dataframe
df = pd.DataFrame(datarows, columns=['country','year','men','women'])
df['year'] = pd.PeriodIndex(df['year'], freq='A-DEC')
df = df.set_index('year', drop=False)
df[['men','women']] = df[['men','women']].astype(int)

print 'For the men and women school life expectancies'
print 'The means are:'
print df[['men','women']].mean()
print 'The medians are:'
print df[['men','women']].median()
print 'and the variances are:'
print df[['men','women']].var()

plt.figure()
df[['men', 'women']].hist()
plt.draw()
plt.savefig('histograms.png')

plt.figure()
df[['men', 'women']].boxplot()
plt.draw()
plt.savefig('boxplots.png')

# the mean for men and women is similar, histogram shows a roughly symmetrical distribution
# the median is higher for women than men, and women have a larger variance than men
# this may suggest some women, if they can continue their eduction, will - while some countries may provide little opportunity for women to continue their education
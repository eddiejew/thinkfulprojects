"""weather_api.py"""

import requests
import sqlite3 as lite
import datetime

my_api_key = '64c420e14c39bed2edd6b570303716d9'
my_api_url = 'https://api.forecast.io/forecast/' + my_api_key

#dictionary of 5 cities
cities = {  "Atlanta": '33.762909,-84.422675',
			"Denver": '39.761850,-104.881105',
			"Miami": '25.775163,-80.208615',
			"Nashville": '36.171800,-86.785002',
			"Seattle": '47.620499,-122.350876' 
		}

# set the time for "now" - the time the code is executed
# this makes the reference time the same for all queries
now = datetime.datetime.now()

# set up the datebase
con = lite.connect('weather.db')
cur = con.cursor()

with con:
	cur.execute('DROP TABLE IF EXISTS temp')
	cur.execute('CREATE TABLE IF NOT EXISTS temp ( day_recorded INT, Atlanta REAL, Denver REAL, Miami REAL, Nashville REAL, Seattle REAL )')

# day_of_query starts 30 days in the past (from now)
day_of_query = now - datetime.timedelta(days=30)

# fill in the days
with con:
	while day_of_query < now:
		cur.execute('INSERT INTO temp ( day_recorded ) VALUES (?)', (int(day_of_query.strftime('%s')),))
		day_of_query += datetime.timedelta(days=1)

# loop through the cities we've chosen and get data using the api
for k, v in cities.iteritems():
	day_of_query = now - datetime.timedelta(days=30)
	while day_of_query < now:
		r = requests.get(my_api_url + '/' + v + ',' + day_of_query.strftime('%Y-%m-%dT12:00:00'))
		with con:
			cur.execute('UPDATE temp SET ' + k + ' = ' + str(r.json()['daily']['data'][0]['temperatureMax']) + ' WHERE day_recorded = ' + day_of_query.strftime('%s'))
		day_of_query += datetime.timedelta(days=1)
con.close()

# r => requests an HTTP file
# response will include a header (status code, encoding (type of format), date + time, cookies)
# response also includes a payload (JSON, file, etc)
# 404 error: the API will not give you a reason why, we have to figure out ourselves
# API key may be incorrect => look up API documentation to see how API server works

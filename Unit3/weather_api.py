"""weather_api.py"""

import requests
import sqlite3 as lite
import datetime

my_api_key = '1c45b275474d4c380bcabf0b395d3b64'
my_api_url = 'https://api.forecast.io/forecast/' + my_api_key

#dictionary of 5 cities
cities = {	"New York": '40.663619,-73.938589',
						"Washington": '38.904103,-77.017229',
						"Austin": '30.303936,-97.754355'
						"Chicago": '41.837551,-87.681844'
						"San Francisco": '37.727239,-123.032229'
			}

# set the time for "now" - the time the code is executed
# this makes the reference time the same for all queries
now = datetime.datetime.now()

# set up the datebase
con = lite.connect('weather_api.db')
cur = con.cursor()

with con:
	cur.execute('CREATE TABLE IF NOT EXISTS temp ( day_recorded INT, New York REAL, Washington REAL, Austin REAL, Chicago REAL, San Francisco REAL )')

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
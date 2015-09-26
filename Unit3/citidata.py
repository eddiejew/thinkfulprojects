"""citidata.py"""

count = 0
while ( count < 60):
		count = count+1

# get the data
import requests
r = requests.get('http://www.citibikenyc.com/stations/json')

# clean the data
key_list = [] # unique list of keys for each station listing
for station in r.json()['stationBeanList']:
	for k in station.keys():
		if k not in key_list:
			key_list.append(k)

# convert data into a pandas DataFrame
from pandas.io.json import json_normalize
df = json_normalize(r.json()['stationBeanList'])

# Are there any test stations?
len(df[df['testStation'] == True].values)
# 0 test stations

# How many stations are 'InService' ? How many stations are 'NotInService' ?
len(df[df['statusValue'] == 'In Service'].values)
# 435 stations in service
len(df[df['statusValue'] == 'Not In Service'].values)
# 73 stations not in service

import numpy as np

# What is the mean number of bikes in a dock?
np.mean(df.availableBikes)
# 10.29 (about 10) bikes in a dock

# What is the median number of bikes in a dock?
np.median(df.availableBikes)
# 6.0 bikes in a dock

# How does this change if we remove the stations that aren't in service?
np.mean(df.availableBikes [df.statusValue == 'In Service'])
# 12.0 bikes in a dock in stations that are in service
np.median(df.availableBikes [df.statusValue == 'In Service'])
# 9.0 bikes in a dock in stations that are in service

import sqlite3 as lite

con = lite.connect('citi_bike.db')
cur = con.cursor()

# create a stationary reference table that will not change with time
with con:
	cur.execute('DROP TABLE IF EXISTS citibike_reference')
	cur.execute('CREATE TABLE citibike_reference (id INT PRIMARY KEY, totalDocks INT, city TEXT, altitude INT, stAddress2 TEXT, longitude NUMERIC, postalCode TEXT, testStation TEXT, stAddress1 TEXT, stationName TEXT, landMark TEXT, latitude NUMERIC, location TEXT );')

sql = "INSERT INTO citibike_reference (id, totalDocks, city, altitude, stAddress2, longitude, postalCode, testStation, stAddress1, stationName, landMark, latitude, location) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"

# for loop to populate values in the database
with con:
	cur = con.cursor()
	for station in r.json()['stationBeanList']:
		cur.execute(sql,(station['id'],staion['totalDocks'],station['city'],station['altitude'],station['stAddress2'],station['longitude'],station['postalCode'],station['testStation'],station['stAddress1'],station['stationName'],station['landMark'],station['latitude'],station['location']))

# extract the column station id and convert it into a data type for sqlite (text)
station_ids = df['id'].tolist() # creating a new list for station_ids

station_ids = ['_' + str(x) + ' INT' for x in station_ids] # converting station_ids to an sqlite usable format

# create the table: concatentating the string and joining all the station ids
with con:
	cur.execute("CREATE TABLE IF NOT EXISTS available_bikes ( execution_time INT, " + ", ".join(station_ids) + ");")
#execution_time: creates a table called availalbe_bikes, creates a column called execution_time, creates a column for the station_ids
#above code creates a column for each execution time, because number of stations does not change

import time
from dateutil.parser import parse
import collections

exec_time = parse(r.json()['executionTime'])
#creating a date_time object - python representation of a date+time: time is stored as an integer (# of seconds since Jan 1 1970)

with con:
	cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', (exec_time.strftime('%s'),))
	# date_time: print this out in a specific format == seconds on the date

id_bikes = collections.defaultdict(int)

# loop through the stations in the station list
for station in r.json()['stationBeanList']:
	id_bikes[station['id']] = station['availableBikes']

# iterate through the defaultdict to update the values in the database
with con:
	for k, v in id_bikes.iteritems():
		cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + str(v) + " WHERE execution_time = " + exec_time.strftime('%s') + ";")
# go to our table, find the row that corresponds to the current execution time (if not, create it) then change the value in the column for the 1 station we care about

time.sleep(60)

con.close()

# running 60 times == for i in xrange: 60, 
# iterations = 0, wall iterations <= 0 - 60
# use timelibrary: time.sleep(60)
# data might take only 1 second to get, time.sleep(59)

import pandas as pd
import sqlite3 as lite

con = lite.connect('citi_bike.db')
cur = con.cursor()

df = pd.read_sql_query("SELECT * FROM available_bikes ORDER BY execution_time",con,index_col='execution_time')

hour_change = collections.defaultdict(int)
for col in df.columns:
	station_vals = df[col].tolist()
	station_id = col[1:]
	station_change = 0
	for k, v in enumerate(station_vals):
		if k < len(station_vals) - 1:
			station_change += abs(station_vals[k] - station_vals[k+1])
	hour_change[int(station_id)] = station_change #convert the station id back to integer

def keyithmaxval(d):
	# create a list of the dictionary's keys and values
	v = list(d.values())
	k = list(d.keys())

	# return the key with the max value
	return k[v.index(max(v))]

# assign the max key to max_station
max_station = keywithmaxval(hour_change)

# query sqlite for reference information
cur.execute("SELECT id, stationname, latitude, longitude FROM citibike_reference WHERE id = ?", (max_station,))
data = cur.fetchone()
print "The most active station is station id %s at %s latitude: %s longitude: %s " % data
print "With " + str(hour_change[379]) + " bicycles coming and going in the hour between " + datetime.datetime.fromtimestamp(int(df.index[0])).strftime('%Y-%m-%dT%H:%M:%S') + " and " + datetime.datetime.fromtimestamp(int(df.index[-1])).strftime('%Y-%m-%dT%H:%M:%S')

import matplotlib.pyplot as plt

plt.bar(hour_change.keys(), hour_change.values())
plt.show()
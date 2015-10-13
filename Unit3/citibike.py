"""citibike.py"""

import requests
from pandas.io.json import json_normalize
import pandas as pd
import time
import sqlite3 as lite
from dateutil.parser import parse
import collections
import datetime

r = requests.get('http://www.citibikenyc.com/stations/json')

key_list = [] # unique list of keys for each station listing
for station in r.json()['stationBeanList']: # loop through the station list
	for k in station.keys():
		if k not in key_list:
			key_list.append(k)

# use json_normalize to convert to a pandas dataframe
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

con = lite.connect('citi_bike.db')
cur = con.cursor()

# create table to store the citibike data we need
with con:
	cur.execute('DROP TABLE IF EXISTS citibike_reference')
	cur.execute('CREATE TABLE IF NOT EXISTS citibike_reference (id INT PRIMARY KEY, totalDocks INT, city TEXT, altitude INT, stAddress2 TEXT, longitude NUMERIC, postalCode TEXT, testStation TEXT, stAddress1 TEXT, stationName TEXT, landMark TEXT, latitude NUMERIC, location TEXT )')

# a prepared SQL statement we're going to execute over and over again
sql = "INSERT INTO citibike_reference (id, totalDocks, city, altitude, stAddress2, longitude, postalCode, testStation, stAddress1, stationName, landMark, latitude, location) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"

# for loop to populate values in the database
with con:
	for station in r.json()['stationBeanList']:
		cur.execute(sql,(station['id'],station['totalDocks'],station['city'],station['altitude'],station['stAddress2'],station['longitude'],station['postalCode'],station['testStation'],station['stAddress1'],station['stationName'],station['landMark'],station['latitude'],station['location']))

# extract the column from the DataFrame and put them into a list
station_ids = df['id'].tolist()
# add the '_' to the station name and also add the data type for SQLite
station_ids = ['_' + str(x) + ' INT' for x in station_ids]
# adding '_' is necessary because column names cannot start with a number

# create the table - concatentating the string and joining all the station ids (now with '_' and 'INT' added)
with con:
	cur.execute("DROP TABLE IF EXISTS available_bikes")
	cur.execute("CREATE TABLE IF NOT EXISTS available_bikes ( execution_time INT, " +  ", ".join(station_ids) + ");")

# take the string and parse it into a Python datetime object
exec_time = parse(r.json()['executionTime'])

# create an entry for execution time by inserting it into the database
with con:
	cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', ((exec_time-datetime.datetime(1970,1,1)).total_seconds(),))

# iterate through the stations in the stationBeanList
id_bikes = collections.defaultdict(int) # defaultdict to store available bikes by station

# loop through the stations in the list
for station in r.json()['stationBeanList']:
	id_bikes[station['id']] = station['availableBikes']

# iterate through the defaultdict to update the values in the database
with con:
	for k, v in id_bikes.iteritems():
		cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + str(v) + " WHERE execution_time = " + str((exec_time-datetime.datetime(1970,1,1)).total_seconds()) + ";")

# repeat the pull and populate steps for 1 hour
for i in range(60):
	r = requests.get('http://www.citibikenyc.com/stations/json')
	exec_time = parse(r.json()['executionTime'])

	cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', ((exec_time-datetime.datetime(1970,1,1)).total_seconds(),))
	con.commit()

	id_bikes = collections.defaultdict(int)
	for station in r.json()['stationBeanList']:
		id_bikes[station['id']] = station['availableBikes']

	for k, v in id_bikes.iteritems():
		cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + str(v) + " WHERE execution_time = " + str((exec_time-datetime.datetime(1970,1,1)).total_seconds()) + ";")
	con.commit()

	time.sleep(60)
con.close()

con = lite.connect('citi_bike.db')
cur = con.cursor()

df = pd.read_sql_query("SELECT * FROM available_bikes ORDER BY execution_time",con,index_col='execution_time')

hour_change = collections.defaultdict(int)
for col in df.columns:
	station_vals = df[col].tolist()
	station_id = col[1:] # trim the "_"
	station_change = 0
	for k, v in enumerate(station_vals):
		if k < len(station_vals) - 1:
				station_change += abs(station_vals[k] - station_vals[k+1])
	hour_change[int(station_id)] = station_change

def keywithmaxval(d):
	# create a list of the dict's keys and values
	v = list(d.values())
	k = list(d.keys())

	# return the key with the max values
	return k[v.index(max(v))]

# assign the max key to max_station
max_station = keywithmaxval(hour_change)

# query sqlite for ref information
cur.execute("SELECT id, stationname, latitude, longitude FROM citibike_reference WHERE id = ?", (max_station,))
data = cur.fetchone()
print "The most active station is station %s at %s latitude: %s longitude: %s " % data
print "With " + str(hour_change[379]) + " bicycles coming and going in the hour between " + datetime.datetime.fromtimestamp(int(df.index[0])).strftime('%Y-%m-%dT%H:%M:%S') + " and " + datetime.datetime.fromtimestamp(int(df.index[-1])).strftime('%Y-%m-%dT%H:%M:%S')
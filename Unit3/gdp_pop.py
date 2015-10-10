"""gdp_pop.py"""

def IntergerOrNull(value):
	try:
			return int(float(value))
	except:
			return None

import sqlite3 as lite
import csv
import pandas as pd

con = lite.connect('education.db')
cur = con.cursor()

# crate the gdp table
cur.execute('DROP TABLE IF EXISTS gdp')
cur.execute('CREATE TABLE IF NOT EXISTS gdp (country TEXT, _1999 INT, _2000 INT, _2001 INT, _2002 INT, _2003 INT, _2004 INT, _2005 INT, _2006 INT, _2007 INT, _2008 INT, _2009 INT, _2010 INT)')
# prepare the gdp data inser statement
sql = "INSERT INTO gdp (country, _1999, _2000, _2001, _2002, _2003, _2004, _2005, _2006, _2007, _2008, _2009, _2010) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"

with open('ny.gdp.mktp.cd_Indicator_en_csv_v2/ny.gdp.mktp.cd_Indicator_en_csv_v2.csv', 'rU') as inputFile:
	next(inputFile)
	next(inputFile)
	header = next(inputFile)
	inputReader = csv.reader(inputFile)
	for line in inputReader:
		with con:
			data = {
						'country' : line[0],
						'1999' : IntergerOrNull(line[43]),
						'2000' : IntergerOrNull(line[44]),
						'2001' : IntergerOrNull(line[45]),
						'2002' : IntergerOrNull(line[46]),
						'2003' : IntergerOrNull(line[47]),
						'2004' : IntergerOrNull(line[48]),
						'2005' : IntergerOrNull(line[49]),
						'2006' : IntergerOrNull(line[50]),
						'2007' : IntergerOrNull(line[51]),
						'2008' : IntergerOrNull(line[52]),
						'2009' : IntergerOrNull(line[53]),
						'2010' : IntergerOrNull(line[54])
					}
			cur.execute(sql,(data['country'],data['1999'],data['2000'],data['2001'],data['2002'],data['2003'],data['2004'],data['2005'],data['2006'],data['2007'],data['2008'],data['2009'],data['2010']))

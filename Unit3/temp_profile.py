"""temp_profile.py"""

import sqlite3 as lite
import pandas as pd

con = lite.connect('weather.db')
with con:
	cur = con.cursor()
	cur.execute("SELECT * FROM temp")
	rows = cur.fetchall()
	cols = [desc[0] for desc in cur.description]
	df = pd.DataFrame(rows, columns=cols)

	# ranges
	print "The range of temps in Atlanta: ", df['Atlanta'].max() - df['Atlanta'].min()
	print "The range of temps in Denver: ", df['Denver'].max() - df['Denver'].min()
	print "The range of temps in Miami: ", df['Miami'].max() - df['Miami'].min()
	print "The range of temps in Nashville: ", df['Nashville'].max() - df['Nashville'].min()
	print "The range of temps in Seattle: ", df['Seattle'].max() - df['Seattle'].min()
	# means
	print "The mean temp for Atlanta: ", df['Atlanta'].mean()
	print "The mean temp for Denver: ", df['Denver'].mean()
	print "The mean temp for Miami: ", df['Miami'].mean()
	print "The mean temp for Nashville: ", df['Nashville'].mean()
	print "The mean temp for Seattle: ", df['Seattle'].mean()
	# variances
	print "The temp variance for Atlanta: ", df['Atlanta'].var()
	print "The temp variance for Denver: ", df['Denver'].var()
	print "The temp variance for Miami: ", df['Miami'].var()
	print "The temp variance for Nashville: ", df['Nashville'].var()
	print "The temp variance for Seattle: ", df['Seattle'].var()
	# largest temp change

"""make_y_sets.py"""

import pandas as pd

df1 = pd.read_csv('samsungdata.csv',header=None,names=["act"])
df2 = pd.read_csv('samsungdata.csv',header=None,names=["act"])

# make y_all
segments = [df1[0:346],df2[0:301],df1[347:687],df2[302:618],df1[688:1903],df2[619:1200],df1[1904:2219],df2[2101:1847],df1[2200:3604],df2[1848:2211],df1[3605:3964],df2[2212:2565],df1[3965:5065],df2[2566:2946],df1[5066:7351]]
df = pd.concat(segments)
df.to_csv("y_all.csv",index=False,header=False)

# make y_train
df = df1[5867:7351]
df.to_csv("y_train.csv",index=False,header=False)

# make y_test
segments = [df1[0:346],df2[0:301],df1[347:687],df2[302:618],df1[688:3314]]
df = pd.concat(segments)
df.to_csv("y_test.csv",index=False,header=False)

# make y_validate
segments = [df1[3965:5065],df2[2566:2946],df1[5066:5866]]
df = pd.concat(segments)
df.to_csv("y_validate.csv",index=False,header=False)
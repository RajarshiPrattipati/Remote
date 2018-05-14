
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import pymssql
import pandas as pd

from datetime import datetime
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cnxn = pymssql.connect( database = 'semi_c3_standardized' ,  server = '52.172.199.52',port = 1433,      user = 'rishi',         password = '9731449873')

#time = 'select accountholderid, trxndatetime  from semi_c3_standardized.dbo.semic3_upi_trxndetails '
time = 'select top 1000  *  from semi_c3_standardized.dbo.semic3_upi_trxndetails '
times = pd.read_sql(time,cnxn)
print(times.describe())
for i in [0,1,6,7] :
	times[times.ix[:,i].apply(lambda x: isinstance(x, (int, np.int64)))]
y = times.ix[:,10]
y = y.apply( lambda x: x.date())

x = times.ix[:,1]
c = times.ix[:,9]
z = times.ix[:,11]	
print(y.min())


def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)
_mad = np.abs(z - np.median(z)) / mad(z)

# Standard deviation
_sd = np.abs(z - np.mean(z)) / np.std(z)

#print(_mad)
#print (_sd)

ax.scatter(x, y, z, c=c, cmap=plt.hot())
plt.show()

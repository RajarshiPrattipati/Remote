import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd  
from sklearn import utils  
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib
from mpl_toolkits.mplot3d import Axes3D 
import pymssql

#from _FUTUTRE_
cnxn = pymssql.connect( database = 'semi_c3_standardized' ,  server = '52.172.199.52',port = 1433, 	user = 'rishi', 	password = '9731449873')
#cnxn = pyodbc.connect("DSN = sqlserver;UID =rishi;PWD=9731449873;TDS_version = 7.0")
sqli = 'select * from SemiC3fileuploads.dbo.SemiC3_Request_XML'
fsql1 =  'select accountno, orderid, orderdate, mobilenumber, ordersource, blockeddatetime from Semi_C3_standardized.dbo.suspect_accounts_blocked_twallet'

fsql2 = 'select distinct accountno,orderid from Semi_C3_standardized.dbo.suspect_accounts_blocked_twallet_05092017'
fsql3 = 'select distinct accountno,orderid from Semi_C3_standardized.dbo.suspect_accounts_blocked_twallet_12032018'

ac1 = pd.read_sql(fsql2,cnxn)
ac2 = pd.read_sql(fsql3,cnxn)
fraudaccnos = ac1.append(ac2)
orders = fraudaccnos['orderid']
creditsql = 'select top 100 FK_TRXN_CR_ACCOUNT_HOLDER_ID, TRXN_CR_AMT  from semi_c3_standardized.dbo.trxns_credit where orderid in ( select orderid from semi_c3_standardized.dbo.suspect_accounts_blocked_twallet_05092017) or orderid in ( select orderid from semi_c3_standardized.dbo.suspect_accounts_blocked_twallet_12032018)'
fraudorders = pd.read_sql(creditsql,cnxn)
accs = fraudorders.loc[:'FK_TRXN_CR_Account_Holder_ID']
#fraudorders['F'] = 1
#print( fraudorders)
creditfull = 'select top 1000 FK_TRXN_CR_ACCOUNT_HOLDER_ID, TRXN_CR_AMT  from semi_c3_standardized.dbo.trxns_credit '

allorders = pd.read_sql(creditfull,cnxn)
allorders[:0] += 2
print (allorders.describe())
#print(accs)
def lbler(acc):
	
#	if acc['FK_TRXN_CR_ACCOUNT_HOLDER_ID'] in accs:
	if acc['FK_TRXN_CR_ACCOUNT_HOLDER_ID'] in accs:
		return 1
	else:
		return 0
allorders['F'] = allorders.apply( lambda  lbl: lbler(lbl) , axis = 1)
#frauds = fraudaccnos['orderid']
#for orderid in allorders['orderid']:
#print (allorders[allorders['F']!=0])
#print (allorders[allorders['F']== 1])

		
#print(allorders)


#print (fraudaccnos)
#fraudblocked = pd.read_sql(fsql1,cnxn)
#mobacc = fraudblocked.loc[:,['mobile','accountno']]


 
#reqdata = pd.read_sql(sqli,cnxn)
sqlo = 'select * from SemiC3fileuploads.dbo.SemiC3_Response_Message'
#resdata = pd.read_sql(sqlo,cnxn)

#accountssql = 'select * from Semi_C3_standardized.dbo.account_holder'
#print(reqdata)
#outliers = pd.read_sql(sqloutlier,cnxn)
#xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
#clf = LocalOutlierFactor(n_neighbors=20)
#y_pred = clf.fit_predict(X)
#y_pred_outliers = y_pred[200:]

#print(outliers.describe())
#print(outliers.columns)
#print(data.describe())
#print(data2.describe())
#kmeans = KMeans(n_clusters=2, random_state=0).fit(data)	

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
#X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[np.loc(allorders[0:500,:1])]
# Generate some regular novel observations
#X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[allorders[500:1000,:1]]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()

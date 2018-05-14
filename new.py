import pyodbc
from scipy import stats
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
import matplotlib.pyplot as plt
def rettime(row):
	return row.date()
#from _FUTUTRE_
cnxn = pymssql.connect( database = 'semi_c3_standardized' ,  server = '52.172.199.52',port = 1433,      user = 'rishi',         password = '9731449873')
sqli = 'select account_holder_id, created_date  from Semi_c3_standardized.dbo.account_holder'
#s = pd.read_sql(sqli,cnxn)	#Master Account list


datasql = 'select top 990 fk_trxn_cr_account_holder_id,trxn_cr_amt from Semi_c3_standardized.dbo.trxns_credit order by trxn_cr_amt desc'
data = pd.read_sql(datasql,cnxn)



outsql = 'select top 10 accountNo, orderamount from Semi_C3_Standardized.dbo.Suspect_Accounts_blocked_TWallet_31012018'
outliers = pd.read_sql(outsql,cnxn)
outliers.iloc[:,0].astype(int) //10000

print (pd.read_sql('select top 3 * from Semi_c3_standardized.dbo.Suspect_accounts_blocked_Twallet_31012018',cnxn ))

print(outliers.shape, data.shape)

n_samples = 1000
rng = np.random.RandomState(42)
outliers_fraction = 0.01
clusters_seperation = [0]#[0,1,2]
#OUTLIER DETECTION CLASSIFIERS
classifiers = {	"One-Class SVM": svm.OneClassSVM(nu= (0.95*outliers_fraction + 0.5),kernel = "rbf", gamma = 0.1),
"Robust Covariance": EllipticEnvelope(contamination = outliers_fraction),
"Isolation Forest": IsolationForest(max_samples = n_samples,contamination = outliers_fraction,random_state = rng ),
"Local Outlier Factor": LocalOutlierFactor(n_neighbors = 3, contamination = outliers_fraction) }
xx,yy = np.meshgrid(np.linspace(1,10000,10), np.linspace(4500,55000,10))
n_inliers = int( (1.- outliers_fraction) * n_samples)
n_outliers = int( outliers_fraction * n_samples)
ground_truth = np.ones(n_samples, dtype = int)
ground_truth[-n_outliers:] = -1

for i, offset in enumerate(clusters_seperation):
	np.random.seed(42)
	#x1 = data['fk_trxn_cr_account_holder_id']
	#X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
    	#X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
    	#X = np.r_[X1, X2]
	    # Add outliers
	#X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

#	X = data.astype(int)
#	Y = outliers.astype(int)
#	X = np.r_[X,Y] 

	data.join(outliers)
	X = data
	#X= X[~np.isnan(X).any(axis=1)].astype(int)
    # Fit the model
	plt.figure(figsize=(9, 7))
	for i, (clf_name, clf) in enumerate(classifiers.items()):
        # fit the data and tag outliers
		if clf_name == "Local Outlier Factor":
			y_pred = clf.fit_predict(X)
			scores_pred = clf.negative_outlier_factor_
		else:
			clf.fit(X)
			scores_pred = clf.decision_function(X)
			y_pred = clf.predict(X)
		threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
		print (threshold)
		n_errors = np.sum(y_pred != ground_truth)
        	# plot the levels lines and the points
		if clf_name == "Local Outlier Factor":
            	# decision_function is private for LOF
			Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
		else:
            		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

		Z = Z.reshape(xx.shape)
		#Z = np.unique(np.sort(Z).astype(int))
		subplot = plt.subplot(2, 2, i + 1)
		subplot.contourf(xx, yy, Z, levels=np.linspace(1, 7, 7),cmap=plt.cm.Blues_r)
		a = subplot.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')
		
		subplot.contourf(xx, yy, Z, levels=[threshold,threshold+10 ],colors='orange')
		b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white',
                            s=20, edgecolor='k')
		c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black',
                            s=20, edgecolor='k')
		subplot.axis('tight')
		subplot.legend( [a.collections[0], b, c], ['learned decision function', 'true inliers', 'true outliers'],
			 prop=matplotlib.font_manager.FontProperties(size=10),
            		loc='lower right')
		subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
		subplot.set_xlim((6000, 10000))
		subplot.set_ylim((4500,55000))	
		plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
		plt.suptitle("Anomaly  detection")

plt.show()


sqlccav = 'select fk_account_holder_id,order_id from semi_c3_standardized.dbo.ccavenue_credit_details'
	# CCAvenue Credit Orders

c3disputes = 'select * from semi_c3_standardized.dbo.imeibaseddisputesaccounts'

imp = ' select fk_sender_ac_number , imps_trxn_date, receiver_ac_number,sender_ifsc,  receiver_ifsc from semi_c3_standardized.dbo.imps_trxns '
#imps = pd.read_sql(imp,cnxn) 		#IMPS transactions

neft = ' select [receiver account no] from semi_c3_standardized.dbo.semic3_neft_file'
#q = pd.read_sql(neft,cnxn)		#NEFT transactions

c3pg = ' select bt, trxndatetime, accountholderid from semi_c3_standardized.dbo.semic3_paymentgateway_trxndetails'
#c3p = pd.read_sql(c3pg, cnxn)		#SemiC3 transactions master

#email = ' select fk_brm_account_holder_id, brm_req_receiving_time from Semi_C3_Standardized.dbo.SemiC3_Registration_Email'
#emails = pd.read_sql(email,cnxn)	


#mob = ' select fk_account_holder_id, brm_req_receiving_time from Semi_C3_Standardized.dbo.SemiC3_Registration_Mobile'
#mobs = pd.read_sql(mob,cnxn)

semic3trxn = ' select accountholderid, toaccount, orderid from Semi_C3_Standardized.dbo.SemiC3_transactiondetails'

upi = 'select accountholderid, orderid, trxndatetime, bt from Semi_c3_standardized.dbo.semic3_upi_trxndetails'

#sklearn.ensemble.IsolationForest(

#Identify abnormal transaction rate
#Identify account numbers of abnormal transactions
#Get Transaction Details
#Flag receiving accounts

#Known Frauds training
fsql2 = 'select distinct accountno,orderid from Semi_C3_standardized.dbo.suspect_accounts_blocked_twallet_05092017'
fsql3 = 'select distinct accountno,orderid from Semi_C3_standardized.dbo.suspect_accounts_blocked_twallet_12032018'

#ac1 = pd.read_sql(fsql2,cnxn)
#ac2 = pd.read_sql(fsql3,cnxn)
#fraudaccnos = ac1.append(ac2)

#Identify time splits
time = 'select accountholderid, trxndatetime  from semi_c3_standardized.dbo.semic3_upi_trxndetails '
#times = pd.read_sql(time,cnxn)

#times2 = pd.to_datetime(times['trxndatetime'], yearfirst=True, format="%y-%m-%d")
#times.assign(times.dt.date, times.dt.time)	

#times['created_date'] =  times.apply( lambda row: rettime(row['trxndatetime']), axis = 1)
#print (times)



#f = imps[ imps['fk_receiver_ac_no'] in fraudaccnos['accountno'] ]
#print(times)


#print('f')


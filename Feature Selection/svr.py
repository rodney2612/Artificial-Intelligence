
def svr(data_frame, task):
	# create X and y
	#print data_frame
	drop_array=[len(data_frame.columns)-1]
	#drop_array.append(len(data_frame.columns))
	X = data_frame.drop(data_frame.columns[drop_array], axis=1) #dropping the last column which is the prediction/class label column
	#print "X:",X
	Y = data_frame.iloc[:,len(data_frame.columns)-1] #getting only the last(the class label column) column  in Y
	#print "Y:",Y

	# split into training and testing sets
	from sklearn.cross_validation import train_test_split
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
	'''
	X_train_list = X_train.values.tolist() #converting data frame to a regular python list of rows and columns
	Y_train_list = Y_train.values.tolist()
	X_test_list = X_test.values.tolist()
	Y_test_list = Y_test.values.tolist()
	'''

	'''print "X train"
	print X_train
	print "y train"
	print y_train'''
	

	from sklearn import metrics
	from sklearn.svm import SVC, SVR
	sv = SVR(kernel='rbf', C=1e3, gamma=0.1)
	if task == 'c': #User wants to perform classification
		sv = SVC(kernel="rbf", C=2.8, gamma=.0073)
	#svc = SVC(kernel='linear')
	#svc = SVC(kernel="rbf", C=2.8, gamma=.0073)
	
	sv.fit(X_train, Y_train)
	Y_pred_class = sv.predict(X_test)
	if task == 'c':
		return metrics.accuracy_score(Y_test, Y_pred_class)
	else:
		return sv.score(X_test, Y_test)
	


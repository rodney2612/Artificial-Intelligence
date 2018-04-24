import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
import time 

def run():
	dir_path = ""

	#Preparing Training data
	train_file_path = dir_path + "train.csv"
	train_file = read_csv(train_file_path,skiprows=1,header=None)

	train_file = train_file.drop(train_file.columns[0],axis=1)
	train_file = train_file.values

	train_X_temp = train_file[5:50000,:-1]
	train_Y = train_file[6:50001,-1]

	#Combining previous 5 time step data into one row
	train_X = np.zeros((train_X_temp.shape[0],8*5))
	for i in range(train_X_temp.shape[0]):
		for j in range(5):
			for k in range(8):
				train_X[i][j*8+k] = train_X_temp[i-j][k]

	print "Train file loaded"

	#Preparing testing data
	test_file_name = dir_path + "test2.csv"
	test_file = read_csv(test_file_name,skiprows=1,header=None)
	test_file = test_file.values
	test_X = np.array(test_file[:,:-1])
	test_y = test_file[:,-1]


	params = {'n_estimators': 500, 'max_depth': 6,
	        'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
	#clf = GradientBoostingRegressor(**params).fit(train_X, train_Y)
	start = time.time()

	#Gradient Boosting Model training and prediction
	clf = GradientBoostingRegressor().fit(train_X, train_Y)
	# make predictions for test data
	prediction = clf.predict(test_X)

	#Calculating and printing results
	mse = np.mean((prediction - test_y)**2)
	print "MSE: ",mse
	print "Score: ",clf.score(test_X,test_y)
	print "Time:",time.time() - start

run()
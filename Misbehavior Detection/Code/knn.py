from sklearn import neighbors
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import time
def run():
	print "k Nearest Neighbor Regression started..."

	#Preparing Training data
	dir_path = ""
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


	#Preparing Testing data
	test_file_name = dir_path + "test2.csv"
	test_file = read_csv(test_file_name,skiprows=1,header=None)
	test_file = test_file.values
	test_X = np.array(test_file[:,:-1])
	test_y = test_file[:,-1]

	start = time.time()

	#k Nearest Neighbor Model training and prediction
	n_neighbors = 6
	knn = neighbors.KNeighborsRegressor(n_neighbors)
	knn.fit(train_X, train_Y)
	prediction = knn.predict(test_X)

	#Calculating and printing results
	mse = np.mean((prediction - test_y)**2)
	print "MSE: ",mse
	# print "Predictions: \n",prediction
	print "Score: ",knn.score(test_X,test_y)
	print "Time: ",time.time() - start
	print "k Nearest Neighbor Regression done...\n"

run()
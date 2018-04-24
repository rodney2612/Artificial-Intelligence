from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor

def run():
	print "Decision Tree Regression started..."

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

	#Preparing testing data
	test_file_name = dir_path + "test2.csv"
	test_file = read_csv(test_file_name,skiprows=1,header=None)
	test_file = test_file.values
	test_X = np.array(test_file[:,:-1])
	test_y = test_file[:,-1]

	#Model training and prediction for different no of trees
	estimators = np.arange(10, 100, 10)
	print "\nBagged Decision Tree:"
	bag_reg = BaggingRegressor(DecisionTreeRegressor(),n_jobs=2,random_state=0).fit(train_X, train_Y)
	scores = []
	prediction = []
	for n in estimators:
	    bag_reg.set_params(n_estimators=n)
	    bag_reg.fit(train_X, train_Y)
	    score = bag_reg.score(test_X, test_y)
	    print score
	    scores.append(score)
	    #prediction.append(bag_reg.predict(test_X))
	
	#plotting the effect of increasing no of trees on accuracy score
	plt.title("Effect of n_estimators")
	plt.xlabel("n_estimator")
	plt.ylabel("score")
	plt.plot(estimators, scores)
	plt.show()
run()

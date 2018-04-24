import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from plot import plot as plot_graph
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def run():
	print "Random Forest Regressor started..."
	dir_path = ""
	train_file_path = dir_path + "train.csv"
	train_file = read_csv(train_file_path,skiprows=1,header=None)

	train_file = train_file.drop(train_file.columns[0],axis=1)
	train_file = train_file.values

	train_X_temp = train_file[5:50000,:-1]
	train_Y = train_file[6:50001,-1]
	train_X = np.zeros((train_X_temp.shape[0],8*5))
	for i in range(train_X_temp.shape[0]):
		for j in range(5):
			for k in range(8):
				train_X[i][j*8+k] = train_X_temp[i-j][k]


	test_file_name = dir_path + "test2.csv"
	test_file = read_csv(test_file_name,skiprows=1,header=None)
	test_file = test_file.values
	test_X = np.array(test_file[:,:-1])
	test_y = test_file[:,-1]

	scores_array = []
	mean_array = []
	variance_array = []
	factors = []
	for i in range(2,11):
		factors.append(i)
		for j in range(1,40):
			model = RandomForestRegressor(n_jobs=-1,max_features=len(train_X[0])/i)
			estimators = [10]
			scores = []
			prediction = []
			for n in estimators:
			    model.set_params(n_estimators=n)
			    model.fit(train_X, train_Y)
			    scores.append(model.score(test_X, test_y))
			    scores_array.append(scores[0])
			    prediction.append(model.predict(test_X))
		mean_array.append(sum(scores_array)/len(scores_array))
		variance_array.append(np.var(scores_array))
	print "Mean array:\n",mean_array
	print "Variance array:\n",variance_array

	plot_graph(factors,variance_array,"Feature division factor","Variance of accuracy scores")
	plot_graph(factors,mean_array,"Feature division factor","Mean of accuracy scores")
run()
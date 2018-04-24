import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor

def run():
	print "Gradient Boosting Regression started..."
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


	params = {'n_estimators': 500, 'max_depth': 6,
	        'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
	maximum_depth = np.arange(2,10,1)
	scores = []
	for depth in maximum_depth:
		#clf = GradientBoostingRegressor(**params).fit(train_X, train_Y)
		gbr = GradientBoostingRegressor(max_depth=depth).fit(train_X, train_Y)
		score = gbr.score(test_X,test_y)
		print score
		scores.append(score)
	plt.title("Effect of max depth")
	plt.xlabel("Max Depth of individual Regression Trees")
	plt.ylabel("Score")
	plt.plot(maximum_depth, scores)
	plt.show()
run()
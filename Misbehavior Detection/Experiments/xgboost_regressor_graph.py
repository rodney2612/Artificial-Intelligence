import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import time

def run():
	print "XGBoost Regressor started..."
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

	scores = []
	learning_rates = np.arange(0.05,0.2,0.01)
	for rate in learning_rates:
		model = XGBRegressor(learning_rate=rate)
		model.fit(train_X, train_Y)
		score = model.score(test_X,test_y)
		print score
		scores.append(score)

	plt.title("Effect of learning rate")
	plt.xlabel("Learning Rate")
	plt.ylabel("Score")
	plt.plot(learning_rates, scores)
	plt.show()

run()
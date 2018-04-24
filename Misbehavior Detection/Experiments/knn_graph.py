from sklearn import neighbors
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

def run():
	print "k Nearest Neighbor Regression started..."
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

	n_neighbors = np.arange(2,25,1)
	for neighbor in n_neighbors:
		knn = neighbors.KNeighborsRegressor(neighbor)
		knn.fit(train_X, train_Y)
		score = knn.score(test_X,test_y)
		print score
		scores.append(score)

	plt.title("Effect of no of neighbours")
	plt.xlabel("No of neighbors")
	plt.ylabel("Score")
	plt.plot(n_neighbors, scores)
	plt.show()
run()

#run()
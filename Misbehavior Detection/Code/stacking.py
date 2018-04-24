import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor 
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

def to_stacked_row(models, row, y):
	stacked_row = list()
	for i in range(len(models)):
		prediction = models[i].predict(row.reshape(1,-1))
		stacked_row.append(prediction[0])
	#stacked_row.append(y)
	#print row
	#print stacked_row
	my_row = np.append(row,stacked_row)
	#my_row.append(stacked_row)
	#print my_row
	return my_row

def stacking(train_X, train_Y, test_X, test_y):
	model_list = [RandomForestRegressor, BaggingRegressor]
	models = list()
	for i in range(len(model_list)):
		model = model_list[i]()
		model.fit(train_X,train_Y)
		models.append(model)
	stacked_dataset = list()
	
	index = 0
	for row in train_X:
		stacked_row = to_stacked_row(models, row, train_Y[index])
		index += 1
		stacked_dataset.append(stacked_row)

	reg = linear_model.LinearRegression()
	print len(stacked_dataset), len(train_Y)
 	reg.fit(stacked_dataset, train_Y)

	stacked_rows = list()
	for row in test_X:
		stacked_rows.append(to_stacked_row(models, row, None))
	prediction = reg.predict(stacked_rows)
	return prediction,reg.score(stacked_rows,test_y)

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

	print stacking(train_X, train_Y, test_X, test_y)

run()
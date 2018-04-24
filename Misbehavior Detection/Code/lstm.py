from math import sqrt
from numpy import concatenate
from numpy import zeros
from numpy import array
from numpy import reshape
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Activation

import numpy as np
import os

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

	#Preparing Testing data
	test_file_name = dir_path + "test2.csv"
	test_file = read_csv(test_file_name,skiprows=1,header=None)
	test_file = test_file.values
	test_X = np.array(test_file[:,:-1])
	test_Y = test_file[:,-1]


	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	print(train_X.shape, train_Y.shape,test_X.shape,test_Y.shape)



	# test_file_name = dir_path + "test2.csv"
	# test_file = read_csv(test_file_name,header=None)
	# test_file = test_file.values
	# test_X = array(test_file[:,:-1])
	# test_y = test_file[:,-1]

	# reshape input to be 3D [samples, timesteps, features]
	# test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
	#print(train_X.shape, train_y.shape)
	# print(test_X.shape, test_y.shape)

	# design network
	model = Sequential()
	model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(100))
	# model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation("relu"))
	model.compile(loss='mse', optimizer='adam',metrics=['mse', 'accuracy'])

	# fit network
	history = model.fit(train_X, train_Y, epochs=100, batch_size=300, verbose=2, shuffle=False)

	# serialize model to JSON
	model_json = model.to_json()
	with open("model2.json", "w") as json_file:
	    json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights("model2.h5")
	print("Saved model to disk")

	#Plotting results
	pyplot.plot(history.history['loss'], label='train')
	# pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	pyplot.show()

	# make a prediction
	yhat = model.predict(test_X[:1,:,:])
	print yhat

run()
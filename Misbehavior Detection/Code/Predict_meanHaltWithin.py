import numpy as np 
import pandas as pd 
from pandas import read_csv
from sklearn import linear_model
from sklearn import neural_network
import cPickle

dir_path = ""
train_file_path = dir_path + "train.csv"
train_file = read_csv(train_file_path,skiprows=1,header=None)

train_file = train_file.drop(train_file.columns[0],axis=1)
train_file = train_file.values

train_X_temp = train_file[5:50000,:-1]
print(train_X_temp.shape)
train_Y = train_file[6:50001,-1]
train_X = np.zeros((train_X_temp.shape[0],8*5))
for i in range(train_X_temp.shape[0]):
	for j in range(5):
		for k in range(8):
			train_X[i][j*8+k] = train_X_temp[i-j][k]


# model = linear_model.SGDRegressor(learning_rate='optimal',eta0=0.0000001,max_iter=5000,shuffle=True)

neural_net = neural_network.MLPRegressor(hidden_layer_sizes=(250,500,100,100,),learning_rate_init=0.001,max_iter=1000)


print(train_X.shape)
print(train_Y.shape)


test_file_name = dir_path + "test2.csv"
test_file = read_csv(test_file_name,skiprows=1,header=None)
test_file = test_file.values
test_X = np.array(test_file[:,:-1])
test_y = test_file[:,-1]


# for i in range(1000):
# model.fit(train_X,train_Y)
print("start")
neural_net.fit(train_X,train_Y)
print("save the model")
print(neural_net.score(test_X,test_y))

# with open(dir_path+"model.pkl",'wb') as p_file:
# 	cPickle.dump(neural_net,p_file)

# print("load the model")

# with open(dir_path+"model.pkl",'rb') as p_file:
# 	loaded_neural_net = cPickle.load(p_file)

# model.partial_fit(train_X,train_Y)
# test_X = np.array(test_file[-1,:])

# print(test_X.shape)

# print(model.predict(test_X))

# print(test_X)
# print(test_file[5,-1])
# print(model.get_params())

# while(1):
# 	test_file_name = raw_input("command: ")
# 	if test_file_name == "exit":
# 		break
# 	else:
# 		print(test_file_name)
# 		test_file_name = dir_path + test_file_name
# 		test_file = pd.read_csv(test_file_name,header=None)
# 		test_file = test_file.values
# 		test_X = np.array(test_file[-1,:-1])
# 		test_X = np.reshape(test_X,(-1,40))
# 		pred_halt = loaded_neural_net.predict(test_X)
# 		actual_halt = test_file[-1,-1]
# 		if abs(pred_halt-actual_halt) >=50:
# 			print("Detector Misbehaviour",pred_halt)
# 		else:
# 			print("Accurate Prediction: ",pred_halt,actual_halt)		

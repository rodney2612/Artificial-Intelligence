import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time

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
test_y = test_file[:,-1]

start = time.time()
model = RandomForestRegressor(n_jobs=-1)

# Try different numbers of n_estimators
#estimators = np.arange(10, 200, 10)

#Model training and prediction
estimators = [20]
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(train_X, train_Y)
    scores.append(model.score(test_X, test_y))
    prediction = model.predict(test_X)
    mse = np.mean((prediction - test_y)**2)
    print "MSE: ",mse
	
'''plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
plt.show()'''
print scores
print "time: ", time.time() - start
# print prediction



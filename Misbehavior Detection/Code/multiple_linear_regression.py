import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics
from pandas import read_csv
import time
 
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
test_file = read_csv(test_file_name,header=None)
test_file = test_file.values
test_X = np.array(test_file[:,:-1])
test_y = test_file[:,-1]


start = time.time()
# create linear regression object
reg = linear_model.LinearRegression()
 
# train the model using the training sets
reg.fit(train_X, train_Y)

prediction = reg.predict(test_X)
mse = np.mean((prediction - test_y)**2)
print "MSE: ",mse

print "Score: ",reg.score(test_X,test_y)

print "Time: ",(time.time() - start)


 

# # regression coefficients
# print('Coefficients: \n', reg.coef_)
 
# # variance score: 1 means perfect prediction
# print('Variance score: {}'.format(reg.score(test_X, test_y)))
 
# # plot for residual error
 
# ## setting plot style
# plt.style.use('fivethirtyeight')
 
# ## plotting residual errors in training data
# plt.scatter(reg.predict(train_X), reg.predict(train_X) - train_Y,
#             color = "green", s = 10, label = 'Train data')
 
# ## plotting residual errors in test data
# plt.scatter(reg.predict(test_X), reg.predict(test_X) - test_y,
#             color = "blue", s = 10, label = 'Test data')
 
# ## plotting line for zero residual error
# plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 
# ## plotting legend
# plt.legend(loc = 'upper right')

# ## plot title
# plt.title("Residual errors")
 
# ## function to show plot
# plt.show()
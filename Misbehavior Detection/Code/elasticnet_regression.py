import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import ElasticNet,ElasticNetCV
import time


def train_EN_model(train_x, train_y, _predict_x):
	'''train_x, predict_x = \
        standarize_feature(_train_x, _predict_x)'''
    #l1_ratios = [1e-4, 1e-3, 1e-2, 1e-1]
    #l1_ratios = [1e-5, 1e-4, 1e-3]
	l1_ratios = [0.9, 0.92, 0.95, 0.97, 0.99]
	#l1_ratios = [0.5]
	min_mse = 1
	best_l1_ratio = 0.95
	best_alpha = 0.5
	for r in l1_ratios:
		t1 = time.time()
		reg_en = ElasticNetCV(l1_ratio=r, cv=5, n_jobs=4, verbose=1, precompute=True)
		reg_en.fit(train_x, train_y)
		n_nonzeros = (reg_en.coef_ != 0).sum()
		_mse = np.mean(reg_en.mse_path_, axis=1)[np.where(reg_en.alphas_ == reg_en.alpha_)[0][0]]
		if _mse < min_mse:
			min_mse = _mse
			best_l1_ratio = r
			best_alpha = reg_en.alpha_
			t2 = time.time()
	return best_l1_ratio,best_alpha

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

#best_l1_ratio, best_alpha = train_EN_model(train_X, train_Y, test_X)
#print "Best L1 ratio, Best alpha",best_l1_ratio,best_alpha
#enet = ElasticNet(l1_ratio=best_l1_ratio, alpha=best_alpha)

start = time.time()
enet = ElasticNet()
enet.fit(train_X, train_Y)
#model = "ElasticNet int %.2f coefs %s" % (enet.intercept_, pprint(enet.coef_))
prediction = enet.predict(test_X)
mse = np.mean((prediction - test_y)**2)
print "MSE: ",mse
# print prediction
print "Score: ",enet.score(test_X,test_y)
print "Time: ",(time.time() - start)
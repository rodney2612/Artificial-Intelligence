import pandas as pd
import numpy as np
from svr import svr

#ascii = {'a':1, 'b':2, 'c': 3, 'd': 4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 
	# 'o':15, 'p':16, 'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v': 22, 'w':23, 'x':24, 'y':25, 'z':26}

#conversion of string to number
def number(x):
	sum = 0
	for c in x:
		sum += ord(c)
	return sum

#Drops all columns having any number of ? or NaN value(s). Also converts strings in data to numbers
def cleanData(data_frame):
	data_frame = data_frame.replace('?', np.nan)
	data_frame=data_frame.dropna(axis=1, how='any')
	#data_frame.iloc[:,len(data_frame.columns)-1] = pd.to_numeric(data_frame.iloc[:,len(data_frame.columns)-1],errors='coerce')
	#print data_frame
	for column in data_frame:
		if data_frame[column].dtype == object:
			data_frame[column] = data_frame[column].apply(lambda x: number(x))
			print data_frame[column]
	return data_frame

def getTask():
	task = ''
	while True:	
		task=raw_input("Regression or classification? Enter r for regression and c for classification: ")
		if task == 'r' or task == 'c':
			break;	
		else:
			print "You have entered wrong input. Please enter either r or c"
	return task

def getInput():
	'''Features are represented by a binary vector where each individual value represents whether the feature is included in the 		current state or not'''	
	task = getTask()
	file_name=raw_input("Enter data file location: ")
	#col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'] #features of the pima data set
	data_frame = pd.read_csv(file_name, header=None)
	#data_frame = data_frame.apply(lambda x: x.str.strip()).replace('?', np.nan)
	data_frame = cleanData(data_frame)
	#print data_frame
	no_of_features=len(data_frame.columns)-1
	initial_state=[0] * no_of_features #initialising initial state with all zeroes i.e initial state doesnt include any feature
	original_accuracy=svr(data_frame, task) #get accuracy with all features present
	print "Original accuracy",original_accuracy
	return data_frame, initial_state, task, no_of_features, original_accuracy


#Drops columns from the dataframe to include only those columns that are in current state
def dropColumns(state,data, no_of_features):
	#print state
	drop_array=[] #Used to store those column numbers which need to be dropped from the data frame
	for i in range(no_of_features):
		if state[i]==0:
			drop_array.append(i)
	data=data.drop(data.columns[drop_array], axis=1) #axis=1 indicates we are dropping olumns and not rows	
	#data=data.iloc[:,drop_array]
	#print data			
	return data
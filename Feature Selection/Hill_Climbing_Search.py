import pandas as pd
import numpy as np
from svr import svr
from Utilities import number, cleanData, getTask, getInput, dropColumns


#data_frame = []
#initial_state = []
#task = ''
#no_of_features = 0

#while best_accuracy<0.99*original_accuracy:

#Hill climbing search
def hillClimbing():
	data_frame, initial_state, task, no_of_features, original_accuracy = getInput()
	backup_state=initial_state
	best_state=initial_state		
	#best_accuracy=0
	current_no_of_features=0
	best_eval_score = 0
	print data_frame
	print no_of_features," columns"
	print "Initial state",initial_state
	print "task",task
	while True:
			'''Used as a terminating criteria i.e. if all children of a current state have lesser evaluation score than the 			current state, then all_children_bad is set to true'''
			all_children_bad = True 
			next_state=list(best_state)
			current_no_of_features += 1
			backup_state=list(next_state)		
			for j in range(no_of_features):		
				new_data_frame=data_frame
				print j
				if next_state[j]==0:
					next_state[j]=1	#adding a feature to the state
					new_data_frame=dropColumns(next_state,new_data_frame, no_of_features)
					accuracy=svr(new_data_frame, task)
					#print "Accuracy",accuracy
					#Giving more importance to states having less no of features
					eval_score = 0.99*accuracy + 0.01*(no_of_features - current_no_of_features)
					print "Next state, Accuracy and best accuracy",next_state,eval_score,best_eval_score
				
					#if accuracy>best_accuracy:
				
					'''changing best state if evaluation score of current state is greater than that of the best state so 					far'''
					if eval_score > best_eval_score:
						all_children_bad = False
						best_eval_score = eval_score
						#best_accuracy=accuracy
						best_state=list(next_state)
						#print best_eval_score

					next_state=list(backup_state)
				print "best state",best_state
			'''We stop either when number of features in the current state becomes greater than 0.7 times the total number of 				features or when all children of current state have lesser evaluation score than current state. It may happen that 				eavluation score of some child of current state is always greater than current state till all features are added. 				Since we do not want to add all the features, we stop when 70 % of the total number of features are added''' 		
			if all_children_bad == True or current_no_of_features>=0.7*no_of_features:
				break;
				
	print "Relevant features:"
	for i in range(no_of_features):
		if best_state[i]==1:
			print i

				

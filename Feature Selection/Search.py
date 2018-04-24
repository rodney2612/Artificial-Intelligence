from Hill_Climbing_Search import hillClimbing
from GeneticAlgorithm import geneticAlgorithm
from Utilities import getTask

search_algo = ''
while True:	
	search_algo=raw_input("Hill Clmbing or GA? Enter h for Hill Climbing and g for GA: ")
	if search_algo == 'h' or search_algo == 'g':
		break;	
	else:
		print "You have entered wrong input. Please enter either h or g"
	
if search_algo == 'h':
	hillClimbing()
else:
	geneticAlgorithm()


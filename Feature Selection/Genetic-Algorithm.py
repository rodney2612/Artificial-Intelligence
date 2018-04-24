import pandas as pd
import random as random
import math as math
import numpy as np
from naive_bayes import bayes

file_name=raw_input("Enter data file location: ")
#col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'] #features of the pima data set
data_frame = pd.read_csv(file_name, header=None)
no_of_features=len(data_frame.columns)-1
print no_of_features," columns"
original_accuracy=bayes(data_frame) #get accuracy with all features present
print "Original accuracy",original_accuracy
uniformProbability = 0.5
mutationProbability = 0.1

#Create a random individual
def createIndividual():
	all_zero = True
	while True:
		genes = np.random.choice([0, 1], size=(no_of_features,), p=[4./5, 1./5])
		for i in range(len(genes)):
			if genes[i] == 1:
				all_zero = False
				break
		if all_zero == False:
			break
	
	return genes

#Features are represented by a binary vector where each individual value represents whether the feature is included in the current state or not


def createInitialPopulation(size):
	population=[]
	for i in range(size):
		population.append(createIndividual())
	return population

def mutate(individual):
	for i in range(len(individual)):
		#Create random gene
		gene = int(round(random.random()))
		individual[i] = gene
	return individual

def selection(population):
        #Create a tournament population
	tournamentSize = 4
	tournament = []
        #For each place in the tournament get a random individual
	randomIds = []
        for i in range(tournamentSize):
        	randomIndex = int(random.random() * len(population))
		#print "random index",randomIndex
        	tournament.append(population[randomIndex])
	fitness = getFitness(tournament)
        #Get the fittest
        fittest = findFittest(tournament,fitness,True)
        return fittest


def crossover(individual1, individual2):
        child = []
        #Loop through genes
        for i in range(len(individual1)):
        	if uniformProbability >= random.random():
        		child.append(individual1[i])
        	else:
        		child.append(individual1[i])
        return child

def evolvePopulation(population):
	newPopulation = []
        #Loop over the population size and create new individuals with
        #crossover
	for i in range(len(population)):
		individual1 = selection(population)            
		individual2 = selection(population)          
		child = crossover(individual1, individual2)
		if mutationProbability >= random.random():
			child = mutate(child)       
		newPopulation.append(child)
	return newPopulation

def findFittest(population, fitness, selection):
	fittest = 0
	for i in range(len(population)):
		if fitness[fittest] <= fitness[i]:
			fittest = i
	print "Fittest",fittest
	if selection == True:
		return population[fittest]
	return fittest

#Drops columns from the dataframe to include only those columns that are in current state
def dropColumns(state,data):
	#print state
	drop_array=[] #Used to store those column numbers which need to be dropped from the data frame
	for i in range(no_of_features):
		if state[i]==0:
			drop_array.append(i)
	data=data.drop(data.columns[drop_array], axis=1) #axis=1 indicates we are dropping olumns and not rows	
	#data=data.iloc[:,drop_array]
	#print data			
	return data

def getFitness(population):
	fitness=[]
	print population
	for i in range(len(population)):
		current_no_of_features = 0
		for j in range(no_of_features):
			if population[i][j] == 1:
				current_no_of_features += 1
		new_data_frame=dropColumns(population[i],data_frame)
		accuracy=bayes(new_data_frame)
		#print "Accuracy",accuracy
		#Giving more importance to states having less no of features
		fitness.append(0.99*accuracy + 0.01*(no_of_features - current_no_of_features))
	return fitness


#print crossover([1,0,1,0,1,0,0,1],[0,0,0,1,1,0,1,0]) #crossover working
#print mutate([1,0,1,0,1,0,0,1]) #mutate working
'''fitness=[]
for i in range(10):
	fitness.append(random.random()*100)
	print fitness[i]
#print "Selected",selection(population,fitness) #selection working'''
#print evolvePopulation(population)
MAX = 30
no_of_generations = 0
population = createInitialPopulation(10)
fitness = getFitness(population)
fittest = findFittest(population, fitness, False)
print "Fittest",population[fittest]
print "Fitness",fitness,fitness[fittest]
#last_3_gen_fittest = []
#i=0
while no_of_generations != MAX:
	no_of_generations +=1
	#print "population:",population
	#fitness = getFitness(population)
	population = evolvePopulation(population)
	fitness = getFitness(population)
	fittest = findFittest(population, fitness, False)
	print "Population",population
	print "Fittest",population[fittest]
	print "Fitness",fitness,fitness[fittest]
	'''last_3_gen_fittest.append(fitness[fittest])
	i += 1
	if('''	
	


print "fitttestt",fittest,population[fittest]			
print "Relevant features:"
for i in range(no_of_features):
	if population[fittest][i]==1:
		print i

				

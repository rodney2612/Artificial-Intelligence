import pandas as pd
import random as random
import math as math
import numpy as np
#from naive_bayes import bayes
from svr import svr
from Utilities import number, cleanData, getTask, getInput, dropColumns

uniformProbability = 0.5
mutationProbability = 0.1
MAX_GENERATIONS = 10
population_size = 10
tournament_size = 4


#Create a random individual
def createIndividual(no_of_features):
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


def createInitialPopulation(size, no_of_features):
	population=[]
	for i in range(size):
		population.append(createIndividual(no_of_features))
	return population

def mutate(individual):
	for i in range(len(individual)):
		#Create random gene
		gene = int(round(random.random()))
		individual[i] = gene
	return individual

def selection(data_frame, population, no_of_features, task):
        #Create a tournament population
	tournament = []
        #For each place in the tournament get a random individual
	randomIds = []
        for i in range(tournament_size):
        	randomIndex = int(random.random() * len(population))
		#print "random index",randomIndex
        	tournament.append(population[randomIndex])
	fitness = getFitness(data_frame, tournament, no_of_features, task)
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

def evolvePopulation(data_frame, population, no_of_features, task):
	newPopulation = []
        #Loop over the population size and create new individuals with
        #crossover
	for i in range(len(population)):
		individual1 = selection(data_frame, population, no_of_features, task)            
		individual2 = selection(data_frame, population, no_of_features, task)          
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
	#print "Fittest",fittest
	if selection == True:
		return population[fittest]
	return fittest


def getFitness(data_frame,population, no_of_features, task):
	fitness=[]
	#print population
	for i in range(len(population)):
		current_no_of_features = 0
		for j in range(no_of_features):
			if population[i][j] == 1:
				current_no_of_features += 1
		new_data_frame=dropColumns(population[i],data_frame, no_of_features)
		accuracy=svr(new_data_frame, task)
		#print "Accuracy",accuracy
		#Giving more importance to states having less no of features
		fitness.append(0.99*accuracy + 0.01*(no_of_features - current_no_of_features))
	return fitness


'''print crossover([1,0,1,0,1,0,0,1],[0,0,0,1,1,0,1,0]) #crossover working
print mutate([1,0,1,0,1,0,0,1]) #mutate working
fitness=[]
for i in range(10):
	fitness.append(random.random()*100)
	print fitness[i]
#print "Selected",selection(population,fitness) #selection working'''

def geneticAlgorithm():
	data_frame, initial_state, task, no_of_features, original_accuracy = getInput()
	no_of_generations = 0
	print no_of_features,"features"
	population = createInitialPopulation(population_size, no_of_features)
	fitness = getFitness(data_frame, population, no_of_features, task)
	fittest = findFittest(population, fitness, False)
	print "Fittest",population[fittest]
	print "Fitness",fitness,fitness[fittest]
	#last_3_gen_fittest = []
	#i=0
	while no_of_generations != MAX_GENERATIONS:
		no_of_generations +=1
		#print "population:",population
		population = evolvePopulation(data_frame, population, no_of_features, task)
		fitness = getFitness(data_frame, population,no_of_features, task)
		fittest = findFittest(population, fitness, False)
		#print "Population",population
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

				
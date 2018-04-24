Algorithms used:

	1. Hill Climbing
	2. Genetic Algorithm
	
Our algorithms support Classification and regression using Machine Learning library scikit-learn.

HOW TO RUN: You need to run Search.py file.
				Command : python Search.py


FUNCTONS USED:

1. Search.py:

	It asks for input from user whether he wants to run hill climbing or genetic algorithm
	(h - hill climbing and g - genetic algorithm)
	On the basis of input, it calls either hill climbing or genetic algorithm function depending on input

2. Utilities.py:

	a. getInput():
			It asks user whether he wants to do classification or regression(us and the file name of the data set
	b. cleanData():
			It preprocesses the dataset removes the undefined symbols like '?' and takes care if dataset contains spaces between the columns for separation.
			NOTE : Our program does not work with the dataset using ';' for data separation.
	c. dropColumns()
			Drops columns from the dataset to include only those columns that are in state passed to the function

3. Hill_Climbing_Search.py:
	hillClimbing():
		This function gets the data set into a data frame structure using getInput().
		It creates state array whose each element refers to a feature in the data set
		It initialises the all the elements of the state zeroes(since at the start we don't select any features)
		With every iteration, we flip the bit in each state position and get the accuracy of the state using 
		svr(support vector) function from svr.py
		The best accuracy from among the above states is selected as the next state and the above process repeats 
		till we find that all successors of current state are worse than the current state

4. GeneticAlgorithm.py:
	
	geneticAlgortihm():
	
		a.	createInitialPopulation(): 
				Creates the initial population of states randomly.
		b. 	getfitness(): 
				It computes the fitness value of every state in the population. 
		c. 	getfittest(): 
				It returns the state with max fitness value.
		d.  evolvePopulation():
				Loop over the population size and create new individuals after creating 2 parents 
				from the selection function using crossover and mutation.
		e.  selection(): 
				It selects 4 individuals randomly from the population and return the fittest state among them.
		f.  crossover(): 
				It swaps the value of an state between two individuals randomly.
		g. 	mutation(): 
				It flips the bits of an individual randomly after some random unit of time.
		  
		
				
	
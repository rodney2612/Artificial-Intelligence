import numpy as np
from random import uniform

epsilon = 0.1
R = 4
C = 16
directions = 4
grid = np.zeros([R, C])
#correct_reading = np.zeroes(R,C,directions)

def initialize_grid():
	grid[0][4] = 1;
	grid[0][10] = 1;
	grid[0][14] = 1;
	grid[1][0] = 1;
	grid[1][1] = 1;
	grid[1][4] = 1;
	grid[1][6] = 1;
	grid[1][7] = 1;
	grid[1][9] = 1;
	grid[1][11] = 1;
	grid[1][13] = 1;
	grid[1][14] = 1;
	grid[1][15] = 1;
	grid[2][0] = 1;
	grid[2][4] = 1;
	grid[2][6] = 1;
	grid[2][7] = 1;
	grid[2][13] = 1;
	grid[2][14] = 1;
	grid[3][2] = 1;
	grid[3][6] = 1;
	grid[3][11] = 1;

empty_squares = []
def find_empty_locations():
	for x in range(0,R):
		for y in range(0,C):
			if grid[x][y] == 0:
				empty_squares.append((x,y))

# Gets all empty neighbours of a given location
def get_empty_neighbours(x,y):
    neighbours=[]
    if x-1 >= 0 and grid[x-1][y] == 0:
        neighbours.append((x-1,y))    
    if x+1 < R and grid[x+1][y] == 0:
        neighbours.append((x+1,y)) 
    if y+1 < C and grid[x][y+1] == 0:
        neighbours.append((x,y+1)) 
    if y-1 >= 0 and grid[x][y-1] == 0:
        neighbours.append((x,y-1))
    return neighbours

T = np.zeros([42,42])
def initialize_transition_matrix():
	for i in range(0,42):
		#print "Iteration: ",i
		(x,y) = empty_squares[i]
		#print "Current square",x,y
		neighbours = get_empty_neighbours(x,y)
		#print "Neighbours",neighbours
		neigh_len = len(neighbours)
		for j in range(0,neigh_len):
			index = empty_squares.index(neighbours[j])
			#print "Index",index
			T[i][index] = 1.0/neigh_len
			#print "Transition value:",T[i][index]

def sensor_reading(x,y):
    reading = [0,0,0,0] 
    if x-1 < 0 or grid[x-1][y] == 1:
        reading[0] = 1   
    if x+1 >= R or grid[x+1][y] == 1:
        reading[1] = 1
    if y+1 >= C or grid[x][y+1] == 1:
        reading[2] = 1
    if y-1 < 0 or grid[x][y-1] == 1:
        reading[3] = 1  
    return reading

def findDIT(percept,actual_percept):
	dit = 0
	for i in range(0,4):
		diff = actual_percept[i] - percept[i]
		if diff < 0:
			diff = -diff
		dit += diff
	return dit

def set_observation_matrix(initial_percept):
	O = np.zeros([42,42])
	for x in range(0,R):
		for y in range(0,C):
			if grid[x][y] == 0:		#if empty square
				#print x,y
				actual_percept = sensor_reading(x,y)
				#print "Actual percept: ", actual_percept
				dit = findDIT(initial_percept,actual_percept)
				#print "DIT: ", dit
				i = empty_squares.index((x,y))
				#print "Index: ", i
				O[i][i] = ((1 - epsilon) ** (4 - dit)) * (epsilon ** dit)
	return O

def printMatrices():
	print "Transition matrix:"
	print T
	print "Observation matrix:"
	print O
	print "Posterior probalibilities: "
	print f

def generate_percept_sequence():
	prob = 0.5
	generated_percept = []
	for i in range(0,4):
		if np.random.uniform(0,1) <= prob:
			generated_percept.append(1)
		else:
			generated_percept.append(0)
	return generated_percept

def normalize(f):
	sum = 0.0	
	for i in range(42):
		sum += f[i][0]
	for i in range(42):
		#print f[i][0],sum
		f[i][0] /= sum
	return f

def initialize_prior():
	f = np.zeros([42,1])
	for j in range(42):
		f[j][0] = 1/42.0
	return f

def viterbi_algo(O):
    prior = initialize_prior()
    g = np.multiply(np.transpose(T),np.transpose(prior))
    return np.argmax(np.matmul(O,np.amax(g,axis=1)))

initialize_grid()
find_empty_locations()
initialize_transition_matrix()
#initial_percept = [0, 0, 1, 1]		#NSW
percept_sequence = []
no_of_observations = 5
#Setting prior probabilities
f = initialize_prior()

most_likely_path = []
for j in range(0,no_of_observations):
	print "Observation no: ",j
	percept_sequence.append(generate_percept_sequence())
	print "Current percept: ", percept_sequence[j]
	O = set_observation_matrix(percept_sequence[j])
	#Computing posterior probabilities
	f = np.matmul(O,np.matmul(np.transpose(T),f)) #filtering equation
	f = normalize(f)
	#print f
	i = np.argmax(f)
	#Printing results
	#printMatrices()
	print "Most probable current location is: ", empty_squares[i]
	l = viterbi_algo(O)
	most_likely_path.append(empty_squares[l])


print "Most likely path: ",most_likely_path

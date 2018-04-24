import matplotlib.pyplot as plt

def plot(x,y,xlabel,ylabel):
	# plotting the points 
	plt.plot(x, y)
	 
	# naming the x axis
	plt.xlabel(xlabel)
	# naming the y axis
	plt.ylabel(ylabel)
	 
	plt.title(xlabel + " vs " + ylabel)
	 
	# function to show the plot
	plt.show()

#plot(x,y,"Feature division factor","Mean of accuracy scores")
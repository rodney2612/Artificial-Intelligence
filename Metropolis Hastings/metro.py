import numpy as np
from matplotlib.pylab import *
from scipy.stats import norm
import math

def p(x):
	return math.exp(-1 * x * x * x * x) * (2 + math.sin(5 * x) + math.sin(-2 * x * x))

def histogram_plot(samples):
	plt.hist(samples, normed=True, bins=30)
	plt.ylabel('Frequency')
	plt.savefig("hist.png")
	'''plot(samples)
	show()'''

N = 1500
sigma = 1
x = -1.0
samples = []
samples.append(x)

no_of_samples = 1
while(no_of_samples <= N):
    candidate = np.random.normal(x, sigma, 1)
    Qx_c = norm.pdf(x, candidate, sigma)
    Qc_x = norm.pdf(candidate, x, sigma)
    accept_prob = min(1.0, (p(candidate) / p(x)) * (Qx_c/Qc_x))
    if np.random.uniform(0,1) <= accept_prob:
        x = candidate
        no_of_samples += 1
        samples.append(x)    
histogram_plot(samples)

"""monte_carlo.py"""

import random
import numpy as np
import matplotlib.pyplot as plt

normdists = []
samps = []
maxvals = []
minvals = []

# create an array of 1000 normal distributions, each having 1000 points
for _ in range (1000):
	dist = np.random.normal(0, 1, 1000)
	normdists.append(dist)
	samps.append(random.choice(dist))
	maxvals.append(np.amax(dist))
	minvals.append(np.amin(dist))

plt.figure()
plt.hist(samps, histtype='bar')
plt.title('random samples from 1000 norm dists')
plt.show()
plt.clf()
# normally distributed, as expected

plt.figure()
plt.hist(maxvals, histtype='bar', label='max vals')
plt.hist(minvals, histtype='bar', label='min vals')
plt.title('max vals from 1000 norm dists')
plt.legend(loc='upper right')
plt.show()
# maxvals distribution is right-skewed
# minvals distribution is left-skewed
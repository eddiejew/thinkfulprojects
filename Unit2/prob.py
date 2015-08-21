"""prob.py"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

x = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9]

plt.boxplot(x)
plt. show()
plt.savefig("unit2_lesson1_boxplot.png")

plt.his(x, histtype='bar')
plt.show()
plt.savefig("unit2_lesson1_histogram.png")

plt.figure()
graph = stats.probplot(x, dist="norm", plot=plt)
plt.show()
plt.savefig("unit2_lesson1_QQplot.png")
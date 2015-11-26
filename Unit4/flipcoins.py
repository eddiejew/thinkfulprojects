"""flipcoins.py"""

import random
from pylab import plot
import matplotlib.pyplot as plt

# create a fair coin to be flipped
class Coin(object):
	# this is a simple fair coin, can be pseudorandomly flipped
	sides = ('heads', 'tails')
	last_result = None

	def flip(self):
		# call coin.flip() to flip the coin and record it as the last last_result
		self.last_result = result = random.choice(self.sides)
		return result

# some auxilliary functions to manipulate the coins
def create_coins(number):
	# create a list of a number of coin objects
	return [Coin() for _ in xrange(number)]

def flip_coins(coins):
	# side effect function, modifies object in place, returns None
	for coin in coins:
		coin.flip()

def count_heads(flipped_coins):
	return sum(coin.last_result == 'heads' for coin in flipped_coins)

def count_tails(flipped_coins):
	return sum(coin.last_result == 'tails' for coin in flipped_coins)

track_heads = []
track_tails = []
def main():
	coins = create_coins(1000)
	for i in range(100):
		flip_coins(coins)
		track_heads.append(count_heads(coins))
		track_tails.append(count_tails(coins))
		print "heads: "
		print count_heads(coins)
	plt.figure()
	plt.hist(track_heads, histtype='bar')
	plt.title('heads')
	plt.show()
	plt.clf()
	# nearly normal distribution of heads
	plt.figure()
	plt.hist(track_tails, histtype='bar')
	plt.title('tails')
	plt.show()
	plt.clf()
	# nearly normal distribution of tails

if __name__ == '__main__':
	main()
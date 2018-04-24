#
# genetic algorithm to make change for a given target value;
# attempts to minize the difference in amounts as well as
# the number of coins given

import os
import random
import pickle

from Chromo import BaseChromo
from GeneticAlg import GenAlg

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

# chromo size is 4 == pennies, nickels, dimes, quarters

# target value we're aiming for:
target = 0.44

class MyChromo(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=4,
			range=(0,10), dtype=int )

	# we'll use the default crossover and mutate functions
	# from BaseChromo (==crossover11 and mutateAll)

	# calculate the fitness function
	def calcFitness( self ):
		ccc = self.data
		val = 0.01*ccc[0] + 0.05*ccc[1] \
				+ 0.10*ccc[2] + 0.25*ccc[3]
		# we really want to make sure we give the correct amount
		# hence the heavier weighting; but we also want to minimize
		# the number of coins (lower weight)
		fitness = -100000.0*(val - target)*(val - target) \
				-1.0*(ccc[0]+ccc[1]+ccc[2]+ccc[3])
		return fitness

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	#
	# if a pickle-file exists, we load it
	if( os.path.isfile('ga_coins.pkl') ):
		with open('ga_coins.pkl','r') as fp:
			ga = pickle.load( fp )
		print( 'Read init data from file')

	else:
		# otherwise, init the gen-alg library from scratch
		ga = GenAlg( size=20,
			elitismPct   = 0.10,
			crossoverPct = 0.30,
			mutationPct  = 0.60,
			parentsPct   = 0.50,
			chromoClass  = MyChromo,
			minOrMax     = 'max',
			showBest     = 0
		)
		ga.initPopulation()

	#
	# Run it !!
	# : we'll just do 10 epochs of 10 steps each
	for i in range(10):
		ga.evolve( 10 )

		# give some running feedback on our progress
		print( str(i) + " best chromo:" )
		for i in range(1):
			print( ga.population[i] )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	for i in range(10):
		print( ga.population[i] )

	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	with open('ga_coins.pkl','w') as fp:
		pickle.dump( ga, fp )
	print('Final data stored to file (rm ga_coins.pkl to start fresh)')

if __name__ == '__main__':
	main()

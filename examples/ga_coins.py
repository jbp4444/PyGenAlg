#
# genetic algorithm to make change for a given target value;
# attempts to minize the difference in amounts as well as
# the number of coins given
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import os
import random

from PyGenAlg import GenAlg, BaseChromo, GenAlgOps, IoOps

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

	ga = GenAlg( size=20,
		elitism      = 0.10,
		crossover    = 0.60,
		pureMutation = 0.30,
		chromoClass  = MyChromo,
		#selectionFcn = GenAlgOps.tournamentSelection,
		#crossoverFcn = GenAlgOps.crossover22,
		#mutationFcn  = GenAlgOps.mutateFew,
		#pureMutationSelectionFcn = GenAlgOps.simpleSelection,
		#pureMutationFcn = GenAlgOps.mutateFew,
		#feasibleSolnFcn = GenAlgOps.disallowDupes,
		minOrMax     = 'max',
		showBest     = 0,
	)

	#
	# if a data-file exists, we load it
	if( os.path.isfile('ga_coins.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_coins.dat' )
		ga.appendToPopulation( pop )
		print( 'Read init data from file ('+str(len(pop))+' chromos)')
	else:
		# otherwise, init the gen-alg library from scratch
		ga.initPopulation()
		print( 'Created random init data' )

	#
	# Run it !!
	# : we'll just do 10 epochs of 10 steps each
	for i in range(10):
		ga.evolve( 10 )

		# give some running feedback on our progress
		print( 'iter '+str(i) + ", best chromo:" )
		for i in range(10):
			print( ga.population[i] )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	for i in range(10):
		print( ga.population[i] )


	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	IoOps.savePopulation( ga, 'ga_coins.dat' )
	print('Final data stored to file (rm ga_coins.dat to start fresh)')

if __name__ == '__main__':
	main()

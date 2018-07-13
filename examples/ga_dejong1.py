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

# de Jong's function #1
# f1(x)=sum(x(i)^2), i=1:n, -5.12<=x(i)<=5.12
# : with n=20
# : solution is all zeros

class MyChromo(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=20,
			range=(-5.12,5.12), dtype=float )

	# calculate the fitness function
	def calcFitness( self ):
		data = self.data
		fitness = 0.0
		for i in range(20):
			fitness = fitness + data[i]**2
		return fitness

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	ga = GenAlg( size=100,
		elitism      = 0.10,
		crossover    = 0.60,
		pureMutation = 0.30,
		parentsPct   = 0.80,
		chromoClass  = MyChromo,
		#selectionFcn = GenAlgOps.tournamentSelection,
		#crossoverFcn = GenAlgOps.crossover22,
		#mutationFcn  = GenAlgOps.mutateFew,
		#pureMutationSelectionFcn = GenAlgOps.simpleSelection,
		#pureMutationFcn = GenAlgOps.mutateFew,
		#feasibleSolnFcn = GenAlgOps.disallowDupes,
		minOrMax     = 'min',
		showBest     = 0,
	)

	# init the gen-alg library from scratch
	ga.initPopulation()
	print( 'Created random init data' )

	#
	# Run it !!
	# : we'll just do 10 epochs of 10 steps each
	for i in range(100):
		ga.evolve( 10 )

		# give some running feedback on our progress
		#print( 'iter '+str(i) + ", best chromo:" )
		#for i in range(10):
		#	print( ga.population[i] )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	for i in range(3):
		print( ga.population[i] )

if __name__ == '__main__':
	main()

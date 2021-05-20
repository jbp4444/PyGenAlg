#
# genetic algorithm to make change for a given target value;
# attempts to minize the difference in amounts as well as
# the number of coins given
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import os
import random
import string

from PyGenAlg import GenAlg, BaseChromo, GenAlgOps, IoOps

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

# chromo size is 4 == pennies, nickels, dimes, quarters
letters = string.ascii_uppercase + string.ascii_lowercase + string.punctuation + ' '

# target value we're aiming for:
solutionWord = 'Hello World!'
solution = []
for i in range(len(solutionWord)):
	j = letters.find( solutionWord[i] )
	solution.append( j )


class MyChromo(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=len(solution),
			range=(0,len(letters)-1), dtype=int )

	def __str__( self ):
		txt = 'data=' + ''.join( letters[i] for i in self.data ) \
				+ ' .. fit=' + str(self.fitness)
		return txt

	def calcFitness( self ):
		rtn = 0
		data = self.data
		for i in range(len(solution)):
			if( solution[i] == data[i] ):
				rtn = rtn + 1
		return rtn

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	ga = GenAlg( size=100,
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
	ga.describe()

	#
	# if a data-file exists, we load it
	if( False and os.path.isfile('ga_hello.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_hello.dat' )
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
	IoOps.savePopulation( ga, 'ga_hello.dat' )
	print('Final data stored to file (rm ga_hello.dat to start fresh)')

if __name__ == '__main__':
	main()

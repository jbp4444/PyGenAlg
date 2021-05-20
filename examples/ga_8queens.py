#
# genetic algorithm to solve the 8-queens puzzle
#

import os
import sys
import random
from copy import deepcopy

from PyGenAlg import GenAlg, BaseChromo, IoOps, GenAlgOps

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

popsize = 1200
outeriter = 20
inneriter = 20

class MyChromo(BaseChromo):
	def __init__( self, params={} ):
		BaseChromo.__init__( self, size=8,
			#range=(0,8-1), dtype=int )
			range=(0,7), dtype=int )

	# the chromo encodes the col-location for each row

	# the calculations for the fitness function
	# : note: this is only called if self.fitness==None
	#   (i.e. we don't recalc known values)
	def calcFitness( self ):
		grid = [ [ 0 for x in range(8) ] for y in range(8) ]

		# count the number of attacks on each queen (no row-attacks since, by defn, only 1 queen per row)
		fitness = 0

		# place the queens on a grid
		data = self.data
		for row in range(8):
			col = data[ row ]
			grid[row][col] = 1

		# assess the fitness metric
		col_count = [ 0 for i in range(8) ]
		diagTL_count = [ 0 for i in range(-8,8) ]
		diagTR_count = [ 0 for i in range(-8,8) ]
		for row in range(8):
			for col in range(8):
				if( grid[row][col] == 1 ):
					# test for same-col
					col_count[col] = col_count[col] + 1
					# test for top-left to bottom-right diagonals
					diagTL = row - col
					diagTL_count[diagTL] = diagTL_count[diagTL] + 1
					# test for top-right to bottom-left diagonals
					diagTR = row + col - 7 
					diagTR_count[diagTR] = diagTR_count[diagTR] + 1

		for i in range(8):
			if( col_count[i] > 1 ):
				fitness = fitness + col_count[i]

		for i in range(-8,8):
			if( diagTL_count[i] > 1 ):
				fitness = fitness + diagTL_count[i]
			if( diagTR_count[i] > 1 ):
				fitness = fitness + diagTR_count[i]

		return fitness

	def showGrid( self ):
		grid = [ [ 0 for x in range(8) ] for y in range(8) ]

		# place the stars
		data = self.data
		for row in range(8):
			col = data[ row ]
			grid[row][col] = 1

		txt = ''
		for row in range(8):
			for col in range(8):
				txt = txt + '%1d '%(grid[row][col])
			txt = txt + '\n'
		
		print( txt )
		return txt


# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	random.seed()

	ga = GenAlg( size=popsize,
		elitism      = 0.01,
		crossover    = 0.49,
		pureMutation = 0.50,
		# selectionFcn = GenAlgOps.simpleSelectionParentPct,
		selectionFcn = GenAlgOps.tournamentSelection,
		crossoverFcn = GenAlgOps.crossover22,
		mutationFcn = GenAlgOps.mutateFew,
		# for pure-mutation of all chromos .. no need to run tournament selection
		pureMutationSelectionFcn = GenAlgOps.nullSelection,
		pureMutationFcn = GenAlgOps.mutateAll,
		chromoClass  = MyChromo,
		minOrMax     = 'min',
		showBest     = 0,
		# optional params ..
		params       = {
			'tournamentSize': 3,
			'mutateNum': 3,    # for mutateFew .. make 3 mutations each time
			'parentPct': 0.50  # for parent-pct .. only top 50% of chromos are eligible as parents
		},
	)

	ga.describe()
	#print( 'random state', random.getstate() )

	#
	# if a data-file exists, we load it
	if( os.path.isfile('ga_8queens.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_8queens.dat' )
		ga.appendToPopulation( pop )
		print( 'Read init data from file ('+str(len(pop))+' chromos)')
	else:
		# otherwise, init the gen-alg library from scratch
		ga.initPopulation()
		print( 'Created random init data' )

	#
	# Run it !!
	# : we'll just do 10 epochs of 10 steps each
	for i in range( outeriter ):
		ga.evolve( inneriter )
		sys.stdout.write( '.' )
		sys.stdout.flush()

	#
	# all done ... output final results
	print( "\nfinal best chromo: " + str(ga.population[0].fitness) )
	ga.population[0].showGrid()
	ga.population[0].calcFitness()

	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	IoOps.savePopulation( ga, 'ga_8queens.dat' )
	print('Final data stored to file (rm ga_8queens.dat to start fresh)')

if __name__ == '__main__':
	main()

	# ch1 = MyChromo()
	# print( ch1 )
	# ch2 = MyChromo()
	# print( ch2 )
	# ch34 = my_crossover( ch1, ch2 )
	# print( ch34[0] )
	# print( ch34[1] )

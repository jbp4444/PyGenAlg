#
# genetic algorithm to create a 2-star (Two Not Touching) puzzle
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

puzzle_size = 10
popsize = 1200
outeriter = 20
inneriter = 20

puz_ranges = []
for i in range(puzzle_size):
	# for each row, don't have 2 stars on top of each other
	puz_ranges.append( (0,puzzle_size-1) )
	puz_ranges.append( (0,puzzle_size-2) )

class MyChromo(BaseChromo):
	def __init__( self, params={} ):
		BaseChromo.__init__( self, size=2*puzzle_size,
			#range=(0,puzzle_size-1), dtype=int )
			range=puz_ranges, dtype=int )

	# the chromo encodes the col-location for each row

	# the calculations for the fitness function
	# : note: this is only called if self.fitness==None
	#   (i.e. we don't recalc known values)
	def calcFitness( self ):
		numgrid = [ [ 0 for x in range(-1,puzzle_size+1) ] for y in range(-1,puzzle_size+1) ]

		# for each row, make sure there are 2 stars
		row_stars = [ 0 for i in range(puzzle_size) ]
		# for each col, make sure there are 2 stars
		col_stars = [ 0 for i in range(puzzle_size) ]

		# place the stars
		data = self.data
		for row in range(puzzle_size):
			col1 = data[ 2*row ]
			idx = data[ 2*row+1 ]
			if( idx < col1 ):
				col2 = idx
			else:
				col2 = idx + 1
			numgrid[row][col1] = 1
			numgrid[row][col2] = 1

		# assess the fitness metric
		fitness = 0
		for row in range(puzzle_size):
			for col in range(puzzle_size):
				if( numgrid[row][col] == 1 ):
					row_stars[row] = row_stars[row] + 1
					col_stars[col] = col_stars[col] + 1

				# look for more than 1 star in a 3x3 region
				ctr = numgrid[row-1][col-1] + numgrid[row-1][col] + numgrid[row-1][col+1] \
						+ numgrid[row][col-1] + numgrid[row][col] + numgrid[row][col+1] \
						+ numgrid[row+1][col-1] + numgrid[row+1][col] + numgrid[row+1][col+1]
				if( ctr >= 2 ):
					fitness = fitness + 100

		for row in range(puzzle_size):
			if( row_stars[row] != 2 ):
				fitness = fitness + 200
		for col in range(puzzle_size):
			if( col_stars[col] != 2 ):
				fitness = fitness + 200

		return fitness

	def showGrid( self ):
		numgrid = [ [ 0 for x in range(puzzle_size) ] for y in range(puzzle_size) ]

		# place the stars
		data = self.data
		dtxt = ''
		for row in range(puzzle_size):
			col1 = data[ 2*row ]
			idx = data[ 2*row+1 ]
			if( idx < col1 ):
				col2 = idx
			else:
				col2 = idx + 1
			numgrid[row][col1] = 1
			numgrid[row][col2] = 1

		txt = ''
		for row in range(puzzle_size):
			for col in range(puzzle_size):
				txt = txt + '%1d '%(numgrid[row][col])
			txt = txt + '\n'
		
		print( txt )
		return txt

# based on crossover22 (2 crossover-points leads to 2 children)
# : but in this case, pairs of chromos work together, so cut on pair-boundaries
def my_crossoverx( mother, father, params={} ):
	child1 = deepcopy(mother)
	child1.fitness = None
	child2 = deepcopy(father)
	child2.fitness = None
	# 2 cutover points
	(index1,index2) = random.sample( range(puzzle_size), k=2 )
	index1 = index1 * 2
	index2 = index2 * 2
	if( index1 > index2 ):
		index1, index2 = index2, index1
	child1.data[index1:index2] = father.data[index1:index2]
	child2.data[index1:index2] = mother.data[index1:index2]
	return [child1,child2]
def my_crossover( mother, father, params={} ):
	child1 = deepcopy(mother)
	child1.fitness = None
	child2 = deepcopy(father)
	child2.fitness = None
	# 2 cutover points
	idx = sorted( random.sample( range(puzzle_size), k=4 ) )
	# continue out to end of chromo
	idx.append( puzzle_size )
	lst = 0
	flipflop = True
	for i in idx:
		i2 = 2*i
		if( flipflop ):
			child1.data[lst:i2] = father.data[lst:i2]
			child2.data[lst:i2] = mother.data[lst:i2]
		else:
			child1.data[lst:i2] = mother.data[lst:i2]
			child2.data[lst:i2] = father.data[lst:i2]
		lst = i2
		flipflop = not flipflop
	return [child1,child2]


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
		crossoverFcn = my_crossover,
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
	if( os.path.isfile('ga_2star.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_2star.dat' )
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
	IoOps.savePopulation( ga, 'ga_2star.dat' )
	print('Final data stored to file (rm ga_2star.dat to start fresh)')

if __name__ == '__main__':
	main()

	# ch1 = MyChromo()
	# print( ch1 )
	# ch2 = MyChromo()
	# print( ch2 )
	# ch34 = my_crossover( ch1, ch2 )
	# print( ch34[0] )
	# print( ch34[1] )

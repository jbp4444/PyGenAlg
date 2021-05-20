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

popsize = 2222
outeriter = 10
inneriter = 10

class MyChromo(BaseChromo):
	def __init__( self, params={} ):
		BaseChromo.__init__( self, size=81,
			range=(0,8), dtype=int )

	# the chromo encodes the col-location for each row

	# the calculations for the fitness function
	# : note: this is only called if self.fitness==None
	#   (i.e. we don't recalc known values)
	def calcFitness( self ):
		TheArray = self.data

		# set fitnesses for columns, rows, and squares initially to 0
		fitnessColumns = 0
		fitnessRows = 0
		fitnessSquares = 0
		# go through each column
		for i in range(9):
			# Go through each cell in a column, add it to the ColumnMap according
			# to the cell value
			ColumnMap = {}
			for j in range(9):
				# check for uniqueness in row
				key = TheArray[ i*9 + j ]
				if( key not in ColumnMap ):
					ColumnMap[key] = 0
				ColumnMap[key] = ColumnMap[key] + 1

			# accumulate the column fitness based on the number of entries in the ColumnMap
			fitnessColumns = fitnessColumns + len(ColumnMap)
			#fitnessColumns = fitnessColumns + (1.0/(10-len(ColumnMap)))
			#fitnessColumns += (float)Math.Exp(ColumnMap.Count*10 - 90)/9
		
		# go through each row next
		for i in range(9):
			# Go through each cell in a row, add it to the RowMap according
			# to the cell value
			RowMap = {}
			for j in range(9):
				# check for uniqueness in row
				key = TheArray[ j*9 + i ]
				if( key not in RowMap ):
					RowMap[key] = 0
				RowMap[key] = RowMap[key] + 1

			# accumulate the row fitness based on the number of entries in the RowMap
			fitnessRows = fitnessRows + len(RowMap)
			# fitnessRows = fitnessRows + (1.0/(10-len(RowMap)))
			# fitnessRows += (float)Math.Exp(RowMap.Count*10 - 90)/9

		# go through next square
		for l in range(3):
			for k in range(3):
				# Go through each cell in a 3 x 3 square, add it to the SquareMap according
				# to the cell value
				SquareMap = {}
				for i in range(3):
					for j in range(3):
						key = TheArray[ (i+k*3)*9 + (j+l*3) ]
						if( key not in SquareMap ):
							SquareMap[key] = 0
						# accumulate the square fitness based on the number of entries in the SquareMap
						SquareMap[key] = SquareMap[key] + 1

				fitnessSquares = fitnessSquares + len(SquareMap)
				# fitnessSquares = fitnessSquares + (1.0/(10-len(SquareMap)))

		# The fitness of the entire Sudoku Grid is the product
		# of the column fitness, row fitness and 3x3 square fitness
		CurrentFitness = fitnessColumns * fitnessRows * fitnessSquares
		return CurrentFitness

	def showGrid( self ):
		grid = [ [ 0 for x in range(9) ] for y in range(9) ]

		# place the numbers
		data = self.data
		i = 0
		for row in range(9):
			for col in range(9):
				grid[row][col] = data[i]
				i = i + 1

		txt = ''
		for row in range(9):
			if( (row%3) == 0 ):
				txt = txt + '+-------+-------+-------+\n'
			for col in range(9):
				if( (col%3) == 0 ):
					txt = txt + '| '
				txt = txt + '%1d '%(grid[row][col]+1)
			txt = txt + '|\n'
		txt = txt + '+-------+-------+-------+\n'
		
		print( txt )
		return txt


# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	random.seed()

	ga = GenAlg( size=popsize,
		elitism      = 0.01,
		crossover    = 0.39,
		pureMutation = 0.60,
		selectionFcn = GenAlgOps.simpleSelectionParentPct,
		#selectionFcn = GenAlgOps.tournamentSelection,
		crossoverFcn = GenAlgOps.crossoverN1,
		mutationFcn = GenAlgOps.mutateRandom,
		# for pure-mutation of all chromos .. no need to run tournament selection
		pureMutationSelectionFcn = GenAlgOps.nullSelection,
		pureMutationFcn = GenAlgOps.mutateAll,
		chromoClass  = MyChromo,
		minOrMax     = 'max',
		showBest     = 0,
		# optional params ..
		params       = {
			'tournamentSize': 3,
			'mutateNum': 3,    # for mutateFew .. make 3 mutations each time
			'chromoMutationPct': 0.01,   # for mutateRandom .. pct chance that each chromo is mutated
			'crossoverNumPts': 3,   # for crossoverN1/N2 .. number of crossover points
			'parentPct': 0.50  # for parent-pct .. only top 50% of chromos are eligible as parents
		},
	)

	ga.describe()
	#print( 'random state', random.getstate() )

	#
	# if a data-file exists, we load it
	if( os.path.isfile('ga_sudoku.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_sudoku.dat' )
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
	IoOps.savePopulation( ga, 'ga_sudoku.dat' )
	print('Final data stored to file (rm ga_sudoku.dat to start fresh)')

if __name__ == '__main__':
	main()

	# ch1 = MyChromo()
	# print( ch1 )
	# ch2 = MyChromo()
	# print( ch2 )
	# ch34 = my_crossover( ch1, ch2 )
	# print( ch34[0] )
	# print( ch34[1] )

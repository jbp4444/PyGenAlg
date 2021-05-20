#
# genetic algorithm to create a math-square problem
#    9 = 8 + 1
#    -   -   +
#    4 + 2 = 6
#    =   =   =
#    5 + 6 ? 7
#
# use all 9 digits, +/-/=, to make all the math work
#

# digit to chromo mapping:
#    1 x 2 x 3
#    x   x   x
#    4 x 5 x 6
#    x   x   x
#    7 x 8 x x
#
# operator to chromo mapping:
#    x  9  x x  x
#    12   13   14
#    x 10  x x  x
#    x     x    x
#    x 11  x x  x


import os
import sys
import random
from copy import deepcopy
import itertools

from PyGenAlg import GenAlg, BaseChromo, IoOps, GenAlgOps

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

operators = [ '?', '+', '-', '=', '=' ]   # fudge the 1-offset
otherops  = [ '?', '=', '=', '+', '-' ]
# for op#1, coord loc=(r=0,c=1), paired with (r=0,c=3); etc.
oplocations = [ [0,1,0,3], [2,1,2,3], [4,1,4,3], [1,0,3,0], [1,2,3,2], [1,4,3,4] ]
maxnum = 12

popsize = 1200
outeriter = 20
inneriter = 20


class MyChromo(BaseChromo):
	def __init__( self, params={} ):
		BaseChromo.__init__( self, size=9+6,
			range=[ (1,maxnum),(1,maxnum-1),(1,maxnum-2),(1,maxnum-3),(1,maxnum-4),(1,maxnum-5),(1,maxnum-6),(1,maxnum-7),(1,maxnum-8),     # digits
					(1,4),(1,4),(1,4),(1,4),(1,4),(1,4) ],  # operators: +/-/=
			dtype=int )

	# calc the digit sequence
	def calc_digits( self ):
		digits = []
		rem_digits = [ (i+1) for i in range(maxnum) ]
		for i in range(9):
			this_dig = rem_digits[ self.data[i]-1 ]
			rem_digits.remove( this_dig )
			digits.append( this_dig )
		return digits

	def calc_grid( self ):
		grid = [ ['x' for i in range(5)] for j in range(5) ]
		grid[1][1] = ' '
		grid[3][1] = ' '
		grid[1][3] = ' '
		grid[3][3] = ' '

		# place the digits
		digits = self.calc_digits()
		for i in range(9):
			row = 2*(i//3)
			col = 2*(i%3)
			grid[row][col] = str(digits[i])

		# add in operators
		for i in range(6):
			row,col,row2,col2 = oplocations[i]
			grid[row][col]    = operators[ self.data[8+i] ]
			grid[row2][col2]  = otherops[ self.data[8+i] ]

		return grid

	# the calculations for the fitness function
	# : note: this is only called if self.fitness==None
	#   (i.e. we don't recalc known values)
	def calcFitness( self ):
		grid = self.calc_grid()
		eqns = [ ' '.join(grid[i]) for i in range(0,5,2) ]
		eqns.extend( [ ' '.join([grid[0][i],grid[1][i],grid[2][i],grid[3][i],grid[4][i]]) for i in range(0,5,2) ] )

		fitness = 0
		for i in range(6):
			eqn = eqns[i].replace( '=', '==' )
			tf = eval( eqn )
			# print( '[%s]->%d'%(eqn,tf) )
			if( tf ):
				fitness = fitness + 1

		return fitness

	def showGrid( self ):
		grid = self.calc_grid()

		# catenate it all into one string
		txt = '\n'.join( [ ' '.join(grid[i]) for i in range(5) ] )
		print( txt )
		return txt

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	random.seed()

	ga = GenAlg( size=popsize,
		elitism      = 0.03,
		crossover    = 0.47,
		pureMutation = 0.50,
		selectionFcn = GenAlgOps.simpleSelectionParentPct,
		crossoverFcn = GenAlgOps.crossover22,
		mutationFcn = GenAlgOps.mutateFew,
		# for pure-mutation of all chromos .. no need to run tournament selection
		# pureMutationSelectionFcn = lambda x: [0,0],
		# pureMutationFcn = GenAlgOps.mutateAll,
		chromoClass  = MyChromo,
		minOrMax     = 'max',
		showBest     = 0,
		# optional params ..
		params       = {
			'mutateNum': 3,    # for mutateFew .. make 2 mutations each time
			'parentPct': 0.50  # for parent-pct .. only top 50% of chromos are eligible as parents
		},
	)

	ga.describe()
	#print( 'random state', random.getstate() )

	#
	# if a data-file exists, we load it
	if( os.path.isfile('ga_mathsq.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_mathsq.dat' )
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
	debuglevel = 100
	ga.population[0].calcFitness()

	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	IoOps.savePopulation( ga, 'ga_mathsq.dat' )
	print('Final data stored to file (rm ga_mathsq.dat to start fresh)')

if __name__ == '__main__':
	main()

	# ch = MyChromo()
	# print( 'chromo', str(ch) )
	# txt = ch.showGrid()
	# print( txt )
	# fit = ch.calcFitness()
	# print( fit )

	# go thru all possible combos
	# myiter = []
	# for i in range(8+6):
	# 	myiter.append( range(ch.dataRange[i][0],ch.dataRange[i][1]+1) )

	# c = 0
	# for ijk in itertools.product(*myiter):
	# 	for i in range(8+6):
	# 		ch.data[i] = ijk[i]
		
	# 	ch.fitness = ch.calcFitness()
	# 	if( ch.fitness == 6 ):
	# 		print( 'soln', ch.data )
	# 		ch.showGrid()

	# 	c = c + 1
	# 	if( (c%100000) == 0 ):
	# 		print( c )


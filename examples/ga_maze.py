#
# genetic algorithm to place circular sprinklers into a field
# to provide the best coverage of that field
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import os
import sys
import random

from PyGenAlg import GenAlg, BaseChromo, IoOps

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

width = 5
height = 5
maxnum = width*height

debuglevel = 0

class MyChromo(BaseChromo):
	def __init__( self, params={} ):
		BaseChromo.__init__( self, size=maxnum,
			range=(0,maxnum-1), dtype=int )

	# the calculations for the fitness function
	# : note: this is only called if self.fitness==None
	#   (i.e. we don't recalc known values)
	def calcFitness( self ):
		fitness = 0.0
		# for each number, make sure it appears exactly once
		appears = [ 0 for i in range(maxnum) ]
		# and make sure it's +1 neighbor is nearby
		in_a_row = 1
		for y in range(height):
			for x in range(width):
				val = self.data[ y*width + x ]
				if( appears[val] == 0 ):
					appears[val] = 1
					fitness = fitness + in_a_row
					in_a_row = in_a_row + 1
				else:
					fitness = fitness - 100
				
				for nlist in [ (1,0), (-1,0), (0,1), (0,-1) ]:
					nx = x + nlist[0]
					ny = y + nlist[1]
					if( (nx>=0) and (nx<width) and (ny>=0) and (ny<height) ):
						ni = ny*width + nx
						nbr = self.data[ni]
						if( nbr == (val+1) ):
							fitness = fitness + 100

		return fitness

	# use PIL to draw a simple PNG image of the fields,
	# colored by a simple colormap
	def showGrid( self ):
		i = 0
		sep = '+----' * width + '+'
		for r in range(height):
			print( sep )
			txt = ''
			for c in range(width):
				txt = txt + '| %2d ' % self.data[i]
				i = i + 1
			print( txt+'|' )
		print( sep )

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():
	global debuglevel

	ga = GenAlg( size=250,
		elitismPct   = 0.10,
		crossoverPct = 0.30,
		mutationPct  = 0.60,
		parentsPct   = 0.50,
		chromoClass  = MyChromo,
		minOrMax     = 'max',
		showBest     = 0
	)

	#
	# if a data-file exists, we load it
	if( os.path.isfile('ga_maze.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_maze.dat' )
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
		ga.evolve( 100 )
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
	IoOps.savePopulation( ga, 'ga_maze.dat' )
	print('Final data stored to file (rm ga_maze.dat to start fresh)')

if __name__ == '__main__':
	main()

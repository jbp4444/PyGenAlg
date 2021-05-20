#
# genetic algorithm to create a maze by putting a "flow field"
# onto a grid and seeing if you can flow across all squares
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
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

width = 10
height = 10
popsize = 4000
outeriter = 10
inneriter = 10

# directions: 0=up, 1=down, 2=left, 3=right
start_x = 3
start_y = 3

class MyChromo(BaseChromo):
	def __init__( self, params={} ):
		BaseChromo.__init__( self, size=width*height,
			range=(0,3), dtype=int )

	# the grid is essentially a flow-field, 
	# a solution = a walk thru the field that connects all points
	# : grid[x,y] = self.data[ y*width + x ]
	def makegrid( self ):
		numgrid = [ [ -1 for x in range(width) ] for y in range(height) ]

		# for each number, make sure it appears exactly once
		appears = [ 0 for i in range(width*height) ]
		# start at given point
		x = start_x
		y = start_y
		for i in range(width*height):
			# mark the current index in numgrid
			if( numgrid[y][x] == -1 ):
				# this space is empty, mark it
				numgrid[y][x] = i
				appears[i] = 1
			else:
				# this space already claimed, ignore it
				pass

			# now, where do we go next?
			current_dir = self.data[ y*width + x ]
			if( current_dir == 0 ):
				y = y - 1
			elif( current_dir == 1 ):
				y = y + 1
			elif( current_dir == 2 ):
				x = x - 1
			elif( current_dir == 3 ):
				x = x + 1

			if( x < 0 ):
				x = 0
			if( x >= width ):
				x = width-1
			if( y < 0 ):
				y = 0
			if( y >= height ):
				y = height-1

		return appears,numgrid

	# the calculations for the fitness function
	# : note: this is only called if self.fitness==None
	#   (i.e. we don't recalc known values)
	def calcFitness( self ):
		fitness = 0.0

		appears,numgrid = self.makegrid()

		for i in range(width*height):
			fitness = fitness + appears[i]

		return fitness

	# use PIL to draw a simple PNG image of the fields,
	# colored by a simple colormap
	def showGrid( self ):
		appears,numgrid = self.makegrid()

		if( True ):
			i = 0
			sep = '+-----'*width + '+'
			for y in range(height):
				print( sep )
				txt = ''
				for x in range(width):
					cdir = '.'
					if( self.data[i] == 0 ):
						cdir = '^'
					elif( self.data[i] == 1 ):
						cdir = 'v'
					elif( self.data[i] == 2 ):
						cdir = '<'
					elif( self.data[i] == 3 ):
						cdir = '>'
					txt = txt + '| %c%2d ' % (cdir,numgrid[y][x])
					i = i + 1
				print( txt+'|' )
			print( sep )

		else:
			i = 0
			sep = '+---' * (2*width*height) + '+'
			for r in range(2*width*height):
				print( sep )
				txt = ''
				for c in range(2*width*height):
					cdir = 'x'
					if( self.data[i] == 0 ):
						cdir = '^'
					elif( self.data[i] == 1 ):
						cdir = 'v'
					elif( self.data[i] == 2 ):
						cdir = '<'
					elif( self.data[i] == 3 ):
						cdir = '>'
					txt = txt + '| %c ' % cdir
					i = i + 1
				print( txt+'|' )
			print( sep )

def paired_mutation( mother, params={} ):
	child = deepcopy(mother)
	child.fitness = None
	for k in range(4):
		# which pair of nodes do we mutate?
		d = random.randint(0,3)
		if( d == 0 ):
			x = random.randint(0,width-1)
			y = random.randint(1,height-1)
			i = y*width + x
			j = (y-1)*width + x
		elif( d == 1 ):
			x = random.randint(0,width-1)
			y = random.randint(0,height-2)
			i = y*width + x
			j = (y+1)*width + x
		elif( d == 2 ):
			x = random.randint(1,width-1)
			y = random.randint(0,height-1)
			i = y*width + x
			j = y*width + x - 1
		elif( d == 3 ):
			x = random.randint(0,width-2)
			y = random.randint(0,height-1)
			i = y*width + x
			j = y*width + x + 1

		child.data[i] = random.randint( mother.dataRange[i][0], mother.dataRange[i][1] )
		child.data[j] = random.randint( mother.dataRange[j][0], mother.dataRange[j][1] )

	return child

def quad_mutation( mother, params={} ):
	child = deepcopy(mother)
	child.fitness = None
	for k in range(2):
		x = random.randint(1,width-1)
		y = random.randint(1,height-1)
		i = y*width + x
		j = (y-1)*width + x
		k = y*width + x - 1
		l = (y-1)*width + x - 1

		child.data[i] = random.randint( mother.dataRange[i][0], mother.dataRange[i][1] )
		child.data[j] = random.randint( mother.dataRange[j][0], mother.dataRange[j][1] )

	return child


# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	ga = GenAlg( size=popsize,
		elitism      = 0.01,
		crossover    = 0.59,
		pureMutation = 0.40,
		mutationFcn  = paired_mutation,
		# for pure-mutation of all chromos .. no need to run tournament selection
		pureMutationSelectionFcn = lambda x: [0,0],
		pureMutationFcn = GenAlgOps.mutateAll,
		chromoClass  = MyChromo,
		minOrMax     = 'max',
		showBest     = 0,
		# optional params ..
		params       = {
			'mutateNum': 10    # for mutateFew .. make 2 mutations each time
		},
	)

	ga.describe()

	#
	# if a data-file exists, we load it
	if( os.path.isfile('ga_maze2.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_maze2.dat' )
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
	IoOps.savePopulation( ga, 'ga_maze2.dat' )
	print('Final data stored to file (rm ga_maze2.dat to start fresh)')

if __name__ == '__main__':
	main()

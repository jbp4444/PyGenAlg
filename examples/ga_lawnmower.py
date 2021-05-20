#
# genetic algorithm to mow an irregular lawn with shortest path
# : TODO: how to add other constraints?
#
# Copyright (C) 2021, John Pormann, Duke University Libraries
#

import os
import math
import random

from PIL import Image, ImageDraw

from PyGenAlg import GenAlg, BaseChromo, GenAlgOps, IoOps

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)
DIR_LFT = -1
DIR_FWD = 0
DIR_RGT = 1

# shape of the field
WIDTH  = 10
HEIGHT = 12
FIELD = [
	[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
	[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
	[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
	[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
]
PATH_LEN = sum( [ sum(x) for x in FIELD ] )
MAX_PATH_LEN = WIDTH*HEIGHT
START_X = 0
START_Y = 0
START_D = 1   # 0=North, 1=East, 2=South, 3=West

class MyChromo(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=MAX_PATH_LEN,
			range=(-1,1), dtype=int )

	# the calculations for the fitness function
	# : note: this is only called if self.fitness==None
	#   (i.e. we don't recalc known values)
	def calcFitness( self ):
		data = self.data

		grid = [ [ 0 for i in range(WIDTH) ] for j in range(HEIGHT) ]

		# is there a path through the space?
		cur_x = START_X
		cur_y = START_Y
		cur_d = START_D
		oob_penalty = 0
		for i in range(PATH_LEN):
			d = data[i]

			new_d = ( cur_d + d )%4
			if( new_d == 0 ):
				# north
				cur_y = cur_y - 1
			elif( new_d == 1 ):
				# east
				cur_x = cur_x + 1
			elif( new_d == 2 ):
				# west
				cur_x = cur_x - 1
			elif( new_d == 3 ):
				# south
				cur_y = cur_y + 1

			if( (cur_x>=0) and (cur_x<WIDTH) and (cur_y>=0) and (cur_y<HEIGHT) ):
				grid[cur_y][cur_x] = grid[cur_y][cur_x] + 1
			else:
				oob_penalty = oob_penalty + 1

		miss_penalty = 0
		for y in range(HEIGHT):
			for x in range(WIDTH):
				if( grid[y][x] == 0 ):
					miss_penalty = miss_penalty + 1

		fit = miss_penalty + oob_penalty
		return fit

	# use PIL to draw a simple PNG image of the fields,
	# colored by a simple colormap
	def drawImage( self ):
		nodes = self.data

		img  = Image.new( 'RGB', (500,500), (0,0,0) )
		draw = ImageDraw.Draw(img)

		scl = 20.0
		ofs = 500/2

		for i in range(nnodes):
			x = nodes[2*i]
			y = nodes[2*i+1]
			draw.rectangle( [scl*x+ofs+1,scl*y+ofs+1, scl*x+ofs-1,scl*y+ofs-1], fill=(255,0,0) )

		for i in range(nbounds):
			x,y = bnodes[i]
			draw.rectangle( [scl*x+ofs,scl*y+ofs, scl*x+ofs-1,scl*y+ofs-1], fill=(0,255,0) )

		img.save( "spread.png" )


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
		minOrMax     = 'min',
		showBest     = 0,
	)

	#
	# if a pickle-file exists, we load it
	if( os.path.isfile('ga_spread.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_spread.dat' )
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
		print( str(i) + " best chromo:" )
		for j in range(1):
			print( ga.population[j] )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	for i in range(10):
		print( ga.population[i] )
	ga.population[0].drawImage()

	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	IoOps.savePopulation( ga, 'ga_spread.dat' )
	print('Final data stored to file (rm ga_spread.dat to start fresh)')

if __name__ == '__main__':
	main()

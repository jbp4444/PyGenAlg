#
# genetic algorithm to place circular regions with max separation
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

# number of items to spread out
# each one has an (x,y) center == 2 chromos,
nnodes = 5
dataRanges = [ (-10,10) for i in range(2*nnodes) ]

# boundary region
twopi = 2*math.pi
nbounds = 100
bnodes = [ (10.0*math.cos(twopi*i/nbounds),10.0*math.sin(twopi*i/nbounds)) for i in range(nbounds) ]

class MyChromo(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=2*nnodes,
			range=dataRanges, dtype=float )

	# the calculations for the fitness function
	# : note: this is only called if self.fitness==None
	#   (i.e. we don't recalc known values)
	def calcFitness( self ):
		nodes = self.data

		# calc some goodness-of-fit metrics
		penalty = 0
		for i in range(nnodes):
			x1 = nodes[2*i]
			y1 = nodes[2*i+1]
			nmin = 9.9e9
			nmax = 0.0
			bmin = 9.9e9
			bmax = 0.0

			dist = y1*y1 + x1*x1
			if( dist > 100 ):
				penalty = penalty + 1

			for j in range(nnodes):
				if( i != j ):
					x2 = nodes[2*j]
					y2 = nodes[2*j+1]
					dx = x2 - x1
					dy = y2 - y1
					dist = math.sqrt(dy*dy + dx*dx)
					if( dist > nmax ):
						nmax = dist
					if( dist < nmin ):
						nmin = dist

			for j in range(nbounds):
				x2,y2 = bnodes[j]
				dx = x2 - x1
				dy = y2 - y1
				dist = math.sqrt(dy*dy + dx*dx)
				if( dist > bmax ):
					bmax = dist
				if( dist < bmin ):
					bmin = dist

		fit = abs(2*bmin-nmin) + 10*abs(nmax-nmin) + 100*penalty

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

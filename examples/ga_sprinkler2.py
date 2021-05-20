#
# genetic algorithm to place circular sprinklers into a field
# to provide the best coverage of that field
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import os
import random
import math

from PIL import Image, ImageDraw

from PyGenAlg import GenAlg, BaseChromo, GenAlgOps, IoOps

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

# assume a 100x100 field with 5 circular water sprinklers
# each one has an (x,y) center == 2 chromos,
#  and a radius == 3rd chromo
#  and start/stop angle = 4th/5th chromos
num_sprayers = 4
# some of them may have smaller/larger spray-radii
dataRanges = [
	(0,100), (0,100), (1,30), (-math.pi,math.pi), (-math.pi,math.pi),
	(0,100), (0,100), (1,30), (-math.pi,math.pi), (-math.pi,math.pi),
	(0,100), (0,100), (1,40), (-math.pi,math.pi), (-math.pi,math.pi),
	(0,100), (0,100), (1,40), (-math.pi,math.pi), (-math.pi,math.pi),
	(0,100), (0,100), (1,50), (-math.pi,math.pi), (-math.pi,math.pi)
]
# fitness = number of squares with at least 1 unit of water
# but negative fitness if more than 2 units;
# i.e. waterlogged plants will die
scoring = [ 0, 10, 7, 2, -2, -5 ]

# a simple colormap for the png output
cmap = [ (0,0,0),
	(255,0,0),(255,255,0),(0,255,0),(0,255,255),
	(0,0,255),(255,0,255),(255,255,255),
	(128,128,128),(192,192,192),
	(128,0,0),(128,128,0),(0,128,0),(128,0,128),
	(0,128,128),(0,0,128),(128,0,128)
]

cutout_region = [ 75, 25 ]

class MyChromo(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=5*num_sprayers,
			range=dataRanges, dtype=float )

	# internal func to count 'units' of water on each
	# block of the grid which defines the field
	# : in this case, we'll zero-out a quadrant to simulate
	#   a non-square field
	def calcWaterfall( self ):
		# how much water hits each grid-square?
		water = [ 0 for i in range(100*100) ]
		for i in range(num_sprayers):
			xc  = self.data[5*i]
			yc  = self.data[5*i+1]
			rad = self.data[5*i+2]
			rad2 = rad*rad
			ang0 = self.data[5*i+3]
			ang1 = self.data[5*i+4]

			for x in range(int(xc-rad-1),int(xc+rad+1)):
				if( (x < 0) or (x >= 100) ):
					continue
				for y in range(int(yc-rad-1),int(yc+rad+1)):
					if( (y < 0) or (y >= 100) ):
						continue
					ang = math.atan2( y-yc, x-xc )
					if( (ang<ang0) or (ang>ang1) ):
						continue
					dd = (x-xc)*(x-xc) + (y-yc)*(y-yc)
					if( dd <= rad2 ):
						water[100*y+x] = water[100*y+x] + 1

			# zero-out any non-field areas (e.g. if the field
			# is not a square) ... 
			# let's make it an L shape with a 50x50 cutout
			for x in range( cutout_region[0] ):
				for y in range( cutout_region[1] ):
					water[100*y+x] = 0

		return water

	# the calculations for the fitness function
	# : note: this is only called if self.fitness==None
	#   (i.e. we don't recalc known values)
	def calcFitness( self ):
		water = self.calcWaterfall()

		sum = 0
		for i in range(100*100):
			sum = sum + scoring[ water[i] ]

		return sum

	# use PIL to draw a simple PNG image of the fields,
	# colored by a simple colormap
	def drawImage( self ):
		img  = Image.new( 'RGB', (500,500), (0,0,0) )
		draw = ImageDraw.Draw(img)

		water = self.calcWaterfall()

		for y in range(100):
			for x in range(100):
				cc = cmap[ water[100*y+x] ]
				draw.rectangle( [5*x,5*y, 5*x+4,5*y+4], fill=cc )

		# mark out the not-a-field area
		for x in range( cutout_region[0] ):
			for y in range( cutout_region[1] ):
				draw.rectangle( [5*x,5*y, 5*x+4,5*y+4], fill=(128,128,128) )

		# put circles where sprinkler-centers are
		for i in range(num_sprayers):
			xc  = self.data[5*i]
			yc  = self.data[5*i+1]
			rad = self.data[5*i+2]
			draw.ellipse( (5*xc-10,5*yc-10,5*xc+10,5*yc+10), outline=(255,255,255), width=1 )
			draw.ellipse( (5*xc-5*rad,5*yc-5*rad,5*xc+5*rad,5*yc+5*rad), outline=(255,255,255), width=1 )

		img.save( "field_spray2.png" )

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

	#
	# if a pickle-file exists, we load it
	if( os.path.isfile('ga_sprinkler2.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_sprinkler2.dat' )
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
	IoOps.savePopulation( ga, 'ga_sprinkler2.dat' )
	print('Final data stored to file (rm ga_sprinkler2.dat to start fresh)')

if __name__ == '__main__':
	main()

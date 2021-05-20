#
# genetic algorithm to place circular sprinklers into a field
# to provide the best coverage of that field
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import os
import random

from PIL import Image, ImageDraw

from PyGenAlg import PsoAlg, BaseChromo

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

# assume a 100x100 field with 5 circular water sprinklers
# each one has an (x,y) center == 2 chromos,
#  and a radius == 3rd chromo
num_sprayers = 5
# some of them may have smaller/larger spray-radii
dataRanges = [
	(0,100), (0,100), (1,30),
	(0,100), (0,100), (1,30),
	(0,100), (0,100), (1,40),
	(0,100), (0,100), (1,40),
	(0,100), (0,100), (1,50)
]
# fitness = number of squares with at least 1 unit of water
# but negative fitness if more than 2 units;
# i.e. waterlogged plants will die
scoring = [ 0, 10, 5, -1, -10, -20 ]

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
		BaseChromo.__init__( self, size=3*num_sprayers,
			range=dataRanges, dtype=float )

	# internal func to count 'units' of water on each
	# block of the grid which defines the field
	# : in this case, we'll zero-out a quadrant to simulate
	#   a non-square field
	def calcWaterfall( self ):
		# how much water hits each grid-square?
		water = [ 0 for i in range(100*100) ]
		for i in range(num_sprayers):
			xc  = self.data[3*i]
			yc  = self.data[3*i+1]
			rad = self.data[3*i+2]
			rad2 = rad*rad

			for x in range(int(xc-rad-1),int(xc+rad+1)):
				if( (x < 0) or (x >= 100) ):
					continue
				for y in range(int(yc-rad-1),int(yc+rad+1)):
					if( (y < 0) or (y >= 100) ):
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
			xc  = self.data[3*i]
			yc  = self.data[3*i+1]
			rad = self.data[3*i+2]
			draw.ellipse( (5*xc-10,5*yc-10,5*xc+10,5*yc+10), outline=(255,255,255), width=1 )
			draw.ellipse( (5*xc-5*rad,5*yc-5*rad,5*xc+5*rad,5*yc+5*rad), outline=(255,255,255), width=1 )

		img.save( "field_spray_pso.png" )

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	pa = PsoAlg( size=200,
		# omega = 0.60,
		# phi_p = 2.05,
		# phi_g = 2.05,
		omega = 0.75,
		phi_p = 1.50,
		phi_g = 1.00,
		learning_rate = 1.0,
		chromoClass  = MyChromo,
		minOrMax     = 'max',
		showBest     = 0,
	)

	pa.initPopulation()
	print( 'Created random init data' )

	#
	# Run it !!
	# : we'll just do 10 epochs of 10 steps each
	for i in range(10):
		pa.evolve( 20 )

		# give some running feedback on our progress
		print( str(i) + " best chromo:" )
		for j in range(1):
			print( pa.population[j] )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	for i in range(10):
		print( pa.population[i] )
	pa.population[0].chromo.drawImage()


if __name__ == '__main__':
	main()



import os
import sys
import time
from multiprocessing import Process

from PIL import Image, ImageDraw

from PyGenAlg import BaseChromo, GenAlg, ParallelMgr


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
	(0,100), (0,100), (1,25),
	(0,100), (0,100), (1,25),
	(0,100), (0,100), (1,25),
	(0,100), (0,100), (1,35),
	(0,100), (0,100), (1,35)
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

		img.save( "field_spray.png" )

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

class the_code( Process ):
	def __init__( self, tid, commMgr ):
		Process.__init__(self)
		self.tid = tid
		self.commMgr = commMgr
		# self.num_pes = parMgr.num_pes
		self.num_pes = commMgr.num_pes
		print( 'TID'+str(tid)+' init (out of '+str(self.num_pes)+')' )
		sys.stdout.flush()

	def run(self):
		print( 'TID start' )
		sys.stdout.flush()

		tid = self.tid
		num_pes = self.num_pes
		commMgr = self.commMgr
		# TODO: maybe this is an MPI_INIT kind of functionality?

		print( 'TID'+str(tid)+' started (out of '+str(num_pes)+')' )
		sys.stdout.flush()

		# TODO: calculate the splitting across the PEs

		# otherwise, init the gen-alg library from scratch
		ga = GenAlg( size=1000,
			elitism      = 0.10,
			crossover    = 0.50,
			pureMutation = 0.35,
			migration    = 0.05,
			migrationSendFcn = self.migrationSendFcn,
			migrationRecvFcn = self.migrationRecvFcn,
			parents      = 0.80,
			chromoClass  = MyChromo,
			minOrMax     = 'max',
			showBest     = 0
		)

		#
		# if a pickle-file exists, we load it
		if( os.path.isfile('ga_sprinkler.dat') ):
			# in a parallel sim, we need to load different chunks of the data-file
			# into each TID .. so we need to manage this through the commMgr
			# pop = IoOps.loadPopulation( ga, 'ga_sprinkler.dat' )
			pop = commMgr.loadPopulation( ga, 'ga_sprinkler.dat' )
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
			print( 'iter '+str(i) + ", best chromo:" )
			for i in range(1):
				print( ga.population[i] )

		# at this point, each PE's population is sorted
		rtn = commMgr.collect( ga.population[:10] )
		# rtn is a list of lists ... flatten it
		bestVals = [ x for y in rtn for x in y ]
		bestVals = ga.sortPopulationList( bestVals )

		#
		# all done ... output final results
		if( tid == 0 ):
			print( "\nfinal best chromos:" )
			for i in range(1):   
				print( bestVals[i] )
			bestVals[0].drawImage()

		commMgr.savePopulation( ga, 'ga_sprinkler.dat' )

	def migrationSendFcn( self, migrants ):
		tid = self.tid
		num_pes = self.num_pes
		commMgr = self.commMgr
		prev_pe = ( tid + num_pes - 1 ) % num_pes
		next_pe = ( tid + 1 ) % num_pes
		# print( 'tid '+str(tid)+' send to tids '+str(prev_pe)+' and '+str(next_pe)+' dsize='+str(len(migrants)) )
		# for add number of migrants, we need the sends to be balanced at the TID level
		# : so all TIDs send the smaller chunk to prev-tid and larger chunk to next-tid
		cutoff = len(migrants)//2
		commMgr.isend( prev_pe, 123, migrants[:cutoff] )
		commMgr.isend( next_pe, 123, migrants[cutoff:] )
		return

	def migrationRecvFcn( self ):
		tid = self.tid
		num_pes = self.num_pes
		commMgr = self.commMgr
		prev_pe = ( tid + num_pes - 1 ) % num_pes
		next_pe = ( tid + 1 ) % num_pes
		x,y,data1 = commMgr.recv( prev_pe, 123 )
		x,y,data2 = commMgr.recv( next_pe, 123 )
		# put all data into one list
		data1.extend( data2 )
		# print( 'tid '+str(tid)+' recv fr tids '+str(prev_pe)+' and '+str(next_pe)+' dsize='+str(len(data1)) )
		return data1


def main():
	parMgr = ParallelMgr( num_pes=12 )

	parMgr.runWorkers( the_code )

	parMgr.finalize()

if __name__ == '__main__':
	main()
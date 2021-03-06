#
# genetic algorithm to create voting districts in North Carolina;
# data is population per zipcode (CSV file: nc_data.zip.txt, includes
# other info too); we create 13 "population-centers" and assign
# each zipcode to the closest center; fitness-func is the difference
# between the biggest (most populous) and smallest (least populous)
# of those pop-centers ... i.e. we want to find an assignment that
# makes all voting districts as close to equal as possible
#
# this work is based on:
# Genetic Algorithms & Optimal Solutions
# Solving for the best congressional redistricting solution in Texas
# By Michael Larson, Dr. Dobb's Journal
# Mar 01, 2004 URL:http://www.ddj.com/dept/architect/184405617
#
# this code is:
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import os
import random
import csv
import sys
import time

from PIL import Image, ImageDraw
from multiprocessing import Process

from PyGenAlg import BaseChromo, GenAlg, ParallelMgr

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

# chromo size is 2 (lat/lon) per district (13) == 26 chromos

# data that is specific to NC districts
# : 13 districts, (lat,lon) coords within a relaxed bounding-box
dataRanges = [
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0),
	(-90.0,-70.0), (32.0,37.0)
]

# data fields in zipcode file:
# "FIPS state","ZIP Census Tabulation Area","State Postal Code",
#  "zipname","Wtd centroid W longitude, degrees",
#  "Wtd centroid latitude, degrees","Total Pop, 2000 census",
#  "state to zcta5 alloc factor"
# : we only use wtd centroid longitude and latitude and Total Pop
#   == columns 4,5,6
nc_data = []
with open('nc_dist12/nc_data.zip.txt') as fp:
	readCSV = csv.reader( fp, delimiter=',' )
	# skip 2 header lines
	next( readCSV, None )
	next( readCSV, None )
	# now read all the 'real' data
	for row in readCSV:
		nc_data.append( row )

num_datapts = len(nc_data)

# a simple colormap for the png output
cmap = [
	(255,0,0),(255,255,0),(0,255,0),(0,255,255),
	(0,0,255),(255,0,255),(255,255,255),
	(128,128,128),(192,192,192),
	(128,0,0),(128,128,0),(0,128,0),(128,0,128),
	(0,128,128),(0,0,128),(128,0,128)
]


class MyChromo(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=26,
			range=dataRanges, dtype=float )

	# internal function to calculate the populations for
	# each proposed voting district (or population-center)
	def population_per_district( self ):
		pop_count = [ 0 for i in range(13) ]

		# grab a copy of all genes for future reference
		glist = self.data

		# accumulate population to closest pop.center/voting distring/gene
		for n in range(num_datapts):
			# compute closest pop.center
			xx = float(nc_data[n][4])
			yy = float(nc_data[n][5])
			pp = int(nc_data[n][6])
			pop_ctr = 0
			min_dist = 9.999e9
			for i in range(13):
				x = glist[2*i]
				y = glist[2*i+1]
				dist = (x-xx)*(x-xx) + (y-yy)*(y-yy)
				if( dist < min_dist ):
					min_dist = dist
					pop_ctr = i
			# accumulate this population into pop.ctr
			pop_count[pop_ctr] += pp

		return pop_count

	# calculate the fitness functions
	# : this is just the difference between biggest and smallest district
	# : one could imagine more involved functions that take into account
	#   current district (limit changes to citizens' current assignment)
	def calcFitness( self ):
		pop_count = self.population_per_district()

		# compute basic stats for fitness fcn
		min_pop  = 9.9e9
		max_pop  = -9.9e9
		sum_pop  = 0.0
		zero_pop = 0
		for i in range(13):
			xx = pop_count[i]
			if( xx < min_pop ):
				min_pop = xx
			if( xx > max_pop ):
				max_pop = xx
			if( xx == 0 ):
				zero_pop = zero_pop + 1
			sum_pop = sum_pop + xx

		# what should the fitness function look like?
		# : add penalty for zero-population districts
		fitness = max_pop - min_pop + 10000000*zero_pop*zero_pop

		return fitness

	# could potentially use this to feed into a geo-mapping package?
	def printCsvResults( self ):
		print "longitude, latitude, district"

		glist = self.data
		for n in range(num_datapts):
			xx = float(nc_data[n][4])
			yy = float(nc_data[n][5])
			pp = int(nc_data[n][6])
			pop_ctr = 0
			min_dist = 9.999e9
			for i in range(13):
				x = glist[2*i]
				y = glist[2*i+1]
				dist = (x-xx)*(x-xx) + (y-yy)*(y-yy)
				if( dist < min_dist ):
					min_dist = dist
					pop_ctr = i
			print str(xx) +','+ str(yy) +','+ str(pop_ctr)

	# use PIL to plot each zipcode, colored by the assigned
	# voting-district
	def drawImage( self ):
		img  = Image.new( 'RGB', (1000,500), (0,0,0) )
		draw = ImageDraw.Draw(img)

		glist = self.data
		for n in range(num_datapts):
			xx = float(nc_data[n][4])
			yy = float(nc_data[n][5])
			pp = int(nc_data[n][6])
			pop_ctr = 0
			min_dist = 9.999e9
			for i in range(13):
				x = glist[2*i]
				y = glist[2*i+1]
				dist = (x-xx)*(x-xx) + (y-yy)*(y-yy)
				if( dist < min_dist ):
					min_dist = dist
					pop_ctr = i

			# ranges: (-90.0,-70.0), (32.0,37.0)
			xx = 50*xx + 4500
			#yy = 100*yy - 3200
			yy = -100*yy + 3700
			cc = cmap[pop_ctr]
			draw.ellipse( [xx-5,yy-5, xx+5,yy+5], fill=cc )

		img.save( "nc_dist_map.png" )

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

class the_code( Process ):
	def __init__( self, tid, parMgr ):
		Process.__init__(self)
		self.tid = tid
		self.parMgr = parMgr
		self.num_pes = parMgr.num_pes
		print( 'TID'+str(tid)+' init (out of '+str(self.num_pes)+')' )
		sys.stdout.flush()

	def run(self):
		tid = self.tid
		num_pes = self.num_pes
		parMgr = self.parMgr
		# TODO: maybe this is an MPI_INIT kind of functionality?
		# : e.g. store this worker-tid into parMgr; init per-worker values
		parMgr.tid = tid

		print( 'TID'+str(tid)+' started (out of '+str(num_pes)+')' )
		sys.stdout.flush()

		# TODO: calculate the splitting across the PEs

		ga = GenAlg( size=250,
			elitismPct   = 0.10,
			crossoverPct = 0.30,
			mutationPct  = 0.50,
			migrationPct = 0.10,
			#selectionFcn = GenAlgOps.tournamentSelection,
			#crossoverFcn = GenAlgOps.crossover22,
			#mutationFcn  = GenAlgOps.mutateFew,
			#pureMutationSelectionFcn = GenAlgOps.simpleSelection,
			#pureMutationFcn = GenAlgOps.mutateFew,
			#feasibleSolnFcn = GenAlgOps.disallowDupes,
			migrationSendFcn = self.migrationSend,
			migrationRecvFcn = self.migrationRecv,
			chromoClass  = MyChromo,
			minOrMax     = 'min',
			showBest     = 0
		)

		# otherwise, init the gen-alg library from scratch
		ga.initPopulation()
		print( 'Created random init data' )

		#
		# Run it !!
		# : we'll just do 10 epochs of 10 steps each
		t0 = time.time()
		for i in range(15):
			ga.evolve( 10 )

			# give some running feedback on our progress
			#print( str(i) + " best chromo:" )
			#for i in range(1):
			#	print( ga.population[i] )

		t1 = time.time()
		print( 'TID'+str(tid)+' time = ' + str(t1-t0) + ' sec' )

		# at this point, each PE's population is sorted
		rtn = parMgr.collect( ga.population[:10] )
		bestVals = [ x for y in rtn for x in y ]
		bestVals.sort()

		#
		# all done ... output final results
		if( tid == 0 ):
			print( "\nfinal best chromos:" )
			for i in range(1):
				pop = bestVals[i].data
				# print( ga.population[i] )
				pop_ct = bestVals[i].population_per_district()
				for j in range(len(pop_ct)):
					ct = pop_ct[j]
					x = pop[2*j]
					y = pop[2*j+1]
					print( '  %10.4f %10.4f : %10d' % (x,y,ct) )

			bestVals[0].drawImage()

		#
		# we'll always save the pickle-file, just delete it
		# if you want to start over from scratch
		#ga.savePopulation( 'ga_voting.dat' )
		#print('Final data stored to file (rm ga_voting.dat to start fresh)')

	def migrationFcn( self, migrants ):
		tid = self.tid
		num_pes = self.num_pes
		parMgr = self.parMgr

		prev_pe = ( tid + num_pes - 1 ) % num_pes
		next_pe = ( tid + 1 ) % num_pes
		#print( 'tid '+str(tid)+' exchanging with tids '+str(next_pe) \
		#    + '/' + str(prev_pe) )

		parMgr.isend( next_pe, 123, migrants )
		x,y,data = parMgr.recv( prev_pe, 123 )

		return data

	def migrationSend( self, migrants ):
		tid = self.tid
		num_pes = self.num_pes
		parMgr = self.parMgr

		prev_pe = ( tid + num_pes - 1 ) % num_pes
		next_pe = ( tid + 1 ) % num_pes
		#print( 'tid '+str(tid)+' exchanging with tids '+str(next_pe) \
		#    + '/' + str(prev_pe) )

		parMgr.isend( next_pe, 234, migrants )

	def migrationRecv( self ):
		tid = self.tid
		num_pes = self.num_pes
		parMgr = self.parMgr

		prev_pe = ( tid + num_pes - 1 ) % num_pes
		next_pe = ( tid + 1 ) % num_pes

		x,y,data = parMgr.recv( prev_pe, 234 )

		return data


def main():
	parMgr = ParallelMgr( num_pes=4 )

	parMgr.runWorkers( the_code )

	parMgr.finalize()

if __name__ == '__main__':
	main()

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

from PIL import Image, ImageDraw

from PyGenAlg import GenAlg, BaseChromo, GenAlgOps, IoOps

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

# data fields in data file:
#     zipcode (really ZCTA), population, 
#     latitude, longitude, current district
nc_data = []
with open('nc_dist12/nc_data_new.csv') as fp:
	readCSV = csv.reader( fp, delimiter=',' )
	for row in readCSV:
		row[1] = int(row[1])
		row[4] = int(row[4])
		nc_data.append( row )

num_datapts = len(nc_data)

# how much to weight the parts of the fitness function
# : pop-diff between proposed/new districts
weight1 = 10.0
# : pop moved between original and proposed/new districts
weight2 = 1.0

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
	def assign_zip_to_district( self ):
		# grab a copy of all genes for future reference
		data = self.data
		# return values
		zip_to_distr = [ 0 for i in range(num_datapts) ]

		# accumulate population to closest pop.center/voting distring/gene
		for n in range(num_datapts):
			# compute closest pop.center
			zip = nc_data[n][0]
			pop = nc_data[n][1]
			xx = float(nc_data[n][2])
			yy = float(nc_data[n][3])
			pop_ctr = 0
			min_dist = 9.999e9
			for i in range(13):
				x = data[2*i]
				y = data[2*i+1]
				dist = (x-xx)*(x-xx) + (y-yy)*(y-yy)
				if( dist < min_dist ):
					min_dist = dist
					pop_ctr = i

			# and assign this zip to the pop_ctr
			zip_to_distr[n] = pop_ctr

		return zip_to_distr

	# calculate the fitness functions
	# : this is just the difference between biggest and smallest district
	# : also adds in a factor for moving people from one district to another
	def calcFitness( self ):
		zip_to_distr = self.assign_zip_to_district()

		# accumulate population per new district
		pop_count = [ 0 for i in range(13) ]
		for n in range(num_datapts):
			pop = nc_data[n][1]
			distr = zip_to_distr[n]
			pop_count[distr] = pop_count[distr] + pop

		# compute how balanced the districts are
		min_pop  = 9.9e9
		max_pop  = -9.9e9
		sum_pop  = 0.0
		for i in range(13):
			xx = pop_count[i]
			if( xx < min_pop ):
				min_pop = xx
			if( xx > max_pop ):
				max_pop = xx
			sum_pop = sum_pop + xx

		# how many people got moved from current district?
		mov_pop = 0
		for n in range(num_datapts):
			pop = nc_data[n][1]
			orig_distr = nc_data[n][4]
			new_distr  = zip_to_distr[n]
			if( orig_distr != new_distr ):
				mov_pop = mov_pop + pop

		# what should the fitness function look like?
		fitness = weight1*(max_pop - min_pop) + weight2*mov_pop

		return fitness

	# could potentially use this to feed into a geo-mapping package?
	def printCsvResults( self ):
		zip_to_distr = self.assign_zip_to_district()

		print "longitude, latitude, district"

		glist = self.data
		for n in range(num_datapts):
			xx = float(nc_data[n][2])
			yy = float(nc_data[n][3])
			pp = int(nc_data[n][1])
			pop_ctr = zip_to_distr[n]
			print str(xx) +','+ str(yy) +','+ str(pop_ctr)

	# use PIL to plot each zipcode, colored by the assigned
	# voting-district
	def drawImage( self, showCurrent=False ):
		img  = Image.new( 'RGB', (1000,500), (0,0,0) )
		draw = ImageDraw.Draw(img)

		if( showCurrent ):
			zip_to_distr = [ 0 for i in range(num_datapts) ]
			# insert current districts into the right slots
			for n in range(num_datapts):
				pop_ctr = int(nc_data[n][4])
				zip_to_distr[n] = pop_ctr
		else:
			zip_to_distr = self.assign_zip_to_district()

		for n in range(num_datapts):
			xx = float(nc_data[n][2])
			yy = float(nc_data[n][3])
			pop_ctr = zip_to_distr[n]

			# ranges: (-90.0,-70.0), (32.0,37.0)
			xx = 50*xx + 4500
			#yy = 100*yy - 3200
			yy = -100*yy + 3700
			cc = cmap[pop_ctr]
			draw.ellipse( [xx-5,yy-5, xx+5,yy+5], fill=cc )

		if( showCurrent ):
			img.save( "nc_dist_map_orig.png" )
		else:
			img.save( "nc_dist_map.png" )

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	ga = GenAlg( size=100,
		elitism      = 0.10,
		crossover    = 0.60,
		pureMutation = 0.30,
		parentsPct   = 0.80,
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
	if( os.path.isfile('ga_voting2.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_voting2.dat' )
		ga.appendToPopulation( pop )
		print( 'Read init data from file ('+str(len(pop))+' chromos)')
		if( len(pop) < ga.population_sz ):
			# we need to fill this out with random data
			pop = IoOps.randomPopulation( ga, ga.population_sz-len(pop) )
			ga.appendToPopulation( pop )
			print( '  appended '+str(len(pop))+' random chromos' )
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
		for i in range(1):
			print( ga.population[i] )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	for i in range(10):
		print( ga.population[i] )
	ga.population[0].drawImage()

	ga.population[0].drawImage(showCurrent=True)
	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	IoOps.savePopulation( ga, 'ga_voting2.dat' )
	print('Final data stored to file (rm ga_voting2.dat to start fresh)')

if __name__ == '__main__':
	main()

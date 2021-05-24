#
# genetic algorithm to place circular sprinklers into a field
# to provide the best coverage of that field
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import os
import sys

import click

from PIL import Image, ImageDraw

import numpy as np
import numba
from numba import cuda
from numpy.core.numeric import _moveaxis_dispatcher

from PyGenAlg import GenAlg, IoOps
import GenAlgCfg as cfg
import GenAlgGPU as gpu

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
	(0.0,100.0), (0.0,100.0), (1.0,20.0),
	(0.0,100.0), (0.0,100.0), (1.0,20.0),
	(0.0,100.0), (0.0,100.0), (1.0,20.0),
	(0.0,100.0), (0.0,100.0), (1.0,20.0),
	(0.0,100.0), (0.0,100.0), (1.0,20.0)
]
# dataRanges = [
# 	(0.0,100.0), (0.0,100.0), (1.0,40.0),
# 	(0.0,100.0), (0.0,100.0), (1.0,30.0),
# 	(0.0,100.0), (0.0,100.0), (1.0,30.0),
# 	(0.0,100.0), (0.0,100.0), (1.0,20.0),
# 	(0.0,100.0), (0.0,100.0), (1.0,20.0)
# ]
# fitness = number of squares with at least 1 unit of water
# but negative fitness if more than 2 units; i.e. waterlogged plants will die
scoring = [ 0.0, 10.0, 5.0, -1.0, -10.0, -20.0 ]
# cutout_region = [ 75, 25 ]
cutout_region = [ -1,-1 ]

# a simple colormap for the png output
cmap = [ (0,0,0),
	(255,0,0),(255,255,0),(0,255,0),(0,255,255),
	(0,0,255),(255,0,255),(255,255,255),
	(128,128,128),(192,192,192),
	(128,0,0),(128,128,0),(0,128,0),(128,0,128),
	(0,128,128),(0,0,128),(128,0,128)
]


# the calculations for the fitness function
@cuda.jit( max_registers=64 )
def calcFitness( popvec, fitvec, config_i,config_f, userdata_i,userdata_f ):
	tid     = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
	grid_sz = cuda.gridDim.x*cuda.blockDim.x
	pop_sz  = config_i[cfg.POPULATION_SIZE]
	chr_sz  = config_i[cfg.CHROMO_SIZE]
	#scoring = ( 0.0, 10.0, 5.0, -1.0, -10.0, -20.0 )
	scoring = userdata_f
	#cutout_region = ( 75, 25 )
	cutout_region = userdata_i

	water = cuda.local.array( 100*100, np.uint8 )

	for i in range(tid,pop_sz,grid_sz):
		# how much water hits each grid-square?
		# : count 'units' of water on each block of the grid which defines the field
		# : in this case, we'll also zero-out a quadrant to simulate a non-square field
		for j in range(100*100):
			water[j] = 0
		for j in range(num_sprayers):
			xc  = popvec[chr_sz*i+3*j]
			yc  = popvec[chr_sz*i+3*j+1]
			rad = popvec[chr_sz*i+3*j+2]
			rad2 = rad*rad

			for x in range(int(xc-rad-1),int(xc+rad+1)):
				for y in range(int(yc-rad-1),int(yc+rad+1)):
					dd = (x-xc)*(x-xc) + (y-yc)*(y-yc)
					if( dd <= rad2 ):
						if( (x >= 0) and (x < 100) and (y >= 0) and (y < 100) ):
							water[100*y+x] = water[100*y+x] + 1

		# zero-out any non-field areas (e.g. if the field
		# is not a square) ... let's make it an L shape with a small cutout
		for x in range( cutout_region[0] ):
			for y in range( cutout_region[1] ):
				water[100*y+x] = 0

		sum = 0.0
		for j in range(100*100):
			sum = sum + scoring[ water[j] ]

		fitvec[i] = sum

@cuda.jit( max_registers=64 )
def crossover( popvec_in, popvec_out, fitvec, config_i,config_f,rng_states,fitstats ):
	tmp       = cuda.local.array( 4, numba.int32 )
	tid       = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
	grid_sz   = cuda.gridDim.x*cuda.blockDim.x
	pop_sz    = config_i[cfg.POPULATION_SIZE]
	num_c     = config_i[cfg.CROSSOVER_COUNT]
	pop_cnt   = config_i[cfg.ELITISM_COUNT]   # offset to where crossover children are stored
	parent_pct = config_f[cfg.PARENT_PCT]
	parent_sz = int( pop_sz * parent_pct )    # only let top 50% of population become parents
	# parent_sz = pop_sz

	for i in range(2*tid,num_c,2*grid_sz):
		ii = i + pop_cnt    # offset by current population cursor

		# tournament selection  (k=3)
		(mother,father) = gpu.selectTournament( parent_sz, fitvec, 3, rng_states,tid,tmp )

		# 2 point crossover leading to 2 children (at ii and ii+1)
		gpu.crossover22( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp )

		# mutate 2 genes per chromo
		gpu.mutateFewF( ii, popvec_out, 2, config_i,config_f,rng_states,tid,tmp )

# use PIL to draw a simple PNG image of the fields,
# colored by a simple colormap
def drawImage( data ):
	img  = Image.new( 'RGB', (500,500), (0,0,0) )
	draw = ImageDraw.Draw(img)

	water = [ 0 for i in range(100*100) ]
	for j in range(num_sprayers):
		xc  = data[3*j]
		yc  = data[3*j+1]
		rad = data[3*j+2]
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
		xc  = data[3*i]
		yc  = data[3*i+1]
		rad = data[3*i+2]
		draw.ellipse( (5*xc-10,5*yc-10,5*xc+10,5*yc+10), outline=(255,255,255), width=1 )
		draw.ellipse( (5*xc-5*rad,5*yc-5*rad,5*xc+5*rad,5*yc+5*rad), outline=(255,255,255), width=1 )

	img.save( "field_spray.png" )

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

@click.group()
@click.option( '-E','--elitism','elitism',type=float,default=0.01,show_default=True,help='elitism (pct or num)' )
@click.option( '-C','--crossover','crossover',type=float,default=0.49,show_default=True,help='crossover (pct or num)' )
@click.option( '-M','--puremutation','puremutation',type=float,default=0.50,show_default=True,help='pure-mutation (pct or num)' )
@click.option( '-P','--parent','parent_pct',type=float,default=1.0,show_default=True,help='top-percent of population that can be parents' )
@click.option( '-s','--size','popsize',default=4096,show_default=True,help='population size' )
@click.option( '-i','--inners','inner_it',default=10,show_default=True,help='inner iterations' )
@click.option( '-e','--epochs','epoch_it',default=10,show_default=True,help='epoch iterations' )
@click.option( '-m','--mutatenum','mutatenum',default=3,show_default=True,help='mutate-num' )
@click.option( '-v','--verbose','verbose',count=True,help='verbose level' )
@click.option( '-V','--veryverbose','veryverbose',count=True,help='verbose level +10' )
@click.pass_context
def cli( ctx, elitism,crossover,puremutation, parent_pct, popsize, inner_it, epoch_it, mutatenum, verbose, veryverbose ):
	ctx.ensure_object(dict)

	ctx.obj['elitism'] = elitism
	ctx.obj['crossover'] = crossover
	ctx.obj['puremutation'] = puremutation
	ctx.obj['parent_pct'] = parent_pct
	ctx.obj['popsize'] = popsize
	ctx.obj['inner_it'] = inner_it
	ctx.obj['epoch_it'] = epoch_it
	ctx.obj['mutatenum'] = mutatenum
	ctx.obj['verbose'] = verbose + 10*veryverbose

@cli.command()
@click.option( '-S','save_out',is_flag=True,help='save output to file' )
@click.option( '-L','load_in',is_flag=True,help='load input from file' )
@click.pass_context
def run( ctx, save_out, load_in ):
	ctxobj = ctx.obj
	verbose = ctxobj.get('verbose')
	epoch_it = ctxobj.get('epoch_it')
	inner_it = ctxobj.get('inner_it')

	ga = GenAlg(
		chromoSize   = 3*num_sprayers,
		dtype        = np.float32,
		range        = dataRanges,
		fitnessFcn   = calcFitness,
		crossoverFcn = crossover,
		userDataI    = cutout_region,  # user-provided cut-out region
		userDataF    = scoring,        # user-provided scoring system
		size         = ctxobj.get('popsize'),
		elitism      = ctxobj.get('elitism'),
		crossover    = ctxobj.get('crossover'),
		pureMutation = ctxobj.get('puremutation'),
		parentPct    = ctxobj.get('parent_pct'),
		minOrMax     = 'max',
		showBest     = 0
	)

	#
	# if a pickle-file exists, we load it
	if( load_in ):
		pop = IoOps.loadPopulation( ga, 'ga_sprinkler.dat' )
		ga.loadPopulation( pop )
		print( 'Read init data from file ('+str(len(pop))+' chromos)')
	else:
		# otherwise, init the gen-alg library from scratch
		ga.initPopulation()
		print( 'Created random init data' )

	if( verbose > 0 ):
		ga.describe()
		print( 'Epoch/Inner iters:', epoch_it, inner_it )

		maxspray = 0.0
		for i in range(num_sprayers):
			maxspray = maxspray + 3.1415*dataRanges[3*i+2][1]*dataRanges[3*i+2][1]
		maxfield = 100*100 - cutout_region[0]*cutout_region[1]
		mx = max(scoring)
		print( 'Max spray: %.2f  ( %d )'%(mx*maxspray,mx*maxfield) )

	#
	# Run it !!
	# : we'll just do 10 epochs of 10 steps each
	with click.progressbar(range(epoch_it)) as bar:
		for i in bar:
			ga.evolve( inner_it )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	print( ga.population[range(3*num_sprayers)], ga.fitnessVals[0] )

	drawImage( ga.population[range(3*num_sprayers)] )

	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	if( save_out ):
		IoOps.savePopulation( ga, 'ga_sprinkler.dat' )
		print('Final data stored to file (rm ga_sprinkler.dat to start fresh)')


@cli.command()
@click.option( '-I','--sweepiters','sweep_iters',default=10,show_default=True,help='number of iterations for the sweep' )
@click.option( '-a','--append','append',type=click.Path(writable=True),help='append data to output file')
@click.option( '-o','--output','outfile',type=click.Path(writable=True),help='write to new output filename' )
@click.pass_context
def sweep( ctx, sweep_iters, append, outfile ):
	ctxobj = ctx.obj
	verbose = ctxobj.get('verbose')
	epoch_it = ctxobj.get('epoch_it')
	inner_it = ctxobj.get('inner_it')

	ga = GenAlg(
		chromoSize   = 3*num_sprayers,
		dtype        = np.float32,
		range        = dataRanges,
		fitnessFcn   = calcFitness,
		crossoverFcn = crossover,
		userDataI    = cutout_region,  # user-provided cut-out region
		userDataF    = scoring,        # user-provided scoring system
		size         = ctxobj.get('popsize'),
		elitism      = ctxobj.get('elitism'),
		crossover    = ctxobj.get('crossover'),
		pureMutation = ctxobj.get('puremutation'),
		parentPct    = ctxobj.get('parent_pct'),
		minOrMax     = 'max',
		showBest     = 0
	)

	if( append ):
		fp_out = open( append, 'a' )
	elif( outfile ):
		fp_out = open( outfile, 'w' )
	else:
		fp_out = sys.stdout

	if( verbose > 0 ):
		print( 'Epoch/Inner iters:', epoch_it, inner_it )
		print( 'Sweep iters:', sweep_iters )

	for m in range(sweep_iters):
		ga.initPopulation()

		with click.progressbar(range(epoch_it)) as bar:
			for i in bar:
				ga.evolve( inner_it )

		fp_out.write( '%d, %d, %d, %f\n'%(ga.population_sz,epoch_it,inner_it,ga.fitnessVals[0]) )

	fp_out.close()


@cli.command()
@click.option( '-I','--sweepiters','sweep_iters',default=10,show_default=True,help='number of iterations for the sweep' )
@click.option( '-t','--thresh','threshold',default=0.99,show_default=True,help='threshold to search for' )
@click.option( '-a','--append','append',type=click.Path(writable=True),help='append data to output file')
@click.option( '-o','--output','outfile',type=click.Path(writable=True),help='write to new output filename' )
@click.pass_context
def thresh( ctx, sweep_iters, threshold, append, outfile ):
	ctxobj = ctx.obj
	verbose = ctxobj.get('verbose')
	epoch_it = ctxobj.get('epoch_it')
	inner_it = ctxobj.get('inner_it')

	if( append ):
		fp_out = open( append, 'a' )
	elif( outfile ):
		fp_out = open( outfile, 'w' )
	else:
		fp_out = sys.stdout

	maxspray = 0.0
	for i in range(num_sprayers):
		maxspray = maxspray + 3.1415*dataRanges[3*i+2][1]*dataRanges[3*i+2][1]
	maxfield = 100*100 - cutout_region[0]*cutout_region[1]
	mx = max(scoring)

	if( threshold < 1 ):
		threshold = threshold * mx * maxspray

	# find all the pairs of vals to test
	test_list = []
	for m in range(sweep_iters):
		for pp in range(1,11):
			for ps in [ 1024, 2048, 4096, 8192, 16384, 32768 ]:
				test_list.append( (ps,pp/10,m) )

	if( verbose > 0 ):
		print( 'Epoch/Inner iters:', epoch_it, inner_it )
		print( 'Sweep iters:', sweep_iters )
		print( 'Threshold: %.2f'%( threshold ) )
		print( 'Max spray: %.2f  ( %d )'%(mx*maxspray,mx*maxfield) )
		print( 'Total num tests:', len(test_list) )

	# run the tests w/ progress bar 
	with click.progressbar(test_list) as bar:
		for params in bar:
			# print( 'params', params )

			ga = GenAlg(
				chromoSize   = 3*num_sprayers,
				dtype        = np.float32,
				range        = dataRanges,
				fitnessFcn   = calcFitness,
				crossoverFcn = crossover,
				userDataI    = cutout_region,  # user-provided cut-out region
				userDataF    = scoring,        # user-provided scoring system
				size         = params[0],
				elitism      = ctxobj.get('elitism'),
				crossover    = ctxobj.get('crossover'),
				pureMutation = ctxobj.get('puremutation'),
				parentPct    = params[1],
				minOrMax     = 'max',
				showBest     = 0
			)

			ga.initPopulation()

			totalits = 0
			flag = 0
			for e in range(epoch_it):
				ga.evolve( inner_it )
				totalits = totalits + inner_it

				if( ga.fitnessVals[0] >= threshold ):
					flag = 1
					break

			fp_out.write( '%d, %d, %d, %d, %.2f, %d, %d, %.2f\n'%(ga.population_sz,ga.elitism,ga.crossover,ga.pureMutation, ga.parent_pct,
				flag, totalits,ga.fitnessVals[0]) )
			fp_out.flush()

	fp_out.close()


@cli.command()
@click.pass_context
def analyze( ctx ):
	ctxobj = ctx.obj

	ga = GenAlg(
		chromoSize   = 3*num_sprayers,
		dtype        = np.float32,
		range        = dataRanges,
		fitnessFcn   = calcFitness,
		crossoverFcn = crossover,
		userDataI    = cutout_region,  # user-provided cut-out region
		userDataF    = scoring,        # user-provided scoring system
		size         = 128,
		elitism      = ctxobj.get('elitism'),
		crossover    = ctxobj.get('crossover'),
		pureMutation = ctxobj.get('puremutation'),
		minOrMax     = 'max',
		showBest     = 0
	)

	# otherwise, init the gen-alg library from scratch
	ga.initPopulation()

	ga.describe()

	# Run it !!
	ga.evolve( 1 )

	ga.cudaAnalysis()


if __name__ == '__main__':
	cli( obj={} )

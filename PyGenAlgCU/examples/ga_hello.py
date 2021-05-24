#
# genetic algorithm to make change for a given target value;
# attempts to minize the difference in amounts as well as
# the number of coins given
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import os
import string
import numpy as np
import numba
from numba import cuda

from PyGenAlg import GenAlg, IoOps
import GenAlgCfg as cfg
import GenAlgGPU as gpu

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

# chromo size is #letters in word
letters = string.ascii_uppercase + string.ascii_lowercase + string.punctuation + ' '

# target value we're aiming for:
solutionWord = 'Hello World!'
solution = np.zeros( len(solutionWord), dtype=np.int32 )
for i in range(len(solutionWord)):
	j = letters.find( solutionWord[i] )
	solution[i] = j
# dev_solution = cuda.device_array_like( solution )

#chromo_type = [ np.uint8 for i in range(len(solution)) ]
chromo_type = np.uint8
chromo_range = [ (0,len(letters)) for i in range(len(solution)) ]

@cuda.jit
def calcFitness( popvec, fitvec, config_i, config_f, userdata_i,userdata_f ):
	tid     = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
	grid_sz = cuda.gridDim.x*cuda.blockDim.x
	pop_sz  = config_i[cfg.POPULATION_SIZE]
	chr_sz  = config_i[cfg.CHROMO_SIZE]

	solution = ( 7, 30, 37, 37, 40, 84, 22, 40, 43, 37, 29, 52 )

	for i in range(tid,pop_sz,grid_sz):
		# NOTE: all chromos are re-calculated every time!
		fitval = 0.0
		for j in range(len(solution)):
			if( solution[j] == popvec[i*chr_sz+j] ):
				fitval = fitval + 1

		fitvec[i] = fitval

@cuda.jit
def crossover1( popvec_in, popvec_out, fitvec, config_i,config_f,rng_states,fitstats ):
	tmp       = cuda.local.array( 4, numba.int32 )
	tid       = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
	grid_sz   = cuda.gridDim.x*cuda.blockDim.x
	pop_sz    = config_i[cfg.POPULATION_SIZE]
	num_c     = config_i[cfg.CROSSOVER_COUNT]
	pop_cnt   = config_i[cfg.ELITISM_COUNT]   # offset to where crossover children are stored
	#parent_sz = int( pop_sz * parent_pct )
	parent_sz = pop_sz

	for i in range(tid,num_c,grid_sz):
		ii = i + pop_cnt    # offset by current population cursor

		# random selection
		#(mother,father) = gpu.selectSimple( parent_sz, rng_states,tid,tmp )
		# tournament selection  (k=3)
		#(mother,father) = gpu.selectTournament( parent_sz, fitvec, 3, rng_states,tid,tmp )
		# roulette wheel selection .. maximization problem with positive vals
		(mother,father) = gpu.selectRouletteMaxPos( parent_sz, fitvec, rng_states,tid,fitstats )

		# 1 point crossover leading to 1 child
		#gpu.crossover11( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp )
		# 2 point crossover leading to 1 child
		gpu.crossover21( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp )

		# mutate 2 genes per chromo (for 1 child)
		gpu.mutateFew( ii, popvec_out, 2, config_i,rng_states,tid,tmp )
		# mutate all genes if rand<0.1
		#gpu.mutateRandom( ii, popvec_out, 0.10, config_i,rng_states,tid )

@cuda.jit
def crossover2( popvec_in, popvec_out, fitvec, config_i,config_f,rng_states,fitstats ):
	tmp       = cuda.local.array( 4, numba.int32 )
	tid       = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
	grid_sz   = cuda.gridDim.x*cuda.blockDim.x
	pop_sz    = config_i[cfg.POPULATION_SIZE]
	num_c     = config_i[cfg.CROSSOVER_COUNT]
	pop_cnt   = config_i[cfg.ELITISM_COUNT]   # offset to where crossover children are stored
	#parent_sz = int( pop_sz * parent_pct )
	parent_sz = pop_sz

	for i in range(2*tid,num_c,2*grid_sz):
		ii = i + pop_cnt    # offset by current population cursor

		# random selection
		#(mother,father) = gpu.selectSimple( parent_sz, rng_states,tid,tmp )
		# tournament selection  (k=3)
		#(mother,father) = gpu.selectTournament( parent_sz, fitvec, 3, rng_states,tid,tmp )
		# roulette wheel selection .. maximization problem with positive vals
		(mother,father) = gpu.selectRouletteMaxPos( parent_sz, fitvec, rng_states,tid,fitstats )

		# 1 point crossover leading to 2 children  (at locations ii and ii+1)
		#gpu.crossover12( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp )
		# 2 point crossover leading to 1 child
		gpu.crossover22( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp )

		# mutate 2 genes per chromo (for each child)
		gpu.mutateFew( ii, popvec_out, 2, config_i,rng_states,tid,tmp )
		gpu.mutateFew( ii+1, popvec_out, 2, config_i,rng_states,tid,tmp )
		# mutate all genes if rand<0.1
		#gpu.mutateRandom( ii, popvec_out, 0.10, config_i,rng_states,tid )

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	ga = GenAlg( 
		chromoSize   = len(solution),
		dtype        = chromo_type,
		range        = chromo_range,
		fitnessFcn   = calcFitness,
		crossoverFcn = crossover2,
		size         = 1024,
		elitism      = 0.10,
		crossover    = 0.60,
		pureMutation = 0.30,
		minOrMax     = 'max',
		showBest     = 0
	)
	ga.describe()

	#
	# if a data-file exists, we load it
	if(  os.path.isfile('ga_hello.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_hello.dat' )
		ga.loadPopulation( pop )
		print( 'Read init data from file ('+str(len(pop)/ga.chromo_sz)+' chromos)')
	else:
		# otherwise, init the gen-alg library from scratch
		ga.initPopulation()
		print( 'Created random init data' )

	# print( "init data best chromo:" )
	# print( ga.population[range(len(solutionWord))], ga.fitnessVals[0] )
	# print( ga.population[range(len(solutionWord),2*len(solutionWord))], ga.fitnessVals[1] )

	#
	# Run it !!
	# : we'll just do 10 epochs of 10 steps each
	for i in range(10):
		ga.evolve( 10 )

		# give some running feedback on our progress
		print( 'iter '+str(i) + ", best chromo:" )
		print( ga.population[range(len(solutionWord))], ga.fitnessVals[0] )
		#print( ga.population[range(len(solutionWord),2*len(solutionWord))], ga.fitnessVals[1] )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	print( ga.population[range(len(solutionWord))], ga.fitnessVals[0] )
	print( ga.population[range(len(solutionWord),2*len(solutionWord))], ga.fitnessVals[1] )

	print( 'solution -  maxchr=', len(letters) )
	print( solution[:] )

	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	IoOps.savePopulation( ga, 'ga_hello.dat' )
	print('Final data stored to file (rm ga_hello.dat to start fresh)')

	ga.cudaAnalysis()

if __name__ == '__main__':
	main()

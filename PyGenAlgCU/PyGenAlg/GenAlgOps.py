#
# some basic gen-alg helper functions
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
#

import math
import numpy as np
import numba
from numba import cuda

import GenAlgCfg as cfg
import GenAlgGPU as gpu

#
# calc basic stats over fitness vector
# : e.g. for use with roulette wheel selection
@cuda.jit
def gpu_calcFitnessStats( config_i, fitvec, results ):
	# splitting is on population_sz (not pvec_size)
	tid     = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
	grid_sz = cuda.gridDim.x*cuda.blockDim.x
	pop_sz  = config_i[cfg.POPULATION_SIZE]

	# calc thread-local sum/min/max
	sum_fitness = 0.0
	sum2_fitness = 0.0
	min_fitness = math.inf
	max_fitness = -math.inf
	for i in range(tid,pop_sz,grid_sz):
		fitval = fitvec[i]
		sum_fitness = sum_fitness + fitval
		sum2_fitness = sum_fitness + fitval*fitval
		min_fitness = min( min_fitness, fitval )
		max_fitness = max( max_fitness, fitval )

	# reduce to get global min/max/sum
	# : store local copies into shared arrays
	cuda.atomic.add( results, cfg.FITSTATS_SUM, sum_fitness )
	cuda.atomic.add( results, cfg.FITSTATS_SUM2, sum2_fitness )
	cuda.atomic.min( results, cfg.FITSTATS_MIN, min_fitness )
	cuda.atomic.max( results, cfg.FITSTATS_MAX, max_fitness )

def calcFitnessStats( gaMgr, fitvec, results ):
	results[cfg.FITSTATS_SUM]  = 0           # for sum calc
	results[cfg.FITSTATS_SUM2] = 0           # for sum^2 calc
	results[cfg.FITSTATS_MIN]  = math.inf    # for min calc
	results[cfg.FITSTATS_MAX]  = -math.inf   # for max calc
	gpu_calcFitnessStats[cfg.INT_GRID_SIZE,cfg.INT_BLOCK_SIZE]( gaMgr.dev_config_i, fitvec, results )

#
# Elitism function ...
#
@cuda.jit
def gpu_appendElitism( config_i, popvec_out, popvec_in ):
	tid     = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
	grid_sz = cuda.gridDim.x*cuda.blockDim.x
	pop_sz  = config_i[cfg.POPULATION_SIZE]
	chr_sz  = config_i[cfg.CHROMO_SIZE]
	num_e   = config_i[cfg.ELITISM_COUNT]

	for i in range(tid,num_e,grid_sz):
		for j in range(chr_sz):
			popvec_out[i*chr_sz+j] = popvec_in[i*chr_sz+j]

def appendElitism( gaMgr, popvec_out, popvec_in ):
	gpu_appendElitism[cfg.INT_GRID_SIZE,cfg.INT_BLOCK_SIZE]( gaMgr.dev_config_i, popvec_out, popvec_in )

#
# Mutation functions
#
# : for now, just one ...

@cuda.jit
def gpu_appendMutationAll( config_i, config_f, rng_states, popvec_out ):
	tid     = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
	grid_sz = cuda.gridDim.x*cuda.blockDim.x
	pop_sz  = config_i[cfg.POPULATION_SIZE]
	chr_sz  = config_i[cfg.CHROMO_SIZE]
	num_m   = config_i[cfg.PUREMUTATION_COUNT]
	# all pure-mutation children are offset in the vector by elitism+crossover counts
	popcnt  = config_i[cfg.ELITISM_COUNT] + config_i[cfg.CROSSOVER_COUNT]

	for i in range(tid,num_m,grid_sz):
		ii = i + popcnt    # offset by current population cursor

		for idx in range(chr_sz):
			min_val = config_i[cfg.DATA_RANGE_I+2*idx]
			max_val = config_i[cfg.DATA_RANGE_I+2*idx+1]
			popvec_out[ii*chr_sz+idx] = gpu.rand_uniform_int( min_val, max_val, rng_states,tid )

@cuda.jit
def gpu_appendMutationAllF( config_i, config_f, rng_states, popvec_out ):
	tid     = cuda.blockDim.x*cuda.blockIdx.x + cuda.threadIdx.x
	grid_sz = cuda.gridDim.x*cuda.blockDim.x
	pop_sz  = config_i[cfg.POPULATION_SIZE]
	chr_sz  = config_i[cfg.CHROMO_SIZE]
	num_m   = config_i[cfg.PUREMUTATION_COUNT]
	# all pure-mutation children are offset in the vector by elitism+crossover counts
	popcnt  = config_i[cfg.ELITISM_COUNT] + config_i[cfg.CROSSOVER_COUNT]

	for i in range(tid,num_m,grid_sz):
		ii = i + popcnt    # offset by current population cursor

		for idx in range(chr_sz):
			min_val = config_f[cfg.DATA_RANGE_F+2*idx]
			max_val = config_f[cfg.DATA_RANGE_F+2*idx+1]
			popvec_out[ii*chr_sz+idx] = gpu.rand_uniform_float( min_val, max_val, rng_states,tid )

def appendPureMutation( gaMgr, popvec_out ):
	if( gaMgr.dtype_num >= cfg.DATATYPE_LIST.index(np.float32) ):
		gpu_appendMutationAllF[cfg.INT_GRID_SIZE,cfg.INT_BLOCK_SIZE]( gaMgr.dev_config_i, gaMgr.dev_config_f, gaMgr.rng_states, popvec_out )
	else:
		gpu_appendMutationAll[cfg.INT_GRID_SIZE,cfg.INT_BLOCK_SIZE]( gaMgr.dev_config_i, gaMgr.dev_config_f, gaMgr.rng_states, popvec_out )


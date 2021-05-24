

import math
import numpy as np
from numba import cuda

import GenAlgCfg as cfg

@cuda.jit
def gpu_sortPopulationIncr( config_i, fitvec,popvec ):
	tid     = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	grid_sz = cuda.gridDim.x * cuda.blockDim.x
	pop_sz  = config_i[cfg.POPULATION_SIZE]
	chr_sz  = config_i[cfg.CHROMO_SIZE]
	psz2    = int( (pop_sz+1)/2 )

	for iter in range(psz2):
		# even phase
		for i in range( tid*2, pop_sz-1, grid_sz*2 ):
			if( fitvec[i] > fitvec[i+1] ):
				# swap( array+i, array+i+1 )
				# print( 'swap', tid, iter, i, fitvec[i], fitvec[i+1] )
				f = fitvec[i]
				fitvec[i] = fitvec[i+1]
				fitvec[i+1] = f
				for j in range(chr_sz):
					k1 = chr_sz*i + j
					k2 = chr_sz*(i+1) + j
					v = popvec[k1]
					popvec[k1] = popvec[k2]
					popvec[k2] = v

		# odd phase
		for i in range( tid*2+1, pop_sz-1, grid_sz*2 ):
			if( fitvec[i] > fitvec[i+1] ):
				# swap( array+i, array+i+1 )
				# print( 'swap', tid, iter, i, fitvec[i], fitvec[i+1] )
				f = fitvec[i]
				fitvec[i] = fitvec[i+1]
				fitvec[i+1] = f
				for j in range(chr_sz):
					k1 = chr_sz*i + j
					k2 = chr_sz*(i+1) + j
					v = popvec[k1]
					popvec[k1] = popvec[k2]
					popvec[k2] = v

@cuda.jit
def gpu_sortPopulationDesc( config_i, fitvec,popvec ):
	tid     = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	grid_sz = cuda.gridDim.x * cuda.blockDim.x
	pop_sz  = config_i[cfg.POPULATION_SIZE]
	chr_sz  = config_i[cfg.CHROMO_SIZE]
	psz2    = int( (pop_sz+1)/2 )

	for iter in range(psz2):
		# even phase
		for i in range( tid*2, pop_sz-1, grid_sz*2 ):
			if( fitvec[i] < fitvec[i+1] ):
				# swap( array+i, array+i+1 )
				# print( 'swap', tid, iter, i, fitvec[i], fitvec[i+1] )
				f = fitvec[i]
				fitvec[i] = fitvec[i+1]
				fitvec[i+1] = f
				for j in range(chr_sz):
					k1 = chr_sz*i + j
					k2 = chr_sz*(i+1) + j
					v = popvec[k1]
					popvec[k1] = popvec[k2]
					popvec[k2] = v

		# odd phasse
		for i in range( tid*2+1, pop_sz-1, grid_sz*2 ):
			if( fitvec[i] < fitvec[i+1] ):
				# swap( array+i, array+i+1 )
				# print( 'swap', tid, iter, i, fitvec[i], fitvec[i+1] )
				f = fitvec[i]
				fitvec[i] = fitvec[i+1]
				fitvec[i+1] = f
				for j in range(chr_sz):
					k1 = chr_sz*i + j
					k2 = chr_sz*(i+1) + j
					v = popvec[k1]
					popvec[k1] = popvec[k2]
					popvec[k2] = v

# when self.minOrMax=='min' ... sort in increasing order
# when self.minOrMax=='max' ... sort in decreasing order
def sortPopulation( gaMgr, fitvec,popvec ):
	if( gaMgr.minOrMax == 'max' ):
		gpu_sortPopulationDesc[cfg.INT_GRID_SIZE,cfg.INT_BLOCK_SIZE]( gaMgr.dev_config_i, fitvec,popvec )
	else:
		gpu_sortPopulationIncr[cfg.INT_GRID_SIZE,cfg.INT_BLOCK_SIZE]( gaMgr.dev_config_i, fitvec,popvec )


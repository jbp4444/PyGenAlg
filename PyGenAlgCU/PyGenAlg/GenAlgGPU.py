
import math
import numpy as np
import numba
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

import GenAlgCfg as cfg

#
# BASIC RANDOM NUMBER FUNCTIONS
#

# random integer from minval to <maxval (not inclusive)
@cuda.jit(device=True)
def rand_uniform_int( minval, maxval, rng_states, tid ):
	rtnval = minval + int( (maxval-minval)*xoroshiro128p_uniform_float32(rng_states,tid) )
	return rtnval

@cuda.jit(device=True)
def rand_uniform_float( minval, maxval, rng_states, tid ):
	rtnval = minval + (maxval-minval)*xoroshiro128p_uniform_float32(rng_states,tid)
	return rtnval

# "Algorithm L" from Wikipedia ... O( k*(1+log(N/k)) )
# : but source array is 0,1,2,3 ... S[i]=i
# @cuda.jit(device=True)
# def rand_sample_L( rtn, n, k, rng_states, tid ):
# 	# fill the reservoir array
# 	for i in range(k):
# 		rtn[i] = i
# 	# random() generates a uniform (0,1) random number
# 	W = math.exp(math.log( xoroshiro128p_uniform_float32(rng_states,tid) ) / k )
# 	while( i < n ):
# 		i = i + math.floor(math.log(xoroshiro128p_uniform_float32(rng_states,tid)) / math.log(1-W) )
# 		if( i < n ):
# 			# replace a random item of the reservoir with item i
# 			rtn[ rand_uniform_int(0,k) ] = i  # random index between 1 and k, inclusive
# 			W = W * math.exp(math.log(xoroshiro128p_uniform_float32(rng_states,tid))/k)

# "Algorithm R" from Wikipedia ... O(N) but uses fixed loops==better for GPUs?
# : https://en.wikipedia.org/wiki/Reservoir_sampling
# : but source array is 0,1,2,3 ... S[i]=i
@cuda.jit(device=True)
def rand_sample( n, k, rng_states,tid,tmp ):
	# TODO: check that len(tmp)>=k
	# fill the reservoir array
	for i in range(k):
		tmp[i] = i
	# now 'shuffle' the array entries (which are just 0:n)
	for i in range(k,n):
		m = rand_uniform_int( 0,i, rng_states,tid )
		if( m < k ):
			tmp[m] = i


#
# SELECTION FUNCTIONS
#

@cuda.jit(device=True)
def selectSimple( popsz, rng_states,tid,tmp ):
	rand_sample( popsz,2, rng_states,tid,tmp )
	mother = tmp[0]
	father = tmp[1]
	return (mother,father)

@cuda.jit(device=True)
def selectTournament( popsz, fitvec, tourn_k,rng_states,tid,tmp ):
	# TODO: check that len(tmp)>=tourn_k
	rand_sample( popsz, tourn_k, rng_states,tid,tmp )
	best_i = tmp[0]
	best_f = fitvec[best_i]
	for k in range(1,tourn_k):
		new_i = tmp[k]
		new_f = fitvec[new_i]
		if( new_f > best_f ):
			best_i = new_i
			best_f = new_f
	mother = best_i
	rand_sample( popsz, tourn_k, rng_states,tid,tmp )
	best_i = tmp[0]
	best_f = fitvec[best_i]
	for k in range(1,tourn_k):
		new_i = tmp[k]
		new_f = fitvec[new_i]
		if( new_f > best_f ):
			best_i = new_i
			best_f = new_f
	father = best_i
	return (mother,father)

@cuda.jit(device=True)
def int_rouletteMaxPos( popsz, fitvec, rng_states,tid,fitstats ):
	# for maximization problems with sum>0 ... xform = lambda x: x
	sumfit = fitstats[cfg.FITSTATS_SUM]
	rtn = -1

	# select random value
	#value = random.random() * abs(sumfit)
	value = rand_uniform_float(0.0,1.0,rng_states,tid) * abs(sumfit)

	# loop over population
	for i in range(popsz):
		value = value - fitvec[i]
		if( value < 0 ):
			rtn = i
			break

	# locate the random value based on the weights
	if( rtn < 0 ):
		rtn = popsz - 1
	return rtn

@cuda.jit(device=True)
def selectRouletteMaxPos( popsz, fitvec, rng_states,tid,sumfit ):
	idx1 = int_rouletteMaxPos( popsz,fitvec, rng_states,tid,sumfit )
	idx2 = int_rouletteMaxPos( popsz,fitvec, rng_states,tid,sumfit )
	return (idx1,idx2)

@cuda.jit(device=True)
def int_rouletteMaxNeg( popsz, fitvec, rng_states,tid,fitstats ):
	# for maximization problems with sum<0 ... xform = lambda x: x - gaMgr.min_fitness + 1
	sumfit = fitstats[cfg.FITSTATS_SUM]
	minfit = fitstats[cfg.FITSTATS_MIN]
	rtn = -1

	# select random value
	#value = random.random() * abs(sumfit)
	value = rand_uniform_float(0.0,1.0,rng_states,tid) * abs(sumfit)

	# loop over population
	for i in range(popsz):
		#value = value - xform(fitvec[i])
		value = value - ( fitvec[i] - minfit + 1)
		if( value < 0 ):
			rtn = i
			break

	# locate the random value based on the weights
	if( rtn < 0 ):
		rtn = popsz - 1
	return rtn

@cuda.jit(device=True)
def selectRouletteMaxNeg( popsz, fitvec, rng_states,tid,fitstats ):
	idx1 = int_rouletteMaxNeg( popsz,fitvec, rng_states,tid,fitstats )
	idx2 = int_rouletteMaxNeg( popsz,fitvec, rng_states,tid,fitstats )
	return (idx1,idx2)

@cuda.jit(device=True)
def int_rouletteMinPos( popsz, fitvec, rng_states,tid,fitstats ):
	# for minimization problems with sum>0 ... xform = lambda x: gaMgr.max_fitness - x + 1
	sumfit = fitstats[cfg.FITSTATS_SUM]
	maxfit = fitstats[cfg.FITSTATS_MAX]
	rtn = -1

	# select random value
	#value = random.random() * abs(sumfit)
	value = rand_uniform_float(0.0,1.0,rng_states,tid) * abs(sumfit)

	# loop over population
	for i in range(popsz):
		#value = value - xform(fitvec[i])
		value = value - ( maxfit - fitvec[i] + 1)
		if( value < 0 ):
			rtn = i
			break

	# locate the random value based on the weights
	if( rtn < 0 ):
		rtn = popsz - 1
	return rtn

@cuda.jit(device=True)
def selectRouletteMinPos( popsz, fitvec, rng_states,tid,fitstats ):
	idx1 = int_rouletteMinPos( popsz,fitvec, rng_states,tid,fitstats )
	idx2 = int_rouletteMinPos( popsz,fitvec, rng_states,tid,fitstats )
	return (idx1,idx2)

@cuda.jit(device=True)
def int_rouletteMinNeg( popsz, fitvec, rng_states,tid,fitstats ):
	# for minimization problems with sum<0 ... xform = lambda x: -x
	sumfit = fitstats[cfg.FITSTATS_SUM]
	rtn = -1

	# select random value
	#value = random.random() * abs(sumfit)
	value = rand_uniform_float(0.0,1.0,rng_states,tid) * abs(sumfit)

	# loop over population
	for i in range(popsz):
		#value = value - xform(fitvec[i])
		value = value + fitvec[i]
		if( value < 0 ):
			rtn = i
			break

	# locate the random value based on the weights
	if( rtn < 0 ):
		rtn = popsz - 1
	return rtn

@cuda.jit(device=True)
def selectRouletteMinNeg( popsz, fitvec, rng_states,tid,fitstats ):
	idx1 = int_rouletteMinNeg( popsz,fitvec, rng_states,tid,fitstats )
	idx2 = int_rouletteMinNeg( popsz,fitvec, rng_states,tid,fitstats )
	return (idx1,idx2)

# https://www.msi.umn.edu/sites/default/files/OptimizingWithGA.pdf
@cuda.jit(device=True)
def int_rankSelection( popsz, fitvec, rng_states,tid,fitstats ):
	rtn = -1

	xform = lambda x: 1.0/(x+1.0)
	sum_xform = 0
	for i in range(popsz):
		sum_xform = sum_xform + xform(i)

	# select random value
	value = rand_uniform_float(0.0,1.0,rng_states,tid) * sum_xform

	# loop over population
	for i in range(popsz):
		value = value - xform(i)
		if( value < 0 ):
			rtn = i
			break

	# locate the random value based on the weights
	if( rtn < 0 ):
		rtn = popsz - 1
	return rtn

@cuda.jit(device=True)
def selectRank( popsz, fitvec, rng_states,tid,fitstats ):
	""" selection-function that uses rank-selection """
	idx1 = int_rankSelection( popsz, fitvec, rng_states,tid,fitstats )
	idx2 = int_rankSelection( popsz, fitvec, rng_states,tid,fitstats )
	return idx1,idx2


#
# CROSSOVER/MUTATION FUNCTIONS
#

@cuda.jit(device=True)
def mutateFew( ii, popvec_out, num_m, config_i,rng_states,tid,tmp ):
	chr_sz = config_i[cfg.CHROMO_SIZE]
	rand_sample( chr_sz, num_m, rng_states,tid,tmp )
	for k in range(num_m):
		idx = tmp[k]
		min_val = config_i[cfg.DATA_RANGE_I+2*idx]
		max_val = config_i[cfg.DATA_RANGE_I+2*idx+1]
		popvec_out[ii*chr_sz+idx] = rand_uniform_int( min_val,max_val, rng_states,tid )

@cuda.jit(device=True)
def mutateRandom( ii, popvec_out, mut_pct, config_i,rng_states,tid ):
	chr_sz  = config_i[cfg.CHROMO_SIZE]
	for idx in range(chr_sz):
		if( rand_uniform_float(0.0,1.0,rng_states,tid) <= mut_pct ):
			min_val = config_i[cfg.DATA_RANGE_I+2*idx]
			max_val = config_i[cfg.DATA_RANGE_I+2*idx+1]
			popvec_out[ii*chr_sz+idx] = rand_uniform_int( min_val,max_val, rng_states,tid )

@cuda.jit(device=True)
def mutateFewF( ii, popvec_out, num_m, config_i,config_f,rng_states,tid,tmp ):
	chr_sz = config_i[cfg.CHROMO_SIZE]
	rand_sample( chr_sz, num_m, rng_states,tid,tmp )
	for k in range(num_m):
		idx = tmp[k]
		min_val = config_f[cfg.DATA_RANGE_F+2*idx]
		max_val = config_f[cfg.DATA_RANGE_F+2*idx+1]
		popvec_out[ii*chr_sz+idx] = rand_uniform_float( min_val,max_val, rng_states,tid )

@cuda.jit(device=True)
def mutateRandomF( ii, popvec_out, mut_pct, config_i,config_f,rng_states,tid ):
	chr_sz  = config_i[cfg.CHROMO_SIZE]
	for idx in range(chr_sz):
		if( rand_uniform_float(0.0,1.0,rng_states,tid) <= mut_pct ):
			min_val = config_f[cfg.DATA_RANGE_F+2*idx]
			max_val = config_f[cfg.DATA_RANGE_f+2*idx+1]
			popvec_out[ii*chr_sz+idx] = rand_uniform_float( min_val,max_val, rng_states,tid )

# TODO: add "creep" mutation options (modify gene by X% instead of uniform random num across whole range)


#
# CROSSOVER/GENE MIXING FUNCTIONS
#

@cuda.jit(device=True)
def crossover11( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp ):
	chr_sz = config_i[cfg.CHROMO_SIZE]

	# crossover11 -- 1 point crossover leading to 1 child
	idx = rand_uniform_int( 0,chr_sz, rng_states,tid )
	for j in range(0,idx):
		popvec_out[ii*chr_sz+j] = popvec_in[mother*chr_sz+j]
	for j in range(idx,chr_sz):
		popvec_out[ii*chr_sz+j] = popvec_in[father*chr_sz+j]

@cuda.jit(device=True)
def crossover12( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp ):
	chr_sz = config_i[cfg.CHROMO_SIZE]

	# crossover12 -- 1 point crossover leading to 2 children
	idx = rand_uniform_int( 0,chr_sz, rng_states,tid )
	for j in range(0,idx):
		popvec_out[ii*chr_sz+j]     = popvec_in[mother*chr_sz+j]
		popvec_out[(ii+1)*chr_sz+j] = popvec_in[father*chr_sz+j]
	for j in range(idx,chr_sz):
		popvec_out[ii*chr_sz+j]     = popvec_in[father*chr_sz+j]
		popvec_out[(ii+1)*chr_sz+j] = popvec_in[mother*chr_sz+j]

@cuda.jit(device=True)
def crossover21( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp ):
	chr_sz = config_i[cfg.CHROMO_SIZE]

	# crossover21 -- 2 point crossover leading to 1 child
	rand_sample( chr_sz, 2, rng_states,tid,tmp )
	if( tmp[0] > tmp[1] ):
		index1 = tmp[1]
		index2 = tmp[0]
	else:
		index1 = tmp[0]
		index2 = tmp[1]
	for j in range(0,index1):
		popvec_out[ii*chr_sz+j] = popvec_in[mother*chr_sz+j]
	for j in range(index1,index2):
		popvec_out[ii*chr_sz+j] = popvec_in[father*chr_sz+j]
	for j in range(index2,chr_sz):
		popvec_out[ii*chr_sz+j] = popvec_in[mother*chr_sz+j]

@cuda.jit(device=True)
def crossover22( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp ):
	chr_sz = config_i[cfg.CHROMO_SIZE]

	# crossover22 -- 2 point crossover leading to 2 children
	rand_sample( chr_sz, 2, rng_states,tid,tmp )
	if( tmp[0] > tmp[1] ):
		index1 = tmp[1]
		index2 = tmp[0]
	else:
		index1 = tmp[0]
		index2 = tmp[1]
	for j in range(0,index1):
		popvec_out[ii*chr_sz+j]     = popvec_in[mother*chr_sz+j]
		popvec_out[(ii+1)*chr_sz+j] = popvec_in[father*chr_sz+j]
	for j in range(index1,index2):
		popvec_out[ii*chr_sz+j]     = popvec_in[father*chr_sz+j]
		popvec_out[(ii+1)*chr_sz+j] = popvec_in[mother*chr_sz+j]
	for j in range(index2,chr_sz):
		popvec_out[ii*chr_sz+j]     = popvec_in[mother*chr_sz+j]
		popvec_out[(ii+1)*chr_sz+j] = popvec_in[father*chr_sz+j]

@cuda.jit(device=True)
def crossoverUniform1( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp ):
	chr_sz = config_i[cfg.CHROMO_SIZE]

	# uniform crossover leading to 1 child
	for j in range(0,chr_sz):
		if( xoroshiro128p_uniform_float32(rng_states,tid) < 0.50 ):
			popvec_out[ii*chr_sz+j] = popvec_in[mother*chr_sz+j]
		else:
			popvec_out[ii*chr_sz+j] = popvec_in[father*chr_sz+j]

@cuda.jit(device=True)
def crossoverUniform2( popvec_in, mother, father, ii, popvec_out, config_i,rng_states,tid,tmp ):
	chr_sz = config_i[cfg.CHROMO_SIZE]

	# uniform crossover leading to 2 children
	for j in range(0,chr_sz):
		if( xoroshiro128p_uniform_float32(rng_states,tid) < 0.50 ):
			popvec_out[ii*chr_sz+j]     = popvec_in[mother*chr_sz+j]
			popvec_out[(ii+1)*chr_sz+j] = popvec_in[father*chr_sz+j]
		else:
			popvec_out[ii*chr_sz+j]     = popvec_in[father*chr_sz+j]
			popvec_out[(ii+1)*chr_sz+j] = popvec_in[mother*chr_sz+j]

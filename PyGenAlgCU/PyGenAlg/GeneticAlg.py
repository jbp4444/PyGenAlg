#
# a "basic" genetic algorithm class
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
#

import math
import numpy as np
import numba
from numba import cuda

# to disable deprecation warnings ...
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

import GenAlgCfg as cfg
import GenAlgOps, GenAlgSort


class GenAlg:
	def __init__( self, **kwargs ):
		# parameters for the chromosome ...
		self.chromo_sz     = kwargs.get( 'chromoSize', 0 )
		self.dataRange     = kwargs.get( 'range', (0,10) )
		self.dataType      = kwargs.get( 'dtype', np.float32 )
		self.fitnessFcn    = kwargs.get( 'fitnessFcn', None )
		self.crossoverFcn  = kwargs.get( 'crossoverFcn', None )

		# specify population-generator-sizes by integer counts or pct ...
		self.elitism       = kwargs.get( 'elitism', 0.10 )
		self.crossover     = kwargs.get( 'crossover', 0.80)
		self.pureMutation  = kwargs.get( 'pureMutation', 0.10 )
		self.migration     = kwargs.get( 'migration', 0.0 )
		# other kwargs ...
		self.population_sz = kwargs.get( 'size', 10 )
		self.minOrMax      = kwargs.get( 'minOrMax', 'max' )
		self.showBest      = kwargs.get( 'showBest', 0 )
		self.parent_pct    = kwargs.get( 'parentPct', 1.0 )
		# hooks for migration
		# self.migrationSendFcn = kwargs.get( 'migrationSendFcn', None )
		# self.migrationRecvFcn = kwargs.get( 'migrationRecvFcn', None )
		# self.migrationSkip    = kwargs.get( 'migrationSkip', 1 )
		# hooks for user-data  (copied/sent to the user-provided fitness function)
		self.userData_i   = kwargs.get( 'userDataI', None )
		self.userData_f   = kwargs.get( 'userDataF', None )

		# check that basic params look good
		if( self.chromo_sz <= 0 ):
			raise ValueError('Chromosome size must be >= 1')

		if( self.dataType == None ):
			self.dataType = np.float32

		# TODO: allow different ranges for each gene
		if( self.dataRange == None ):
			self.dataRange = [ (0,10) for i in range(self.chromo_sz) ]
		elif( type(self.dataRange) is tuple ):
			# TODO: check that it is a 2-tuple
			dr = self.dataRange
			self.dataRange = [ dr for i in range(self.chromo_sz) ]
		# TODO: check that it is a list of 2-tuples

		# convert percentages to integer numbers of chromos
		# TODO: could/should check if float and < 1.0 then assume it's a percentage
		#       (otherwise can't test with pct=100% since that would not be < 1)
		if( self.elitism < 1 ):
			self.elitism = int( self.population_sz * self.elitism + 0.5 )
		else:
			self.elitism      = int( self.elitism )
		if( self.crossover < 1 ):
			self.crossover = int( self.population_sz * self.crossover + 0.5 )
		else:
			self.crossover    = int( self.crossover )
		if( self.migration < 1 ):
			self.migration = int( self.population_sz * self.migration + 0.5 )
		else:
			self.migration = int( self.migration )
		if( self.pureMutation < 1 ):
			self.pureMutation = int( self.population_sz * self.pureMutation + 0.5 )
		else:
			self.pureMutation = int( self.pureMutation )

		# check that sum(elitism+crossover+migration+pureMutation) == pop_sz
		# TODO: adjust counts if needed
		if( (self.elitism+self.crossover+self.pureMutation) != self.population_sz ):
			print( '* ERROR: population size and elitism/crossover/mutation counts do not match' )

		if( (self.minOrMax!='min') and (self.minOrMax!='max') ):
			raise ValueError('minOrMax must be min or max')

		# if( self.migration > 0 ):
		# 	if( not callable(self.migrationSendFcn) ):
		# 		raise ValueError('migrationSendFcn is not callable')
		# 	if( not callable(self.migrationRecvFcn) ):
		# 		raise ValueError('migrationRecvFcn is not callable')
		# 	self.migrationCounter = 0

		# the computation vectors ...
		self.pvecSize    = self.population_sz * self.chromo_sz
		self.population  = np.zeros( self.pvecSize, dtype=self.dataType )
		self.fitnessVals = np.zeros( self.population_sz, dtype=np.float32 )

		# pre-allocate 2 pop-vectors on the device
		self.dev_pvec = [	cuda.device_array( shape=(self.pvecSize,), dtype=self.dataType ), 
							cuda.device_array( shape=(self.pvecSize,), dtype=self.dataType ) ]
		self.dev_fitvec =   cuda.device_array( shape=(self.population_sz,), dtype=np.float32 )

		# just to be sure we get different random numbers
		# : user can always override with random.setState
		self.rng = np.random.default_rng()
		# init and save the CUDA random number state
		self.rng_states = cuda.random.create_xoroshiro128p_states( cfg.INT_GRID_SIZE*cfg.INT_BLOCK_SIZE, seed=self.rng.integers(0,1000000) )

		self.dtype_num = cfg.DATATYPE_LIST.index( self.dataType )
		# TODO: only store data-ranges for int or float (now we do both)
		self.config_i = [ 0 for i in range(cfg.CFG_I_LEN+2*self.chromo_sz) ]
		self.config_f = [ 0.0 for i in range(cfg.CFG_F_LEN+2*self.chromo_sz) ]
		self.config_i[cfg.POPULATION_SIZE] = self.population_sz
		self.config_i[cfg.CHROMO_SIZE]     = self.chromo_sz
		self.config_i[cfg.DATA_TYPE]       = self.dtype_num
		self.config_i[cfg.ELITISM_COUNT]   = self.elitism
		self.config_i[cfg.CROSSOVER_COUNT] = self.crossover
		self.config_i[cfg.PUREMUTATION_COUNT]  = self.pureMutation
		self.config_i[cfg.MIGRATION_COUNT] = 0
		if( self.minOrMax == 'min' ):
			self.config_i[cfg.MIN_OR_MAX]  = 0
		else:
			self.config_i[cfg.MIN_OR_MAX]  = 1
		
		self.config_f[cfg.PARENT_PCT] = self.parent_pct
		if( self.dtype_num >= cfg.DATATYPE_LIST.index(np.float32) ):
			for i in range(self.chromo_sz):
				self.config_f[cfg.DATA_RANGE_F+2*i]   = self.dataRange[i][0]
				self.config_f[cfg.DATA_RANGE_F+2*i+1] = self.dataRange[i][1]
		else:
			for i in range(self.chromo_sz):
				self.config_i[cfg.DATA_RANGE_I+2*i]   = self.dataRange[i][0]
				self.config_i[cfg.DATA_RANGE_I+2*i+1] = self.dataRange[i][1]

		# push user- and config-data out to device
		self.dev_config_i = cuda.to_device( self.config_i )
		self.dev_config_f = cuda.to_device( self.config_f )
		self.dev_userdata_i = cuda.to_device( self.userData_i )
		self.dev_userdata_f = cuda.to_device( self.userData_f )

	# maybe this should be __repr__ or __str__?
	def describe(self):
		print( 'Genetic Algorithm object:' )
		print( '   pop size: '+str(self.population_sz) )
		print( '   chromo size: '+str(self.chromo_sz) )
		print( '   chromo datatype: ', self.dataType )
		# print( '   chromo data-range: ', self.dataRange )
		print( '   elitism: %d :: %0.1f%%' % (self.elitism,float(100*self.elitism)/self.population_sz) )
		print( '   crossover: %d :: %0.1f%%' % (self.crossover,float(100*self.crossover)/self.population_sz) )
		print( '   mutation: %d :: %0.1f%%' % (self.pureMutation,float(100*self.pureMutation)/self.population_sz) )
		print( '   parent pct: %0.1f%%' % (100.0*self.parent_pct) )
		# print( '   migration: %d :: %0.1f%%' % (self.migration,float(100*self.migration)/self.population_sz) )
		print( '   min_or_max: '+self.minOrMax )

	def reseedRng(self):
		# init and save the CUDA random number state
		self.rng_states = cuda.random.create_xoroshiro128p_states( cfg.INT_GRID_SIZE*cfg.INT_BLOCK_SIZE, seed=self.rng.integers(0,1000000) )

	# start with random data
	def initPopulation(self):
		# NOTE: right now, chromo is a single data-type and the code below doesn't make sense (inefficient)
		#       at some point, we'll want to allow multiple data-types and maybe this makes sense then
		pop = self.population
		for p in range(self.population_sz):
			for i in range(self.chromo_sz):
				if( (self.dataType is np.float32) or (self.dataType is np.float64) ):
					pop[p*self.chromo_sz+i] = np.random.uniform( self.dataRange[i][0], self.dataRange[i][1] )
				else:
					pop[p*self.chromo_sz+i] = np.random.randint( self.dataRange[i][0], self.dataRange[i][1] )

		self.population = pop

	def loadPopulation( self, items ):
		# TODO: check that item-size <= population_sz*chromo_sz
		item_sz   = len(items)
		for i in range(item_sz):
			self.population[i] = items[i]

	def evolve( self, iters ):
		# push population vector out to device
		cuda.to_device( self.population, to=self.dev_pvec[0] )

		prev_vec = 0
		active_vec = 0

		# make sure the chromo's are sorted first ... requires that we calc fitness too
		self.fitnessFcn[cfg.USER_GRID_SIZE,cfg.USER_BLOCK_SIZE]( self.dev_pvec[active_vec], self.dev_fitvec, self.dev_config_i, self.dev_config_f, self.dev_userdata_i,self.dev_userdata_f )

		results = np.zeros( cfg.FITSTATS_LEN, np.float32 )
		GenAlgOps.calcFitnessStats( self, self.dev_fitvec, results )

		GenAlgSort.sortPopulation( self, self.dev_fitvec,self.dev_pvec[active_vec] )

		for iter in range(iters):
			prev_vec = active_vec
			active_vec = (active_vec+1)%2
			# print( 'evolve', iter, prev_vec, active_vec )
			
			# while we add elitism population "first", we can
			# send any migrants out now, to minimize any network slowness
			# NOTE: this func does not remove the migrant from the
			#       current population, it makes a copy to send to
			#       the remote population
			# migrants_out = []
			# migrants_idx_out = []
			# if( self.migration > 0 ):
			# 	# only do migrations every N generations
			# 	if( self.migrationCounter == 0 ):
			# 		for i in range(0,self.migration):
			# 			# TODO: migrant should be removed from population (if present)
			# 			idx1 = random.randrange(self.population_sz)
			# 			migrants_out.append( pop[ idx1 ] )
			# 			migrants_idx_out.append( idx1 )
			# 		self.migrationSendFcn( migrants_out )

			# TODO: process these with 'feasibleSolnFcn' to make sure they are checksummed/hashed/etc.

			# first group is best-N chromos (elitism)
			GenAlgOps.appendElitism( self, self.dev_pvec[active_vec], self.dev_pvec[prev_vec] )

			# next group are computed from crossover and mutation
			self.crossoverFcn[cfg.USER_GRID_SIZE,cfg.USER_BLOCK_SIZE]( self.dev_pvec[prev_vec], self.dev_pvec[active_vec], self.dev_fitvec, self.dev_config_i,self.dev_config_f,self.rng_states,results )

			# last group are pure-mutation
			GenAlgOps.appendPureMutation( self, self.dev_pvec[active_vec] )

			# if present, do migration (callback to user-code)
			# migrants_in = []
			# if( self.migration > 0 ):
			# 	# only do migrations every N generations
			# 	if( self.migrationCounter == 0 ):
			# 		migrants_in = self.migrationRecvFcn()
			# 	# else:
			# 	# 	print( 'skipped migration' )
				
			# 	# handle the update of the migration-counter
			# 	self.migrationCounter = self.migrationCounter + 1
			# 	if( self.migrationCounter == self.migrationSkip ):
			# 		self.migrationCounter = 0

			# calculate fitness function for all chromos
			self.fitnessFcn[cfg.USER_GRID_SIZE,cfg.USER_BLOCK_SIZE]( self.dev_pvec[active_vec], self.dev_fitvec, self.dev_config_i,self.dev_config_f, self.dev_userdata_i,self.dev_userdata_f )

			# calc basic stats (for roulette wheel selection)
			GenAlgOps.calcFitnessStats( self, self.dev_fitvec, results )

			# sort the population by fitness vals
			GenAlgSort.sortPopulation( self, self.dev_fitvec,self.dev_pvec[active_vec] )

			# # show a progress report?
			# if( self.showBest > 0 ):
			# 	print( "best chromo:" )
			# 	for i in range(self.showBest):
			# 		print( self.population[i] )

		# copy data back to host for further analysis
		self.population = self.dev_pvec[active_vec].copy_to_host()
		self.fitnessVals = self.dev_fitvec.copy_to_host()

	# at end of a full 'evolve' run, we can inspect the CUDA code for regs per thread
	def cudaAnalysis( self ):
		# turn off deprecation warnings ...
		warnings.simplefilter( 'ignore', category=NumbaDeprecationWarning )
		warnings.simplefilter( 'ignore', category=NumbaPendingDeprecationWarning )

		# from:  https://stackoverflow.com/questions/63823395/how-can-i-get-the-number-of-cuda-cores-in-my-gpu-using-python-and-numba
		cc_cores_per_SM_dict = {
			(2,0) : 32,
			(2,1) : 48,
			(3,0) : 192,
			(3,5) : 192,
			(3,7) : 192,
			(5,0) : 128,
			(5,2) : 128,
			(6,0) : 64,
			(6,1) : 128,
			(7,0) : 64,
			(7,5) : 64,
			(8,0) : 64,
			(8,6) : 128
			}

		# TODO: should only be 1 entry in each of these reg/thr hash tables!
		calcFitness_regs = list(self.fitnessFcn.get_regs_per_thread().items())[0][1]
		crossover_regs = list(self.crossoverFcn.get_regs_per_thread().items())[0][1]
		calcFitnessStats_regs = list(GenAlgOps.gpu_calcFitnessStats.get_regs_per_thread().items())[0][1]
		appendElitism_regs = list(GenAlgOps.gpu_appendElitism.get_regs_per_thread().items())[0][1]
		if( self.dtype_num >= cfg.DATATYPE_LIST.index(np.float32) ):
			appendMutation_regs = list(GenAlgOps.gpu_appendMutationAllF.get_regs_per_thread().items())[0][1]
		else:
			appendMutation_regs = list(GenAlgOps.gpu_appendMutationAll.get_regs_per_thread().items())[0][1]
		if( self.minOrMax == 'max' ):
			sortPopulation_regs = list(GenAlgSort.gpu_sortPopulationDesc.get_regs_per_thread().items())[0][1]
		else:
			sortPopulation_regs = list(GenAlgSort.gpu_sortPopulationIncr.get_regs_per_thread().items())[0][1]

		dev = cuda.get_current_device()
		ctx = cuda.current_context()
		gpumem = ctx.get_memory_info()

		cores_per_sm = cc_cores_per_SM_dict.get( dev.compute_capability )
		total_cores = cores_per_sm * dev.MULTIPROCESSOR_COUNT

		def get_ptx_shared_mem( fcn ):
			nfuncs = 0
			smem = 0
			lmem = 0
			rawptx = fcn.inspect_asm()
			ptx = str(rawptx)
			print( 'ptx', len(ptx) )

			# find 'Generated by NVIDIA NVVM Compiler' comments
			# as a proxy for number of functions inside this ptx block
			i = 0
			while( True ):
				# look for ".shared .align"
				try:
					i = ptx.index( 'Generated by NVIDIA NVVM Compiler', i )
					print( 'found func start' )
					nfuncs = nfuncs + 1
					i = i + 32
				except ValueError as e:
					break

			# now find '.shared .align' definitions (use .align to avoid ld.shared op-codes)
			i = 0
			while( True ):
				# look for ".shared .align"
				try:
					i = ptx.index( '.shared .align', i )
					j = ptx.index( ';', i+8 )
					k1 = ptx.index( '[', i+8 ) + 1
					k2 = ptx.index( ']', k1 )
					print( 's-found', ptx[i:j], ptx[k1:k2] )
					smem = smem + int(ptx[k1:k2])
					i = j + 1
				except ValueError as e:
					break

			# now find '.local .align' definitions (use .align to avoid ld.local op-codes)
			i = 0
			while( True ):
				# look for ".local .align"
				try:
					i = ptx.index( '.local .align', i )
					j = ptx.index( ';', i+8 )
					k1 = ptx.index( '[', i+8 ) + 1
					k2 = ptx.index( ']', k1 )
					print( 'l-found', ptx[i:j], ptx[k1:k2] )
					lmem = lmem + int(ptx[k1:k2])
					i = j + 1
				except ValueError as e:
					break
			smem = int( smem/nfuncs )
			lmem = int( lmem/nfuncs )
			return (smem,lmem)

		(calcFitness_smem,calcFitness_lmem) = get_ptx_shared_mem( self.fitnessFcn )
		(crossover_smem,crossover_lmem) = get_ptx_shared_mem( self.crossoverFcn )

		# TODO: refactor with print-fmts
		print( 'cuda analysis:' )
		print( '   current device:' )
		print( '      name:', dev.name )
		print( '      compute capability:', dev.compute_capability )
		print( '      processor count:', dev.MULTIPROCESSOR_COUNT )
		print( '      total cores:', total_cores, '  (',cores_per_sm,')' )
		print( '      gpu memory:', int(gpumem.total/1024/1024/1024), 'GB' )
		print( '      max block size:', dev.MAX_BLOCK_DIM_X, dev.MAX_BLOCK_DIM_Y, dev.MAX_BLOCK_DIM_Z )
		print( '      max grid size:', dev.MAX_GRID_DIM_X, dev.MAX_GRID_DIM_Y, dev.MAX_GRID_DIM_Z )
		print( '      threads per block:', dev.MAX_THREADS_PER_BLOCK )
		print( '      shared mem per block:', dev.MAX_SHARED_MEMORY_PER_BLOCK )
		print( '   user functions:' )
		print( '      launch config: ', cfg.USER_GRID_SIZE, 'x', cfg.USER_BLOCK_SIZE )
		print( '         min pop_sz: ', cfg.USER_GRID_SIZE*cfg.USER_BLOCK_SIZE )
		print( '      fitnessFcn:       reg/thr=', calcFitness_regs, '  smem=', calcFitness_smem, '  lmem=', calcFitness_lmem )
		print( '      crossoverFcn:     reg/thr=', crossover_regs, '  smem=', crossover_smem, '  lmem=', crossover_lmem )
		print( '   internal functions:' )
		print( '      launch config: ', cfg.INT_GRID_SIZE, 'x', cfg.INT_BLOCK_SIZE )
		print( '         min pop_sz: ', cfg.INT_GRID_SIZE*cfg.INT_BLOCK_SIZE )
		print( '      calcFitnessStats: reg/thr=', calcFitnessStats_regs )
		print( '      appendElitism:    reg/thr=', appendElitism_regs )
		print( '      appendMutation:   reg/thr=', appendMutation_regs, '  (',self.dataType,')' )
		print( '      sortPopulation:   reg/thr=', sortPopulation_regs, '  (',self.minOrMax,')' )
	

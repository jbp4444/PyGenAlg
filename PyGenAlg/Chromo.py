#
# a "basic" chromosome class that you should subclass;
#
# you MUST override calcFitness()
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import random
from copy import deepcopy
import struct
import base64

# TODO: create a class method .dcopy() to do a deepcopy of data
#       but leave pointers to other arrays, etc

class BaseChromo(object):
	def __init__( self, **kwargs ):

		# parameters for this chromosome ...
		self.chromo_sz     = kwargs.get( 'size', None )
		self.dataRange     = kwargs.get( 'range', (0,10) )
		self.dataType      = kwargs.get( 'dtype', float )
		self.mutateNum     = kwargs.get( 'mutateNum', 3 )
		self.mutateNumPct  = kwargs.get( 'mutateNumPct', None )
		self.mutatePct     = kwargs.get( 'mutatePct', 0.25 )
		self.crossover_fcn = kwargs.get( 'crossover', 'crossover11' )
		self.mutate_fcn    = kwargs.get( 'mutate', 'mutateAll' )

		in_data = kwargs.get( 'data', None )

		# calculated values
		self.fitness     = None

		if( self.chromo_sz <= 0 ):
			raise ValueError('Chromosome size must be >= 1')

		if( self.dataType == None ):
			self.dataType = [ float for i in range(self.chromo_sz) ]
		elif( self.dataType is float ):
			self.dataType = [ float for i in range(self.chromo_sz) ]
		elif( self.dataType is int ):
			self.dataType = [ int for i in range(self.chromo_sz) ]
		else:
			if( len(self.dataType) != self.chromo_sz ):
				raise ValueError('DataType must be float, int, or array of size chromo_sz')

		if( self.dataRange == None ):
			self.dataRange = [ (0,10) for i in range(self.chromo_sz) ]
		elif( type(self.dataRange) is tuple ):
			# TODO: check that it is a 2-tuple
			dr = self.dataRange
			self.dataRange = [ dr for i in range(self.chromo_sz) ]
		# TODO: check that it is a list of 2-tuples

		# TODO: check mutateNum and mutateNumPct
		# TODO: check mutatePct ... naming is confusing too

		# start with random data
		if( in_data != None ):
			self.data = in_data
		else:
			self.data = []
			for i in range(self.chromo_sz):
				if( self.dataType[i] is float ):
					self.data.append( random.uniform( self.dataRange[i][0], self.dataRange[i][1] ) )
				elif( self.dataType[i] is int ):
					self.data.append( random.randint( self.dataRange[i][0], self.dataRange[i][1] ) )

	def setInitData( self, data ):
		if( len(data) != self.chromo_sz ):
			raise ValueError('Data should be of size chromo_sz' )
		# TODO: check that data is valid (vs data-type)
		self.data = data

	def zeroOut( self ):
		for i in range(self.chromo_sz):
			if( self.dataType[i] is float ):
				self.data[i] = 0.0
			elif( self.dataType[i] is int ):
				self.data[i] = 0

	def getFitness( self ):
		return self.fitness

	# NEED TO OVERRIDE
	def calcFitness( self ):
		self.fitness = None

	def __str__( self ):
		txt = 'data=' + ','.join( str(i) for i in self.data ) \
				+ ' .. fit=' + str(self.fitness)
		return txt

	# pack just the data into a text/base64 format
	# format is:  fmt-string==base64data==
	# where fmt-string is the struct.pack format string
	# based on the Chromo's dataType entries
	def packData(self):
		fmt = '<'
		for tp in self.dataType:
			if( tp is int ):
				fmt = fmt + 'i'
			elif( tp is float ):
				fmt = fmt + 'f'
			# TODO: add double, char, etc.
		return fmt + '==' + base64.b64encode( struct.pack( fmt, *self.data ) )

	# unpack the data from the text/base64 format
	def unpackData(self,data):
		i = data.index('==')
		if( i < 0 ):
			# this is the wrong format, no struct-fmt string prepended
			# TODO: throw error?
			return
		#else:
		in_fmt = data[:i]
		udata = base64.b64decode( data[i+2:] )
		fmt = '<'
		for tp in self.dataType:
			if( tp is int ):
				fmt = fmt + 'i'
			elif( tp is float ):
				fmt = fmt + 'f'
			# TODO: add double, char, etc.
		if( in_fmt != fmt ):
			# this fmt string does not match this Chromo;
			# maybe you grabbed the wrong file for this GA?
			# TODO: throw error?
			return
		# store into the 'data' array
		vals = struct.unpack( fmt, udata )
		for i in range(self.chromo_sz):
			self.data[i] = vals[i]

	#
	# Crossover functions
	#

	# 1 crossover-point leads to 1 child
	def crossover11( self, father ):
		# single cutover
		mother = self
		child = deepcopy(mother)
		child.fitness = None
		# cutover at what location?
		idx = random.randint(0,mother.chromo_sz-1)
		child.data[idx:] = father.data[idx:]
		return child

	# 1 crossover-point leads to 2 children
	def crossover12( self, father ):
		# single cutover
		mother = self
		child1 = deepcopy(mother)
		child1.fitness = None
		# cutover at what location?
		idx = random.randint(0,mother.chromo_sz-1)
		child1.data[idx:] = father.data[idx:]
		child2 = deepcopy(father)
		child2.fitness = None
		child2.data[:idx] = mother.data[:idx]
		return (child1,child2)

	# 2 crossover-points leads to 1 child
	def crossover21( self, father ):
		mother = self
		index1 = random.randint(0,self.chromo_sz-1)
		index2 = random.randint(0,self.chromo_sz-1)
		if( index1 > index2 ):
			index1, index2 = index2, index1
		child = deepcopy(mother)
		child.fitness = None
		child.data[index1:index2] = father.data[index1:index2]
		return child

	# 2 crossover-points leads to 2 children
	def crossover22( self, father ):
		mother = self
		index1 = random.randint(0,self.chromo_sz-1)
		index2 = random.randint(0,self.chromo_sz-1)
		if( index1 > index2 ):
			index1, index2 = index2, index1
		child1 = deepcopy(mother)
		child1.fitness = None
		child1.data[index1:index2] = father.data[index1:index2]
		child2 = deepcopy(father)
		child2.fitness = None
		child2.data[index1:index2] = mother.data[index1:index2]
		return (child1,child2)

	# you can use this break-out function, or just
	# directly call one of the crossover* funcs in your chromo
	def crossover( self, father ):
		if( self.crossover_fcn == 'crossover11' ):
			return self.crossover11( father )
		elif( self.crossover_fcn == 'crossover21' ):
			return self.crossover21( father )
		elif( self.crossover_fcn == 'crossover22' ):
			return self.crossover22( father )

	#
	# Mutate functions
	#
	def mutateAll( self ):
		child = deepcopy(self)
		child.fitness = None
		for i in range(self.chromo_sz):
			# TODO: range for variation could be a function of data-range?
			if( self.dataType[i] is float ):
				x = self.data[i] + random.uniform(-2.0,2.0)
			elif( self.dataType[i] is int ):
				x = self.data[i] + random.randint(-2,2)

			if( x < self.dataRange[i][0] ):
				child.data[i] = self.dataRange[i][0]
			elif( x > self.dataRange[i][1] ):
				child.data[i] = self.dataRange[i][1]
			else:
				child.data[i] = x
		return child

	def mutateFew( self ):
		child = deepcopy(self)
		child.fitness = None
		for i in range(self.chromo_sz):
			child.data[i] = self.data[i]
		for k in range(self.mutateNum):
			idx = random.randint(0,self.chromo_sz-1)
			# TODO: range for variation could be a function of data-range?
			if( self.dataType[i] is float ):
				x = self.data[i] + random.uniform(-2.0,2.0)
			elif( self.dataType[i] is int ):
				x = self.data[i] + random.randint(-2,2)

			if( x < self.dataRange[i][0] ):
				child.data[i] = self.dataRange[i][0]
			elif( x > self.dataRange[i][1] ):
				child.data[i] = self.dataRange[i][1]
			else:
				child.data[i] = x
		return child

	def mutateRandom( self ):
		pct = self.mutatePct
		child = deepcopy(self)
		child.fitness = None
		for i in range(self.chromo_sz):
			child.data[i] = self.data[i]
			if( random.uniform(0.0,1.0) <= pct ):
				# TODO: range for variation could be a function of data-range?
				if( self.dataType[i] is float ):
					x = self.data[i] + random.uniform(-2.0,2.0)
				elif( self.dataType[i] is int ):
					x = self.data[i] + random.randint(-2,2)

				if( x < self.dataRange[i][0] ):
					child.data[i] = self.dataRange[i][0]
				elif( x > self.dataRange[i][1] ):
					child.data[i] = self.dataRange[i][1]
				else:
					child.data[i] = x
		return child

	# you can use this break-out function, or just
	# directly call one of the mutate* funcs in your chromo
	def mutate( self ):
		if( self.mutate_fcn == 'mutateAll' ):
			return self.mutateAll()
		elif( self.mutate_fcn == 'mutateFew' ):
			return self.mutateFew()
		elif( self.mutate_fcn == 'mutateRandom' ):
			return self.mutateRandom()

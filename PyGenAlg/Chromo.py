#
# a "basic" chromosome class that you should subclass;
#
# you MUST override calcFitness()
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
#

# TODO: add support for double, unsigned-int, byte, etc.

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
		self.mutateNum     = kwargs.get( 'mutateNum', 1 )
		self.mutateNumPct  = kwargs.get( 'mutateNumPct', None )
		self.mutatePct     = kwargs.get( 'mutatePct', 0.25 )
		self.crossoverFcn  = kwargs.get( 'crossover', 'crossover11' )
		self.mutateFcn     = kwargs.get( 'mutate', 'mutateFew' )

		in_data = kwargs.get( 'data', None )

		# calculated values
		self.fitness = None

		if( self.chromo_sz <= 0 ):
			raise ValueError('Chromosome size must be >= 1')

		if( self.dataType == None ):
			self.dataType = float
		
		if( self.dataType is float ):
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

		# calculate the struct formatting string
		fmt = '<'
		for tp in self.dataType:
			if( tp is int ):
				fmt = fmt + 'i'
			elif( tp is float ):
				fmt = fmt + 'f'
		self.struct_fmt = fmt

		if( in_data != None ):
			self.data = in_data
		else:
			# start with all chromos==min-value
			# self.data = [ self.dataRange[i][0] for i in range(self.chromo_sz) ]

			# start with random data
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
	def toBytes(self):
		# TODO: could try to memoize this, but since we use deepcopy elsewhere
		#       we'd need to manually zero it out for every child-obj created
		return struct.pack( self.struct_fmt, *self.data )
	def packData(self):
		# base64encode returns bytes .. convert to str and trim off the b' prefix
		b64 = str(base64.b64encode( self.toBytes() ))
		return self.struct_fmt + '==' + b64[1:]

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
		if( in_fmt != self.struct_fmt ):
			# this fmt string does not match this Chromo;
			# maybe you grabbed the wrong file for this GA?
			# TODO: throw error?
			print( "** ERROR: file-format does not match (%s vs %s)"%(in_fmt,self.struct_fmt) )
			return
		# store into the 'data' array
		vals = struct.unpack( self.struct_fmt, udata )
		for i in range(self.chromo_sz):
			self.data[i] = vals[i]


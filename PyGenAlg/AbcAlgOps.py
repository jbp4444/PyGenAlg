#
# a "basic" bee class (mostly used internally)
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
#

import random
from copy import deepcopy

# TODO: create a class method .dcopy() to do a deepcopy of data
#       but leave pointers to other arrays, etc

class FoodSource(object):
	def __init__( self, **kwargs ):

		# parameters for this food-source (what is the underlying chromosome)
		self.chromoClass = kwargs.get( 'chromoClass', None )
		self.minOrMax    = kwargs.get( 'minOrMax', 'max' )
		self.chromo      = kwargs.get( 'chromoData', None )

		# calculated values
		self.fitness = None
		self.counter = 0

		# start with random data
		# : TODO: should be Gaussian distrib, not uniform
		if( self.chromo is None ):
			self.chromo = self.chromoClass()
		# else:
		#	TODO: check that user-provided chromo matches others?

		self.chromo_sz = self.chromo.chromo_sz

	def setInitData( self, data ):
		self.chromo.setInitData( data )
		# TODO: check for errors/exceptions

	def zeroOut( self ):
		self.chromo.zeroOut()

	def getFitness( self ):
		return self.fitness

	def calcFitness( self ):
		self.fitness = self.chromo.calcFitness()
		return self.fitness

	def testSolution( self, temp ):
		# TODO: check if min or max
		if( self.minOrMax == 'max' ):
			# trying to maximize fitness:
			if( temp.fitness > self.fitness ):
				# this is the new best-solution
				for i in range(self.chromo_sz):
					self.chromo.data[i] = temp.data[i]
				self.chromo.fitness = temp.fitness
				self.counter = 0
			else:
				self.counter = self.counter + 1
		else:
			# trying to minimize fitness:
			if( temp.fitness < self.fitness ):
				# this is the new best-solution
				for i in range(self.chromo_sz):
					self.chromo.data[i] = temp.data[i]
				self.chromo.fitness = temp.fitness
				self.counter = 0
			else:
				self.counter = self.counter + 1

	def __str__( self ):
		txt = 'data=' + ','.join( str(i) for i in self.chromo.data ) \
				+ ' .. fit=' + str(self.fitness) + ' .. ctr='+str(self.counter)
		return txt

	# NOTE: this does not store ALL AbcAlg data, just the chromo data
	def packData( self ):
		return self.chromo.packData()
	def unpackData( self, data ):
		return self.chromo.unpackData(data)

def generateTestSolution( first, second ):
	temp = deepcopy( first.chromo )
	# which index to modify?
	dim = random.randint(0,first.chromo_sz-1)

	dtype = temp.dataType[dim]
	rng   = temp.dataRange[dim]

	# TODO: how to handle integers?
	temp.data[dim] = temp.data[dim] + random.uniform(-1,1)*( temp.data[dim] - second.chromo.data[dim] )
	if( dtype is int ):
		temp.data[dim] = int( temp.data[dim] + 0.50 )

	# clamp the value to chromo's bounds
	if( temp.data[dim] > rng[1] ):
		temp.data[dim] = rng[1]
	elif( temp.data[dim] < rng[0] ):
		temp.data[dim] = rng[0]

	temp.fitness = temp.calcFitness()

	return temp

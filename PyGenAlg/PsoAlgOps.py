#
# a "basic" particle class (mostly used internally)
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
#

import sys
import random
from copy import deepcopy

# TODO: create a class method .dcopy() to do a deepcopy of data
#       but leave pointers to other arrays, etc

class BaseParticle(object):
	def __init__( self, **kwargs ):

		# parameters for this particle (what is the underlying chromosome)
		self.chromoClass   = kwargs.get( 'chromoClass', None )
		self.minOrMax      = kwargs.get( 'minOrMax', 'max' )
		self.chromo        = kwargs.get( 'chromoData', None )
		# self.velocity      = kwargs.get( 'velocity', None )
		# self.best_chromo   = kwargs.get( 'bestChromo', None )

		# calculated values
		self.fitness  = None
		self.velocity = []
		self.best_pos = []
		if( self.minOrMax == 'max' ):
			self.best_fit = -sys.float_info.max
		else:
			self.best_fit = sys.float_info.max

		# start with random data
		if( self.chromo is None ):
			self.chromo = self.chromoClass()
		# else:
		#	TODO: check that user-provided chromo matches others?

		self.chromo_sz = self.chromo.chromo_sz

		# TODO: how to handle integer data??

		for i in range(self.chromo_sz):
			diff = self.chromo.dataRange[i][1] - self.chromo.dataRange[i][0]
			self.velocity.append( random.uniform( -diff,diff ) )
			# self.velocity.append( random.uniform(-1,1) )

	def setInitData( self, data ):
		self.chromo.setInitData( data )
		# TODO: check for errors/exceptions

	def zeroOut( self ):
		self.chromo.zeroOut()

	def getFitness( self ):
		return self.fitness

	def calcFitness( self ):
		self.fitness = self.chromo.calcFitness()
		if( self.minOrMax == 'max' ):
			# trying to maximize fitness:
			if( self.fitness > self.best_fit ):
				# new best .. copy data over to best_pos
				self.best_fit = self.fitness
				self.best_pos = deepcopy( self.chromo.data )
		else:
			# trying to minimize fitness:
			if( self.fitness < self.best_fit ):
				# new best .. copy data over to best_pos
				self.best_fit = self.fitness
				self.best_pos = deepcopy( self.chromo.data )
		return self.fitness

	def update_vel( self, omega, phi_p, phi_g, swarm_best ):
		# TODO: how to vectorize this calc for better speed
		vel = self.velocity
		best_pos = self.best_pos
		data = self.chromo.data
		for i in range(self.chromo_sz):
			rn1 = random.random()
			rn2 = random.random()
			vel_p = best_pos[i] - data[i]
			vel_g = swarm_best[i] - data[i]

			vel[i] = omega*vel[i] + rn1*phi_p*vel_p + rn2*phi_g*vel_g

			# TODO: velocity clamping?

	def update_pos( self, learning_rate ):
		vel = self.velocity
		pos = self.chromo.data
		for i in range(self.chromo_sz):
			pos[i] = pos[i] + learning_rate*vel[i]

			dtype = self.chromo.dataType[i]
			if( dtype is int ):
				pos[i] = int( pos[i] + 0.50 )

			if( pos[i] < self.chromo.dataRange[i][0] ):
				pos[i] = self.chromo.dataRange[i][0]
			elif( pos[i] > self.chromo.dataRange[i][1] ):
				pos[i] = self.chromo.dataRange[i][1]

	def __str__( self ):
		# txt = 'data=' + ','.join( str(i) for i in self.chromo.data ) \
		# 		+ ' .. vel=' + ','.join( str(i) for i in self.velocity ) \
		# 		+ ' .. fit=' + str(self.fitness)
		txt = 'data=' + ','.join( str(i) for i in self.chromo.data[0:2] ) \
				+ ' .. vel=' + ','.join( str(i) for i in self.velocity[0:2] ) \
				+ ' .. fit=' + str(self.fitness)
		# txt = 'data=' + ','.join( str(i) for i in self.chromo.data ) \
		# 		+ ' .. fit=' + str(self.fitness)
		return txt

	# NOTE: this does not store ALL Particle data, just the chromo data
	def packData( self ):
		return self.chromo.packData()
	def unpackData( self, data ):
		return self.chromo.unpackData(data)

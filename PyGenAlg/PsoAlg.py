#
# a "basic" particle swarm optimization algorithm class
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
#

import sys
import math
import random
from copy import deepcopy

from PsoAlgOps import BaseParticle

# TODO: import default crossover and mutation funcs from GenOps

# TODO: is there a better way to insert data into an already sorted list?
# : right now, we calc "all" data (skipping vals we previously calc'd),
#   then we sort the whole list ... then repeat

# based on pseudo-code in https://en.wikipedia.org/wiki/Particle_swarm_optimization
# : https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/

class PsoAlg:
	def __init__( self, **kwargs ):

		# specify kwargs by float (or pct?) ...
		self.omega  = kwargs.get( 'omega', 1.00 )
		self.phi_p  = kwargs.get( 'phi_p', 0.50 )
		self.phi_g  = kwargs.get( 'phi_g', 0.50 )
		self.learning_rate = kwargs.get( 'learning_rate', 1.0 )
		# other kwargs ...
		self.chromoClass   = kwargs.get( 'chromoClass', None )
		self.population_sz = kwargs.get( 'size', 10 )
		self.minOrMax      = kwargs.get( 'minOrMax', 'max' )
		self.showBest      = kwargs.get( 'showBest', 0 )
		self.removeDupes   = kwargs.get( 'removeDupes', False )
		# optional params to be passed to functions
		self.params           = kwargs.get( 'params', {} )

		# calculated/to-be-calculated values
		self.population = []
		if( self.minOrMax == 'max' ):
			self.swarm_best_fit = -sys.float_info.max
		else:
			self.swarm_best_fit = sys.float_info.max
		self.swarm_best_pos = None

		if( (self.minOrMax!='min') and (self.minOrMax!='max') ):
			raise ValueError('minOrMax must be min or max')

		# TODO: check that chromoClass is a suitable class
		a = self.chromoClass()
		if( not callable(a.calcFitness) ):
			raise ValueError('chromoClass does not have calcFitness')
		# TODO: check that chromoClass has packData/unpackData/etc.

		#print( 'genalg:', self.population_sz,'=',self.elitism,self.crossover,self.mutation )
		self.is_sorted = False

	# maybe this should be __repr__ or __str__?
	def describe(self):
		print( 'Particle Swarm Optimization Algorithm object:' )
		print( '  pop size: '+str(self.population_sz) )
		print( '  omega: %0.3f%%' % (self.omega) )
		print( '  phi_p: %0.3f%%' % (self.phi_p) )
		print( '  phi_g: %0.3f%%' % (self.phi_g) )
		print( '  l_rate: %0.3f%%' % (self.learning_rate) )
		print( '  min_or_max: '+self.minOrMax )
		print( '  remove_dupes: '+str(self.removeDupes) )

	def initPopulation(self):
		pop = []
		for i in range(self.population_sz):
			pop.append( BaseParticle( chromoClass=self.chromoClass, minOrMax=self.minOrMax ) )
		self.population = pop
		self.is_sorted = False

	# NOTE: items is a list of chromo's (not food-src's)
	def appendToPopulation( self, items ):
		actual_sz = len(self.population)
		item_sz   = len(items)
		if( (actual_sz+item_sz) <= self.population_sz ):
			# create a food-src obj to hold data
			for x in items:
				part = BaseParticle( chromoClass=self.chromoClass, minOrMax=self.minOrMax, chromoData=x )
				# now append to pop
				self.population.append( part )
			self.is_sorted = False
			return 0
		#else:
		#	TOO MANY ITEMS
		return -1

	def bestChromo(self):
		# assume idx=1 is the min/max
		pop = self.population
		if( self.minOrMax == 'max' ):
			compare = lambda a,b: a>b
		else:
			compare = lambda a,b: a<b
		idx = 0
		#mm = pop[idx].getFitness()
		mm = pop[idx].fitness
		for i in range(1,self.population_sz):
			#fit = pop[i].getFitness()
			fit = pop[i].fitness
			if( compare(fit,mm) ):
				mm = fit
				idx = i
		return pop[idx]

	def sortPopulation(self):
		psz = self.population_sz
		pop = self.population
		if( self.minOrMax == 'max' ):
			compare = lambda a,b: a>b
		else:
			compare = lambda a,b: a<b
		for i in range(psz):
			fit1 = pop[i].getFitness()
			idx = i
			for j in range(i+1,psz):
				fit2 = pop[j].getFitness()
				if( compare(fit2,fit1) ):
					fit1 = fit2
					idx = j
			if( idx != i ):
				t = pop[i]
				pop[i] = pop[idx]
				pop[idx] = t
		self.is_sorted = True

	def evolve( self, iters ):

		# initial calc of fitness
		for i in range(self.population_sz):
			part = self.population[i]

			# calc fitness and update individual-best
			fit = part.calcFitness()

			if( self.minOrMax == 'max' ):
				# trying to maximize fitness:
				if( fit > self.swarm_best_fit ):
					self.swarm_best_fit = fit
					self.swarm_best_pos = deepcopy( part.chromo.data )
			else:
				# trying to minimize fitness:
				if( fit < self.swarm_best_fit ):
					self.swarm_best_fit = fit
					self.swarm_best_pos = deepcopy( part.chromo.data )

		# now the main loop
		for iter in range(iters):
			pop = self.population

			# for each particle, update the velocity and position
			for i in range(self.population_sz):
				pop[i].update_vel( self.omega,self.phi_p,self.phi_g, self.swarm_best_pos )
				pop[i].update_pos( self.learning_rate )

			# for each particle, take one step w/ current velocity
			for i in range(self.population_sz):
				# calc fitness and update individual-best
				fit = pop[i].calcFitness()

				if( self.minOrMax == 'max' ):
					# trying to maximize fitness:
					if( fit > self.swarm_best_fit ):
						self.swarm_best_fit = fit
						self.swarm_best_pos = deepcopy( pop[i].chromo.data )
				else:
					# trying to minimize fitness:
					if( fit < self.swarm_best_fit ):
						self.swarm_best_fit = fit
						self.swarm_best_pos = deepcopy( pop[i].chromo.data )

			# show a progress report?
			if( self.showBest > 0 ):
				print( "best chromo:" )
				self.sortPopulation()
				for i in range(self.showBest):
					print( self.population[i] )

		# sort the best soln's to the top
		self.sortPopulation()


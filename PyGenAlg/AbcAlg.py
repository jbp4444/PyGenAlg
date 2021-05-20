#
# a "basic" artificial bee colony optimization algorithm class
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
#

import sys
import math
import random
from copy import deepcopy

from GenAlgOps import int_rouletteWheelSelection
from AbcAlgOps import FoodSource, generateTestSolution

# https://en.wikipedia.org/wiki/Artificial_bee_colony_algorithm
# https://github.com/rwuilbercq/Hive/blob/master/Hive/Hive.py
# https://github.com/andaviaco/abc/blob/master/src/swarm.py

class AbcAlg:
	def __init__( self, **kwargs ):

		# specify kwargs by float (or pct?) ...
		self.chromoClass   = kwargs.get( 'chromoClass', None )
		self.population_sz = kwargs.get( 'size', 10 )
		self.onlooker_sz   = kwargs.get( 'onlookerSize', 10 )
		self.trial_limit   = kwargs.get( 'trialLimit', 10 )
		self.minOrMax      = kwargs.get( 'minOrMax', 'max' )
		self.showBest      = kwargs.get( 'showBest', 0 )
		self.removeDupes   = kwargs.get( 'removeDupes', False )
		# optional params to be passed to functions
		self.params           = kwargs.get( 'params', {} )

		if( (self.minOrMax!='min') and (self.minOrMax!='max') ):
			raise ValueError('minOrMax must be min or max')

		# calculated/to-be-calculated values
		self.population = []

		# TODO: check that chromoClass is a suitable class
		a = self.chromoClass()
		if( not callable(a.calcFitness) ):
			raise ValueError('chromoClass does not have calcFitness')
		# TODO: check that chromoClass has packData/unpackData/etc.
		self.chromo_sz = len(a.data)

		# for roulette wheel selection
		self.min_fitness = sys.float_info.max
		self.max_fitness = -sys.float_info.max
		self.is_sorted = False

	# maybe this should be __repr__ or __str__?
	def describe(self):
		print( 'Artificial Bee Colony Algorithm object:' )
		print( '  pop size: '+str(self.population_sz) )
		print( '  onlookers: '+str(self.onlooker_sz) )
		print( '  trial limit: '+str(self.trial_limit) )
		print( '  min_or_max: '+self.minOrMax )
		print( '  remove_dupes: '+str(self.removeDupes) )

	def initPopulation(self):
		pop = []
		for i in range(self.population_sz):
			pop.append( FoodSource( chromoClass=self.chromoClass, minOrMax=self.minOrMax ) )
		self.population = pop
		self.is_sorted = False

	# NOTE: items is a list of chromo's (not food-src's)
	def appendToPopulation( self, items ):
		actual_sz = len(self.population)
		item_sz   = len(items)
		if( (actual_sz+item_sz) <= self.population_sz ):
			# create a food-src obj to hold data
			for x in items:
				fsrc = FoodSource( chromoClass=self.chromoClass, minOrMax=self.minOrMax, chromoData=x )
				# now append to pop
				self.population.append( fsrc )
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
		pop = self.population

		# initial calc of fitness for food sources
		for i in range(self.population_sz):
			pop[i].calcFitness()

		# now the main loop
		for iter in range(iters):
			# reset stats for roulette wheel search
			self.sum_fitness = 0
			self.min_fitness = sys.float_info.max
			self.max_fitness = -sys.float_info.max

			# for each employee-bee, do the local-search update process
			for i in range(self.population_sz):
				other = random.randrange(0,self.population_sz-1)
				# TODO: check that other!=i (not same bee selected)

				# create new chromo from current and other food-sources
				temp = generateTestSolution( pop[i], pop[other] )
				pop[i].testSolution( temp )
				if( temp.fitness > pop[i].fitness ):
					self.sum_fitness = self.sum_fitness + temp.fitness
				else:
					self.sum_fitness = self.sum_fitness + pop[i].fitness

				# accum. total fitness info for roulette wheel selection
				if( temp.fitness > self.max_fitness ):
					self.max_fitness = temp.fitness
				if( temp.fitness < self.min_fitness ):
					self.min_fitness = temp.fitness

			# for each onlooker-bee
			for i in range(self.onlooker_sz):
				# choose initial food-source
				i = int_rouletteWheelSelection( self )
				# choose 2nd food-source
				other = random.randrange(0,self.population_sz-1)

				# create new chromo from current and other food-sources
				temp = generateTestSolution( pop[i], pop[other] )
				pop[i].testSolution( temp )

			# check for 'bad' food sources and send out scouts
			for i in range(self.population_sz):
				if( pop[i].counter > self.trial_limit ):
					temp = FoodSource( chromoClass=self.chromoClass, minOrMax=self.minOrMax )
					pop[i] = temp
					pop[i].calcFitness()

			# show a progress report?
			if( self.showBest > 0 ):
				print( "best chromo:" )
				self.sortPopulation()
				for i in range(self.showBest):
					print( self.population[i] )

		# sort the best soln's to the top
		self.sortPopulation()


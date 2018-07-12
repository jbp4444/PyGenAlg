#
# a "basic" genetic algorithm class
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import math
import random

import GenAlgOps

# TODO: import default crossover and mutation funcs from GenOps

# TODO: is there a better way to insert data into an already sorted list?
# : right now, we calc "all" data (skipping vals we previously calc'd),
#   then we sort the whole list ... then repeat

class GenAlg:
	def __init__( self, **kwargs ):

		# specify kwargs by integer counts or pct ...
		self.elitism       = kwargs.get( 'elitism', 0.10 )
		self.crossover     = kwargs.get( 'crossover', 0.80)
		self.pureMutation  = kwargs.get( 'pureMutation', 0.10 )
		self.migration     = kwargs.get( 'migration', 0.0 )
		self.parents       = kwargs.get( 'parents', 0.50 )
		# other kwargs ...
		self.chromoClass   = kwargs.get( 'chromoClass', None )
		self.population_sz = kwargs.get( 'size', 10 )
		self.minOrMax      = kwargs.get( 'minOrMax', 'max' )
		self.showBest      = kwargs.get( 'showBest', 0 )
		self.removeDupes   = kwargs.get( 'removeDupes', False )
		# selection, crossover, and mutation functions
		self.selectionFcn  = kwargs.get( 'selectionFcn', GenAlgOps.tournamentSelection )
		self.crossoverFcn  = kwargs.get( 'crossoverFcn', GenAlgOps.crossover12 )
		self.mutationFcn   = kwargs.get( 'mutationFcn', GenAlgOps.mutateFew )
		self.pureMutationSelectionFcn = kwargs.get( 'pureMutationSelectionFcn', GenAlgOps.simpleSelection )
		self.pureMutationFcn = kwargs.get( 'pureMutationFcn', GenAlgOps.mutateFew )
		self.feasibleSolnFcn = kwargs.get( 'feasibleSolnFcn', GenAlgOps.alwaysTrue )
		self.selectionParams = kwargs.get( 'selectionParams', {} )
		# hooks for migration
		self.migrationSendFcn = kwargs.get( 'migrationSendFcn', None )
		self.migrationRecvFcn = kwargs.get( 'migrationRecvFcn', None )

		# calculated/to-be-calculated values
		self.population = []

		if( self.elitism < 1 ):
			self.elitism = int( self.population_sz * self.elitism + 0.5 )
		if( self.crossover < 1 ):
			self.crossover = int( self.population_sz * self.crossover + 0.5 )
		if( self.migration < 1 ):
			self.migration = int( self.population_sz * self.migration + 0.5 )
		if( self.pureMutation < 1 ):
			self.pureMutation = int( self.population_sz * self.pureMutation + 0.5 )

		# TODO: check that sum(elitism+crossover+migration+pureMutation) == pop_sz
		#       and adjust if needed

		if( self.parents < 1 ):
			self.parents = int( self.population_sz * self.parents + 0.5 )

		if( (self.minOrMax!='min') and (self.minOrMax!='max') ):
			raise ValueError('minOrMax must be min or max')

		if( not callable(self.selectionFcn) ):
			raise ValueError('selectionFcn is not callable')
		if( not callable(self.crossoverFcn) ):
			raise ValueError('crossoverFcn is not callable')
		if( not callable(self.mutationFcn) ):
			raise ValueError('mutationFcn is not callable')
		if( not callable(self.feasibleSolnFcn) ):
			raise ValueError('feasibleSolnFcn is not callable')

		if( self.migration > 0 ):
			if( not callable(self.migrationSendFcn) ):
				raise ValueError('migrationSendFcn is not callable')
			if( not callable(self.migrationRecvFcn) ):
				raise ValueError('migrationRecvFcn is not callable')

		# TODO: check that chromoClass is a suitable class
		a = self.chromoClass()
		if( not callable(a.calcFitness) ):
			raise ValueError('chromoClass does not have calcFitness')
		# TODO: check that chromoClass has packData/unpackData/etc.

		# if chromo-crossover returns tuples (2-vec) then we may need
		#   to adjust the value of self.crossover to be even
		#   and adjust self.mutation to match
		b = self.chromoClass()
		cross_rtn = self.crossoverFcn( a, b )
		if( type(cross_rtn) is tuple ):
			self.cross_step = 2
		else:
			self.cross_step = 1

		#print( 'genalg:', self.population_sz,'=',self.elitism,self.crossover,self.mutation )
		self.is_sorted = False

	def initPopulation(self):
		pop = []
		chrClass = self.chromoClass
		for i in range(self.population_sz):
			pop.append( chrClass() )
		self.population = pop
		self.is_sorted = False

	def appendToPopulation( self, items ):
		actual_sz = len(self.population)
		item_sz   = len(items)
		if( (actual_sz+item_sz) <= self.population_sz ):
			# just append to pop
			self.population.extend( items )
			self.is_sorted = False
			return 0
		#else:
		return -1

	def calcFitness(self):
		pop = self.population
		# sum of fitness values needed for roulette wheel selection
		sum_fitness = 0.0
		if( pop[0].fitness == None ):
				pop[0].fitness = pop[0].calcFitness()
		min_fitness = pop[0].fitness 
		max_fitness = pop[0].fitness
		for i in range(self.population_sz):
			if( pop[i].fitness == None ):
				pop[i].fitness = pop[i].calcFitness()

			# track some basic stats on the fitness values of the population
			sum_fitness = sum_fitness + pop[i].fitness
			if( pop[i].fitness > max_fitness ):
				max_fitness = pop[i].fitness
			if( pop[i].fitness < min_fitness ):
				min_fitness = pop[i].fitness
		self.sum_fitness = sum_fitness
		self.min_fitness = min_fitness
		self.max_fitness = max_fitness
		self.is_sorted = False

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
		# make sure the chromo's are sorted first
		self.calcFitness()
		self.sortPopulation()

		for iter in range(iters):
			pop = self.population

			# while we add elitism population "first", we can
			# send any migrants out now, to minimize any network slowness
			# NOTE: this func does not remove the migrant from the
			#       current population, it makes a copy to send to
			#       the remote population
			migrants_out = []
			if( self.migration > 0 ):
				# TODO: only do migration every N iterations
				for i in range(0,self.migration):
					# simple selection from all "parents"
					#   parents == top X% of population with the best fitness
					# TODO: migrant should be removed from population (if present)
					idx1 = random.randint(0,self.parents-1)
					migrants_out.append( pop[ idx1 ] )
				self.migrationSendFcn( migrants_out )

			# first group is best-N chromos (elitism)
			pop_e = []
			for i in range(self.elitism):
				pop_e.append( pop[i] )

			# next group are computed from crossover and mutation
			pop_c = []
			i = 0
			while( i < self.crossover ):
				idx1,idx2 = self.selectionFcn( self )
				mother = pop[idx1]
				father = pop[idx2]
				children = self.crossoverFcn( mother, father )
				for child in children:
					child = self.mutationFcn( child )
					# test if child is feasible sol'n
					if( self.feasibleSolnFcn(self,child) ):
						pop_c.append( child )
						i = i + 1

			# last group are pure-mutation
			pop_m = []
			i = 0
			while( i < self.pureMutation ):
				idx1,idx2 = self.pureMutationSelectionFcn( self )
				parent = pop[idx1]
				child = self.pureMutationFcn( parent )
				# test if child is feasible sol'n
				if( self.feasibleSolnFcn(self,child) ):
					pop_m.append( child )
					i = i + 1

			# if present, do migration (callback to user-code)
			# NOTE: this func does not remove the migrant from the
			#       current population, it makes a copy to send to
			#       the remote population
			migrants_in = []
			if( self.migration > 0 ):
				migrants_in = self.migrationRecvFcn()

			# now look at the new-population subsets and
			# assemble them into the next generation
			# TODO: should we throw away equivalent chromos
			#   in the population?  i.e. if pop[0]==pop[1],
			#   then we lose diversity in the population
			#   (since they're sorted, this shouldn't be too hard;
			#   but would be chromo-specific)
			self.population = pop_e
			self.population.extend( pop_c )
			self.population.extend( pop_m )
			# TODO: add in migrants

			self.calcFitness()
			self.sortPopulation()

			# show a progress report?
			if( self.showBest > 0 ):
				print( "best chromo:" )
				for i in range(self.showBest):
					print( self.population[i] )

#
# a "basic" genetic algorithm class
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
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
		# other kwargs ...
		self.chromoClass   = kwargs.get( 'chromoClass', None )
		self.population_sz = kwargs.get( 'size', 10 )
		self.minOrMax      = kwargs.get( 'minOrMax', 'max' )
		self.showBest      = kwargs.get( 'showBest', 0 )
		# selection, crossover, and mutation functions
		self.selectionFcn  = kwargs.get( 'selectionFcn', GenAlgOps.tournamentSelection )
		self.crossoverFcn  = kwargs.get( 'crossoverFcn', GenAlgOps.crossover12 )
		self.mutationFcn   = kwargs.get( 'mutationFcn', GenAlgOps.mutateFew )
		self.pureMutationSelectionFcn = kwargs.get( 'pureMutationSelectionFcn', GenAlgOps.simpleSelection )
		self.pureMutationFcn = kwargs.get( 'pureMutationFcn', GenAlgOps.mutateFew )
		self.feasibleSolnFcn = kwargs.get( 'feasibleSolnFcn', GenAlgOps.allowAll )
		self.selectionParams = kwargs.get( 'selectionParams', {} )
		# hooks for migration
		self.migrationSendFcn = kwargs.get( 'migrationSendFcn', None )
		self.migrationRecvFcn = kwargs.get( 'migrationRecvFcn', None )
		self.migrationSkip    = kwargs.get( 'migrationSkip', 1 )
		# optional params to be passed to functions
		self.params           = kwargs.get( 'params', {} )

		# calculated/to-be-calculated values
		self.population = []

		# convert percentages to integer numbers of chromos
		# TODO: could/should check if float and < 1.0 then assume it's a percentage
		#       (otherwise can't test with pct=100% since that would not be < 1)
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
			self.migrationCounter = 0

		# TODO: check that chromoClass is a suitable class
		a = self.chromoClass()
		if( not callable(a.calcFitness) ):
			raise ValueError('chromoClass does not have calcFitness')
		# TODO: check that chromoClass has packData/unpackData/etc.

		self.is_sorted = False

		# just to be sure we get different random numbers
		# : user can always override with random.setState
		random.seed()

	# maybe this should be __repr__ or __str__?
	def describe(self):
		print( 'Genetic Algorithm object:' )
		print( '   pop size: '+str(self.population_sz) )
		print( '   elitism: %d :: %0.1f%%' % (self.elitism,float(100*self.elitism)/self.population_sz) )
		print( '   crossover: %d :: %0.1f%%' % (self.crossover,float(100*self.crossover)/self.population_sz) )
		print( '      selection function: %s.%s: %s'%(self.selectionFcn.__module__,self.selectionFcn.__name__,str(self.selectionFcn.__doc__)) )
		print( '      crossover function: %s.%s: %s'%(self.crossoverFcn.__module__,self.crossoverFcn.__name__,self.crossoverFcn.__doc__) )
		print( '      mutation function: %s.%s: %s'%(self.mutationFcn.__module__,self.mutationFcn.__name__,self.mutationFcn.__doc__) )
		print( '   mutation: %d :: %0.1f%%' % (self.pureMutation,float(100*self.pureMutation)/self.population_sz) )
		print( '      pure-mutation selection function: %s.%s: %s'%(self.pureMutationSelectionFcn.__module__,self.pureMutationSelectionFcn.__name__,self.pureMutationSelectionFcn.__doc__) )
		print( '      pure-mutation function: %s.%s: %s'%(self.pureMutationFcn.__module__,self.pureMutationFcn.__name__,self.pureMutationFcn.__doc__) )
		print( '   feasible-soln function: %s.%s: %s'%(self.feasibleSolnFcn.__module__,self.feasibleSolnFcn.__name__,self.feasibleSolnFcn.__doc__) )
		print( '   migration: %d :: %0.1f%%' % (self.migration,float(100*self.migration)/self.population_sz) )
		print( '   min_or_max: '+self.minOrMax )
		print( '   optional params: '+str(self.params) )

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
		#	TOO MANY ITEMS
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
			if( pop[i].fitness is None ):
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
		if( self.minOrMax == 'max' ):
			rev = True
		else:
			rev = False
		self.population.sort( key=lambda x: x.fitness, reverse=rev )
		self.is_sorted = True

	def evolve( self, iters ):
		# make sure the chromo's are sorted first
		self.calcFitness()
		self.sortPopulation()

		for iter in range(iters):
			pop = self.population

			self.feasibleSolnFcn( self, None, newgen=True )

			# while we add elitism population "first", we can
			# send any migrants out now, to minimize any network slowness
			# NOTE: this func does not remove the migrant from the
			#       current population, it makes a copy to send to
			#       the remote population
			migrants_out = []
			migrants_idx_out = []
			if( self.migration > 0 ):
				# only do migrations every N generations
				if( self.migrationCounter == 0 ):
					for i in range(0,self.migration):
						# TODO: migrant should be removed from population (if present)
						idx1 = random.randrange(self.population_sz)
						migrants_out.append( pop[ idx1 ] )
						migrants_idx_out.append( idx1 )
					self.migrationSendFcn( migrants_out )

			# first group is best-N chromos (elitism)
			# : process these with 'feasibleSolnFcn' to make sure they are checksummed/hashed/etc.
			pop_e = []
			i = 0
			while( i < self.elitism ):
				if( self.feasibleSolnFcn(self,pop[i]) ):
					pop_e.append( pop[i] )
					i = i + 1

			# next group are computed from crossover and mutation
			pop_c = []
			i = 0
			while( i < self.crossover ):
				idx1,idx2 = self.selectionFcn( self )
				mother = pop[idx1]
				father = pop[idx2]
				children = self.crossoverFcn( mother, father, self.params )
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
			migrants_in = []
			if( self.migration > 0 ):
				# only do migrations every N generations
				if( self.migrationCounter == 0 ):
					migrants_in = self.migrationRecvFcn()
				# else:
				# 	print( 'skipped migration' )
				
				# handle the update of the migration-counter
				self.migrationCounter = self.migrationCounter + 1
				if( self.migrationCounter == self.migrationSkip ):
					self.migrationCounter = 0

			# now look at the new-population subsets and
			# assemble them into the next generation
			# : due to dedup/infeasible sol'ns, this may not add up, so we have to check each time
			len_e = len(pop_e)
			len_c = len(pop_c)
			len_m = len(pop_m)
			len_mi = len(migrants_in)
			#print( 'len', len_e, len_c, len_m, len_mi )
			# : we always add the elite population in full
			self.population = pop_e
			# : and we'll always take the migrant population (or else they could be lost)
			self.population.extend( migrants_in )
			# : for crossover population, add as many as we can (until full-pop)
			if( (len_e+len_mi+len_c) < self.population_sz ):
				self.population.extend( pop_c )
			else:
				i = self.population_sz - len_e - len_mi
				self.population.extend( pop_c[:i] )
			# : mutation-only population, again, take as many as we can
			if( (len_e+len_mi+len_c+len_m) < self.population_sz ):
				self.population.extend( pop_m )
			else:
				i = self.population_sz - len_e - len_mi - len_c
				self.population.extend( pop_m[:i] )
			#print( 'pop size', self.population_sz, len(self.population), len(pop_e), len(pop_c), len(pop_m), len(migrants_in) )

			self.calcFitness()
			self.sortPopulation()

			# show a progress report?
			if( self.showBest > 0 ):
				print( "best chromo:" )
				for i in range(self.showBest):
					print( self.population[i] )

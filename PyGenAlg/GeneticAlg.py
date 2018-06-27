#
# a "basic" genetic algorithm class
#
# Copyright (C) 2018, John Pormann, Duke University Libraries
#

import math
import random

# TODO: is there a better way to insert data into an already sorted list?
# : right now, we calc "all" data (skipping vals we previously calc'd),
#   then we sort the whole list ... then repeat

class GenAlg:
	def __init__( self, **kwargs ):

		# specify kwargs by integer counts ...
		self.elitism       = kwargs.get( 'elitism', None )
		self.callback1     = kwargs.get( 'callback1', None )
		self.crossover     = kwargs.get( 'crossover', None )
		self.mutation      = kwargs.get( 'mutation', None )
		self.migration     = kwargs.get( 'migration', None )
		self.parents       = kwargs.get( 'parents', None )
		# or specify by percent of population (default)
		self.elitismPct    = kwargs.get( 'elitismPct', 0.10 )
		self.crossoverPct  = kwargs.get( 'crossoverPct', 0.30 )
		self.mutationPct   = kwargs.get( 'mutationPct', 0.60 )
		self.migrationPct  = kwargs.get( 'migrationPct', 0.0 )
		self.parentsPct    = kwargs.get( 'parentsPct', 0.50 )
		# other kwargs ...
		self.chromoClass   = kwargs.get( 'chromoClass', None )
		self.population_sz = kwargs.get( 'size', 10 )
		self.minOrMax      = kwargs.get( 'minOrMax', 'max' )
		self.showBest      = kwargs.get( 'showBest', 0 )
		self.useRoulette   = kwargs.get( 'useRoulette', False )
		self.replaceDupes  = kwargs.get( 'replaceDupes', False )
		# hooks for migration
		self.migrationFcn     = kwargs.get( 'migrationFcn', None )
		self.migrationSendFcn = kwargs.get( 'migrationSendFcn', None )
		self.migrationRecvFcn = kwargs.get( 'migrationRecvFcn', None )

		# calculated/to-be-calculated values
		self.population = []

		if( (self.elitism == None) and (self.elitismPct != None) ):
			self.elitism = int( self.population_sz * self.elitismPct + 0.5 )
		if( (self.crossover == None) and (self.crossoverPct != None) ):
			self.crossover = int( self.population_sz * self.crossoverPct + 0.5 )
		if( (self.mutation == None) and (self.mutationPct != None) ):
			self.mutation = int( self.population_sz * self.mutationPct + 0.5 )

		if( (self.migration == None) and (self.migrationPct != None) ):
			self.migration = int( self.population_sz * self.migrationPct + 0.5 )

		# TODO: make sure pop_sz = elitism+crossover+mutation+callback1/2/3

		#if( (self.elitism != None) and (self.crossover != None) ):
		#	self.mutation = self.population_sz - self.elitism - self.crossover
		#elif( (self.elitism != None) and (self.mutation != None) ):
		#	self.crossover = self.population_sz - self.elitism - self.mutation
		#else:
		#	self.elitism = self.population_sz - self.crossover - self.mutation

		if( (self.parents == None) and (self.parentsPct != None) ):
			self.parents = int( self.population_sz * self.parentsPct + 0.5 )

		if( (self.minOrMax!='min') and (self.minOrMax!='max') ):
			raise ValueError('minOrMax must be min or max')

		# TODO: check that chromoClass is a suitable class?

		# if chromo-crossover returns tuples (2-vec) then we may need
		#   to adjust the value of self.crossover to be even
		#   and adjust self.mutation to match
		a = self.chromoClass()
		b = self.chromoClass()
		cross_rtn = a.crossover( b )
		if( type(cross_rtn) is tuple ):
			self.cross_step = 2
			if( (self.crossover%2) == 1 ):
				self.crossover = self.crossover + 1
				self.mutation  = self.mutation  - 1
			print( 'adjusting crossover to match two-child return value' )
		else:
			self.cross_step = 1

		# TODO: check that callback functions are given, if needed

		#print( 'genalg:', self.population_sz,'=',self.elitism,self.crossover,self.mutation )
		self.is_sorted = False

	def initPopulation(self):
		pop = []
		chrClass = self.chromoClass
		for i in range(self.population_sz):
			pop.append( chrClass() )
		self.population = pop
		self.is_sorted = False

	# for parallel runs, use start/finish to read just the pieces of data
	# each PE needs for it's local population (file==global population)
	def loadPopulation( self, filename, start=0, finish=None ):
		# TODO: refactor to do this the right way
		pop = []
		if( finish is None ):
			finish = self.population_sz
		# how big is each chromosome?
		chrClass = self.chromoClass
		# read the file, one chromo at a time
		with open(filename,'r') as fp:
			c = 0
			# skip over un-needed values
			while( c < start ):
				data = fp.readline()
				c = c + 1
			while( c < finish ):
				line = fp.readline()
				p = chrClass()
				p.unpackData( line )
				pop.append( p )
				c = c + 1
		self.population = pop
		self.is_sorted = False    # we can't know if it's sorted

	# for parallel runs, first PE sets mode='w' to create the file
	# then other PEs set mode='a' to just append to the end of file
	# : TODO: may need to channel all I/O through one task to ensure
	#   that the file point (end-of-file) is accurate
	def savePopulation( self, filename, mode='w' ):
		with open(filename,mode) as fp:
			for p in self.population:
				txt = p.packData()
				fp.write( txt + '\n' )

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

	def checkDuplicates( self ):
		if( not self.is_sorted ):
			self.sortPopulation()
		ndup = 0
		pop = self.population
		for i in range(1,self.population_sz):
			d1 = pop[i-1].data
			d2 = pop[i].data
			values_equal = True
			for j in range(len(d1)):
				if( d1[j] != d2[j] ):
					values_equal = False
					break
			if( values_equal ):
				#print( 'dupe found: '+str(i-1)+' '+str(i) )
				ndup = ndup + 1
		#print( 'found '+str(ndup)+' dupes' )

	def replaceDuplicates( self ):
		# this modifies self.population in-place
		if( not self.is_sorted ):
			self.sortPopulation()
		ndup = 0
		pop = self.population
		chrClass = self.chromoClass
		for i in range(1,self.population_sz):
			d1 = pop[i-1].data
			d2 = pop[i].data
			values_equal = True
			for j in range(len(d1)):
				if( d1[j] != d2[j] ):
					values_equal = False
					break
			if( values_equal ):
				#print( 'dupe found: '+str(i-1)+' '+str(i) )
				ndup = ndup + 1
				# replace i-1 copy so that i+1 can match i
				# and doctor up the sum_fitness value so that
				# roulette wheel selection still works
				newguy = chrClass()
				self.sum_fitness = self.sum_fitness - pop[i-1].fitness
				pop[i-1] = newguy
				pop[i-1].fitness = pop[i-1].calcFitness()
				self.sum_fitness = self.sum_fitness + pop[i-1].fitness
		#print( 'found '+str(ndup)+' dupes' )

	def randomRouletteWheel( self ):
		rtn = -1
		if( self.minOrMax == 'max' ):
			if( self.sum_fitness > 0 ):
				# maximize fitness, and fitness values are positive
				value = random.random() * self.sum_fitness
				offset = 0
			else:
				# maximize fitness, and fitness values are negative
				value = random.random() * abs(self.sum_fitness)
				# : make all values "look" positive
				offset = self.min_fitness
			# loop over population
			for i in range(self.population_sz):
				value = value - (self.population[i].fitness + offset)
				if( value < 0 ):
					rtn = i
					break
		else:
			# TODO: these are not working yet
			if( self.sum_fitness > 0 ):
				# minimize fitness, and fitness values are positive
				value = random.random() * self.sum_fitness * (-1)
				offset = self.max_fitness
			else:
				# minimize fitness, and fitness values are negative
				value = random.random() * self.sum_fitness
				offset = 0
			for i in range(self.population_sz):
				value = value - (self.population[i].fitness + offset)
				if( value > 0 ):
					rtn = i
					break
		# locate the random value based on the weights
		if( rtn < 0 ):
			rtn = self.population_sz - 1
		return rtn

	def evolve( self, iters ):
		psz  = self.population_sz

		# make sure they're sorted first
		self.calcFitness()
		self.sortPopulation()

		for iter in range(iters):
			pop1 = self.population
			pop2 = []

			if( self.replaceDupes ):
				# this modifies population (pop1) in-place
				self.replaceDuplicates()

			# if present, do migration (callback to user-code)
			# NOTE: this func does not remove the migrant from the
			#       current population, it makes a copy to send to
			#       the remote population
			if( (self.migration > 0) and (self.migrationSendFcn!=None) ):
				# TODO: only do migration every N iterations
				migrants = []
				for i in range(0,self.migration):
					# simple selection from all "parents"
					#   parents == top X% of population with the best fitness
					# TODO: migrant should be removed from population (if present)
					idx1 = random.randint(0,self.parents-1)
					migrants.append( pop1[ idx1 ] )
				add_pop = self.migrationSendFcn( migrants )

			# first/best N chromos are kept (elitism)
			for i in range(self.elitism):
				pop2.append( pop1[i] )

			# next group are computed from crossover
			for i in range(0,self.crossover,self.cross_step):
				if( self.useRoulette ):
					# TODO: roulette wheel selection
					idx1 = self.randomRouletteWheel()
					idx2 = self.randomRouletteWheel()
				else:
					# simple selection from all "parents"
					# potential parents == top X% of population with the best fitness
					idx1 = random.randint(0,self.parents-1)
					idx2 = random.randint(0,self.parents-1)
				mother = pop1[ idx1 ]
				father = pop1[ idx2 ]
				if( self.cross_step == 2 ):
					children = mother.crossover( father )
					pop2.extend( children )
				else:
					child = mother.crossover( father )
					pop2.append( child )

			# last group are computed from mutation
			for i in range(self.mutation):
				# crossover and mutation
				idx1 = random.randint(0,self.parents-1)
				idx2 = random.randint(0,self.parents-1)
				#print( "c-idx=",idx1,idx2 )
				mother = pop1[ idx1 ]
				father = pop1[ idx2 ]
				if( self.cross_step == 2 ):
					children = mother.crossover( father )
					# TODO: can this be a for loop? (to allow for more children)
					child0 = children[0].mutate()
					child1 = children[1].mutate()
					pop2.append( child0 )
					pop2.append( child1 )
				else:
					newchr = mother.crossover( father )
					newchr = newchr.mutate()
					pop2.append( newchr )

			# if present, do migration (callback to user-code)
			# NOTE: this func does not remove the migrant from the
			#       current population, it makes a copy to send to
			#       the remote population
			if( self.migration > 0 ):
				# TODO: only do migration every N iterations
				add_pop = []
				if( self.migrationRecvFcn != None ):
					add_pop = self.migrationRecvFcn()
				else:
					migrants = []
					for i in range(0,self.migration):
						# simple selection from all "parents"
						#   parents == top X% of population with the best fitness
						# TODO: migrant should be removed from population (if present)
						idx1 = random.randint(0,self.parents-1)
						migrants.append( pop1[ idx1 ] )
					add_pop = self.migrationFcn( migrants )
				pop2.extend( add_pop )

			# TODO: we should look at pop2 fitnesses and
			#   only keep those better than in pop1
			self.population = pop2

			self.calcFitness()
			self.sortPopulation()

			# TODO: should we throw away equivalent chromos
			#   in the population?  i.e. if pop[0]==pop[1],
			#   then we lose diversity in the population
			#   (since they're sorted, this shouldn't be too hard;
			#   but would be chromo-specific)

			if( self.showBest > 0 ):
				print( "best chromo:" )
				for i in range(self.showBest):
					print( self.population[i] )

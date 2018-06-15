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
		self.crossover     = kwargs.get( 'crossover', None )
		self.mutation      = kwargs.get( 'mutation', None )
		self.parents       = kwargs.get( 'parents', None )
		# or specify by percent of population (default)
		self.elitismPct    = kwargs.get( 'elitismPct', 0.10 )
		self.crossoverPct  = kwargs.get( 'crossoverPct', 0.30 )
		self.mutationPct   = kwargs.get( 'mutationPct', 0.60 )
		self.parentsPct    = kwargs.get( 'parentsPct', 0.50 )
		# other kwargs ...
		self.chromoClass   = kwargs.get( 'chromoClass', None )
		self.population_sz = kwargs.get( 'size', 10 )
		self.minOrMax      = kwargs.get( 'minOrMax', 'max' )
		self.showBest      = kwargs.get( 'showBest', 0 )

		# calculated/to-be-calculated values
		self.population = []

		if( (self.elitism == None) and (self.elitismPct != None) ):
			self.elitism = int( self.population_sz * self.elitismPct + 0.5 )
		if( (self.crossover == None) and (self.crossoverPct != None) ):
			self.crossover = int( self.population_sz * self.crossoverPct + 0.5 )
		if( (self.mutation == None) and (self.mutationPct != None) ):
			self.mutation = int( self.population_sz * self.mutationPct + 0.5 )

		if( (self.elitism != None) and (self.crossover != None) ):
			self.mutation = self.population_sz - self.elitism - self.crossover
		elif( (self.elitism != None) and (self.mutation != None) ):
			self.crossover = self.population_sz - self.elitism - self.mutation
		else:
			self.elitism = self.population_sz - self.crossover - self.mutation

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


		# TODO: double-check that elitism+crossover+mutation == population_sz

		#print( 'genalg:', self.population_sz,'=',self.elitism,self.crossover,self.mutation )

	def initPopulation(self):
		pop = []
		chrClass = self.chromoClass
		for i in range(self.population_sz):
			pop.append( chrClass() )
		self.population = pop

	def calcFitness(self):
		pop = self.population
		for i in range(self.population_sz):
			if( pop[i].fitness == None ):
				pop[i].fitness = pop[i].calcFitness()

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

	def evolve( self, iters ):
		psz  = self.population_sz

		# make sure they're sorted first
		self.calcFitness()
		self.sortPopulation()

		for iter in range(iters):
			pop1 = self.population
			pop2 = []

			# first/best N chromos are kept (elitism)
			for i in range(self.elitism):
				pop2.append( pop1[i] )

			# next group are computed from crossover
			for i in range(0,self.crossover,self.cross_step):
				# simple selection from all "parents"
				#   parents == top X% of population with the best fitness
				idx1 = random.randint(1,self.parents-1)
				idx2 = random.randint(1,self.parents-1)
				#print( "c-idx=",idx1,idx2 )
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
				idx1 = random.randint(1,self.parents-1)
				idx2 = random.randint(1,self.parents-1)
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

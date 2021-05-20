#
# some basic gen-alg helper functions
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
#

import random
from copy import deepcopy

#
# base fcn for testing feasible solutions
# : every child is a feasible solution
def allowAll( gaMgr, child, newgen=False ):
	""" feasible-solution-function that allows all values, including duplicates """
	return True

# TODO: this dedupe func may NOT be re-entrant/parallel-aware
#       see https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
#       for a class-based approach to 'static' vars
# returns True if child is not a dupe of existing chromo
fitHistory = {}
fitDupeCount = 0
def disallowDupes( gaMgr, child, newgen=False ):
	""" feasible-solution-function that disallows duplicate values """
	global fitHistory, fitDupeCount
	rtn = True   # assume NOT a dupe
	if( newgen ):
		#print( 'fitDupeCount=',fitDupeCount )
		fitHistory = {}
		fitDupeCount = 0
	else:
		strdat = child.toBytes()
		if( strdat in fitHistory ):
			rtn = False
			fitDupeCount = fitDupeCount + 1
		else:
			fitHistory[strdat] = 1
	return rtn


#
# Crossover functions
#

# 1 crossover-point leads to 1 child
def crossover11( mother, father, params={} ):
	""" crossover-function that uses 1 crossover-point and provides 1 child """
	child = deepcopy(mother)
	child.fitness = None
	# cutover at what location?
	idx = random.randint(0,mother.chromo_sz-1)
	child.data[idx:] = father.data[idx:]
	return [child]

# 1 crossover-point leads to 2 children
def crossover12( mother, father, params={} ):
	""" crossover-function that uses 1 crossover-point and provides 2 children """
	child1 = deepcopy(mother)
	child1.fitness = None
	child2 = deepcopy(father)
	child2.fitness = None
	# cutover at what location?
	idx = random.randint(0,mother.chromo_sz-1)
	child1.data[idx:] = father.data[idx:]
	child2.data[idx:] = mother.data[idx:]
	return [child1,child2]

# 2 crossover-points leads to 1 child
def crossover21( mother, father, params={} ):
	""" crossover-function that uses 2 crossover-points and provides 1 child """
	child = deepcopy(mother)
	child.fitness = None
	# 2 cutover points
	(index1,index2) = random.sample( range(mother.chromo_sz), k=2 )
	# index1 = random.randint(0,mother.chromo_sz-1)
	# index2 = random.randint(0,mother.chromo_sz-1)
	if( index1 > index2 ):
		index1, index2 = index2, index1
	child.data[index1:index2] = father.data[index1:index2]
	return [child]

# 2 crossover-points leads to 2 children
def crossover22( mother, father, params={} ):
	""" crossover-function that uses 2 crossover-points and provides 2 children """
	child1 = deepcopy(mother)
	child1.fitness = None
	child2 = deepcopy(father)
	child2.fitness = None
	# 2 cutover points
	(index1,index2) = random.sample( range(mother.chromo_sz), k=2 )
	# index1 = random.randint(0,mother.chromo_sz-1)
	# index2 = random.randint(0,mother.chromo_sz-1)
	if( index1 > index2 ):
		index1, index2 = index2, index1
	child1.data[index1:index2] = father.data[index1:index2]
	child2.data[index1:index2] = mother.data[index1:index2]
	return [child1,child2]

# N crossover-points leads to 1 child
def crossoverN1( mother, father, params={} ):
	""" crossover-function that uses N crossover-points and provides 1 child """
	npts   = params.get( 'crossoverNumPts', 3 )
	child = deepcopy(mother)
	child.fitness = None
	# N cutover points
	indicies = sorted( random.sample( range(mother.chromo_sz), k=npts ) )
	# continue to end of chromo
	indicies.append( mother.chromo_sz )
	lst = 0
	flipflop = True
	for i in indicies:
		if( flipflop ):
			child.data[lst:i] = mother.data[lst:i]
		else:
			child.data[lst:i] = father.data[lst:i]
		lst = i
		flipflop = not flipflop
	return [child]

# N crossover-points leads to 2 children
def crossoverN2( mother, father, params={} ):
	""" crossover-function that uses N crossover-points and provides 2 children """
	npts   = params.get( 'crossoverNumPts', 3 )
	child1 = deepcopy(mother)
	child1.fitness = None
	child2 = deepcopy(father)
	child2.fitness = None
	# N cutover points
	indicies = sorted( random.sample( range(mother.chromo_sz), k=npts ) )
	# continue to end of chromo
	indicies.append( mother.chromo_sz )
	lst = 0
	flipflop = True
	for i in indicies:
		if( flipflop ):
			child1.data[lst:i] = mother.data[lst:i]
			child2.data[lst:i] = father.data[lst:i]
		else:
			child1.data[lst:i] = father.data[lst:i]
			child2.data[lst:i] = mother.data[lst:i]
		lst = i
		flipflop = not flipflop
	return [child1,child2]


#
# Mutate functions
#
# mutateNone = useful for crossover-only children
def mutateNone( mother, params={} ):
	""" mutation-function that does NO mutation """
	child = deepcopy(mother)
	child.fitness = None
	return child

def mutateAll( mother, params={} ):
	""" mutation-function that overwrites all chromos in a given parent """
	child = deepcopy(mother)
	child.fitness = None
	for i in range(mother.chromo_sz):
		# TODO: range for variation could be a function of data-range?
		if( mother.dataType[i] is float ):
			child.data[i] = random.uniform( mother.dataRange[i][0], mother.dataRange[i][1] )
		elif( mother.dataType[i] is int ):
			child.data[i] = random.randint( mother.dataRange[i][0], mother.dataRange[i][1] )
	return child

# simple mutation of a few items .. but it can re-randomize the same chromo
def mutateFewSimple( mother, params={} ):
	""" mutation-function that overwrites a few chromos in a given parent """
	num = params.get( 'mutateNum', 1 )
	child = deepcopy(mother)
	child.fitness = None
	for k in range(num):
		i = random.randint(0,mother.chromo_sz-1)
		# TODO: range for variation could be a function of data-range?
		if( mother.dataType[i] is float ):
			child.data[i] = random.uniform( mother.dataRange[i][0], mother.dataRange[i][1] )
		elif( mother.dataType[i] is int ):
			child.data[i] = random.randint( mother.dataRange[i][0], mother.dataRange[i][1] )
	return child

def mutateFew( mother, params={} ):
	""" mutation-function that overwrites a few chromos in a given parent """
	num = params.get( 'mutateNum', 1 )
	child = deepcopy(mother)
	child.fitness = None
	chrlist = random.sample( range(mother.chromo_sz), k=num )
	for i in chrlist:
		# TODO: range for variation could be a function of data-range?
		if( mother.dataType[i] is float ):
			child.data[i] = random.uniform( mother.dataRange[i][0], mother.dataRange[i][1] )
		elif( mother.dataType[i] is int ):
			child.data[i] = random.randint( mother.dataRange[i][0], mother.dataRange[i][1] )
	return child

def mutateRandom( mother, params={} ):
	""" mutation-function that potentially overwrites every chromo based on per-chromo pct """
	pct = params.get( 'chromoMutationPct', 0.1 )
	child = deepcopy(mother)
	child.fitness = None
	for i in range(mother.chromo_sz):
		if( random.uniform(0.0,1.0) <= pct ):
			# TODO: range for variation could be a function of data-range?
			if( mother.dataType[i] is float ):
				child.data[i] = random.uniform( mother.dataRange[i][0], mother.dataRange[i][1] )
			elif( mother.dataType[i] is int ):
				child.data[i] = random.randint( mother.dataRange[i][0], mother.dataRange[i][1] )
	return child


# # # # # # # # # # # # #
#  selection functions  #
# # # # # # # # # # # # #

def int_tournamentSelection( gaMgr ):
	k = gaMgr.params.get( 'tournamentSize', 3 )
	pop = gaMgr.population
	k_list = random.sample( range(gaMgr.population_sz), k=k )
	best_i = k_list[0]
	best_f = pop[best_i].fitness
	for i in range(1,k):
		new_i = k_list[i]
		new_f = pop[new_i].fitness
		if( new_f > best_f ):
			best_i = new_i
			best_f = new_f
	return best_i

def int_tournamentSelection0( gaMgr ):
	k = gaMgr.params.get( 'tournamentSize', 3 )
	pop = gaMgr.population
	best_i = random.randrange(gaMgr.population_sz)
	best_f = pop[best_i].fitness
	for i in range(k-1):
		new_i = random.randrange(gaMgr.population_sz)
		new_f = pop[new_i].fitness
		if( new_f > best_f ):
			best_i = new_i
			best_f = new_f
	return best_i

def tournamentSelection( gaMgr ):
	""" selection-function for tournament selection """
	idx1 = int_tournamentSelection( gaMgr )
	idx2 = int_tournamentSelection( gaMgr )
	return idx1,idx2

def int_rouletteWheelSelection0( gaMgr ):
	rtn = -1
	if( gaMgr.minOrMax == 'max' ):
		if( gaMgr.sum_fitness > 0 ):
			# maximize fitness, and fitness values are positive
			value = random.random() * gaMgr.sum_fitness
			offset = 0
		else:
			# maximize fitness, and fitness values are negative
			value = random.random() * abs(gaMgr.sum_fitness)
			# : make all values "look" positive
			offset = gaMgr.min_fitness
		# loop over population
		for i in range(gaMgr.population_sz):
			value = value - (gaMgr.population[i].fitness + offset)
			if( value < 0 ):
				rtn = i
				break

	else:
		# TODO: these are not working yet
		if( gaMgr.sum_fitness > 0 ):
			# minimize fitness, and fitness values are positive
			value = random.random() * gaMgr.sum_fitness * (-1)
			offset = gaMgr.max_fitness
			#print( 'min/positive: value=',value,' offset=',offset )
		else:
			# minimize fitness, and fitness values are negative
			value = random.random() * gaMgr.sum_fitness
			offset = 0
			#print( 'min/negative: value=',value,' offset=',offset )
		# loop over population
		for i in range(gaMgr.population_sz):
			value = value - (gaMgr.population[i].fitness + offset)
			if( value > 0 ):
				rtn = i
				break

	# locate the random value based on the weights
	if( rtn < 0 ):
		rtn = gaMgr.population_sz - 1
	return rtn

def int_rouletteWheelSelection( gaMgr ):
	rtn = -1

	# NOTE: xform function must be range-preserving
	# TODO: could we add an input-argument for xform?

	if( gaMgr.minOrMax == 'max' ):
		if( gaMgr.sum_fitness > 0 ):
			# maximize fitness, and fitness values are positive
			#print( 'max/positive' )
			xform = lambda x: x
		else:
			# maximize fitness, and fitness values are negative
			#print( 'max/negative', gaMgr.max_fitness, gaMgr.min_fitness )
			xform = lambda x: x - gaMgr.min_fitness + 1
	else:
		# TODO: these are not working yet
		if( gaMgr.sum_fitness > 0 ):
			# minimize fitness, and fitness values are positive
			#print( 'min/positive' )
			xform = lambda x: gaMgr.max_fitness - x + 1
		else:
			# minimize fitness, and fitness values are negative
			#print( 'min/negative' )
			xform = lambda x: -x

	# select random value
	value = random.random() * abs(gaMgr.sum_fitness)

	#print( '     sum(F)=',gaMgr.sum_fitness,' value=',value )

	# loop over population
	for i in range(gaMgr.population_sz):
		value = value - xform(gaMgr.population[i].fitness)
		if( value < 0 ):
			rtn = i
			break

	# locate the random value based on the weights
	if( rtn < 0 ):
		rtn = gaMgr.population_sz - 1
	return rtn

def rouletteWheelSelection( gaMgr ):
	""" selection-function for roulette-wheel selection """
	idx1 = int_rouletteWheelSelection( gaMgr )
	idx2 = int_rouletteWheelSelection( gaMgr )
	return idx1,idx2

def simpleSelectionParentPct( gaMgr ):
	""" selection-function that randomly pulls from all parents in top pct of population """
	# simple selection from all "parents"
	# potential parents == top X% of population with the best fitness
	pct = gaMgr.params.get( 'parentPct', 0.50 )
	parent_pop_sz = int( pct * len( gaMgr.population ) )
	(idx1,idx2) = random.sample( range(parent_pop_sz), k=2 )
	return idx1,idx2

def simpleSelection( gaMgr ):
	""" selection-function that randomly pulls from the whole population """
	# simple selection from whole population
	(idx1,idx2) = random.sample( range(gaMgr.population_sz), k=2 )
	return idx1,idx2

# https://www.msi.umn.edu/sites/default/files/OptimizingWithGA.pdf
def int_rankSelection( gaMgr ):
	rtn = -1

	# TODO: could we add an input-argument for xform?
	xform = lambda x: 1.0/(x+1.0)
	sum_xform = 0
	for i in range(gaMgr.population_sz):
		sum_xform = sum_xform + xform(i)

	# select random value
	value = random.random() * sum_xform

	# loop over population
	for i in range(gaMgr.population_sz):
		value = value - xform(i)
		if( value < 0 ):
			rtn = i
			break

	# locate the random value based on the weights
	if( rtn < 0 ):
		rtn = gaMgr.population_sz - 1
	return rtn

def rankSelection( gaMgr ):
	""" selection-function that uses rank-selection """
	idx1 = int_rankSelection( gaMgr )
	idx2 = int_rankSelection( gaMgr )
	return idx1,idx2

# mostly used for mutateAll function .. no need to do selection at all
def nullSelection( gaMgr ):
	""" selection-function that returns the top parent """
	return [0,0]

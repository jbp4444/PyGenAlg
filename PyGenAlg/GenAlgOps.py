
import random
from copy import deepcopy

#
# dummy fcn for testing feasible solutions
def alwaysTrue( gaMgr, child ):
	return True

# returns True if child is not a dupe of existing chromo
def disallowDupes( gaMgr, child ):
	rtn = True
	pop = gaMgr.population
	i = 0
	while( i < len(gaMgr.population) ):
		pp = pop[i]
		all_values_equal = True
		# easier/faster check .. are fitness vals the same?
		if( pp.fitness == child.fitness ):
			# now do a deeper check on data in chromos
			d1 = pp.data
			d2 = child.data
			for j in range(len(d1)):
				if( d1[j] != d2[j] ):
					all_values_equal = False
					break
		else:
			all_values_equal = False
		
		if( all_values_equal ):
			rtn = False
			break
		else:
			i = i + 1

	return rtn

#
# Crossover functions
#

# 1 crossover-point leads to 1 child
def crossover11( mother, father ):
	# single cutover
	mother = mother
	child = deepcopy(mother)
	child.fitness = None
	# cutover at what location?
	idx = random.randint(0,mother.chromo_sz-1)
	child.data[idx:] = father.data[idx:]
	return [child]

# 1 crossover-point leads to 2 children
def crossover12( mother, father ):
	# single cutover
	mother = mother
	child1 = deepcopy(mother)
	child1.fitness = None
	# cutover at what location?
	idx = random.randint(0,mother.chromo_sz-1)
	child1.data[idx:] = father.data[idx:]
	child2 = deepcopy(father)
	child2.fitness = None
	child2.data[:idx] = mother.data[:idx]
	return [child1,child2]

# 2 crossover-points leads to 1 child
def crossover21( mother, father ):
	mother = mother
	index1 = random.randint(0,mother.chromo_sz-1)
	index2 = random.randint(0,mother.chromo_sz-1)
	if( index1 > index2 ):
		index1, index2 = index2, index1
	child = deepcopy(mother)
	child.fitness = None
	child.data[index1:index2] = father.data[index1:index2]
	return [child]

# 2 crossover-points leads to 2 children
def crossover22( mother, father ):
	mother = mother
	index1 = random.randint(0,mother.chromo_sz-1)
	index2 = random.randint(0,mother.chromo_sz-1)
	if( index1 > index2 ):
		index1, index2 = index2, index1
	child1 = deepcopy(mother)
	child1.fitness = None
	child1.data[index1:index2] = father.data[index1:index2]
	child2 = deepcopy(father)
	child2.fitness = None
	child2.data[index1:index2] = mother.data[index1:index2]
	return [child1,child2]

#
# Mutate functions
#
def mutateAll( mother ):
	child = deepcopy(mother)
	child.fitness = None
	for i in range(mother.chromo_sz):
		# TODO: range for variation could be a function of data-range?
		if( mother.dataType[i] is float ):
			#x = mother.data[i] + random.uniform(-2.0,2.0)
			x = random.uniform( mother.dataRange[i][0], mother.dataRange[i][1] )
		elif( mother.dataType[i] is int ):
			#x = mother.data[i] + random.randint(-2,2)
			x = random.randint( mother.dataRange[i][0], mother.dataRange[i][1] )

		if( x < mother.dataRange[i][0] ):
			child.data[i] = mother.dataRange[i][0]
		elif( x > mother.dataRange[i][1] ):
			child.data[i] = mother.dataRange[i][1]
		else:
			child.data[i] = x
	return child

def mutateFew( mother ):
	child = deepcopy(mother)
	child.fitness = None
	for i in range(mother.chromo_sz):
		child.data[i] = mother.data[i]
	for k in range(mother.mutateNum):
		i = random.randint(0,mother.chromo_sz-1)
		# TODO: range for variation could be a function of data-range?
		if( mother.dataType[i] is float ):
			#x = mother.data[i] + random.uniform(-2.0,2.0)
			x = random.uniform( mother.dataRange[i][0], mother.dataRange[i][1] )
		elif( mother.dataType[i] is int ):
			#x = mother.data[i] + random.randint(-2,2)
			x = random.randint( mother.dataRange[i][0], mother.dataRange[i][1] )

		if( x < mother.dataRange[i][0] ):
			child.data[i] = mother.dataRange[i][0]
		elif( x > mother.dataRange[i][1] ):
			child.data[i] = mother.dataRange[i][1]
		else:
			child.data[i] = x
	return child

def mutateRandom( mother ):
	pct = mother.mutatePct
	child = deepcopy(mother)
	child.fitness = None
	for i in range(mother.chromo_sz):
		child.data[i] = mother.data[i]
		if( random.uniform(0.0,1.0) <= pct ):
			# TODO: range for variation could be a function of data-range?
			if( mother.dataType[i] is float ):
				#x = mother.data[i] + random.uniform(-2.0,2.0)
				x = random.uniform( mother.dataRange[i][0], mother.dataRange[i][1] )
			elif( mother.dataType[i] is int ):
				#x = mother.data[i] + random.randint(-2,2)
				x = random.randint( mother.dataRange[i][0], mother.dataRange[i][1] )

			if( x < mother.dataRange[i][0] ):
				child.data[i] = mother.dataRange[i][0]
			elif( x > mother.dataRange[i][1] ):
				child.data[i] = mother.dataRange[i][1]
			else:
				child.data[i] = x
	return child

def int_tournamentSelection( gaMgr ):
	k = gaMgr.selectionParams.get( 'k', 3 )
	pop = gaMgr.population
	best_i = random.randint(0,gaMgr.population_sz-1)
	best_f = pop[best_i].fitness
	for i in range(k-1):
		new_i = random.randint(0,gaMgr.population_sz-1)
		new_f = pop[new_i].fitness
		if( new_f > best_f ):
			best_i = new_i
			best_f = new_f
	return best_i

def tournamentSelection( gaMgr ):
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
	idx1 = int_rouletteWheelSelection( gaMgr )
	idx2 = int_rouletteWheelSelection( gaMgr )
	return idx1,idx2

def simpleSelection2( gaMgr ):
	# simple selection from all "parents"
	# potential parents == top X% of population with the best fitness
	parent_pop_sz = len( gaMgr.population )
	if( gaMgr.parents < parent_pop_sz ):
		parent_pop_sz = gaMgr.parents
	idx1 = random.randint(0,parent_pop_sz-1)
	idx2 = random.randint(0,parent_pop_sz-1)
	return idx1,idx2

def simpleSelection( gaMgr ):
	# simple selection from whole population
	parent_pop_sz = len( gaMgr.population )
	idx1 = random.randint(0,parent_pop_sz-1)
	idx2 = random.randint(0,parent_pop_sz-1)
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
	idx1 = int_rankSelection( gaMgr )
	idx2 = int_rankSelection( gaMgr )
	return idx1,idx2

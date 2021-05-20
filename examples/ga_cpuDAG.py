#
# genetic algorithm to find a program that adds 2 numbers
#
# we'll set the input stream to the 2 nums, expect output stream to have sum
#
# fitness func will be to run the code against 5+ inputs, #right answers = fitness val

import os
import time
import random
from copy import deepcopy

import click

from PyGenAlg import GenAlg, PsoAlg, BaseChromo, GenAlgOps, IoOps

import math
from cpuDAG import CPU

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

# target values we're aiming for:
inputs = []
for i in range(100):
	for j in range(100):
		if( (i+j) < 255 ):
			inputs.append( [i,j,1] )
data_size = len(inputs)

def my_func( xyz ):
	return 2*xyz[0] - 3*xyz[1] + 1

input_size = 3
num_levels = 3
nodes_per_level = 3

def mean(someList):
    total = 0
    for a in someList:
        total += float(a)
    mean = total/len(someList)
    return mean
def standDev(someList):
    listMean = mean(someList)
    dev = 0.0
    for i in range(len(someList)):
        dev += (someList[i]-listMean)**2
    dev = dev**(1/2.0)
    return dev
def corrCoef(someList1, someList2):
    # First establish the means and standard deviations for both lists.
    xMean = mean(someList1)
    yMean = mean(someList2)
    xStandDev = standDev(someList1)
    yStandDev = standDev(someList2)
    # r numerator
    rNum = 0.0
    for i in range(len(someList1)):
        rNum += (someList1[i]-xMean)*(someList2[i]-yMean)

    # r denominator
    rDen = xStandDev * yStandDev

    r =  rNum/rDen
    return r

class MyChromo(BaseChromo):
	def __init__( self ):
		self.cpu = CPU( num_levels=num_levels, nodes_per_level=nodes_per_level, input_size=input_size )
		self.total_instrs = len(self.cpu.OPSinstrs)

		# ranges for each chromo ..
		rlist = []
		for ll in range(num_levels):
			lvl_start = ll*nodes_per_level + input_size
			for nn in range(nodes_per_level):
				rlist.append( (0,self.total_instrs-1) )   # function-identifier
				rlist.append( (0,lvl_start-1) )           # op1=node-num less than level-start
				rlist.append( (0,lvl_start-1) )           # op2=node-num less than level-start
		# output node
		lvl_start = num_levels*nodes_per_level + input_size
		rlist.append( (0,self.total_instrs-1) )   # function-identifier
		rlist.append( (0,lvl_start-1) )           # op1=node-num less than level-start
		rlist.append( (0,lvl_start-1) )           # op2=node-num less than level-start

		BaseChromo.__init__( self, size=3*num_levels*nodes_per_level+3,
			range=rlist, dtype=int )

	# calculate the fitness function
	def calcFitness( self ):
		# load the chromos as instruction-memory
		cpu = self.cpu

		# the chromo-values ARE the program
		err = cpu.load_imem( self.data )
		if( err < 0 ):
			print( 'error in cpu.load_imem' )

		fitness = 0

		# now run the test on the input data
		calc_vals = []
		act_vals  = []
		for ij in inputs:
			cpu.reset()
			cpu.load_inputs( ij )
			cpu.run()
			
			try:
				out = cpu.nodevals[-1]
			except Exception as e:
				# probably IndexError .. no data?
				out = 0
			calc_vals.append( out )
			act_vals.append( my_func(ij) )

		# print( 'found vals', len(calc_vals), len(act_vals) )
		# print( '  ', calc_vals[0:5] )
		# print( '  ', act_vals[0:5] )

		try:
			#cor = corrCoef( calc_vals, act_vals )
			# print( 'corrCoef', cor )
			#fitness = fitness + 100*len(inputs)*abs(cor-1)

			for i in range(len(calc_vals)):
				diff = calc_vals[i] - act_vals[i]
				fitness = fitness + (diff*diff)

		except Exception as e:
			# print( 'corrCoef error: '+str(e) )
			# print( '   ',len(calc_vals),len(act_vals) )
			fitness = fitness + 1000*len(inputs)

		return fitness


# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #


@click.group()
@click.option( '-L','--numlevels','numlevels',default=1,show_default=True,help='number of levels' )
@click.option( '-N','--nodesper','nodesperlevel',default=10,show_default=True,help='nodes per level' )
@click.option( '-E','--elitism','elitism',type=float,default=0.05,show_default=True,help='elitism (pct or num)' )
@click.option( '-C','--crossover','crossover',type=float,default=0.45,show_default=True,help='crossover (pct or num)' )
@click.option( '-M','--puremutation','puremutation',type=float,default=0.50,show_default=True,help='pure-mutation (pct or num)' )
@click.option( '-s','--size','popsize',default=20,show_default=True,help='population size' )
@click.option( '-i','--inners','inner_it',default=10,show_default=True,help='inner iterations' )
@click.option( '-e','--epochs','epoch_it',default=10,show_default=True,help='epoch iterations' )
@click.option( '-m','--mutatenum','mutatenum',default=3,show_default=True,help='mutate-num' )
@click.option( '-W','save_out',is_flag=True,help='write/save output to file' )
@click.option( '-R','load_in',is_flag=True,help='read/load input from file' )
@click.option( '-v','--verbose','verbose',count=True,help='verbose level' )
@click.option( '-V','--veryverbose','veryverbose',count=True,help='verbose level +10' )
@click.pass_context
def cli( ctx, numlevels, nodesperlevel, elitism,crossover,puremutation, popsize, inner_it, epoch_it, mutatenum, save_out, load_in, verbose, veryverbose ):
	ctx.ensure_object(dict)

	global num_levels, nodes_per_level
	num_levels = numlevels
	nodes_per_level = nodesperlevel

	ctx.obj['elitism'] = elitism
	ctx.obj['crossover'] = crossover
	ctx.obj['puremutation'] = puremutation
	ctx.obj['popsize'] = popsize
	ctx.obj['inner_it'] = inner_it
	ctx.obj['epoch_it'] = epoch_it
	ctx.obj['mutatenum'] = mutatenum
	ctx.obj['load_in'] = load_in
	ctx.obj['save_out'] = save_out
	ctx.obj['verbose'] = verbose + 10*veryverbose


@cli.command()
@click.pass_context
def run( ctx ):
	ctxobj = ctx.obj
	verbose = ctxobj.get('verbose')
	epoch_it = ctxobj.get('epoch_it')
	inner_it = ctxobj.get('inner_it')

	# ga = GenAlg( size= ctxobj.get('popsize'),
	# 	elitism      = ctxobj.get('elitism'),
	# 	crossover    = ctxobj.get('crossover'),
	# 	pureMutation = ctxobj.get('puremutation'),
	# 	chromoClass  = MyChromo,
	# 	#selectionFcn = GenAlgOps.tournamentSelection,
	# 	crossoverFcn = GenAlgOps.crossover12,
	# 	mutationFcn  = GenAlgOps.mutateNone,
	# 	# for pure-mutation of all chromos .. no need to run tournament selection
	# 	#pureMutationSelectionFcn = lambda x: [0,0],
	# 	#pureMutationFcn = GenAlgOps.mutateAll,
	# 	pureMutationSelectionFcn = GenAlgOps.simpleSelection,
	# 	pureMutationFcn = GenAlgOps.mutateAll,
	# 	#feasibleSolnFcn = GenAlgOps.disallowDupes,
	# 	minOrMax     = 'min',
	# 	showBest     = 0,
	# 	# optional params ..
	# 	params = {
	# 		'mutateNum': ctxobj.get('mutatenum'),
	# 		'parentPct': 0.50,
	# 	}
	# )
	ga = PsoAlg( size= ctxobj.get('popsize'),
		omega = 0.75,
		phi_p = 1.50,
		phi_g = 1.00,
		learning_rate = 1.0,
		chromoClass  = MyChromo,
		minOrMax     = 'min',
		showBest     = 0,
		# optional params ..
		params = {
			'mutateNum': ctxobj.get('mutatenum'),
			'parentPct': 0.50,
		}
	)

	#
	# if a data-file exists, we load it
	if( ctxobj.get('load_in') ):
		pop = IoOps.loadPopulation( ga, 'ga_cpuDAG.dat' )
		ga.appendToPopulation( pop )
		print( 'Read init data from file ('+str(len(pop))+' chromos)')
	else:
		# otherwise, init the gen-alg library from scratch
		ga.initPopulation()
		print( 'Created random init data' )

	if( verbose > 0 ):
		ga.describe()
		print( 'Chromo size: %d :: %d %d'%(len(ga.population[0].chromo.data),num_levels,nodes_per_level) )
		print( 'Epoch/Inner iters:', epoch_it, inner_it )
		print( 'Instruction set:', ' '.join(ga.population[0].chromo.cpu.PARSEops.keys()) )

	#
	# Run it !!
	# : we'll just do 10 epochs of 10 steps each
	for i in range( epoch_it ):
		print( 'it=',i, time.time() )
		ga.evolve( inner_it )

		# give some running feedback on our progress
		txt = ''
		for j in range(10):
			txt = txt + ' %d'%(ga.population[j].fitness)
		print( 'iter '+str(i) + ", best fitnesses:" + txt )
		print( '    '+ga.population[0].chromo.cpu.show_prog( show_pc=False, nl='/' ) )
		print( '    '+ga.population[0].chromo.cpu.show_prog_as_func() )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	for i in range(5):
		#print( ga.population[i] )
		print( '  fit=%d'%(ga.population[i].fitness) )
		print( '    '+ga.population[i].chromo.cpu.show_prog( show_pc=False, nl='/' ) )
		print( '    '+ga.population[i].chromo.cpu.show_prog_as_func() )


	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	if( ctxobj.get('save_out') ):
		IoOps.savePopulation( ga, 'ga_cpuDAG.dat' )
		print('Final data stored to file (rm ga_cpuDAG.dat to start fresh)')

@cli.command()
@click.pass_context
def test1( ctx ):
	ctxobj = ctx.obj
	verbose = ctxobj.get('verbose')

	ga = GenAlg( size= 4,
		elitism      = ctxobj.get('elitism'),
		crossover    = ctxobj.get('crossover'),
		pureMutation = ctxobj.get('puremutation'),
		chromoClass  = MyChromo,
		#selectionFcn = GenAlgOps.tournamentSelection,
		crossoverFcn = GenAlgOps.crossover12,
		mutationFcn  = GenAlgOps.mutateNone,
		# for pure-mutation of all chromos .. no need to run tournament selection
		#pureMutationSelectionFcn = lambda x: [0,0],
		#pureMutationFcn = GenAlgOps.mutateAll,
		pureMutationSelectionFcn = GenAlgOps.simpleSelectionParentPct,
		pureMutationFcn = GenAlgOps.mutateAll,
		#feasibleSolnFcn = GenAlgOps.disallowDupes,
		minOrMax     = 'min',
		showBest     = 0,
		# optional params ..
		params = {
			'mutateNum': ctxobj.get('mutatenum'),
			'parentPct': 0.50,
		}
	)

	# otherwise, init the gen-alg library from scratch
	ga.initPopulation()
	print( 'Created random init data' )

	mother = ga.population[0]
	mother.fitness = mother.calcFitness()
	print( 'mother', str(mother) )
	#print( '   ', mother.dataRange )
	print( '   ', mother.cpu.show_prog_as_func() )

	father = ga.population[1]
	father.fitness = father.calcFitness()
	print( 'father', str(father) )
	#print( '   ', father.dataRange )
	print( '   ', father.cpu.show_prog_as_func() )

	# simulate a cross-over
	children = ga.crossoverFcn( mother, father, ga.params )
	for child in children:
		child.fitness = child.calcFitness()
		print( 'child', str(child) )


if __name__ == '__main__':
	cli( obj={} )

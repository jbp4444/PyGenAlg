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

from PyGenAlg import GenAlg, BaseChromo, GenAlgOps, IoOps

from cpuStack2 import CPU

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

# chromo size is 8 == num cpu instructions

# target values we're aiming for:
inputs = []
for i in range(100):
	for j in range(100):
		if( (i+j) < 255 ):
			inputs.append( [i,j] )
data_size = len(inputs)

def my_func( xyz ):
	return 2*xyz[0] - 3*xyz[1] + 1
# with proper consts stored in ROM ..
# ROM0, XYZ0, MPY, ROM1, XYZ1, MPY, ADD, ROM2, ADD = 9 instr

prog_size  = 16
rom_size   = 3
input_size = 2

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
		# cpuSimple:  self.cpu = CPU( data_memory_size=2, register_set_size=2, instruction_memory_size=2+2 )
		self.cpu = CPU( instruction_size=prog_size, input_size=input_size, rom_size=rom_size )
		self.total_instrs = len(self.cpu.OPSinstrs)

		# ranges for each chromo ..
		rlist = [ (0,self.total_instrs-1) for i in range(prog_size) ]
		rlist.extend( [ (-5,5) for i in range(rom_size) ] )
		# TODO: datatypes for each chromo ..

		BaseChromo.__init__( self, size=prog_size+rom_size,
			range=rlist, dtype=int )

	# calculate the fitness function
	def calcFitness( self ):
		# load the chromos as instruction-memory
		cpu = self.cpu

		# the chromo-values ARE the program
		err = cpu.load_imem( self.data[0:prog_size] )
		if( err < 0 ):
			print( 'error in cpu.load_imem' )
		#err = cpu.load_rommemory( self.data[prog_size:] )
		err = cpu.load_rommemory( [1,2,-3] )
		if( err < 0 ):
			print( 'error in cpu.load_rommemory' )

		# print( 'imem', cpu.dump_imem(nl=':') )
		# print( 'rmem', cpu.dump_rommemory() )

		fitness = 0

		# basic "program analysis"
		# : do we ref every input?
		if( False ):
			for i in range(input_size):
				code = 'XYZ%d'%(i)
				op = self.cpu.PARSEops[code]
				is_used = False
				for j in self.data[0:prog_size]:
					if( j == op ):
						is_used = True
						break
				if( is_used == False ):
					fitness = fitness + 5000
			# # : do we ref every rom-location?
			for i in range(rom_size):
				code = 'ROM%d'%(i)
				op = self.cpu.PARSEops[code]
				is_used = False
				for j in self.data[0:prog_size]:
					if( j == op ):
						is_used = True
						break
				if( is_used == False ):
					fitness = fitness + 1000

			# NOTE: we're assuming info about the cpu model
			#       : e.g. XYZ0 and up are all input/add-to-stack op-codes
			xx = self.cpu.PARSEops['XYZ0']
			stack_add = 0
			stack_pop = 0
			for j in self.data[0:prog_size]:
				op = self.data[j]
				if( op == 0 ):
					# no-op
					pass
				elif( op >= xx ):
					# this op will add to the stack
					stack_add = stack_add + 1
				else:
					# this op will pop 2 vals off the stack
					stack_pop = stack_pop + 2
			if( stack_add < stack_pop ):
				fitness = 5000*( stack_pop - stack_add )

		# now run the test on the input data
		calc_vals = []
		act_vals  = []
		for ij in inputs:
			cpu.reset()
			cpu.load_inputs( ij )
			cpu.run()
			
			try:
				out = cpu.execstack.pop()
			except Exception as e:
				# probably IndexError .. no data in exec-stack
				out = 0
			calc_vals.append( out )
			act_vals.append( my_func(ij) )

		# print( 'found vals', len(calc_vals), len(act_vals) )
		# print( '  ', calc_vals[0:5] )
		# print( '  ', act_vals[0:5] )

		try:
			cor = corrCoef( calc_vals, act_vals )
			# print( 'corrCoef', cor )
			fitness = fitness + 100*len(inputs)*abs(cor-1)

			for i in range(len(calc_vals)):
				diff = calc_vals[i] - act_vals[i]
				fitness = fitness + (diff*diff)*0.3

		except Exception as e:
			fitness = fitness + 1000*len(inputs)


		# make an adjustment for 'crufty' programs 
		# : data from last run is still in cpu.*
		# .. attempts to access non-existant data on stack?
		#fitness = fitness + 10*cpu.cruftcount
		# .. excess data on the stack?
		#fitness = fitness + 10*len(cpu.execstack)

		return fitness

# specialized crossover function
# : clip out a "program clause" (set of N instrs) and swap it with another loc
def MyCrossover1( mother, father ):
	mother = mother
	# rearrange clauses in the program part
	idx = random.randint(0,prog_size-1)
	num = random.randint(0,prog_size//2)
	ofs = random.randint(0,prog_size-num-1)
	child = deepcopy(mother)
	child.fitness = None
	for i in range(num):
		ii = (idx+i) % prog_size
		jj = (idx+num+ofs+i) % prog_size
		child.data[ii] = mother.data[jj]
		child.data[jj] = mother.data[ii]
	# then 1-pt crossover for the constants/rom-memory side
	idx = prog_size + random.randint(0,rom_size-1)
	child.data[prog_size:] = father.data[prog_size:]
	return [child]

# 1 crossover-point in program-code and 1 crossover-point in rom-memory
# leads to 1 child
def MyCrossover111( mother, father, params={} ):
	child = deepcopy(mother)
	child.fitness = None
	# program-code crossover ...
	index1 = random.randrange(prog_size)
	child.data[index1:prog_size] = father.data[index1:prog_size]
	# rom-memory crossover ...
	index2 = prog_size + random.randrange(rom_size)
	child.data[index2:] = father.data[index2:]
	return [child]
# 1 crossover-point in program-code and 1 crossover-point in rom-memory
# leads to 2 children
def MyCrossover112( mother, father, params={} ):
	child1 = deepcopy(mother)
	child1.fitness = None
	child2 = deepcopy(father)
	child2.fitness = None
	# program-code crossover ...
	index1 = random.randrange(prog_size)
	child1.data[index1:prog_size] = father.data[index1:prog_size]
	child2.data[index1:prog_size] = mother.data[index1:prog_size]
	# rom-memory crossover ...
	index2 = prog_size + random.randrange(rom_size)
	child1.data[index2:] = father.data[index2:]
	child2.data[index2:] = mother.data[index2:]
	return [child1,child2]
# 2 crossover-point in program-code and 1 crossover-point in rom-memory
# leads to 2 children
def MyCrossover212( mother, father, params={} ):
	child1 = deepcopy(mother)
	child1.fitness = None
	child2 = deepcopy(father)
	child2.fitness = None
	# program-code crossover ...
	(index1,index2) = random.sample( range(prog_size), k=2 )
	if( index1 > index2 ):
		index1, index2 = index2, index1
	child1.data[index1:index2] = father.data[index1:index2]
	child2.data[index1:index2] = mother.data[index1:index2]
	# rom-memory crossover ...
	index3 = prog_size + random.randrange(rom_size)
	child1.data[index3:] = father.data[index3:]
	child2.data[index3:] = mother.data[index3:]
	# print( 'index', index1, index2, index3 )
	return [child1,child2]

# mutate a few prog-elements and a few rom-elements
def MyMutate( mother, params={} ):
	num = params.get( 'mutateNum', 1 )
	child = deepcopy(mother)
	child.fitness = None
	# most of the randomness will be within the program
	chrlist = random.sample( range(prog_size), k=num-1 )
	# and make sure one is in the ROM-memory
	chrlist.append( prog_size+random.randrange(rom_size) )
	for i in chrlist:
		# TODO: range for variation could be a function of data-range?
		if( mother.dataType[i] is float ):
			child.data[i] = random.uniform( mother.dataRange[i][0], mother.dataRange[i][1] )
		elif( mother.dataType[i] is int ):
			child.data[i] = random.randint( mother.dataRange[i][0], mother.dataRange[i][1] )
	return child


# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #


@click.group()
@click.option( '-P','--progsize','progsize',default=16,show_default=True,help='program size' )
@click.option( '-R','--romsize','romsize',default=3,show_default=True,help='rom-memory size' )
@click.option( '-E','--elitism','elitism',type=float,default=0.05,show_default=True,help='elitism (pct or num)' )
@click.option( '-C','--crossover','crossover',type=float,default=0.70,show_default=True,help='crossover (pct or num)' )
@click.option( '-M','--puremutation','puremutation',type=float,default=0.25,show_default=True,help='pure-mutation (pct or num)' )
@click.option( '-s','--size','popsize',default=250,show_default=True,help='population size' )
@click.option( '-i','--inners','inner_it',default=10,show_default=True,help='inner iterations' )
@click.option( '-e','--epochs','epoch_it',default=10,show_default=True,help='epoch iterations' )
@click.option( '-m','--mutatenum','mutatenum',default=3,show_default=True,help='mutate-num' )
@click.option( '-S','save_out',is_flag=True,help='save output to file' )
@click.option( '-L','load_in',is_flag=True,help='load input from file' )
@click.option( '-v','--verbose','verbose',count=True,help='verbose level' )
@click.option( '-V','--veryverbose','veryverbose',count=True,help='verbose level +10' )
@click.pass_context
def cli( ctx, progsize, romsize, elitism,crossover,puremutation, popsize, inner_it, epoch_it, mutatenum, save_out, load_in, verbose, veryverbose ):
	ctx.ensure_object(dict)

	global prog_size, rom_size
	prog_size = progsize
	rom_size  = romsize

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

	ga = GenAlg( size= ctxobj.get('popsize'),
		elitism      = ctxobj.get('elitism'),
		crossover    = ctxobj.get('crossover'),
		pureMutation = ctxobj.get('puremutation'),
		chromoClass  = MyChromo,
		#selectionFcn = GenAlgOps.tournamentSelection,
		crossoverFcn = MyCrossover212,
		mutationFcn  = GenAlgOps.mutateNone,
		# for pure-mutation of all chromos .. no need to run tournament selection
		#pureMutationSelectionFcn = lambda x: [0,0],
		#pureMutationFcn = GenAlgOps.mutateAll,
		pureMutationSelectionFcn = GenAlgOps.simpleSelectionParentPct,
		pureMutationFcn = MyMutate,
		feasibleSolnFcn = GenAlgOps.disallowDupes,
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
		pop = IoOps.loadPopulation( ga, 'ga_cpu.dat' )
		ga.appendToPopulation( pop )
		print( 'Read init data from file ('+str(len(pop))+' chromos)')
	else:
		# otherwise, init the gen-alg library from scratch
		ga.initPopulation()
		print( 'Created random init data' )

		# add some 'good' candidate solns ... ROM0*XYZ0, ROM1*XYZ1, etc.
		cpu = ga.population[0].cpu
		prog = cpu.compile( [ 'ROM0', 'XYZ0', 'MPY', 'ROM1', 'XYZ1', 'MPY' ] )
		for i in range(0,prog_size-6):
			ga.population[i].data[-len(prog):] = prog

	if( verbose > 0 ):
		ga.describe()
		print( 'Chromo size: %d :: %d %d'%(len(ga.population[0].data),prog_size,rom_size) )
		print( 'Epoch/Inner iters:', epoch_it, inner_it )
		print( 'Instruction set:', ' '.join(ga.population[0].cpu.PARSEops.keys()) )

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
		print( '    '+ga.population[0].cpu.show_prog( show_pc=False, nl='/' )+' :: '+ga.population[0].cpu.dump_rommemory() )
		print( '    '+ga.population[0].cpu.show_prog_as_func() )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	for i in range(5):
		#print( ga.population[i] )
		print( '  fit=%d'%(ga.population[i].fitness) )
		print( '    '+ga.population[i].cpu.show_prog( show_pc=False, nl='/' )+' :: '+ga.population[i].cpu.dump_rommemory() )
		print( '    '+ga.population[i].cpu.show_prog_as_func() )


	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	if( ctxobj.get('save_out') ):
		IoOps.savePopulation( ga, 'ga_cpu.dat' )
		print('Final data stored to file (rm ga_cpu.dat to start fresh)')

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
		crossoverFcn = MyCrossover212,
		mutationFcn  = GenAlgOps.mutateFew,
		# for pure-mutation of all chromos .. no need to run tournament selection
		#pureMutationSelectionFcn = lambda x: [0,0],
		#pureMutationFcn = GenAlgOps.mutateAll,
		pureMutationSelectionFcn = GenAlgOps.simpleSelection2,
		pureMutationFcn = MyMutate,
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

	# simulate a cross-over
	mother = ga.population[0]
	print( 'mother', str(mother) )
	father = ga.population[1]
	print( 'father', str(father) )
	children = ga.crossoverFcn( mother, father, ga.params )
	for child in children:
		print( 'child', str(child) )

	asm_prog = [ 'ROM0', 'XYZ0', 'MPY', 'ROM1', 'XYZ1', 'MPY', 'ADD', 'ROM2', 'ADD' ]
	prog = ga.population[2].cpu.compile( asm_prog )
	print( 'prog', prog )
	prog.extend( [1,1,1] )  # add the 3 rom/coeffs
	# but make these the last N steps of prog, so the output _is_ the prog output
	ga.population[2].data[-len(prog):] = prog
	fit = ga.population[2].calcFitness()
	print( 'code', ga.population[2].cpu.show_prog(show_pc=False,nl='/') )
	print( 'code', ga.population[2].cpu.show_prog_as_func() )
	print( 'fitness', fit )

if __name__ == '__main__':
	cli( obj={} )

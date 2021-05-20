#
# genetic algorithm to find regression coeffs
#
# we'll set the input stream to the 2 nums, expect output stream to have sum
#
# fitness func will be to run the code against 5+ inputs, #right answers = fitness val

import os
import random

from PyGenAlg import GenAlg, BaseChromo, GenAlgOps, IoOps

# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

# domain-specific data (for the problem at hand)

# solution matrix:  1*a + 1*a^2 + 2*b + 2*b^2 + 3
S_Ca  = 1
S_Ca2 = 1
S_Cb  = 2
S_Cb2 = 2
S_Cc  = 3

# target values we're aiming for:
# : while we want random #s, we need them to be fixed across runs
random.seed( 1234567 )
inputs = []
for i in range(10000):
	a = random.randint(0,100)
	b = random.randint(0,100)
	inputs.append( [a,b] )
#print( len(inputs) )

class MyChromo(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=5, dtype=float )

	# calculate the fitness function
	def calcFitness( self ):
		# load the chromos as coeffs
		Ca  = self.data[0]
		Ca2 = self.data[1]
		Cb  = self.data[2]
		Cb2 = self.data[3]
		Cc  = self.data[4]  # constant

		fitness = 0
		for (a,b) in inputs:
			y_good = S_Ca*a + S_Ca2*a*a + S_Cb*b + S_Cb2*b*b + S_Cc
			y_calc =   Ca*a +   Ca2*a*a +   Cb*b +   Cb2*b*b +   Cc

			fitness = fitness + (y_good-y_calc)*(y_good-y_calc)

		return fitness


# # # # # # # # # # # # # # # # # # # #
## # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # #

def main():

	ga = GenAlg( size=500,
		elitism      = 0.05,
		crossover    = 0.55,
		pureMutation = 0.40,
		chromoClass  = MyChromo,
		#selectionFcn = GenAlgOps.tournamentSelection,
		#crossoverFcn = GenAlgOps.crossover22,
		#mutationFcn  = GenAlgOps.mutateFew,
		#pureMutationSelectionFcn = GenAlgOps.simpleSelection,
		#pureMutationFcn = GenAlgOps.mutateFew,
		#feasibleSolnFcn = GenAlgOps.disallowDupes,
		minOrMax     = 'min',
		showBest     = 0,
	)

	#
	# if a data-file exists, we load it
	if( os.path.isfile('ga_regr.dat') ):
		pop = IoOps.loadPopulation( ga, 'ga_regr.dat' )
		ga.appendToPopulation( pop )
		print( 'Read init data from file ('+str(len(pop))+' chromos)')
	else:
		# otherwise, init the gen-alg library from scratch
		ga.initPopulation()
		print( 'Created random init data' )

	#
	# Run it !!
	# : we'll just do 10 epochs of 10 steps each
	for i in range(10):
		ga.evolve( 10 )

		# give some running feedback on our progress
		txt = ''
		for j in range(10):
			txt = txt + ' %d'%(ga.population[j].fitness)
		print( 'iter '+str(i) + ", best fitnesses:" + txt )

	#
	# all done ... output final results
	print( "\nfinal best chromos:" )
	for i in range(5):
		print( ga.population[i] )


	#
	# we'll always save the pickle-file, just delete it
	# if you want to start over from scratch
	IoOps.savePopulation( ga, 'ga_regr.dat' )
	print('Final data stored to file (rm ga_regr.dat to start fresh)')

if __name__ == '__main__':
	main()

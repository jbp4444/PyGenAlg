
import os
import random
import string

from PyGenAlg import GenAlg, BaseChromo, GenAlgOps


# for roulette wheel selection:
#	max w/ positive values == default case:
#		fitness = F .. prob = F/sum(F)
#   min w/ negative values:
#		fitness = F .. prob = F/sum(F)

class MyChromoPos(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=1,
			range=(1,10), dtype=int )

	def __str__( self ):
		txt = 'data=' + str(self.data ) + ' .. fit=' + str(self.fitness)
		return txt

	def calcFitness( self ):
		rtn = self.data[0]
		return rtn

class MyChromoNeg(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=1,
			range=(-10,-1), dtype=int )

	def __str__( self ):
		txt = 'data=' + str(self.data ) + ' .. fit=' + str(self.fitness)
		return txt

	def calcFitness( self ):
		rtn = self.data[0]
		return rtn

# # # # # # # # # # # # # # # # # # # # # #

def printGaPopulation( gaMgr ):
	pop = [ gaMgr.population[i].data[0] for i in range(gaMgr.population_sz) ]
	print( 'pop='+str(pop) )

def main():

	sims = [ ('max',MyChromoPos), ('min',MyChromoPos), ('max',MyChromoNeg), ('min',MyChromoNeg) ]

	for s in sims:
		print( s )
		ga = GenAlg( size=10,
			elitism      = 0.10,
			crossover    = 0.60,
			pureMutation = 0.30,
			parentsPct   = 0.80,
			chromoClass  = s[1],
			#selectionFcn = GenAlgOps.tournamentSelection,
			#crossoverFcn = GenAlgOps.crossover22,
			#mutationFcn  = GenAlgOps.mutateFew,
			#pureMutationSelectionFcn = GenAlgOps.simpleSelection,
			#pureMutationFcn = GenAlgOps.mutateFew,
			#feasibleSolnFcn = GenAlgOps.disallowDupes,
			minOrMax     = s[0],
			showBest     = 0,
		)

		ga.initPopulation()
		for i in range(10):
			if( s[1] == MyChromoNeg ):
				ga.population[i].data[0] = -i - 1
			else:
				ga.population[i].data[0] = i + 1
		ga.calcFitness()
		ga.sortPopulation()
		printGaPopulation( ga )

		counts = [ 0 for i in range(10) ]

		for i in range(10000):
			idx1,idx2 = GenAlgOps.rouletteWheelSelection(ga)
			counts[idx1] = counts[idx1]+1
			counts[idx2] = counts[idx2]+1
		print( '     roulette counts = '+str(counts) )

		counts = [ 0 for i in range(10) ]

		for i in range(10000):
			idx1,idx2 = GenAlgOps.rankSelection(ga)
			counts[idx1] = counts[idx1]+1
			counts[idx2] = counts[idx2]+1
		print( '     rank counts = '+str(counts) )


if __name__ == '__main__':
	main()

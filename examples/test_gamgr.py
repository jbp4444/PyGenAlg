
import os
import random
import string

from PyGenAlg import GenAlg, BaseChromo, GenAlgOps


class MyChromo(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=1,
			range=(1,10), dtype=int )

	def __str__( self ):
		txt = 'data=' + str(self.data ) + ' .. fit=' + str(self.fitness)
		return txt

	def calcFitness( self ):
		rtn = self.data[0]
		return rtn


# # # # # # # # # # # # # # # # # # # # # #

def main():

    ga = GenAlg( size=10,
        elitism      = 0.10,
        crossover    = 0.60,
        pureMutation = 0.30,
        parentsPct   = 0.80,
        chromoClass  = MyChromo,
        #selectionFcn = GenAlgOps.tournamentSelection,
        #crossoverFcn = GenAlgOps.crossover22,
        #mutationFcn  = GenAlgOps.mutateFew,
        #pureMutationSelectionFcn = GenAlgOps.simpleSelection,
        #pureMutationFcn = GenAlgOps.mutateFew,
        feasibleSolnFcn = GenAlgOps.disallowDupes,
        minOrMax     = 'max',
        showBest     = 0,
    )

    ga.initPopulation()
    ga.calcFitness()

    child = MyChromo()
    child.data[0] = ga.population[4].data[0]
    child.fitness = ga.population[4].fitness
    feas = GenAlgOps.disallowDupes( ga, child )
    print( 'unsorted population; data present; feas=',feas )
    child.data[0] = ga.population[4].data[0] + 9
    child.fitness = ga.population[4].fitness + 9
    feas = GenAlgOps.disallowDupes( ga, child )
    print( 'unsorted population; data not present; feas=',feas )

    ga.sortPopulation()
    child.data[0] = ga.population[4].data[0]
    child.fitness = ga.population[4].fitness
    feas = GenAlgOps.disallowDupes( ga, child )
    print( 'sorted population; data at pos=4; feas=',feas )
    child.data[0] = ga.population[4].data[0] + 9
    child.fitness = ga.population[4].fitness + 9
    feas = GenAlgOps.disallowDupes( ga, child )
    print( 'sorted population; data not present; feas=',feas )


if __name__ == '__main__':
    main()

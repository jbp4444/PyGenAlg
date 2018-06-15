

import sys
import time
from multiprocessing import Process

from PyGenAlg import BaseChromo, GenAlg, ParallelMgr

# target value we're aiming for:
target = 0.44

class MyChromo(BaseChromo):
	def __init__( self ):
		BaseChromo.__init__( self, size=4,
			range=(0,10), dtype=int )

	# we'll use the default crossover and mutate functions
	# from BaseChromo (==crossover11 and mutateAll)

	# calculate the fitness function
	def calcFitness( self ):
		ccc = self.data
		val = 0.01*ccc[0] + 0.05*ccc[1] \
				+ 0.10*ccc[2] + 0.25*ccc[3]
		# we really want to make sure we give the correct amount
		# hence the heavier weighting; but we also want to minimize
		# the number of coins (lower weight)
		fitness = -100000.0*(val - target)*(val - target) \
				-1.0*(ccc[0]+ccc[1]+ccc[2]+ccc[3])
		return fitness


class the_code( Process ):
    def __init__( self, tid, parMgr ):
        Process.__init__(self)
        self.tid = tid
        self.parMgr = parMgr
        self.num_pes = parMgr.num_pes
        print( 'tid '+str(tid)+' init (out of '+str(self.num_pes)+')' )
        sys.stdout.flush()

    def run(self):
        tid = self.tid
        num_pes = self.num_pes
        parMgr = self.parMgr
        # TODO: maybe this is an MPI_INIT kind of functionality?
        # : e.g. store this worker-tid into parMgr; init per-worker values
        parMgr.tid = tid

        print( 'tid '+str(tid)+' started (out of '+str(num_pes)+')' )
        sys.stdout.flush()

        # TODO: calculate the splitting across the PEs

        # otherwise, init the gen-alg library from scratch
        ga = GenAlg( size=100,
            elitismPct   = 0.10,
            crossoverPct = 0.30,
            mutationPct  = 0.50,
            migrationPct = 0.10,
            migrationFcn = self.migrationFcn,
            parentsPct   = 0.50,
            chromoClass  = MyChromo,
            minOrMax     = 'max',
            showBest     = 0
        )
        ga.initPopulation()
        #ga.loadPopulation( 'ga_data.dat' )

        #
        # Run it !!
        # : we'll just do 10 epochs of 10 steps each
        for i in range(10):
            print( 'pop_sz='+str(len(ga.population)))
            ga.evolve( 10 )

            # give some running feedback on our progress
            print( 'iter '+str(i) + ", best chromo:" )
            for i in range(1):
                print( ga.population[i] )

        # at this point, each PE's population is sorted
        rtn = parMgr.collect( ga.population[:10] )
        bestVals = [ x for y in rtn for x in y ]
        bestVals.sort()

        #
        # all done ... output final results
        print( "\nfinal best chromos:" )
        for i in range(10):   
            print( bestVals[i] )

        #ga.savePopulation( 'ga_data.dat' )

    def migrationFcn( self, migrants ):
        tid = self.tid
        num_pes = self.num_pes
        parMgr = self.parMgr

        prev_pe = ( tid + num_pes - 1 ) % num_pes
        next_pe = ( tid + 1 ) % num_pes
        #print( 'tid '+str(tid)+' exchanging with tids '+str(next_pe) \
        #    + '/' + str(prev_pe) )

        parMgr.isend( next_pe, 123, migrants )
        x,y,data = parMgr.recv( prev_pe, 123 )

        return data


def main():
    parMgr = ParallelMgr( num_pes=2 )

    parMgr.runWorkers( the_code )

    parMgr.finalize()

if __name__ == '__main__':
    main()
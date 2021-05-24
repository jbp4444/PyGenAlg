# PyGenAlg
--------

A basic genetic algorithm classes to play around with ... using Numba's CUDA interface.

* GeneticAlg.py - contains a GenAlg class which runs the algorithm itself
  * You give it a chromosome size, type (all genes must be the same), and range (different genes can have different ranges)
  * InitPopulation - creates the population
  * evolve - the main method; it goes through N iterations of the algorithm (elitism, crossover, mutation)
    * can show a fixed number of best chromosomes after each iter (set showBest to 0 to quiet this)

* GenAlgSort - GPU-based code to sort the population by fitness values and calculate some basic stats (stats might be needed by roulette wheel selection, e.g.)
  * sortPopulation - checks MinOrMax problem spec and then sorts accordingly (max=descending, min=ascending)
  * There are two internal functions to do the asc/desc sorts

* GenAlgCfg - "config" values for passing data into the user-defined calcFitness and crossover functions
  * config_i is sent by the GA-Manager to the user function and includes basic parameters of this run:
    * population size, chromo size, data type, elitism count, crossover count (new children), pure-mutation count (no crossover, just random)
	* the data-range for each gene, if integer
  * config_f is much smaller:
    * parent-percentage - used to limit the parents that are looked at for selection
	* the data-range for each gene, if float
  * when the user receives the config_i array, they can find:
    * pop_size = config_i[ cfg.POPULATION_SIZE ]

* GenAlgGPU.py - contains some basic operations that can be used by user functions (below)
  * rand_uniform_int, rand_uniform_float - basic random number operations (uses CUDA's xoroshiro rng)
  * rand_sample - selects K numbers from 1..N (CUDA/xoroshiro, requires user-provided local-tmp array)
  * selectSimple - selects 2 individuals from population
  * selectTournament - for each parent: selects K individuals from population and returns top ranked candidate for each
  * selectRouletteMaxPos - for each parent, perform roulette selection based on fitness vals
    * there are several options (MaxPos, MaxNeg, MinPos, MinNeg) based on whether you're maximizing/minimizing the fitness function, and whether values are positive/negative
  * selectRank - for each parent, perform rank-based selection
  * mutateFew - integer genes - for a new child, mutate K genes (across full range of that gene)
    * mutateFewF - float genes - as above, but for floating point-based chromos
  * mutateRandom - integer genes - for a new child, scan all genes and mutate each one if rand value is <= mutation-percent (user-supplied threshold)
    * mutateRandomF - float genes - as above, but for floatin point-based chromos
  * crossover11 - for a pair of parents, do 1-point crossover to produce 1 child
  * crossover12 - for a pair of parents, do 1-point crossover to produce 2 children
  * crossover21 - for a pair of parents, do 2-point crossover to produce 1 child
  * crossover22 - for a pair of parents, do 2-point crossover to produce 2 children
  * crossoverUniform1 - for a pair of parents, do uniform crossover of each gene to produce 1 child
  * crossoverUniform2 - for a pair of parents, do uniform crossover of each gene to produce 2 children


* The programmer supplies:
  * calcFitness - @cuda.jit function - iterates through the population to calculate any unknown fitness values
    * the user can find config values in the config_i/config_f arrays
    * the user must create a local-tmp array for use with some internal functions (cuda.local.array)
  * crossover - @cuda.jit function - iterates through the population to calculate crossover and mutation
    * the usual case is that you'll call GenAlgGPU functions, but you can make your own unique crossover function too

* Examples:
  * ga_coins.py - uses a GA to calculate change for a given target value
    * genes are number of pennies, nickels, dimes, quarters
    * fitness function includes weighting factors for how close is the value and how many coins are needed
  * ga_hello.py - uses a GA to calculate "Hello World!"
  * ga_sprinkler.py - uses a GA to place 5 circular sprinklers in a square field so that the field is uniformly covered in water
    * can 'zero out' part of the square to make an odd-shaped field
    * can alter the 'scoring' for how well-covered the field is; no water=0, 1-unit of water=10, 2-units=5 (wasting water), 3-units=-1 (waterlogging the crops), etc.
    * uses PIL to draw a PNG image of the field and coverage
    * uses the click library to provide a more "full featured" command line interface
      * basic run:  python ga_sprinkler.py -v run
      * change params:  python ga_sprinkler.py -v -s POP_SZ -E NUM_ELITISM -C NUM_CROSSOVER run


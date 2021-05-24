# PyGenAlg
--------

A set of basic genetic algorithm classes to play around with.

NEW: refactored the code to better "fit" with Numba/CUDA's approach (e.g. the population is one large vector); but note that that means the CUDA (GPU) and CPU code are NOT equivalent and should not be compared to each other (for run-times)

* Chromo.py - includes a BaseChromo class that should be subclassed
  * Includes several crossover methods
    * 1 crossover point producing 1 child - child contains mother[:idx] and father[idx:]
    * 1 crossover point producing 2 children - child-1 contains mother[:idx] and father[idx:]; child-2 contains father[:idx] and mother[idx:]
    * 2 crossover point producing 1 child - child contains mother[:idx1], father[idx1:idx2], mother[idx2:]
    * 2 crossover point producing 2 children - child-1 contains mother[:idx1], father[idx1:idx2], mother[idx2:]; child-2 contains father[:idx1], mother[idx1:idx2], father[idx2:]
  * Includes several mutation methods
    * mutateAll - every chromosome (index) undergoes some degree of mutation
	* mutateFew - between 1 and mutateNum chromosomes undergo mutation
	* mutateRandom - each chromosome has a mutatePct percent chance of mutation

* GeneticAlg.py - contains a GenAlg class which runs the algorithm itself
  * You give it a chromosome class (subclass of BaseChromo) which can create new members without any arguments (__init__ needs no arguments)
  * InitPopulation - creates the population
  * calcFitness - iterates through the population to calculate any unknown fitness values
    * if chromo.fitness == None, then the chromo.calcFitness function will be called
    * otherwise, it will re-use the previously computed value
  * evolve - the main method; it goes through N iterations of the algorithm (elitism, crossover, mutation)
    * can show a fixed number of best chromosomes after each iter (set showBest to 0 to quiet this)

* Examples:
  * ga_coins.py - uses a GA to calculate change for a given target value
    * chromosomes are number of pennies, nickels, dimes, quarters
    * fitness function includes weighting factors for how close is the value and how many coins are needed
  * ga_sprinkler.py - uses a GA to place 5 circular sprinklers in a square field so that the field is uniformly covered in water
    * can 'zero out' part of the square to make an odd-shaped field
    * can alter the 'scoring' for how well-covered the field is; no water=0, 1-unit of water=10, 2-units=5 (wasting water), 3-units=-1 (waterlogging the crops), etc.
    * uses PIL to draw a PNG image of the field and coverage
  * ga_voting.py - taken from an old Dr. Dobb's magazine article, uses a GA to calculate voting districts to ensure each district is compact and has roughly the same number of citizens
    * uses US government data on population in each zipcode in NC
    * chromosomes are (lat,lon) coords for the centroid of a voting district; each zipcode is matched to the closest centroid, and that determines the shape of the district
    * fitness function is just the difference in population between the most- and least-populous voting districts
  * all examples read/write to data files to save one run's data to prime the next run; i.e. running the code several times should produce better results
    * just delete the ga_*.dat file to start over from scratch

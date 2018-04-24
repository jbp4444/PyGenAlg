PyGenAlg
--------

A pair of basic genetic algorithm classes to play around with.

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

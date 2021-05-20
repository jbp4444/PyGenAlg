#
# some basic gen-alg helper functions
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
#

# for parallel runs, use start/finish to read just the pieces of data
# each PE needs for it's local population (file==global population)
def loadPopulation( gaMgr, filename, start=0, finish=None ):
	pop = []
	
	if( finish is None ):
		finish = gaMgr.population_sz

	# how big is each chromosome?
	chrClass = gaMgr.chromoClass

	# read the file, one chromo at a time
	with open(filename,'r') as fp:
		c = 0
		# skip over un-needed values
		while( c < start ):
			data = fp.readline()
			c = c + 1
		try:
			while( c < finish ):
				line = fp.readline()
				# print( 'line ['+line+']' )
				p = chrClass()
				p.unpackData( line )
				pop.append( p )
				c = c + 1
		except Exception as e:
			print( '** ERROR: cannot convert line: '+str(e) )
	
	return pop

# in case you need to fill-out a population with random
# members (e.g. if loaded file is too small)
def randomPopulation( gaMgr, num ):
	pop = []
	chrClass = gaMgr.chromoClass
	for i in range(num):
		pop.append( chrClass() )
	return pop


# for parallel runs, first PE sets mode='w' to create the file
# then other PEs set mode='a' to just append to the end of file
# : TODO: may need to channel all I/O through one task to ensure
#   that the file point (end-of-file) is accurate
def savePopulation( gaMgr, filename, mode='w' ):
	with open(filename,mode) as fp:
		for p in gaMgr.population:
			txt = p.packData()
			fp.write( txt + '\n' )


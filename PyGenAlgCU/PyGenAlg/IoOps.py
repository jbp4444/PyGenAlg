#
# some basic gen-alg helper functions
#
# Copyright (C) 2018-2020, John Pormann, Duke University Libraries
#

import struct
import base64

import numpy as np
from numpy.testing._private.utils import nulp_diff

import GenAlgCfg as cfg

def calcFormatString( gaMgr ):
	# calculate the struct formatting string
	fmtltr = cfg.DATATYPE_TO_FMT[ gaMgr.dataType ]
	fmt = '<' + fmtltr*gaMgr.chromo_sz
	return fmt

# pack just the data into a text/base64 format
# : fmt-string is the struct.pack format string,
#   based on the Chromo's dataType entries
def packData( fmt, data ):
	chr_sz = len(fmt)
	the_bytes = struct.pack( fmt, *data )
	b64 = base64.b64encode( the_bytes )
	return b64

# unpack the data from the text/base64 format
def unpackData( fmt, data ):
	udata = base64.b64decode( data )
	vals = struct.unpack( fmt, udata )
	return vals

# for parallel runs, use start/finish to read just the pieces of data
# each PE needs for it's local population (file==global population)
def loadPopulation( gaMgr, filename, start=0, finish=None ):
	pop = []
	
	if( finish is None ):
		finish = gaMgr.population_sz

	# read the file, one chromo at a time
	with open(filename,'r') as fp:
		header1 = fp.readline()    # <== the data format string
		# check header/format info with gaMgr
		header1 = header1[:-1]
		if( header1 != calcFormatString(gaMgr) ):
			# this fmt string does not match this Chromo;
			# maybe you grabbed the wrong file for this GA?
			# TODO: throw error?
			print( "** ERROR: file-format does not match (%s vs %s)"%(header1,calcFormatString(gaMgr)) )
			return

		# TODO: read/skip lines until divider (==========)
		header2 = fp.readline()

		# read in the population data
		c = 0
		while( c < start ):
			# skip some chromos at front of file
			data = fp.readline()
			c = c + 1
		try:
			# read in lines for desired pop_sz
			while( c < finish ):
				line = fp.readline()
				# print( 'line ['+line+']' )
				p = unpackData( header1, line )
				pop.extend( p )
				c = c + 1
		except Exception as e:
			print( '** ERROR: cannot convert line: '+str(e) )
	
	return pop

# for parallel runs, first PE sets mode='w' to create the file
# then other PEs set mode='a' to just append to the end of file
# : TODO: may need to channel all I/O through one task to ensure
#   that the file point (end-of-file) is accurate
def savePopulation( gaMgr, filename ):
	nl_bytes = '\n'.encode('utf-8')
	fmt = calcFormatString(gaMgr)
	chr_sz = gaMgr.chromo_sz
	with open(filename,'wb') as fp:
		# print header line (format info)
		fp.write( fmt.encode('utf-8') )
		fp.write( '\n==========\n'.encode('utf-8') )
		for p in range(gaMgr.population_sz):
			idx1 = p * chr_sz
			idx2 = idx1 + chr_sz
			data = gaMgr.population[idx1:idx2]
			txt = packData( fmt, data )
			fp.write( txt )
			fp.write( nl_bytes )


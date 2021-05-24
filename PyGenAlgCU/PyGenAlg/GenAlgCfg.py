
import numba
import numpy as np

# cuda launch config
# : internal functions ...
INT_GRID_SIZE  = 32
INT_BLOCK_SIZE = 64
# : user functions ...
USER_GRID_SIZE = 32
USER_BLOCK_SIZE = 64

# NOTE: at blocksz=64, we can go to 64 reg/thr and 4KB/thr and still have 100% occupancy

# cuda.local.array() size
# NOTE: TOURNAMENT_SIZE must be <= LOCAL_VEC_MAX
# NOTE: MUTATION_NUM_PTS must be <= LOCAL_VEC_MAX
# NOTE: crossover (rand_sample) must be <= LOCAL_VEC_MAX
# TODO: THIS IS NOT WORKING YET .. it just "informational" (changing it doesn't do anything)
LOCAL_VEC_MAX   = 4

# integer config settings
# : indices into config_i array
POPULATION_SIZE = 0
CHROMO_SIZE     = 1
DATA_TYPE       = 2
ELITISM_COUNT   = 3
CROSSOVER_COUNT = 4
PUREMUTATION_COUNT  = 5
MIGRATION_COUNT = 6
MIN_OR_MAX      = 7
CFG_I_LEN       = 8
# start of data-range pairs...
DATA_RANGE_I    = 8

# floating-point config settings
# : indices into config_f array
PARENT_PCT      = 0
CFG_F_LEN       = 1
# start of data-range pairs...
DATA_RANGE_F    = 1

DATATYPE_LIST = [ 
	np.uint8, np.uint16, np.uint32, np.uint64,
	np.int8, np.int16, np.int32, np.int64,
	np.float32, np.float64
	# np.complex64, np.complex128
	]

DATATYPE_TO_FMT = {
	np.uint8:   'B', 
	np.uint16:  'H', 
	np.uint32:  'I',
	np.uint64:  'Q',
	np.int8:    'b',
	np.int16:   'h',
	np.int32:   'i',
	np.int64:   'q',
	np.float32: 'f',
	np.float64: 'd'
	# np.complex64: , 
	# np.complex128: 
}

FITSTATS_SUM  = 0
FITSTATS_SUM2 = 1
FITSTATS_MIN  = 2
FITSTATS_MAX  = 3
FITSTATS_LEN  = 4
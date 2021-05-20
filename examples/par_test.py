

import sys
import time
from PyGenAlg import ParallelMgr
from multiprocessing import Process

class the_code( Process ):
    def __init__( self, tid, parMgr ):
        Process.__init__(self)
        self.tid = tid
        self.parMgr = parMgr
        self.num_pes = parMgr.num_pes
        print( 'tid '+str(tid)+' init (out of '+str(self.num_pes)+')' )
        sys.stdout.flush()

    def run(self):
        print( 'tid start' )
        sys.stdout.flush()

        tid = self.tid
        num_pes = self.num_pes
        parMgr = self.parMgr
        # TODO: maybe this is an MPI_INIT kind of functionality?
        # : e.g. store this worker-tid into parMgr; init per-worker values
        parMgr.tid = tid

        print( 'tid '+str(tid)+' started (out of '+str(num_pes)+')' )
        sys.stdout.flush()

        # send/recv a bunch of msgs
        for i in range(num_pes):
            parMgr.isend( i, 44, 'hello from tid '+str(tid) )
        for i in range(num_pes):
            rem_pe,recv_tag,data = parMgr.recv( i, 44 )
            print( 'tid '+str(tid)+' recv fr '+str(rem_pe)+' data='+str(data) )
        
        print( 'tid '+str(tid)+' enter barrier' )
        sys.stdout.flush()
        parMgr.barrier()
        print( 'tid '+str(tid)+' left barrier' )
        sys.stdout.flush()

        if( tid == 0 ):
            parMgr.send( 1, 55, 'blocking send from '+str(tid) )
            print( 'tid '+str(tid)+' blocking send to 1' )
            sys.stdout.flush()
        elif( tid == 1 ):
            time.sleep( 1 )
            rem_pe,recv_tag,data = parMgr.recv( 0, 55 )
            print( 'tid '+str(tid)+' recv fr '+str(rem_pe)+' data='+str(data) )
        
        vals = parMgr.collect( 10.0-float(tid) )
        print( 'tid '+str(tid)+' collect='+str(vals) )

def main():
    parMgr = ParallelMgr( num_pes=2 )

    parMgr.runWorkers( the_code )

    parMgr.finalize()

if __name__ == '__main__':
    main()


import sys
import multiprocessing
from multiprocessing import Process, Manager

import IoOps

# basic parallel operations:
#   who-ami-i, how-big-is-the-world
#   block/non-blocking send and recv
#   broadcast, gather/scatter

# we'll use negative tags for "internal" communication
# : other than:
ANY_TAG = -1
ANY_PE  = -1

class CommMgr():
    def __init__( self, **kwargs ):
        self.tid = kwargs.get( 'tid', -1 )
        self.ns  = kwargs.get( 'namespace', None )
        self.num_pes = self.ns.num_pes

    # non-blocking send
    # : internal, allows negative tags
    def int_send( self, other_pe, tag, data ):
        comm = self.ns.msgLists[other_pe]
        comm.append( (self.tid,tag,data) )
        # TODO: check for errors
        return (0,0)

    # : external/blocking send, disallows negative tags other than -1==ANY_TAG
    def send( self, other_pe, tag, data ):
        if( tag < -1 ):
            return (-1,'ERROR_BAD_TAG')
        # send the data
        rtn = self.int_send( other_pe, tag, data )
        # wait for it to be grabbed/removed
        comm = self.ns.msgLists[other_pe]
        #msg = (other_pe,tag,data)
        all_done = False
        while not all_done:
            #if( not msg in comm ):
            #    break
            all_done = True
            for msg in comm:
                if( (msg[0]==ns.tid) and (msg[1]==tag) \
                        and (msg[2]==data) ):
                    all_done = False
            # TODO: add a throttle to this spin-loop
        return rtn

    # : external/non-blocking send, disallows negative tags other than -1==ANY_TAG
    def isend( self, other_pe, tag, data ):
        if( tag < -1 ):
            return (-1,'ERROR_BAD_TAG')
        return self.int_send( other_pe, tag, data )

    # blocking and non-blocking recv
    # : internal, allows negative tags
    def int_recv( self, blocking, rem_pe, tag ):
        recv_pe = -1
        recv_tag = -1
        recv_data = None

        comm = self.ns.msgLists[self.tid]
        # if blocking==false, then set all_done=True,
        #    then we'll only do a single trip thru loop
        all_done = not blocking
        while not all_done:
            #for msg in comm:
            for n in range(len(comm)):
                msg = comm[n]
                # match on remote-pe?
                match_pe = False
                if( rem_pe == -1 ):
                    match_pe = True
                elif( msg[0] == rem_pe ):
                    match_pe = True

                # match on tag?
                match_tag = False
                if( tag == -1 ):
                    match_tag = True
                elif( msg[1] == tag ):
                    match_tag = True

                if( match_pe and match_tag ):    
                    (recv_pe,recv_tag,recv_data) = msg
                    #print( 'msg='+str(msg) )
                    # pull this msg out of the list
                    #comm.remove( msg )
                    comm.pop( n )
                    all_done = True
                    break
            
            # TODO: add a throttle to this spin-loop

        # TODO: check for errors
        return (recv_pe,recv_tag,recv_data)

    # : external/blocking recv, disallows negative tags other than -1==ANY_TAG
    def recv( self, rem_pe, tag ):
        if( tag < -1 ):
            return (-1,-1,'ERROR_BAD_TAG')
        return self.int_recv(True,rem_pe,tag)

    # : external/non-blocking recv, disallows negative tags other than -1==ANY_TAG
    def irecv( self, rem_pe, tag ):
        if( tag < -1 ):
            return (-1,-1,-1)
        return self.int_recv(False,rem_pe,tag)

    def barrier( self ):
        # post the sends
        for i in range(self.num_pes):
            if( i != self.tid ):
                self.int_send( i, -11, self.tid )
        # wait for recvs (this is a terrible way to do it)
        # TODO: wait for any matching message as they come in
        for i in range(self.num_pes):
            if( i != self.tid ):
                self.int_recv( True, i, -11 )
        return

    def collect( self, value ):
        rtn = [ 0 for i in range(self.num_pes) ]
        for i in range(self.num_pes):
            if( i != self.tid ):
                self.int_send( i, -22, value )
        for i in range(self.num_pes):
            if( i == self.tid ):
                rtn[i] = value
            else:
                msg = self.int_recv( True, i, -22 )
                rtn[i] = msg[2]
        return rtn

    #
    # For load/save to disk .. we ASSUME that all TIDs have same amount of data!
    def loadPopulation( self, gaMgr, filename ):
        start = self.tid * gaMgr.population_sz
        finish = start + gaMgr.population_sz
        pop = IoOps.loadPopulation( gaMgr, filename, start, finish)
        return pop

    # sort of 'for completeness' ...
    def randomPopulation( self, gaMgr, num ):
        return IoOps.randomPopulation( gaMgr, num )

    # dumb/brute-force approach ... all IO is done by TID=0
    # : this could have major memory and performance implications!
    def savePopulation( self, gaMgr, filename, mode='w' ):
        if( self.tid == 0 ):
            with open(filename,mode) as fp:
                # write TID-0 data first:
                for p in gaMgr.population:
                    txt = p.packData()
                    fp.write( txt + '\n' )
                # now for each other TID in sequence ...
                for i in range(1,self.num_pes):
                    x,y,data = self.int_recv( True, i, -123 )
                    for p in data:
                        txt = p.packData()
                        fp.write( txt + '\n' )
        else:
            # every other TID sends to TID=0
            self.int_send( 0, -123, gaMgr.population )



# # # # # # # # # # # # # # # # # # # #

class ParallelMgr():
    def __init__( self, **kwargs ):
        num_pes = kwargs.get( 'num_pes', 1 )

        # TODO: compare num_pes to multiprocessing.cpu_count()

        # manager for manager-to-worker communication & shared namespace
        self.mpMgr = Manager()
        self.namespace = self.mpMgr.Namespace()
        # need a copy of num_pes
        self.namespace.num_pes = num_pes

        # manager for manager-to-worker communication
        self.namespace.masterList = [ self.mpMgr.list() for i in range(num_pes) ]
        # self.masterList = [ self.commMgr.list() for i in range(self.num_pes) ]

        # separate lists for direct PE-to-PE communication
        self.namespace.msgLists = [ self.mpMgr.list() for i in range(num_pes) ]
        # self.msgLists = [ self.commMgr.list() for i in range(self.num_pes) ]

        print( 'ParallelMgr init' )
        sys.stdout.flush()

    def runWorkers( self, workerModule ):
        # self.peList = [ workerModule(i,self.namespace) for i in range(self.namespace.num_pes) ]
        self.peList = []
        for i in range(self.namespace.num_pes):
            commMgr = CommMgr( tid=i, namespace=self.namespace )
            pid = workerModule( i, commMgr )
            self.peList.append( pid )
        for pe in self.peList:
            print( 'starting worker '+str(pe) )
            pe.start()
        # TODO: check for errors

    def finalize( self ):
        print( 'waiting for workers to finish' )
        for pe in self.peList:
            print( 'worker '+str(pe)+' is_alive='+str(pe.is_alive()) )
            
        for pe in self.peList:
            pe.join()
        # TODO: check for errors


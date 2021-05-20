import multiprocessing as mp

def foo(q,i):
	q.put('hello'+str(i))

if __name__ == '__main__':
	mp.set_start_method('spawn')
	q = mp.Queue()
	p1 = mp.Process(target=foo, args=(q,1,))
	p1.start()
	p2 = mp.Process(target=foo, args=(q,2,))
	p2.start()
	print( q.get() )
	print( q.get() )
	p1.join()
	p2.join()

if __name__ == 'xx__main__':
	ctx = mp.get_context('spawn')
	q = ctx.Queue()
	p1 = ctx.Process(target=foo, args=(q,1,))
	p1.start()
	p2 = ctx.Process(target=foo, args=(q,2,))
	p2.start()
	print(q.get())
	print(q.get())
	p2.join()
	p2.join()

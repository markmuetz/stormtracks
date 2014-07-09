import time
import datetime as dt
from multiprocessing import Pool

from stormtracks.c20data import C20Data
from stormtracks.detect import GlobalCyclones

NUM_TS = 146

def f(i):
    try:
	c20data = C20Data(2005, verbose=False)
    except Exception, e:
	print e
    gdata = GlobalCyclones(c20data, i)
    gdata.track_vort_maxima(gdata.dates[0], gdata.dates[NUM_TS])
	# Must be picklable.
    return gdata.vortmax_time_series

def main():
    for num_workers in range(1, 3):
	start = time.time()

	pool = Pool(processes=num_workers)              # start worker processes
	res = pool.map(f, range(num_workers))

	t = time.time() - start
	print('Completed {0} timesteps with {1} workers in {2} ({3} ts/s)'.format(NUM_TS, num_workers, t, NUM_TS * num_workers / t))

if __name__ == '__main__':
    main()


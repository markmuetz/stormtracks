import time
import datetime as dt
from multiprocessing import Pool

import detect

NUM_TS = 146

def f(i):
    ncdata = detect.NCData(2001, verbose=False)
    gdata = detect.GlobalCyclones(ncdata)
    gdata.track_vort_maxima(i, gdata.dates[0], gdata.dates[NUM_TS])
    # Must be picklable.
    return gdata.vortmax_time_series

def main():
    for num_workers in range(1, 10):
	start = time.time()

	pool = Pool(processes=num_workers)              # start worker processes
	res = pool.map(f, range(num_workers))

	t = time.time() - start
	print('Completed {0} timesteps with {1} workers in {2} ({3} ts/s)'.format(NUM_TS, num_workers, t, NUM_TS * num_workers / t))

if __name__ == '__main__':
    main()


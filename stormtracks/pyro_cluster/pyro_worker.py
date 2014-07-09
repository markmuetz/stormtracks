import sys
sys.path.insert(0, '/home/ucfamue/DATA/.local/lib/python2.7/site-packages/')
import socket
import datetime as dt
import time

import Pyro4

from stormtracks.c20data import C20Data
from stormtracks.detect import GlobalCyclones
import stormtracks.match as match
from stormtracks.ibtracsdata import IbtracsData
from pyro_settings import worker_servers, is_ucl

class PyroWorker(object):
    def __init__(self):
	self.tracks_by_year = {}

    def do_work(self, year, ensemble_member):
	if year in self.tracks_by_year.keys():
	    tracks = self.tracks_by_year[year]
	else:
	    print('Loading tracks for year {0}'.format(year))
            ibt = IbtracsData()
	    tracks = ibt.load_ibtracks_year(year)
	    self.tracks_by_year[year] = tracks
	
	start = time.time()

	print('Received request for matches from year {0} ensemble {1}'.format(year, ensemble_member))
	c20data = C20Data(year, verbose=False)
	gdata = GlobalCyclones(c20data, ensemble_member)
	print('Processing')
	gdata.track_vort_maxima(dt.datetime(year, 6, 1), dt.datetime(year, 12, 1))
	matches = match.match2(gdata.vort_tracks_by_date, tracks)
	print('Returning matches')

	end = time.time()

	return 'Found {0} matches in {1}s'.format(len(matches.values()), end - start)


def main():
    hostname = socket.gethostname()
    short_hostname = hostname.split('.')[0]
    worker = PyroWorker()

    if is_ucl:
	daemon = Pyro4.Daemon(host=short_hostname)   
    else:
	daemon = Pyro4.Daemon(host='192.168.0.2')  
    ns = Pyro4.locateNS()
    uri = daemon.register(worker)   # register the greeting object as a Pyro object
    ns.register('stormtracks.worker_{0}'.format(short_hostname), uri)
    print('stormtracks.worker_{0}'.format(short_hostname))

    print "Ready. Object uri =", uri      # print the uri so we can use it in the client later
    daemon.requestLoop()                  # start the event loop of the server to wait for calls

if __name__ == '__main__':
    main()

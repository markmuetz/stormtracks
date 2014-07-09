import sys
sys.path.insert(0, '/home/ucfamue/DATA/.local/lib/python2.7/site-packages/')
import socket
import datetime as dt
import time

import Pyro4

import detect
import match
import load

class PyroWorker(object):
    def __init__(self):
	self.tracks_by_year = {}

    def do_work(self, year, ensemble_member):
	if year in self.tracks_by_year.keys():
	    tracks = self.tracks_by_year[year]
	else:
	    print('Loading tracks for year {0}'.format(year))
	    tracks, cou = load.load_ibtracks_year(year)
	    self.tracks_by_year[year] = tracks
	
	start = time.time()

	print('Received request for matches from year {0} ensemble {1}'.format(year, ensemble_member))
	ncdata = detect.NCData(year, verbose=False)
	gdata = detect.GlobalCyclones(ncdata, ensemble_member)
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

    daemon = Pyro4.Daemon(host=short_hostname)                 # make a Pyro daemon
    ns = Pyro4.locateNS()
    uri = daemon.register(worker)   # register the greeting object as a Pyro object
    ns.register('stormtracks.worker_{0}'.format(short_hostname), uri)

    print "Ready. Object uri =", uri      # print the uri so we can use it in the client later
    daemon.requestLoop()                  # start the event loop of the server to wait for calls

if __name__ == '__main__':
    main()

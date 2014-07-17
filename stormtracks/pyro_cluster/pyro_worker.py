#!/usr/bin/python
import socket
import datetime as dt
import time

import Pyro4

from stormtracks.c20data import C20Data, GlobalEnsembleMember
from stormtracks.tracking import VortmaxFinder, VortmaxNearestNeighbourTracker
from stormtracks.match import match
from stormtracks.ibtracsdata import IbtracsData
from stormtracks.load_settings import pyro_settings
from stormtracks.results import StormtracksResultsManager


class PyroWorker(object):
    def __init__(self):
        self.best_tracks_by_year = {}
        self.results_manager = StormtracksResultsManager()

    def do_work(self, year, ensemble_member, task):
        print('Task received')

        if task != 'vort_track':
            raise Exception('Unkown task {0}'.format(task))

        try:
            print('Received request for matches for year {0} ensemble {1}'.format(
                year, ensemble_member))

            if year in self.best_tracks_by_year.keys():
                best_tracks = self.best_tracks_by_year[year]
            else:
                print('Loading best_tracks for year {0}'.format(year))
                ibt = IbtracsData(verbose=False)
                best_tracks = ibt.load_ibtracks_year(year)
                self.best_tracks_by_year[year] = best_tracks

            results_manager = self.results_manager

            start = time.time()

            c20data = C20Data(year, verbose=False)
            gdata = GlobalEnsembleMember(c20data, ensemble_member)

            print('Processing')
            vort_finder = VortmaxFinder(gdata)
            vort_finder.find_vort_maxima(dt.datetime(year, 6, 1), dt.datetime(year, 12, 1))

            tracker = VortmaxNearestNeighbourTracker()
            tracker.track_vort_maxima(vort_finder.vortmax_time_series)

            matches = match(tracker.vort_tracks_by_date, best_tracks)
            # Quick to execute, no need to store.
            # good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]

            print('Saving data')
            results_manager.add_result(year, ensemble_member, 'vortmax_time_series',
                                       vort_finder.vortmax_time_series)
            results_manager.add_result(year, ensemble_member, 'vort_tracks_by_date',
                                       tracker.vort_tracks_by_date)
            results_manager.add_result(year, ensemble_member, 'matches',
                                       matches)

            results_manager.save()
            end = time.time()

            print('Found {0} matches in {1}s'.format(len(matches.values()), end - start))

            response = {
                'status': 'complete',
                'time_taken': end - start,
                }

            return response

        except Exception, e:
            response = {
                'status': 'failure',
                'exception': e
                }
            return response


def main():
    hostname = socket.gethostname()
    short_hostname = hostname.split('.')[0]
    worker = PyroWorker()

    if pyro_settings.is_ucl:
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

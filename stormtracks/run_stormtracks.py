#!/usr/bin/python
import time
import datetime as dt

from stormtracks.load_settings import settings
from stormtracks.results import StormtracksResultsManager
from stormtracks.ibtracsdata import IbtracsData
from stormtracks.c20data import C20Data, GlobalEnsembleMember
from stormtracks.tracking import VortmaxFinder, VortmaxNearestNeighbourTracker
from stormtracks.match import match


def main():
    results_manager = StormtracksResultsManager()
    num_ensemble_members = 56
    year = 2005
    results_name = 'main_results_{0}_{1}'.format(num_ensemble_members, year)

    ibdata = IbtracsData(verbose=False)
    print('Loaded best tracks for {0}'.format(year))
    best_tracks = ibdata.load_ibtracks_year(year)
    c20data = C20Data(year, verbose=False)

    if results_name not in results_manager.list():
        print('No saved results. Analysing...')
        start = time.time()

        for i in range(num_ensemble_members):
            print('Ensemble member {0} of {1}'.format(i + 1, num_ensemble_members))

            gdata = GlobalEnsembleMember(c20data, i)
            vort_finder = VortmaxFinder(gdata)

            vort_finder.find_vort_maxima(dt.datetime(year, 6, 1), dt.datetime(year, 12, 1))

            tracker = VortmaxNearestNeighbourTracker()
            tracker.track_vort_maxima(vort_finder.vortmax_time_series)

            matches = match(tracker.vort_tracks_by_date, best_tracks)
            good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]

            results_manager.add_result('vortmax_time_series_{0}'.format(i),
                                       vort_finder.vortmax_time_series)
            results_manager.add_result('vort_tracks_by_date_{0}'.format(i),
                                       tracker.vort_tracks_by_date)
            results_manager.add_result('matches_{0}'.format(i), matches)

        results_manager.save(results_name)
        end = time.time()
        print('Analysed and saved results to {0} in {1}s'.format(results_name, end - start))
    else:
        print('Loading saved results from {0}...'.format(results_name))
        start = time.time()
        results_manager.load(results_name)
        end = time.time()
        print('Loaded in {0}s'.format(end - start))

    all_good_matches = []
    for i in range(num_ensemble_members):
        print('Analysing ensemble member {0} of {1}'.format(i + 1, num_ensemble_members))

        vortmax_time_series = results_manager.results['vortmax_time_series_{0}'.format(i)]
        vort_tracks_by_date = results_manager.results['vort_tracks_by_date_{0}'.format(i)]
        matches = results_manager.results['matches_{0}'.format(i)]

        good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]
        all_good_matches.append(good_matches)


if __name__ == '__main__':
    main()

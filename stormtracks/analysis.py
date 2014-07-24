from __future__ import print_function

from collections import Counter
import datetime as dt

import numpy as np

from results import StormtracksResultsManager
from ibtracsdata import IbtracsData
from c20data import C20Data, GlobalEnsembleMember
from tracking import VortmaxFinder, VortmaxNearestNeighbourTracker, VortmaxKalmanFilterTracker
from match import match
from plotting import Plotter


class TrackingAnalysis(object):
    def __init__(self, year, ensemble_member):
        self.year = year
        self.ensemble_member = ensemble_member

        self.ibdata = IbtracsData(verbose=False)
        self.best_tracks = self.ibdata.load_ibtracks_year(year)
        self.results = {}

        self.results_manager = StormtracksResultsManager('analysis')

    def run_analysis(self, force_regen=False, load_only=False):
        scales = [1, 2, 3]
        pressure_levels = [995, 850, 250]
        # trackers = ['nearest_neighbour', 'kalman']
        trackers = ['nearest_neighbour']

        # For each set of config options, run a tracking analysis and store the results.
        for scale in scales:
            for pressure_level in pressure_levels:
                for tracker_name in trackers:
                    result_key = 'scale:{0};pl:{1};tracker:{2}'.format(scale,
                                                                       pressure_level,
                                                                       tracker_name)
                    saved_results = self.results_manager.list_results(self.year,
                                                                      self.ensemble_member)
                    if (result_key not in saved_results or force_regen) and not load_only:
                        print('Running analysis: {0}'.format(result_key))
                        result = self.run_individual_analysis(scale, pressure_level, tracker_name)
                        self.results_manager.add_result(self.year,
                                                        self.ensemble_member,
                                                        result_key,
                                                        result)
                        self.results_manager.save()
                    elif result_key in saved_results:
                        print('Loading saved result: {0}'.format(result_key))
                        self.results_manager.load(self.year,
                                                  self.ensemble_member,
                                                  result_key)

        # Run through the results and works out how many times each best track is represented.
        best_track_counter = Counter()
        results = self.results_manager.get_results(self.year, self.ensemble_member).items()
        return results

    def list_results(self):
        results = self.results_manager.get_results(self.year, self.ensemble_member).items()
        for i, (key, result) in enumerate(results):
            print('{0:2d}: {1}'.format(i, key))

    def setup_display(self, active_results):
        self.active_results = active_results
        results = self.results_manager.get_results(self.year, self.ensemble_member).items()
        self.plotters = []
        c20data = C20Data(self.year, verbose=False)
        for i, (key, result) in enumerate(results):
            if i in self.active_results:
                plotter = Plotter(key, self.best_tracks, c20data, [result])
                plotter.load('match_comp_1', is_plot=False)
                plotter.layout['figure'] = i + 1
                self.plotters.append(plotter)
        self.best_track_index = 0

    def next_best_track(self):
        self.best_track_index += 1
        self.plot()

    def prev_best_track(self):
        self.best_track_index -= 1
        self.plot()

    def plot(self):
        best_track = self.best_tracks[self.best_track_index]
        for plotter in self.plotters:
            print('{0} - '.format(plotter.title), end='')
            plotter.plot_match_from_best_track(best_track)

    def print_stats(self, sort_on='overlap'):
        sort_col = {'overlap': 1, 'cumdist': 2, 'avgdist': 3}[sort_on]
        results = self.results_manager.get_results(self.year, self.ensemble_member).items()
        stats = []
        for i in self.active_results:
            key, good_matches = results[i]
            sum_overlap = np.sum([m.overlap for m in good_matches])
            sum_cum_dist = np.sum([m.cum_dist for m in good_matches])
            sum_av_dist = np.sum([m.av_dist() for m in good_matches])
            stats.append((key, sum_overlap, sum_cum_dist, sum_av_dist))

        for stat in sorted(stats, key=lambda x: x[sort_col]):
            key, sum_overlap, sum_cum_dist, sum_av_dist = stat
            print(key)
            print('  sum overlap: {0}'.format(sum_overlap))
            print('  sum cumdist: {0}'.format(sum_cum_dist))
            print('  sum avgdist: {0}'.format(sum_av_dist))

    def run_individual_analysis_from_result_key(self, key):
        config = key.split(';')
        scale = int(config[0].split(':')[1])
        pressure_level = int(config[1].split(':')[1])
        tracker_name = config[2].split(':')[1]
        result = self.run_individual_analysis(scale, pressure_level, tracker_name)
        return result

    def run_individual_analysis(self, scale, pressure_level, tracker_name):
        print('Scale: {0}; Press level: {1}; Tracker: {2}'.format(
            scale, pressure_level, tracker_name))

        if scale == 1:
            upscaling = False
        else:
            upscaling = True

        c20data = C20Data(self.year, verbose=False,
                          pressure_level=pressure_level,
                          upscaling=upscaling,
                          scale_factor=scale)

        if tracker_name == 'nearest_neighbour':
            tracker = VortmaxNearestNeighbourTracker()
        elif tracker_name == 'kalman':
            tracker = VortmaxKalmanFilterTracker()

        gem = GlobalEnsembleMember(c20data, self.ensemble_member)
        vort_finder = VortmaxFinder(gem)

        vort_finder.find_vort_maxima(dt.datetime(self.year, 6, 1),
                                     dt.datetime(self.year, 12, 1),
                                     use_upscaled=upscaling)

        tracker.track_vort_maxima(vort_finder.vortmax_time_series)

        matches = match(tracker.vort_tracks_by_date, self.best_tracks)
        good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]

        return good_matches

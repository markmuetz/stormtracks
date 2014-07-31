from __future__ import print_function

from collections import Counter, OrderedDict
import datetime as dt

import numpy as np

from results import StormtracksResultsManager
from ibtracsdata import IbtracsData
from c20data import C20Data, GlobalEnsembleMember
from tracking import VortmaxFinder, VortmaxNearestNeighbourTracker, VortmaxKalmanFilterTracker
import matching
from plotting import Plotter
from logger import setup_logging, get_logger

SORT_COLS = {
    'overlap': 1,
    'cumdist': 2,
    'cumoveroverlap': 3,
    'avgdist': 4,
    'avgdistovermatches': 5,
    }


class TrackingAnalysis(object):
    def __init__(self, year, verbose=False):
        self.year = year
        self.verbose = verbose

        self.ibdata = IbtracsData(verbose=False)
        self.best_tracks = self.ibdata.load_ibtracks_year(year)
        self.results = {}

        self.results_manager = StormtracksResultsManager('pyro_analysis')

        self.setup_analysis()
        self.analysis_loaded = False

        # self.log = get_logger('analysis', console_level_str='INFO')

        filename = 'analysis.log'
        self.log = setup_logging('analysis', filename=filename, console_level_str='INFO')

    def __say(self, message):
        if self.verbose:
            print(message)

    def setup_analysis(self):
        self.analysis_config_options = []

        scales = [1, 2, 3]
        pressure_levels = [995, 850]
        trackers = ['nearest_neighbour']
        # pressure_levels = [995, 850, 250]
        # trackers = ['nearest_neighbour', 'kalman']

        for scale in scales:
            for pressure_level in pressure_levels:
                for tracker_name in trackers:
                    config = {
                        'scale': scale,
                        'pressure_level': pressure_level,
                        'tracker': tracker_name,
                        }
                    self.analysis_config_options.append(config)

    def run_cross_ensemble_analysis(self, sort_on='avgdist', active_results=None, show_wld=False):
        if not self.analysis_loaded:
            for i in range(56):
                self.load_analysis(i)
            self.analysis_loaded = True

        if active_results:
            self.active_results = active_results
        else:
            self.active_results = range(len(self.analysis_config_options))

        if show_wld:
            if len(self.active_results) != 2:
                raise Exception('if printing wins/loses/draws must be called with 2 active results')
            wlds = []
        else:
            wlds = None

        if not show_wld:
            cross_ensemble_results = OrderedDict()
            for i in self.active_results:
                config = self.analysis_config_options[i]
                cross_ensemble_results[self.good_matches_key(config)] = Counter()

            for i in range(56):
                stats = self.list_stats(i, sort_on)
                for stat_pos, stat in enumerate(stats):
                    cross_ensemble_results[stat[0]][stat_pos] += 1

            pos_title = 'Position on {0}'.format(sort_on)
            if self.log:
                self.log.info(pos_title)
                self.log.info('=' * len(pos_title))
                self.log.info('')
                for k, v in cross_ensemble_results.items():
                    self.log.info(k)
                    self.log.info('  {0}'.format(v.items()))
                self.log.info('')
        else:
            for i in range(56):
                key0, key1, wld = self.win_lose_draw(0,
                                                     self.active_results[0],
                                                     self.active_results[1])
                wlds.append(wld)

            sum0 = 0
            sum1 = 0
            sum_draw = 0
            for wld in wlds:
                sum0 += wld['w0']
                sum1 += wld['w1']
                sum_draw += wld['d']

            if self.log:
                self.log.info('Win Lose Draw')
                self.log.info('=============')
                self.log.info('')
                self.log.info('{0} won: {1}'.format(key0, sum0))
                self.log.info('{0} won: {1}'.format(key1, sum1))
                self.log.info('draw: {0}'.format(sum_draw))
                self.log.info('')

    def _result_key(self, config):
        return 'scale:{scale};pl:{pressure_level};tracker:{tracker}'.format(**config)

    def good_matches_key(self, config):
        return 'good_matches-{0}'.format(self._result_key(config))

    def vort_tracks_by_date_key(self, config):
        return 'vort_tracks_by_date-{0}'.format(self._result_key(config))

    def load_analysis(self, ensemble_member=0):
        for config in self.analysis_config_options:
            good_matches_key = self.good_matches_key(config)

            self.load_analysis_for_config(ensemble_member, config, good_matches_key)

    def load_analysis_for_config(self, ensemble_member, config, key):
        saved_results = self.results_manager.list_results(self.year, ensemble_member)

        if key in saved_results:
            self.__say('Loading saved result: {0}'.format(key))
            self.results_manager.load(self.year,
                                      ensemble_member,
                                      key)

    def run_ensemble_matches_analysis(self, force_regen=False):
        config = self.analysis_config_options[5]
        self.log.info('Using {0}'.format(self._result_key(config)))
        vort_tracks_by_date_key = self.vort_tracks_by_date_key(config)

        num_ensemble_members = 10
        for ensemble_member in range(num_ensemble_members):
            # self.load_analysis_for_config(ensemble_member, config, vort_tracks_by_date_key)
            saved_results = self.results_manager.list_results(self.year, ensemble_member)

            if (vort_tracks_by_date_key not in saved_results or force_regen):
                self.log.info('Analysing ensemble member {0}'.format(ensemble_member))
                self.__say('Running analysis: {0}'.format(vort_tracks_by_date_key))
                good_matches, vort_tracks_by_date = \
                    self.run_individual_analysis(ensemble_member, config)
                self.results_manager.add_result(self.year,
                                                ensemble_member,
                                                vort_tracks_by_date_key,
                                                vort_tracks_by_date)
                self.results_manager.save()
            elif vort_tracks_by_date_key in saved_results:
                self.log.info('Loading ensemble member {0}'.format(ensemble_member))
                self.__say('Loading saved result: {0}'.format(vort_tracks_by_date_key))
                self.results_manager.load(self.year, ensemble_member, vort_tracks_by_date_key)

        vort_tracks = []
        for ensemble_member in range(num_ensemble_members):
            vort_tracks.append(self.results_manager.get_results(self.year, 
                                                                ensemble_member).items()[0][1])

        self.log.info('Matching all tracks')
        ensemble_matches = matching.match_ensemble_vort_tracks_by_date(vort_tracks)
        self.log.info('Done')

        best_track_matches = \
                matching.match_best_track_to_ensemble_match(self.best_tracks, ensemble_matches)

        return ensemble_matches, best_track_matches

    def run_analysis(self, ensemble_member=0, force_regen=False):
        # For each set of config options, run a tracking analysis and store the results.
        for config in self.analysis_config_options:
            good_matches_key = self.good_matches_key(config)
            saved_results = self.results_manager.list_results(self.year, ensemble_member)

            if (good_matches_key not in saved_results or force_regen):
                self.__say('Running analysis: {0}'.format(good_matches_key))
                result = self.run_individual_analysis(ensemble_member, config)
                self.results_manager.add_result(self.year,
                                                ensemble_member,
                                                good_matches_key,
                                                result)
                self.results_manager.save()
            elif good_matches_key in saved_results:
                self.__say('Loading saved result: {0}'.format(good_matches_key))
                self.results_manager.load(self.year, ensemble_member, good_matches_key)

        results = self.results_manager.get_results(self.year, ensemble_member).items()
        return results

    def list_results(self, ensemble_member):
        results = self.results_manager.get_results(self.year, ensemble_member).items()
        for i, (key, result) in enumerate(results):
            print('{0:2d}: {1}'.format(i, key))

    def setup_display(self, ensemble_member, active_results):
        self.active_results = active_results
        results = self.results_manager.get_results(self.year, ensemble_member).items()
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
            self.__say('{0} - '.format(plotter.title), end='')
            plotter.plot_match_from_best_track(best_track)

    def print_win_lose_draw(self, ensemble_member, result0, result1):
        key0, key1, wld = self.win_lose_draw(ensemble_member, result0, result1)
        print('{0}: won: {1}'.format(key0, wld['w0']))
        print('{0}: won: {1}'.format(key1, wld['w1']))
        print('draws: {0}'.format(wld['d']))

    def win_lose_draw(self, ensemble_member, result0, result1):
        wld = Counter()

        results = self.results_manager.get_results(self.year, ensemble_member).items()

        key0, gm0 = results[result0]
        key1, gm1 = results[result1]

        # import ipdb; ipdb.set_trace()
        for bt in self.best_tracks:
            m0 = None
            for m in gm0:
                if m.best_track.name == bt.name:
                    m0 = m
                    break

            m1 = None
            for m in gm1:
                if m.best_track.name == bt.name:
                    m1 = m
                    break

            if m0 and m1:
                if m0.av_dist() < m1.av_dist() - 0.02:
                    wld['w0'] += 1
                elif m1.av_dist() < m0.av_dist() - 0.02:
                    wld['w1'] += 1
                else:
                    wld['d'] += 1
            elif m0:
                wld['w0'] += 1
            elif m1:
                wld['w1'] += 1
            else:
                wld['d'] += 1

        return key0, key1, wld

    def list_stats(self, ensemble_member=0, sort_on='overlap'):
        sort_col = SORT_COLS[sort_on]
        results = self.results_manager.get_results(self.year, ensemble_member).items()
        stats = []
        for i in self.active_results:
            key, good_matches = results[i]
            len_matches = len(good_matches)
            sum_overlap = np.sum([m.overlap for m in good_matches])
            sum_cum_dist = np.sum([m.cum_dist for m in good_matches])
            cum_over_overlap = sum_cum_dist / sum_overlap
            sum_av_dist = np.sum([m.av_dist() for m in good_matches])
            sum_av_dist_over_len_matches = np.sum([m.av_dist()/len_matches for m in good_matches])
            stats.append((key,
                          sum_overlap,
                          sum_cum_dist,
                          cum_over_overlap,
                          sum_av_dist,
                          sum_av_dist_over_len_matches))

        return sorted(stats, key=lambda x: x[sort_col])

    def print_stats(self, ensemble_member=0, sort_on='avgdist'):
        stats = self.list_stats(ensemble_membe, sort_on)

        for stat in stats:
            key, sum_overlap, sum_cum_dist, sum_av_dist = stat
            print(key)
            print('  sum overlap: {0}'.format(sum_overlap))
            print('  sum cumdist: {0}'.format(sum_cum_dist))
            print('  sum avgdist: {0}'.format(sum_av_dist))

    def run_individual_analysis(self, ensemble_member, config):
        msg = 'Scale: {scale}, press level: {pressure_level}, tracker:{tracker}'.format(**config)
        self.__say(msg)

        if config['scale'] == 1:
            upscaling = False
        else:
            upscaling = True

        c20data = C20Data(self.year, verbose=False,
                          pressure_level=config['pressure_level'],
                          upscaling=upscaling,
                          scale_factor=config['scale'])

        if config['tracker'] == 'nearest_neighbour':
            tracker = VortmaxNearestNeighbourTracker()
        elif config['tracker'] == 'kalman':
            tracker = VortmaxKalmanFilterTracker()

        gem = GlobalEnsembleMember(c20data, ensemble_member)
        vort_finder = VortmaxFinder(gem)

        vort_finder.find_vort_maxima(dt.datetime(self.year, 6, 1),
                                     dt.datetime(self.year, 12, 1),
                                     use_upscaled=upscaling)

        tracker.track_vort_maxima(vort_finder.vortmax_time_series)

        matches = matching.match(tracker.vort_tracks_by_date, self.best_tracks)
        good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]

        return good_matches, tracker.vort_tracks_by_date


def main(year, analysis, log=None):
    tracking_analysis = TrackingAnalysis(year)
    if not log:
        filename = 'analysis.log'
        log = setup_logging('analysis', filename=filename, console_level_str='INFO')

    tracking_analysis.log = log
    log.info('Running analysis {0} for year {1}'.format(analysis, year))

    if analysis == 'ensemble':
        tracking_analysis.run_ensemble_matches_analysis()
    elif analysis == 'tracking':
        for sort_col in SORT_COLS.keys():
            log.info('Run analysis on col {0}'.format(sort_col))

            log.info('Run full analysis\n')
            tracking_analysis.run_cross_ensemble_analysis(sort_on=sort_col,
                                                          active_results=(0, 1, 2, 3, 4, 5))

            log.info('Run 995 analysis\n')
            tracking_analysis.run_cross_ensemble_analysis(sort_on=sort_col,
                                                          active_results=(0, 2, 4))
            log.info('Run 850 analysis\n')
            tracking_analysis.run_cross_ensemble_analysis(sort_on=sort_col,
                                                          active_results=(1, 3, 5))

            log.info('Run scale 1 analysis\n')
            tracking_analysis.run_cross_ensemble_analysis(sort_on=sort_col,
                                                          active_results=(0, 1))
            log.info('Run scale 2 analysis\n')
            tracking_analysis.run_cross_ensemble_analysis(sort_on=sort_col,
                                                          active_results=(2, 3))
            log.info('Run scale 3 analysis\n')
            tracking_analysis.run_cross_ensemble_analysis(sort_on=sort_col,
                                                          active_results=(4, 5))

        log.info('Run scale 1 wld\n')
        tracking_analysis.run_cross_ensemble_analysis(active_results=(0, 1),
                                                      show_wld=True)
        log.info('Run scale 2 wld\n')
        tracking_analysis.run_cross_ensemble_analysis(active_results=(2, 3),
                                                      show_wld=True)
        log.info('Run scale 3 wld\n')
        tracking_analysis.run_cross_ensemble_analysis(active_results=(4, 5),
                                                      show_wld=True)
    return log


if __name__ == '__main__':
    log = main(2005, 'ensemble')
    # for year in range(2003, 2008):
        # log = main(year, log)

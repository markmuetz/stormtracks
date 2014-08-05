from __future__ import print_function

from collections import Counter, OrderedDict
import datetime as dt
from argparse import ArgumentParser

import numpy as np

from results import StormtracksResultsManager, ResultNotFound
from ibtracsdata import IbtracsData
from c20data import C20Data, GlobalEnsembleMember
from tracking import VortmaxFinder, VortmaxNearestNeighbourTracker, VortmaxKalmanFilterTracker
import matching
from plotting import Plotter
from logger import setup_logging, get_logger
from utils.utils import geo_dist

SORT_COLS = {
    'overlap': 1,
    'cumdist': 2,
    'cumoveroverlap': 3,
    'avgdist': 4,
    'avgdistovermatches': 5,
    }


class TrackingAnalysis(object):
    '''Provides a variety of ways of analysing tracking performance

    Makes extensive use of its results_manager to load/save results.
    Used by the pyro code to farm out jobs across the cluster.
    To a large extent replaces the manual analysis steps.
    :param year: year on which to run analysis
    :param is_setup_logging: whether to setup logging (useful in ipython)
    '''
    def __init__(self, year, is_setup_logging=False):
        self.set_year(year)
        self.setup_analysis()

        if is_setup_logging:
            filename = 'analysis.log'
            self.log = setup_logging('analysis', filename=filename, console_level_str='INFO')
        else:
            self.log = get_logger('analysis', console_logging=False)

    def set_year(self, year):
        '''Sets the year, loading best_tracks and setting up results_manager appropriately'''
        self.year = year
        self.ibdata = IbtracsData(verbose=False)
        self.best_tracks = self.ibdata.load_ibtracks_year(year)

        self.results_manager = StormtracksResultsManager('pyro_tracking_analysis')

    def setup_analysis(self):
        '''Sets up the current configuration options'''
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

    def get_matching_configs(self, **kwargs):
        '''Allows for easy filtering of current config options'''
        configs = []
        for config in self.analysis_config_options:
            should_add = True
            for k, v in kwargs.items():
                if config[k] != v:
                    should_add = False
                    break
            if should_add:
                configs.append(config)

        return configs

    def _result_key(self, config):
        return 'scale:{scale};pl:{pressure_level};tracker:{tracker}'.format(**config)

    def good_matches_key(self, config):
        '''Returns the good_matches key for the config options'''
        return 'good_matches-{0}'.format(self._result_key(config))

    def vort_tracks_by_date_key(self, config):
        '''Returns the vort_tracks_by_date key for the config options'''
        return 'vort_tracks_by_date-{0}'.format(self._result_key(config))

    def run_individual_analysis(self, ensemble_member, config):
        '''Runs a given analysis based on config dict'''
        msg = 'Scale: {scale}, press level: {pressure_level}, tracker:{tracker}'.format(**config)
        self.log.info(msg)

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

        good_matches = matching.good_matches(matches)

        return good_matches, tracker.vort_tracks_by_date

    def run_wld_analysis(self, active_configs={}, num_ensemble_members=56):
        '''Runs a win/lose/draw analysis on all ensemble members

        If a track from a particular analysis has a lower average dist it is said to have 'won'
        i.e. track for Wilma in pressure_level:850/scale:2/tracker:nearest neighbour has a lower
        av dist than pl:995/../.. .
        '''
        wlds = []
        for i in range(num_ensemble_members):
            configs = self.get_matching_configs(**active_configs)
            key0 = self.good_matches_key(configs[0])
            key1 = self.good_matches_key(configs[1])

            wld = self._win_lose_draw(0, key0, key1)
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

    def run_position_analysis(self, sort_on='avgdist', active_configs={},
                              force_regen=False, num_ensemble_members=56):
        '''Runs a positional analysis on the given sort_on col

        If sort_on is e.g. avg_dist, summed av dist for each of the active configs are
        calc'd and they are ranked in terms of which is lowest
        '''
        # import ipdb; ipdb.set_trace()
        self.log.info('Analysing {0} ensemble members'.format(num_ensemble_members))

        cross_ensemble_results = OrderedDict()
        for config in self.get_matching_configs(**active_configs):
            cross_ensemble_results[self.good_matches_key(config)] = Counter()

        for ensemble_member in range(num_ensemble_members):
            stats = self.list_stats(ensemble_member, sort_on, active_configs)
            for stat_pos, stat in enumerate(stats):
                cross_ensemble_results[stat[0]][stat_pos] += 1

        pos_title = 'Position on {0}'.format(sort_on)

        self.log.info(pos_title)
        self.log.info('=' * len(pos_title))
        self.log.info('')
        for k, v in cross_ensemble_results.items():
            self.log.info(k)
            self.log.info('  {0}'.format(v.items()))
        self.log.info('')

    def run_ensemble_matches_analysis(self, num_ensemble_members=56, force_regen=False):
        '''Looks at a particular config and runs a matching algortihm against each track'''
        config = self.analysis_config_options[5]
        self.log.info('Using {0}'.format(self._result_key(config)))
        vort_tracks_by_date_key = self.vort_tracks_by_date_key(config)

        if not force_regen:
            try:
                # Cut to the chase, just load the results from disk.
                ensemble_matches = self.results_manager.get_result(self.year,
                                                                   'all',
                                                                   'ensemble_matches')
                best_track_matches = self.results_manager.get_result(self.year,
                                                                     'all',
                                                                     'best_track_matches')
            except ResultNotFound:
                force_regen = True

        if force_regen:
            vort_tracks = []
            for ensemble_member in range(num_ensemble_members):
                if not force_regen:
                    try:
                        vort_tracks_by_date = \
                            self.results_manager.get_result(year,
                                                            ensemble_member,
                                                            vort_tracks_by_date_key)
                        self.log.info('Loaded ensemble member {0}'.format(ensemble_member))
                    except ResultNotFound:
                        force_regen = True

                if force_regen:
                    self.log.info('Analysing ensemble member {0}'.format(ensemble_member))
                    good_matches, vort_tracks_by_date = \
                        self.run_individual_analysis(ensemble_member, config)
                    self.results_manager.add_result(self.year,
                                                    ensemble_member,
                                                    vort_tracks_by_date_key,
                                                    vort_tracks_by_date)
                    self.results_manager.save()

                vort_tracks.append(vort_tracks_by_date)

            self.log.info('Matching all tracks')
            ensemble_matches = matching.match_ensemble_vort_tracks_by_date(vort_tracks)
            self.log.info('Done')

            best_track_matches = \
                matching.match_best_track_to_ensemble_match(self.best_tracks, ensemble_matches)

            self.results_manager.add_result(self.year,
                                            'all',
                                            'ensemble_matches',
                                            ensemble_matches)
            self.results_manager.add_result(self.year,
                                            'all',
                                            'best_track_matches',
                                            best_track_matches)
            self.results_manager.save()

        return ensemble_matches, best_track_matches

    def run_analysis(self, ensemble_member, force_regen=False):
        '''For each set of config options, run a tracking analysis and store the results'''
        results = {}
        for config in self.analysis_config_options:
            good_matches_key = self.good_matches_key(config)
            vort_tracks_by_date_key = self.vort_tracks_by_date_key(config)

            if not force_regen:
                try:
                    good_matches = self.results_manager.get_result(self.year,
                                                                   ensemble_member,
                                                                   good_matches_key)
                    self.log.info('Loaded saved result: {0}'.format(good_matches_key))
                except ResultNotFound:
                    force_regen = True

            if force_regen:
                self.log.info('Running analysis: {0}'.format(good_matches_key))
                good_matches, vort_tracks_by_date = \
                    self.run_individual_analysis(ensemble_member, config)
                self.results_manager.add_result(self.year,
                                                ensemble_member,
                                                good_matches_key,
                                                good_matches)
                self.results_manager.add_result(self.year,
                                                ensemble_member,
                                                vort_tracks_by_date_key,
                                                vort_tracks_by_date)
                self.results_manager.save()
            results[good_matches_key] = good_matches

        return results

    def _win_lose_draw(self, ensemble_member, key0, key1):
        wld = Counter()

        gm0 = self.results_manager.get_result(self.year, ensemble_member, key0)
        gm1 = self.results_manager.get_result(self.year, ensemble_member, key1)

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

        return wld

    def get_good_matches(self, ensemble_member, config):
        '''Either loads or generates (and saves) good_matches'''
        key = self.good_matches_key(config)
        try:
            good_matches = self.results_manager.get_result(self.year, ensemble_member, key)
        except ResultNotFound:
            good_matches, vort_tracks_by_date = \
                self.run_individual_analysis(ensemble_member, config)

            self.results_manager.add_result(self.year,
                                            ensemble_member,
                                            key,
                                            good_matches)
            self.results_manager.save()
        return good_matches

    def get_vort_tracks_by_date(self, ensemble_member, config):
        '''Either loads or generates (and saves) vort_tracks_by_date'''
        key = self.vort_tracks_by_date_key(config)
        try:
            vort_tracks_by_date = self.results_manager.get_result(self.year, ensemble_member, key)
        except ResultNotFound:
            good_matches, vort_tracks_by_date = \
                self.run_individual_analysis(ensemble_member, config)

            self.results_manager.add_result(self.year,
                                            ensemble_member,
                                            vort_tracks_by_date_key,
                                            vort_tracks_by_date)
            self.results_manager.save()
        return vort_tracks_by_date

    def list_stats(self, ensemble_member=0, sort_on='avgdist', active_configs={}):
        '''Runs through all statistics for the requested ensemble member and compiles stats

        sorts on the requested column and only looks at the active_configs. This makes it easy to
        e.g. compare all scale 1 or all 850 configuration options.'''
        sort_col = SORT_COLS[sort_on]
        configs = self.get_matching_configs(**active_configs)
        stats = []
        for config in configs:
            key = self.good_matches_key(config)
            good_matches = self.get_good_matches(ensemble_member, config)

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
        '''Prints the stats'''
        stats = self.list_stats(ensemble_membe, sort_on)

        for stat in stats:
            key, sum_overlap, sum_cum_dist, sum_av_dist = stat
            print(key)
            print('  sum overlap: {0}'.format(sum_overlap))
            print('  sum cumdist: {0}'.format(sum_cum_dist))
            print('  sum avgdist: {0}'.format(sum_av_dist))

    def setup_display(self, ensemble_member, active_configs={}):
        '''Sets up plotters for displaying of results'''
        configs = self.get_matching_configs(**active_configs)
        self.plotters = []
        c20data = C20Data(self.year, verbose=False)
        for i, config in enumerate(configs):
            key = self.good_matches_key(config)
            good_matches = self.get_good_matches(ensemble_member, key)

            plotter = Plotter(key, self.best_tracks, c20data, [good_matches])
            plotter.load('match_comp_1', is_plot=False)
            plotter.layout['figure'] = i + 1
            self.plotters.append(plotter)

        self.best_track_index = 0

    def next_best_track(self):
        '''Moves each plotter's best track on by one'''
        self.best_track_index += 1
        self.plot()

    def prev_best_track(self):
        '''Moves each plotter's best track back by one'''
        self.best_track_index -= 1
        self.plot()

    def plot(self):
        '''Uses each plotter to plot the current scene'''
        best_track = self.best_tracks[self.best_track_index]
        for plotter in self.plotters:
            self.log.info('{0} - '.format(plotter.title), end='')
            plotter.plot_match_from_best_track(best_track)


def run_ensemble_analysis(tracking_analysis, year):
    '''Performs a full enesmble analysis on the given year

    Searches through and tries to match all tracks across ensemble members **without** using
    any best tracks info.'''
    tracking_analysis.set_year(year)
    tracking_analysis.run_ensemble_matches_analysis(56)


def run_tracking_stats_analysis(tracking_analysis, year):
    '''Runs a complete tracking analysis, comparing the performance of each configuration option

    Compares performance in a variety of ways, e.g. within pressure level or just scale 1.'''

    tracking_analysis.set_year(year)

    log = tracking_analysis.log
    log.info('Running tracking stats analysis for year {0}'.format(year))

    for sort_col in SORT_COLS.keys():
        if sort_col in ['overlap', 'cumdist']:
            continue

        log.info('Run analysis on col {0}'.format(sort_col))

        log.info('Run full analysis\n')
        tracking_analysis.run_position_analysis(sort_on=sort_col)

        log.info('Run 995 analysis\n')
        tracking_analysis.run_position_analysis(sort_on=sort_col,
                                                active_configs={'pressure_level': 995})
        log.info('Run 850 analysis\n')
        tracking_analysis.run_position_analysis(sort_on=sort_col,
                                                active_configs={'pressure_level': 850})

        log.info('Run scale 1 analysis\n')
        tracking_analysis.run_position_analysis(sort_on=sort_col,
                                                active_configs={'scale': 1})
        log.info('Run scale 2 analysis\n')
        tracking_analysis.run_position_analysis(sort_on=sort_col,
                                                active_configs={'scale': 2})
        log.info('Run scale 3 analysis\n')
        tracking_analysis.run_position_analysis(sort_on=sort_col,
                                                active_configs={'scale': 3})
        log.info('Run scale 4 analysis\n')
        tracking_analysis.run_position_analysis(sort_on=sort_col,
                                                active_configs={'scale': 4})
        log.info('Run scale 5 analysis\n')
        tracking_analysis.run_position_analysis(sort_on=sort_col,
                                                active_configs={'scale': 5})

    log.info('Run scale 1 wld\n')
    tracking_analysis.run_wld_analysis(active_configs={'scale': 1})

    log.info('Run scale 2 wld\n')
    tracking_analysis.run_wld_analysis(active_configs={'scale': 2})

    log.info('Run scale 3 wld\n')
    tracking_analysis.run_wld_analysis(active_configs={'scale': 3})

    log.info('Run scale 4 wld\n')
    tracking_analysis.run_wld_analysis(active_configs={'scale': 4})

    log.info('Run scale 5 wld\n')
    tracking_analysis.run_wld_analysis(active_configs={'scale': 5})


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-a', '--analysis', default='ensemble')
    parser.add_argument('-s', '--start-year', type=int, default=2005)
    parser.add_argument('-e', '--end-year', type=int, default=2005)
    args = parser.parse_args()

    years = range(args.start_year, args.end_year + 1)
    # import ipdb; ipdb.set_trace()
    tracking_analysis = TrackingAnalysis(years[0])
    if args.analysis == 'ensemble':
        for year in years:
            run_ensemble_analysis(tracking_analysis, year)
    elif args.analysis == 'stats':
        for year in years:
            run_tracking_stats_analysis(tracking_analysis, year)
    else:
        raise Exception('One of ensemble or stats should be chosen')

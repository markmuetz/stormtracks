from __future__ import print_function

import sys
import os
from collections import Counter, OrderedDict
import time
import datetime as dt
from argparse import ArgumentParser
import random

import numpy as np
import pylab as plt

from load_settings import settings
from results import StormtracksResultsManager, ResultNotFound, StormtracksNumpyResultsManager
from ibtracsdata import IbtracsData
from c20data import C20Data, GlobalEnsembleMember
from tracking import VortmaxFinder, VortmaxNearestNeighbourTracker,\
    VortmaxKalmanFilterTracker, FieldFinder
import matching
from categorisation import CutoffCategoriser, get_cyclone_attr, SCATTER_ATTRS
from plotting import Plotter
import plotting
from logger import setup_logging, get_logger
from utils.utils import geo_dist

SORT_COLS = {
    'overlap': 1,
    'cumdist': 2,
    'cumoveroverlap': 3,
    'avgdist': 4,
    'avgdistovermatches': 5,
    }


class StormtracksAnalysis(object):
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
            tracker = VortmaxNearestNeighbourTracker(ensemble_member)
        elif config['tracker'] == 'kalman':
            tracker = VortmaxKalmanFilterTracker()

        gem = GlobalEnsembleMember(c20data, ensemble_member)
        vort_finder = VortmaxFinder(gem)

        vort_finder.find_vort_maxima(dt.datetime(self.year, 6, 1),
                                     dt.datetime(self.year, 12, 1),
                                     use_upscaled=upscaling)

        tracker.track_vort_maxima(vort_finder.vortmax_time_series)

        matches = matching.match_vort_tracks_by_date_to_best_tracks(tracker.vort_tracks_by_date,
                                                                    self.best_tracks)

        good_matches = matching.good_matches(matches)

        return good_matches, tracker.vort_tracks_by_date

    def run_individual_field_collection(self, ensemble_member):
        self.log.info('Collecting fields for {0}'.format(ensemble_member))
        c20data = C20Data(self.year, verbose=False,
                          pressure_level=995,
                          upscaling=False,
                          scale_factor=1)

        self.log.info('c20data created')
        tracking_config = {'pressure_level': 850, 'scale': 3, 'tracker': 'nearest_neighbour'}
        key = self.vort_tracks_by_date_key(tracking_config)

        self.log.info('Loading key: {0}'.format(key))
        vms = self.results_manager.get_result(self.year, ensemble_member, key)

        self.log.info('Finding fields')
        field_finder = FieldFinder(c20data, vms, ensemble_member)
        field_finder.collect_fields()

        return field_finder.cyclone_tracks.values()

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
        stats = self.list_stats(ensemble_member, sort_on)

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


def score_matchup(matchup):
    score = 0
    score += matchup['tp'] * 4
    # score += matchup['unmatched_tn'] * 0.1
    # score += matchup['tn'] * 1
    score -= matchup['fn'] * 2
    score -= matchup['fp'] * 2
    score -= matchup['unmatched_fp'] * 2
    score -= matchup['missed'] * 2
    return score


class CategorisationAnalysis(object):
    def __init__(self):
        self.results_manager = StormtracksResultsManager('pyro_field_collection_analysis')
        self.plot_results_manager = StormtracksResultsManager('plot_results')
        self.all_best_tracks = {}
        self.hurricanes_in_year = {}
        self.cutoff_cat = CutoffCategoriser()

    def cat_results_key(self, name, years, ensemble_members):
        years_str = '-'.join(map(str, years))
        em_str = '-'.join(map(str, ensemble_members))
        return '{0}_{1}_{2}'.format(name, years_str, em_str)

    def run_categorisation_analysis(self, years, ensemble_members=(0, ), plot_mode=None, save=False):
        total_cat_data = None
        total_are_hurricanes = None
        numpy_results_manager = StormtracksNumpyResultsManager('cat_data')
        for year in years:
            for ensemble_member in ensemble_members:
                print('{0}-{1}'.format(year, ensemble_member))

                # matches, unmatched = self.run_individual_analysis(year, ensemble_member, plot_mode, save)
                cat_data, are_hurricanes = self.build_cat_data(year, ensemble_member)
                if total_cat_data is not None:
                    total_cat_data = np.concatenate((total_cat_data, cat_data))
                else:
                    total_cat_data = cat_data

                if total_are_hurricanes is not None:
                    total_are_hurricanes = np.concatenate((total_are_hurricanes, are_hurricanes))
                else:
                    total_are_hurricanes = are_hurricanes

                # cat_matchup, unmatched_fp = self.run_individual_categorisation_analysis(year, ensemble_member, matches, unmatched)

                if plot_mode:
                    print('Length of unmatched fps: {0}'.format(len(unmatched_fp)))
                    self.plot(year, ensemble_member, matches, unmatched_fp, plot_mode, save)

        numpy_results_manager.save(self.cat_results_key('cat_data', years, ensemble_members), total_cat_data)
        numpy_results_manager.save(self.cat_results_key('are_hurr', years, ensemble_members), total_are_hurricanes)

        return total_cat_data, total_are_hurricanes

    def lon_filter(self, total_cat_data, total_are_hurricanes):
        i = SCATTER_ATTRS['lon']['index']
        mask = total_cat_data[:, i] > 260
        return total_cat_data[mask], total_are_hurricanes[mask]

    def apply_all_cutoffs(self, total_cat_data, total_are_hurricanes):
        total_cat_data, total_are_hurricanes = self.lon_filter(total_cat_data, total_are_hurricanes)
        hf = total_cat_data[total_are_hurricanes].copy()
        nhf = total_cat_data[~total_are_hurricanes].copy()

        t850_cutoff = 287
        t995_cutoff = 297
        vort_cutoff = 0.0003 # 0.00035 might be better.

        hf = hf[hf[:, 5] > t850_cutoff]
        hf = hf[hf[:, 4] > t995_cutoff]
        hf = hf[hf[:, 0] > vort_cutoff]

        nhf = nhf[nhf[:, 5] > t850_cutoff]
        nhf = nhf[nhf[:, 4] > t995_cutoff]
        nhf = nhf[nhf[:, 0] > vort_cutoff]

        plt.clf();
        ci1 = 0
        ci2 = 1
        plt.plot(nhf[:, ci1], nhf[:, ci2], 'b+', zorder=1);
        plt.plot(hf[:, ci1], hf[:, ci2], 'ro', zorder=0)

        return hf, nhf

    def plot_total_cat_data(self, total_cat_data, total_are_hurricanes, var1, var2):
        i1 = SCATTER_ATTRS[var1]['index']
        i2 = SCATTER_ATTRS[var2]['index']

        plt.xlabel(var1)
        plt.ylabel(var2)

        plt.plot(total_cat_data[:, i1][~total_are_hurricanes], 
                 total_cat_data[:, i2][~total_are_hurricanes], 'b+', zorder=3)
        plt.plot(total_cat_data[:, i1][total_are_hurricanes], 
                 total_cat_data[:, i2][total_are_hurricanes], 'ro', zorder=2)

    def run_individual_analysis(self, year, ensemble_member, plot_mode=None, save=False):
        results_manager = self.results_manager
        if year not in self.all_best_tracks:
            ibdata = IbtracsData(verbose=False)
            best_tracks = ibdata.load_ibtracks_year(year)
            self.all_best_tracks[year] = best_tracks

            total_hurricane_count = 0
            for best_track in best_tracks:
                for cls in best_track.cls:
                    if cls == 'HU':
                        total_hurricane_count += 1
            self.hurricanes_in_year[year] = total_hurricane_count

        best_tracks = self.all_best_tracks[year]

        cyclones = results_manager.get_result(year, ensemble_member, 'cyclones')
        matches, unmatched = matching.match_best_tracks_to_cyclones(best_tracks, cyclones)

        return cyclones, matches, unmatched

    def _make_cat_data_row(self, year, ensemble_member, date, cyclone, variables):
        cat_data_row = []
        for variable in SCATTER_ATTRS.key():
            attr = SCATTER_ATTRS[variable]
            x = get_cyclone_attr(cyclone, attr, date)
            cat_data_row.append(x)
        cat_data_row.append(year)
        cat_data_row.append(ensemble_member)
        return cat_data_row

    def build_cat_data(self, year, ensemble_member, unmatched_sample_size=None):
        cyclones, matches, unmatched = self.run_individual_analysis(year, ensemble_member)
        cat_data = []
        are_hurricanes = []

        if unmatched_sample_size:
            unmatched_samples = random.sample(unmatched, min(unmatched_sample_size, len(unmatched)))
        else:
            unmatched_samples = unmatched

        for cyclone in unmatched_samples:
            for date in cyclone.dates:
                if cyclone.pmins[date]:
                    cat_data.append(self._make_cat_data_row(year, ensemble_member, date, cyclone))
                    are_hurricanes.append(False)

        added_dates = []
        for match in matches:
            best_track = match.best_track
            cyclone = match.cyclone

            for date, cls in zip(best_track.dates, best_track.cls):
                if date in cyclone.dates and cyclone.pmins[date]:
                    added_dates.append(date)
                    if cls == 'HU':
                        cat_data.append(self._make_cat_data_row(year, ensemble_member, date, cyclone))
                        are_hurricanes.append(True)
                    else:
                        cat_data.append(self._make_cat_data_row(year, ensemble_member, date, cyclone))
                        are_hurricanes.append(False)

            for date in cyclone.dates:
                if date not in added_dates and cyclone.pmins[date]:
                    cat_data.append(self._make_cat_data_row(year, ensemble_member, date, cyclone))
                    are_hurricanes.append(False)

        return np.array(cat_data), np.array(are_hurricanes)

    def run_individual_categorisation_analysis(self, year, ensemble_member, matches, unmatched):
        results_manager = self.results_manager
        cyclones = results_manager.get_result(year, ensemble_member, 'cyclones')
        cutoff_cat = self.cutoff_cat

        for cyclone in cyclones:
            cutoff_cat.categorise(cyclone)

        cat_matchup = Counter()
        # Run through all matches and work out whether the current categorisation produced:
        # * False Positive (fp)
        # * False Negative (fn)
        # * True Positive (tp)
        # * True Negative (tn)
        for match in matches:
            cyclone = match.cyclone
            cyclone.cat_matches = OrderedDict()
            for cls, date in zip(match.best_track.cls, match.best_track.dates):
                if date in cyclone.hurricane_cat:
                    if cyclone.hurricane_cat[date] and cls != 'HU':
                        key = 'fp'
                    elif not cyclone.hurricane_cat[date] and cls == 'HU':
                        key = 'fn'
                    elif cyclone.hurricane_cat[date] and cls == 'HU':
                        key = 'tp'
                    else:
                        key = 'tn'
                    cyclone.cat_matches[date] = key
                    cat_matchup[key] += 1

        # Any hurricanes in the unmatched cyclones are false positives,
        # everything else are true negatives.
        unmatched_fp = []
        for cyclone in unmatched:
            for date in cyclone.dates:
                if cyclone.hurricane_cat[date]:
                    cat_matchup['unmatched_fp'] += 1
                    if cyclone not in unmatched_fp:
                        unmatched_fp.append(cyclone)
                else:
                    cat_matchup['unmatched_tn'] += 1

        print(cat_matchup)

        total_hurricane_count = self.hurricanes_in_year[year]
        print('Total hurricanes: {0}'.format(total_hurricane_count))

        # TODO: this calc is not working, can give -ve number for e.g. 2003, 1
        # Not sure above is still true, think fixing double counting bug may have fixed.
        # Still giving -ve numbers.
        missed_huricanes = total_hurricane_count - cat_matchup['tp'] - cat_matchup['fn']
        print('Missed hurricanes: {0}'.format(missed_huricanes))

        cat_matchup['missed'] = missed_huricanes
        print('Matchup score {0}'.format(score_matchup(cat_matchup)))

        return cat_matchup, unmatched_fp

    def gen_plotting_scatter_data(self, matches, unmatched, var1, var2):
        plotted_dates = []
        ps = {'unmatched': {'xs': [], 'ys': []},
              'hu': {'xs': [], 'ys': []},
              'ts': {'xs': [], 'ys': []},
              'no': {'xs': [], 'ys': []}}

        attr1 = SCATTER_ATTRS[var1]
        attr2 = SCATTER_ATTRS[var2]

        for cyclone in unmatched:
            for date in cyclone.dates:
                if cyclone.pmins[date]:
                    x = get_cyclone_attr(cyclone, attr1, date)
                    y = get_cyclone_attr(cyclone, attr2, date)
                    ps['unmatched']['xs'].append(x)
                    ps['unmatched']['ys'].append(y)

        for match in matches:
            best_track = match.best_track
            cyclone = match.cyclone

            for date, cls in zip(best_track.dates, best_track.cls):
                if date in cyclone.dates and cyclone.pmins[date]:
                    plotted_dates.append(date)
                    if cls == 'HU':
                        ps['hu']['xs'].append(get_cyclone_attr(cyclone, attr1, date))
                        ps['hu']['ys'].append(get_cyclone_attr(cyclone, attr2, date))
                    else:
                        ps['ts']['xs'].append(get_cyclone_attr(cyclone, attr1, date))
                        ps['ts']['ys'].append(get_cyclone_attr(cyclone, attr2, date))

            for date in cyclone.dates:
                if date not in plotted_dates and cyclone.pmins[date]:
                    ps['no']['xs'].append(get_cyclone_attr(cyclone, attr1, date))
                    ps['no']['ys'].append(get_cyclone_attr(cyclone, attr2, date))

        return ps

    def gen_plotting_error_data(self, matches, unmatched, var1, var2):
        plotted_dates = []
        ps = {'fp': {'xs': [], 'ys': []},
              'fn': {'xs': [], 'ys': []},
              'tp': {'xs': [], 'ys': []},
              'tn': {'xs': [], 'ys': []},
              'un': {'xs': [], 'ys': []}}

        attr1 = SCATTER_ATTRS[var1]
        attr2 = SCATTER_ATTRS[var2]

        for date in cyclone.dates:
            xs = get_cyclone_attr(cyclone, attr1, date)
            ys = get_cyclone_attr(cyclone, attr2, date)
            if date in cyclone.cat_matches:
                ps[cyclone.cat_matches[date]]['xs'].append(xs)
                ps[cyclone.cat_matches[date]]['ys'].append(ys)
            else:
                ps['un']['xs'].append(xs)
                ps['un']['ys'].append(ys)
        return ps

    def plot_scatters(self, years, ensemble_members, var1='vort', var2='pmin'):
        for year in years:
            for ensemble_member in ensemble_members:
                self.plot_scatter(year, ensemble_member, var1=var1, var2=var2)

    def plot_scatter(self, year, ensemble_member, matches=None, unmatched=None, var1='vort', var2='pmin'):
        if not matches or not unmatched:
            matches, unmatched = self.run_individual_analysis(year, ensemble_member)

        key = 'scatter_{0}_{1}'.format(var1, var2)
        try:
            ps = self.plot_results_manager.get_result(year, ensemble_member, key)
        except ResultNotFound:
            ps = self.gen_plotting_scatter_data(matches, unmatched, var1, var2)
            self.plot_results_manager.add_result(year, ensemble_member, key, ps)
            self.plot_results_manager.save()
        plotting.plot_2d_scatter(ps, var1, var2)

    def plot_error(self, year, ensemble_member, matches=None, unmatched=None, var1='vort', var2='pmin'):
        if not matches or not unmatched:
            matches, unmatched = self.run_individual_analysis(year, ensemble_member)

        key = 'error_{0}_{1}'.format(var1, var2)
        try:
            ps = self.plot_results_manager.get_result(year, ensemble_member, key)
        except ResultNotFound:
            ps = self.gen_plotting_error_data(matches, unmatched, var1, var2)
            self.plot_results_manager.add_result(year, ensemble_member, key, ps)
            self.plot_results_manager.save()

        plotting.plot_2d_error_scatter(ps, var1, var2)

    def plot(self, year, ensemble_member, matches, unmatched, plot_mode, save):
        output_path = os.path.join(settings.OUTPUT_DIR, 'hurr_scatter_plots')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plot_variables = (
            'pmin', 
            'pambdiff', 
            'max_windspeed',
            't995', 
            't850', 
            't_anom',
            'mindist',
            'max_windspeed_dist',
            'max_windspeed_dir',
            'lon',
            'lat',
            )

        var1 = 'vort'

        if plot_mode in ('scatter', 'both'):
            title = 'scatter {0}-{1}'.format(year, ensemble_member)
            plt.figure(title)
            plt.clf()
            plt.title(title)
            for i, var2 in enumerate(plot_variables):
                plt.subplot(3, 4, i + 1)
                self.plot_scatter(year, ensemble_member, matches, unmatched, var1, var2)
            if save:
                plt.savefig(os.path.join(output_path, '{0}.png'.format(title)))

        if plot_mode in ('error', 'both'):
            title = 'error scatter {0}-{1}'.format(year, ensemble_member)
            plt.figure(title)
            plt.clf()
            plt.title(title)
            for i, var2 in enumerate(plot_variables):
                plt.subplot(3, 4, i + 1)
                self.plot_error(year, ensemble_member, matches, unmatched, var1, var2)
            if save:
                plt.savefig(os.path.join(output_path, '{0}.png'.format(title)))


def run_ensemble_analysis(stormtracks_analysis, year):
    '''Performs a full enesmble analysis on the given year

    Searches through and tries to match all tracks across ensemble members **without** using
    any best tracks info.'''
    stormtracks_analysis.set_year(year)
    stormtracks_analysis.run_ensemble_matches_analysis(56)


def run_tracking_stats_analysis(stormtracks_analysis, year):
    '''Runs a complete tracking analysis, comparing the performance of each configuration option

    Compares performance in a variety of ways, e.g. within pressure level or just scale 1.'''

    stormtracks_analysis.set_year(year)

    log = stormtracks_analysis.log
    log.info('Running tracking stats analysis for year {0}'.format(year))
    include_extra_scales = False

    for sort_col in SORT_COLS.keys():
        if sort_col in ['overlap', 'cumdist']:
            continue

        log.info('Run analysis on col {0}'.format(sort_col))

        log.info('Run full analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col)

        log.info('Run 995 analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   active_configs={'pressure_level': 995})
        log.info('Run 850 analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   active_configs={'pressure_level': 850})

        log.info('Run scale 1 analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   active_configs={'scale': 1})
        log.info('Run scale 2 analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   active_configs={'scale': 2})
        log.info('Run scale 3 analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   active_configs={'scale': 3})
        if include_extra_scales:
            log.info('Run scale 4 analysis\n')
            stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                       active_configs={'scale': 4})
            log.info('Run scale 5 analysis\n')
            stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                       active_configs={'scale': 5})

    log.info('Run scale 1 wld\n')
    stormtracks_analysis.run_wld_analysis(active_configs={'scale': 1})

    log.info('Run scale 2 wld\n')
    stormtracks_analysis.run_wld_analysis(active_configs={'scale': 2})

    log.info('Run scale 3 wld\n')
    stormtracks_analysis.run_wld_analysis(active_configs={'scale': 3})

    if include_extra_scales:
        log.info('Run scale 4 wld\n')
        stormtracks_analysis.run_wld_analysis(active_configs={'scale': 4})

        log.info('Run scale 5 wld\n')
        stormtracks_analysis.run_wld_analysis(active_configs={'scale': 5})


def run_field_collection(stormtracks_analysis, year, num_ensemble_members=56):
    stormtracks_analysis.set_year(year)
    for ensemble_member in range(num_ensemble_members):
        stormtracks_analysis.run_individual_field_collection(ensemble_member)


def run_wilma_katrina_analysis(show_plots=False, num_ensemble_members=56):
    c20data = C20Data(year, verbose=False)
    ibdata = IbtracsData(verbose=False)
    wilma_bt, katrina_bt = ibdata.load_wilma_katrina()
    # plt.plot(wilma_bt.dates, wilma_bt.pressures)
    if show_plots:
        plt.plot(katrina_bt.pressures * 100)

    results_manager = StormtracksResultsManager('pyro_tracking_analysis')

    cyclones = []
    for i in range(num_ensemble_members):
        start = time.time()
        print(i)

        gms = results_manager.get_result(2005,
                                         i,
                                         'good_matches-scale:3;pl:850;tracker:nearest_neighbour')
        for gm in gms:
            if gm.best_track.name == katrina_bt.name:
                wilma_match = gm
                break

        d = OrderedDict()

        for date in gm.vort_track.dates:
            d[date] = [gm.vort_track]
        field_finder = FieldFinder(c20data, d, i)
        field_finder.collect_fields()
        cyclone_track = field_finder.cyclone_tracks.values()[0]
        cyclones.append(cyclone_track)

        if show_plots:
            pmin_values = []
            for pmin in cyclone_track.pmins.values():
                if pmin:
                    pmin_values.append(pmin[0])
                else:
                    pmin_values.append(104000)
                plt.plot(pmin_values)

    if show_plots:
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-a', '--analysis', default='ensemble')
    parser.add_argument('-s', '--start-year', type=int, default=2005)
    parser.add_argument('-e', '--end-year', type=int, default=2005)
    args = parser.parse_args()

    years = range(args.start_year, args.end_year + 1)
    if args.analysis == 'scatter_plots':
        for year in years:
            print(year)
            run_scatter_plot_output(year)
        sys.exit(0)

    stormtracks_analysis = StormtracksAnalysis(years[0])
    if args.analysis == 'ensemble':
        for year in years:
            run_ensemble_analysis(stormtracks_analysis, year)
    elif args.analysis == 'stats':
        for year in years:
            run_tracking_stats_analysis(stormtracks_analysis, year)
    elif args.analysis == 'collection':
        for year in years:
            run_field_collection(stormtracks_analysis, year)
    elif args.analysis == 'wilma_katrina':
        for year in years:
            run_wilma_katrina_analysis(year)
    else:
        raise Exception('One of ensemble or stats should be chosen')

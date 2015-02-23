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
import classification
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

CAL_YEARS = range(1990, 2009, 2)
VAL_YEARS = range(1991, 2010, 2)
ENSEMBLE_RANGE = range(56)


class StormtracksAnalysis(object):
    """Provides a variety of ways of analysing tracking performance

    Makes extensive use of its results_manager to load/save results.
    Used by the pyro code to farm out jobs across the cluster.
    To a large extent replaces the manual analysis steps.
    :param year: year on which to run analysis
    :param is_setup_logging: whether to setup logging (useful in ipython)
    """
    def __init__(self, year, is_setup_logging=False):
        self.set_year(year)
        self.setup_analysis()

        if is_setup_logging:
            filename = 'analysis.log'
            self.log = setup_logging('analysis', filename=filename, console_level_str='INFO')
        else:
            self.log = get_logger('analysis', console_logging=False)

    def set_year(self, year):
        """Sets the year, loading best_tracks and setting up results_manager appropriately"""
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

        # Set up a c20 object with the specified config options.
        # N.B. for tracking only vorticity (which uses u, v fields) is needed.
        c20data = C20Data(self.year, verbose=False,
                          pressure_level=config['pressure_level'],
                          upscaling=upscaling,
                          scale_factor=config['scale'],
                          fields=['u', 'v'])

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

        return key0, sum0, key1, sum1, sum_draw

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

        return cross_ensemble_results

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


class ClassificationAnalysis(object):
    def __init__(self):
        self.results_manager = StormtracksResultsManager('pyro_field_collection_analysis')
        self.plot_results_manager = StormtracksResultsManager('plot_results')
        self.all_best_tracks = {}
        self.hurricanes_in_year = {}
        self.cal_cd = None
        self.val_cd = None
        self.classifiers = None

    def get_trained_classifiers(self):
        if not self.cal_cd:
            self.cal_cd = self.load_cal_classification_data()
        if not self.val_cd:
            self.val_cd = self.load_val_classification_data()

        if self.classifiers:
            return self.cal_cd, self.val_cd, self.classifiers

        cc = classification.CutoffClassifier()
        ldc = classification.LDAClassifier()
        qdc = classification.QDAClassifier()
        sgdc = classification.SGDClassifier()
        qdc_chain = classification.QDAClassifier()
        cc_chain = classification.CutoffClassifier()
        chain_qdc_cc = classification.ClassifierChain([qdc_chain, cc_chain])

        sgdc_chain = classification.SGDClassifier()
        cc_chain2 = classification.CutoffClassifier()
        chain_sgdc_cc = classification.ClassifierChain([sgdc_chain, cc_chain2])

        classifiers = ((cc, 'cc_best', 'g^', 'Threshold'),
                       (ldc, 'ldc_best', 'm+', 'LDA'),
                       (qdc, 'qdc_best', 'bx', 'QDA'),
                       (sgdc, 'sgdc_best', 'ro', 'SGD'),
                       # (chain_qdc_cc, 'chain_qdc_cc_best', 'cs', 'QDC/Thresh.'),
                       # (chain_sgdc_cc, 'chain_sgdc_cc_best', 'cs', 'Combined SGDC/Thresh.'),
                       )

        for i, (classifier, settings, fmt, name) in enumerate(classifiers):
            cat_settings = classifier.load(settings)
            classifier.train(self.cal_cd, **cat_settings)
            # classifier.predict(cal_cd, plot='all', fig=(i + 1) * 10, fmt=fmt)
            # classifier.predict(val_cd, plot='all', fig=(i + 1) * 10 + 2, fmt=fmt)

        self.classifiers = classifiers
        return self.cal_cd, self.val_cd, self.classifiers

    def run_yearly_analysis(self, classifier, start_year=1890, end_year=2010):
        ems = range(56)
        ib_hurrs = []
        ib_pdis = []
        cla_hurrs = []
        cla_pdis = []

        for start_year in range(start_year, end_year, 10):
            print(start_year)
            years = range(start_year, start_year + 10)
            cla_data = self.load_classification_data('{0}s'.format(start_year), years, ems)

            classifier.predict(cla_data)
            pred_hurr = cla_data.data[classifier.are_hurr_pred]

            for year in years:
                cla_hurr = []
                cla_pdi = []
                ib_hurr = self.get_total_hurrs([year])
                ib_hurrs.append(ib_hurr)

                if year not in self.all_best_tracks:
                    self.load_ibtracs_year(year)

                ib_pdi = 0
                for bt in self.all_best_tracks[year]:
                    for cls, ws in zip(bt.cls, bt.winds):
                        if cls == 'HU':
                            ib_pdi += ws ** 3

                ib_pdis.append(ib_pdi)

                year_mask = (pred_hurr[:, 15] == year)
                for em in ems:
                    em_mask = pred_hurr[year_mask][:, 16] == em
                    em_hurr = (em_mask).sum()
                    mws_index = classification.SCATTER_ATTRS['maxwindspeed']['index']
                    em_pdi = (pred_hurr[year_mask][em_mask][:, mws_index] ** 3).sum()
                    cla_hurr.append(em_hurr)
                    cla_pdi.append(em_pdi)

                cla_hurrs.append(cla_hurr)
                cla_pdis.append(cla_pdi)

            del cla_data

        return np.array(ib_hurrs), np.array(ib_pdis), np.array(cla_hurrs), np.array(cla_pdis)

    def cat_results_key(self, name, years, ensemble_members):
        years_str = '-'.join(map(str, years))
        em_str = '-'.join(map(str, ensemble_members))
        return '{0}_{1}_{2}'.format(name, years_str, em_str)

    def load_ibtracs_year(self, year):
        ibdata = IbtracsData(verbose=False)
        best_tracks = ibdata.load_ibtracks_year(year)
        self.all_best_tracks[year] = best_tracks

        hurricanes_in_year = 0
        for best_track in best_tracks:
            for cls in best_track.cls:
                if cls == 'HU':
                    hurricanes_in_year += 1
        self.hurricanes_in_year[year] = hurricanes_in_year

    def miss_count(self, years, num_ensemble_members, hurr_counts):
        total_hurrs = self.get_total_hurrs(years)
        expexted_hurrs = total_hurrs * num_ensemble_members

        all_tracked_hurricanes = hurr_counts[:, 2].sum()

        return expexted_hurrs - all_tracked_hurricanes

    def get_total_hurrs(self, years):
        total_hurricanes = 0

        for year in years:
            if year not in self.all_best_tracks:
                self.load_ibtracs_year(year)
            total_hurricanes += self.hurricanes_in_year[year]
        return total_hurricanes

    def run_categorisation_analysis(self, years, ensemble_members=(0, ),
                                    plot_mode=None, save=False):
        total_classification_data = None
        total_are_hurricanes = None
        total_dates = None
        total_hurr_counts = []
        numpy_res_man = StormtracksNumpyResultsManager('classification_data')
        total_hurr_count = 0
        for year in years:
            for ensemble_member in ensemble_members:
                print('{0}-{1}'.format(year, ensemble_member))

                # matches, unmatched = self.run_individual_analysis(year, ensemble_member,
                #                                                   plot_mode, save)
                classification_data, are_hurricanes, dates, hurr_count, double_count =\
                    self.build_classification_data(year, ensemble_member)
                if double_count > 5:
                    print('Hi double count for year/em: {0}, {1}'.format(year, ensemble_member))

                hurr_counts = np.array((year, ensemble_member, hurr_count, double_count))

                total_hurr_count += hurr_count
                if total_classification_data is not None:
                    total_classification_data = np.concatenate((total_classification_data,
                                                                classification_data))
                else:
                    total_classification_data = classification_data

                if total_are_hurricanes is not None:
                    total_are_hurricanes = np.concatenate((total_are_hurricanes, are_hurricanes))
                else:
                    total_are_hurricanes = are_hurricanes

                if total_dates is not None:
                    total_dates = np.concatenate((total_dates, dates))
                else:
                    total_dates = dates

                total_hurr_counts.append(hurr_counts)

        total_hurr_counts = np.array(total_hurr_counts)
        numpy_res_man.save(self.cat_results_key('classification_data', years, ensemble_members),
                           total_classification_data)
        numpy_res_man.save(self.cat_results_key('are_hurr', years, ensemble_members),
                           total_are_hurricanes)
        numpy_res_man.save(self.cat_results_key('dates', years, ensemble_members),
                           total_dates)
        numpy_res_man.save(self.cat_results_key('hurr_counts', years, ensemble_members),
                           total_hurr_counts)

        miss_count = self.miss_count(years, len(ensemble_members), total_hurr_counts)

        return classification.ClassificationData('calcd', total_classification_data,
                                                 total_are_hurricanes, total_dates,
                                                 total_hurr_counts, miss_count)

    def load_classification_data(self, name, years, ensemble_members, should_lon_filter=False):
        numpy_res_man = StormtracksNumpyResultsManager('classification_data')

        total_classification_data = numpy_res_man.load(self.cat_results_key('classification_data',
                                                                            years,
                                                                            ensemble_members))
        total_are_hurricanes = numpy_res_man.load(self.cat_results_key('are_hurr',
                                                                       years,
                                                                       ensemble_members))
        total_dates = numpy_res_man.load(self.cat_results_key('dates',
                                                              years,
                                                              ensemble_members))
        total_hurr_counts = numpy_res_man.load(self.cat_results_key('hurr_counts',
                                                                    years,
                                                                    ensemble_members))

        miss_count = self.miss_count(years, len(ensemble_members), total_hurr_counts)

        if should_lon_filter:
            total_classification_data, total_are_hurricanes, total_dates = \
                self.lon_filter(total_classification_data, total_are_hurricanes, total_dates)

        return classification.ClassificationData(name, total_classification_data,
                                                 total_are_hurricanes, total_dates,
                                                 total_hurr_counts, miss_count)

    def load_cal_classification_data(self):
        classification_data = self.load_classification_data('Calibration',
                                                            CAL_YEARS,
                                                            ENSEMBLE_RANGE)
        return classification_data

    def load_val_classification_data(self):
        classification_data = self.load_classification_data('Validation', VAL_YEARS, ENSEMBLE_RANGE)
        return classification_data

    def optimize_cutoff_cat(self, classification_data, are_hurr, dates):
        self.cutoff_cat.best_so_far()
        vort_lo_dist = 0.00001
        vort_lo_start = self.cutoff_cat.cutoffs['vort_lo']

        t995_lo_dist = 0.1
        t995_lo_start = self.cutoff_cat.cutoffs['t995_lo']

        t850_lo_dist = 0.1
        t850_lo_start = self.cutoff_cat.cutoffs['t850_lo']

        maxwindspeed_lo_dist = 0.2
        maxwindspeed_lo_start = self.cutoff_cat.cutoffs['maxwindspeed_lo']

        pambdiff_lo_dist = 0.2
        pambdiff_lo_start = self.cutoff_cat.cutoffs['pambdiff_lo']

        lowest_score = 1e99

        n = 3
        for vort_lo in np.arange(vort_lo_start - vort_lo_dist * n,
                                 vort_lo_start + vort_lo_dist * n,
                                 vort_lo_dist):
            self.cutoff_cat.cutoffs['vort_lo'] = vort_lo
            # for t995_lo in np.arange(t995_lo_start - t995_lo_dist * n,
            #                          t995_lo_start + t995_lo_dist * n,
            #                          t995_lo_dist):
            #     self.cutoff_cat.cutoffs['t995_lo'] = t995_lo
            for maxwindspeed_lo in np.arange(maxwindspeed_lo_start - maxwindspeed_lo_dist * n,
                                             maxwindspeed_lo_start + maxwindspeed_lo_dist * n,
                                             maxwindspeed_lo_dist):
                self.cutoff_cat.cutoffs['maxwindspeed_lo'] = maxwindspeed_lo
            #     for t850_lo in np.arange(t850_lo_start - t850_lo_dist * n,
            #                              t850_lo_start + t850_lo_dist * n,
            #                              t850_lo_dist):
            #         self.cutoff_cat.cutoffs['t850_lo'] = t850_lo
                for pambdiff_lo in np.arange(pambdiff_lo_start - pambdiff_lo_dist * n,
                                             pambdiff_lo_start + pambdiff_lo_dist * n,
                                             pambdiff_lo_dist):
                    self.cutoff_cat.cutoffs['pambdiff_lo'] = pambdiff_lo
                    score = self.cutoff_cat.predict(classification_data, are_hurr)
                    if score < lowest_score:
                        print('New low score: {0}'.format(score))
                        lowest_score = score
                        print(self.cutoff_cat.cutoffs)

    def lon_filter(self, total_classification_data, total_are_hurricanes, total_dates):
        i = classification.SCATTER_ATTRS['lon']['index']
        mask = total_classification_data[:, i] > 260
        # return total_classification_data[mask], total_are_hurricanes[mask]
        return total_classification_data[mask], total_are_hurricanes[mask], total_dates[mask]

    def apply_all_cutoffs(self, total_classification_data, total_are_hurricanes):
        total_classification_data, total_are_hurricanes = self.lon_filter(total_classification_data,
                                                                          total_are_hurricanes)
        hf = total_classification_data[total_are_hurricanes].copy()
        nhf = total_classification_data[~total_are_hurricanes].copy()

        t850_cutoff = 287
        t995_cutoff = 297
        vort_cutoff = 0.0003  # 0.00035 might be better.

        hf = hf[hf[:, 5] > t850_cutoff]
        hf = hf[hf[:, 4] > t995_cutoff]
        hf = hf[hf[:, 0] > vort_cutoff]

        nhf = nhf[nhf[:, 5] > t850_cutoff]
        nhf = nhf[nhf[:, 4] > t995_cutoff]
        nhf = nhf[nhf[:, 0] > vort_cutoff]

        plt.clf()
        ci1 = 0
        ci2 = 1
        plt.plot(nhf[:, ci1], nhf[:, ci2], 'bx', zorder=1)
        plt.plot(hf[:, ci1], hf[:, ci2], 'ko', zorder=0)

        return hf, nhf

    def plot_total_classification_data(self, total_classification_data, total_are_hurricanes,
                                       var1, var2, fig=1):
        plt.figure(fig)
        plt.clf()
        i1 = classification.SCATTER_ATTRS[var1]['index']
        i2 = classification.SCATTER_ATTRS[var2]['index']

        plt.xlabel(var1)
        plt.ylabel(var2)

        plt.plot(total_classification_data[:, i1][~total_are_hurricanes],
                 total_classification_data[:, i2][~total_are_hurricanes], 'bx', zorder=3)
        plt.plot(total_classification_data[:, i1][total_are_hurricanes],
                 total_classification_data[:, i2][total_are_hurricanes], 'ko', zorder=2)

    def run_individual_cla_analysis(self, year, ensemble_member):
        results_manager = self.results_manager
        if year not in self.all_best_tracks:
            self.load_ibtracs_year(year)

        best_tracks = self.all_best_tracks[year]

        cyclones = results_manager.get_result(year, ensemble_member, 'cyclones')
        matches, unmatched = matching.match_best_tracks_to_cyclones(best_tracks, cyclones)

        return cyclones, matches, unmatched

    def calc_total_hurr(self, hurr_counts):
        num_ensemble_members = len(set(hurr_counts[:, 1]))
        all_hurricanes = hurr_counts[:, 2].sum()
        return 1. * all_hurricanes / num_ensemble_members

    def _make_classification_data_row(self, year, ensemble_member, date, cyclone):
        classification_data_row = []
        for variable in classification.SCATTER_ATTRS.keys():
            attr = classification.SCATTER_ATTRS[variable]
            x = classification.get_cyclone_attr(cyclone, attr, date)
            classification_data_row.append(x)
        classification_data_row.append(year)
        classification_data_row.append(ensemble_member)
        return classification_data_row

    def build_classification_data(self, year, ensemble_member, unmatched_sample_size=None):
        cyclones, matches, unmatched = self.run_individual_cla_analysis(year, ensemble_member)
        dates = []
        class_data = []
        are_hurricanes = []

        if unmatched_sample_size:
            unmatched_samples = random.sample(unmatched, min(unmatched_sample_size, len(unmatched)))
        else:
            unmatched_samples = unmatched

        for cyclone in unmatched_samples:
            for date in cyclone.dates:
                if cyclone.pmins[date]:
                    class_data.append(self._make_classification_data_row(year,
                                                                         ensemble_member,
                                                                         date, cyclone))
                    dates.append(date)
                    are_hurricanes.append(False)

        added_dates = []
        # Stops a double count of matched hurrs.
        matched_best_tracks = Counter()
        for match in matches:
            best_track = match.best_track
            cyclone = match.cyclone

            for date, cls in zip(best_track.dates, best_track.cls):
                if date in cyclone.dates and cyclone.pmins[date]:
                    added_dates.append(date)
                    if cls == 'HU':
                        matched_best_tracks[(best_track.name, date)] += 1
                        class_data.append(self._make_classification_data_row(year,
                                                                             ensemble_member,
                                                                             date, cyclone))
                        dates.append(date)
                        are_hurricanes.append(True)
                    else:
                        class_data.append(self._make_classification_data_row(year,
                                                                             ensemble_member,
                                                                             date, cyclone))
                        dates.append(date)
                        are_hurricanes.append(False)

            for date in cyclone.dates:
                if date not in added_dates and cyclone.pmins[date]:
                    class_data.append(self._make_classification_data_row(year,
                                                                         ensemble_member,
                                                                         date, cyclone))
                    dates.append(date)
                    are_hurricanes.append(False)

        double_count = sum(matched_best_tracks.values()) - len(matched_best_tracks)
        return np.array(class_data), np.array(are_hurricanes), np.array(dates),\
            len(matched_best_tracks), double_count

    def gen_plotting_scatter_data(self, matches, unmatched, var1, var2):
        plotted_dates = []
        ps = {'unmatched': {'xs': [], 'ys': []},
              'hu': {'xs': [], 'ys': []},
              'ts': {'xs': [], 'ys': []},
              'no': {'xs': [], 'ys': []}}

        attr1 = classification.SCATTER_ATTRS[var1]
        attr2 = classification.SCATTER_ATTRS[var2]

        for cyclone in unmatched:
            for date in cyclone.dates:
                if cyclone.pmins[date]:
                    x = classification.get_cyclone_attr(cyclone, attr1, date)
                    y = classification.get_cyclone_attr(cyclone, attr2, date)
                    ps['unmatched']['xs'].append(x)
                    ps['unmatched']['ys'].append(y)

        for match in matches:
            best_track = match.best_track
            cyclone = match.cyclone

            for date, cls in zip(best_track.dates, best_track.cls):
                if date in cyclone.dates and cyclone.pmins[date]:
                    plotted_dates.append(date)
                    if cls == 'HU':
                        ps['hu']['xs'].append(classification.get_cyclone_attr(cyclone, attr1, date))
                        ps['hu']['ys'].append(classification.get_cyclone_attr(cyclone, attr2, date))
                    else:
                        ps['ts']['xs'].append(classification.get_cyclone_attr(cyclone, attr1, date))
                        ps['ts']['ys'].append(classification.get_cyclone_attr(cyclone, attr2, date))

            for date in cyclone.dates:
                if date not in plotted_dates and cyclone.pmins[date]:
                    ps['no']['xs'].append(classification.get_cyclone_attr(cyclone, attr1, date))
                    ps['no']['ys'].append(classification.get_cyclone_attr(cyclone, attr2, date))

        return ps

    def gen_plotting_error_data(self, matches, unmatched, var1, var2):
        plotted_dates = []
        ps = {'fp': {'xs': [], 'ys': []},
              'fn': {'xs': [], 'ys': []},
              'tp': {'xs': [], 'ys': []},
              'tn': {'xs': [], 'ys': []},
              'un': {'xs': [], 'ys': []}}

        attr1 = classification.SCATTER_ATTRS[var1]
        attr2 = classification.SCATTER_ATTRS[var2]

        for date in cyclone.dates:
            xs = classification.get_cyclone_attr(cyclone, attr1, date)
            ys = classification.get_cyclone_attr(cyclone, attr2, date)
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

    def plot_scatter(self, year, ensemble_member, matches=None, unmatched=None,
                     var1='vort', var2='pmin'):
        if not matches or not unmatched:
            matches, unmatched = self.run_individual_cla_analysis(year, ensemble_member)

        key = 'scatter_{0}_{1}'.format(var1, var2)
        try:
            ps = self.plot_results_manager.get_result(year, ensemble_member, key)
        except ResultNotFound:
            ps = self.gen_plotting_scatter_data(matches, unmatched, var1, var2)
            self.plot_results_manager.add_result(year, ensemble_member, key, ps)
            self.plot_results_manager.save()
        plotting.plot_2d_scatter(ps, var1, var2)

    def plot_error(self, year, ensemble_member, matches=None, unmatched=None,
                   var1='vort', var2='pmin'):
        if not matches or not unmatched:
            matches, unmatched = self.run_individual_cla_analysis(year, ensemble_member)

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


def run_ensemble_analysis(stormtracks_analysis, year, num_ensemble_members):
    '''Performs a full enesmble analysis on the given year

    Searches through and tries to match all tracks across ensemble members **without** using
    any best tracks info.'''
    stormtracks_analysis.set_year(year)
    stormtracks_analysis.run_ensemble_matches_analysis(num_ensemble_members)


def run_tracking_stats_analysis(stormtracks_analysis, year, num_ensemble_members=56):
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
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   num_ensemble_members=num_ensemble_members)

        log.info('Run 995 analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   active_configs={'pressure_level': 995},
                                                   num_ensemble_members=num_ensemble_members)
        log.info('Run 850 analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   active_configs={'pressure_level': 850},
                                                   num_ensemble_members=num_ensemble_members)

        log.info('Run scale 1 analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   active_configs={'scale': 1},
                                                   num_ensemble_members=num_ensemble_members)
        log.info('Run scale 2 analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   active_configs={'scale': 2},
                                                   num_ensemble_members=num_ensemble_members)
        log.info('Run scale 3 analysis\n')
        stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                   active_configs={'scale': 3},
                                                   num_ensemble_members=num_ensemble_members)
        if include_extra_scales:
            log.info('Run scale 4 analysis\n')
            stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                       active_configs={'scale': 4},
                                                       num_ensemble_members=num_ensemble_members)
            log.info('Run scale 5 analysis\n')
            stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                       active_configs={'scale': 5},
                                                       num_ensemble_members=num_ensemble_members)

    log.info('Run scale 1 wld\n')
    stormtracks_analysis.run_wld_analysis(active_configs={'scale': 1},
                                          num_ensemble_members=num_ensemble_members)

    log.info('Run scale 2 wld\n')
    stormtracks_analysis.run_wld_analysis(active_configs={'scale': 2},
                                          num_ensemble_members=num_ensemble_members)

    log.info('Run scale 3 wld\n')
    stormtracks_analysis.run_wld_analysis(active_configs={'scale': 3},
                                          num_ensemble_members=num_ensemble_members)

    if include_extra_scales:
        log.info('Run scale 4 wld\n')
        stormtracks_analysis.run_wld_analysis(active_configs={'scale': 4},
                                              num_ensemble_members=num_ensemble_members)

        log.info('Run scale 5 wld\n')
        stormtracks_analysis.run_wld_analysis(active_configs={'scale': 5},
                                              num_ensemble_members=num_ensemble_members)


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


def analyse_ibtracs_data(plot=True):
    '''Adds up a freq. distribution and a yearly total of hurricane-timesteps'''
    ibdata = IbtracsData(verbose=False)
    yearly_hurr_distribution = Counter()
    hurr_per_year = OrderedDict()
    for year in range(1890, 2010):
        print(year)
        best_tracks = ibdata.load_ibtracks_year(year)
        hurr_count = 0
        for best_track in best_tracks:
            for date, cls in zip(best_track.dates, best_track.cls):
                if cls == 'HU':
                    hurr_count += 1
                    day_of_year = date.timetuple().tm_yday
                    if date.year == year + 1:
                        # Takes into account leap years.
                        day_of_year += dt.datetime(year, 12, 31).timetuple().tm_yday
                    elif date.year != year:
                        raise Exception('{0} != {1}'.format(date.year, year))

                    yearly_hurr_distribution[day_of_year] += 1
        hurr_per_year[year] = hurr_count

    start_doy = dt.datetime(2001, 6, 1).timetuple().tm_yday
    end_doy = dt.datetime(2001, 12, 1).timetuple().tm_yday

    if plot:
        plt.figure(1)
        plt.title('Hurricane Distribution over the Year')
        plt.plot(yearly_hurr_distribution.keys(), yearly_hurr_distribution.values())
        plt.plot((start_doy, start_doy), (0, 250), 'k--')
        plt.plot((end_doy, end_doy), (0, 250), 'k--')
        plt.xlabel('Day of Year')
        plt.ylabel('Hurricane-timesteps')

        plt.figure(2)
        plt.title('Hurricanes per Year')
        plt.plot(hurr_per_year.keys(), hurr_per_year.values())
        plt.xlabel('Year')
        plt.ylabel('Hurricane-timesteps')

    return yearly_hurr_distribution, hurr_per_year


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-a', '--analysis', default='ensemble')
    parser.add_argument('-s', '--start-year', type=int, default=2005)
    parser.add_argument('-e', '--end-year', type=int, default=2005)
    parser.add_argument('-n', '--num-ensemble-members', type=int, default=56)
    args = parser.parse_args()

    years = range(args.start_year, args.end_year + 1)
    if args.analysis == 'scatter_plots':
        for year in years:
            print(year)
            run_scatter_plot_output(year)
        sys.exit(0)

    stormtracks_analysis = StormtracksAnalysis(years[0], True)
    if args.analysis == 'ensemble':
        for year in years:
            run_ensemble_analysis(stormtracks_analysis, year, args.num_ensemble_members)
    elif args.analysis == 'stats':
        for year in years:
            run_tracking_stats_analysis(stormtracks_analysis, year, args.num_ensemble_members)
    elif args.analysis == 'collection':
        for year in years:
            run_field_collection(stormtracks_analysis, year, args.num_ensemble_members)
    elif args.analysis == 'wilma_katrina':
        for year in years:
            run_wilma_katrina_analysis(year, args.num_ensemble_members)
    else:
        raise Exception('One of ensemble or stats should be chosen')

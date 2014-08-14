from collections import OrderedDict
from copy import copy

import numpy as np
import pylab as plt
try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.lda import LDA
except ImportError:
    print 'IMPORT mlpy/scikit failed: must be on UCL computer'

from utils.utils import geo_dist

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def calc_t_anom(cyclone, date):
    return cyclone.t850s[date] - cyclone.t995s[date]


def get_vort(cyclone, date):
    return cyclone.vortmax_track.vortmax_by_date[date].vort
    # return cyclone.get_vort(date)


def calc_ws_dist(cyclone, date):
    return geo_dist(cyclone.max_windspeed_positions[date], cyclone.get_vmax_pos(date))


def calc_ws_dir(cyclone, date):
    p1, p2 = (cyclone.max_windspeed_positions[date], cyclone.get_vmax_pos(date))
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    return np.arctan2(dy, dx)


def calc_lat(cyclone, date):
    pos = cyclone.get_vmax_pos(date)
    return pos[1]


def calc_lon(cyclone, date):
    pos = cyclone.get_vmax_pos(date)
    return pos[0]


def get_cyclone_attr(cyclone, attr, date):
    if attr['name'] != 'calc':
        return getattr(cyclone, attr['name'])[date]
    else:
        val = attr['calc'](cyclone, date)
        return val


SCATTER_ATTRS = OrderedDict([
    # Normally x.
    ('vort', {'name': 'calc', 'calc': get_vort, 'index': 0}),
    ('pmin', {'name': 'pmins', 'index': 1}),
    ('pambdiff', {'name': 'p_ambient_diffs', 'index': 2}),
    ('mindist', {'name': 'min_dists', 'index': 3}),
    ('t995', {'name': 't995s', 'index': 4}),
    ('t850', {'name': 't850s', 'index': 5}),
    ('tanom', {'name': 'calc', 'calc': calc_t_anom, 'index': 6}),
    ('maxwindspeed', {'name': 'max_windspeeds', 'index': 7}),
    ('maxwindspeeddist', {'name': 'calc', 'calc': calc_ws_dist, 'index': 8}),
    ('maxwindspeeddir', {'name': 'calc', 'calc': calc_ws_dir, 'index': 9}),
    ('cape', {'name': 'capes', 'index': 10}),
    ('pwat', {'name': 'pwats', 'index': 11}),
    ('rh995', {'name': 'rh995s', 'index': 12}),
    ('lon', {'name': 'calc', 'calc': calc_lon, 'index': 13}),
    ('lat', {'name': 'calc', 'calc': calc_lat, 'index': 14}),
])


class Categoriser(object):
    def __init__(self, missed_count):
        self.highest_score = 0
        self.missed_count = missed_count
        self.settings = OrderedDict()
        self.prev_fn = 0
        self.best_settings = None

        self.res = {}

    def chain(self, categoriser, **kwargs):
        self.prev_fn = categoriser.res['fn']
        self.try_cat(categoriser.cat_data[categoriser.are_hurr_pred],
                     categoriser.are_hurr_actual[categoriser.are_hurr_pred],
                     **kwargs)

    def try_cat(self, cat_data, are_hurr_actual, **kwargs):
        self.cat_data = cat_data
        self.are_hurr_actual = are_hurr_actual

        try:
            plot = kwargs.pop('plot')
        except KeyError:
            plot = None

        for k, v in kwargs.items():
            self.settings[k] = v

        self.are_hurr_pred = self.categorise(cat_data)
        self.compare(are_hurr_actual, self.are_hurr_pred)

        if plot == 'remaining':
            self.plot_remaining_actual('vort', 'pmin')
        elif plot == 'confusion':
            self.plot_confusion('vort', 'pmin')
        curr_score = self.print_score()

        plt.figure(0)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.xlabel('sensitivity')
        plt.ylabel('ppv')
        plt.plot(self.sensitivity, self.ppv, 'b+')

        return curr_score
    
    def plot_remaining_actual(self, var1, var2):
        plt.figure(self.fig)
        plt.clf()
        cd = self.cat_data[self.are_hurr_pred]
        h = self.are_hurr_actual[self.are_hurr_pred]

        i1 = SCATTER_ATTRS[var1]['index']
        i2 = SCATTER_ATTRS[var2]['index']

        plt.xlabel(var1)
        plt.ylabel(var2)

        plt.plot(cd[:, i1][~h], 
                 cd[:, i2][~h], 'bx', zorder=3)
        plt.plot(cd[:, i1][h], 
                 cd[:, i2][h], 'ko', zorder=2)
        return self.fig
    
    def plot_confusion(self, var1, var2, show_true_negs=False):
        plt.figure(self.fig + 1)
        plt.clf()

        plt.xlabel(var1)
        plt.ylabel(var2)

        i1 = SCATTER_ATTRS[var1]['index']
        i2 = SCATTER_ATTRS[var2]['index']

        cd = self.cat_data

        tp = self.are_hurr_pred & self.are_hurr_actual
        tn = ~self.are_hurr_pred & ~self.are_hurr_actual
        fp = self.are_hurr_pred & ~self.are_hurr_actual
        fn = ~self.are_hurr_pred & self.are_hurr_actual

        hs = ((tp, 'go', 1), 
              (fp, 'ro', 3),
              (fn, 'rx', 4),
              (tn, 'gx', 2))

        if not show_true_negs:
            hs = hs[:-1]

        for h, fmt, order in hs:
            plt.plot(cd[:, i1][h], cd[:, i2][h], fmt, zorder=order)

        return self.fig
    
    def compare(self, are_hurr_actual, are_hurr_pred):
        self.res['fn'] = self.missed_count + self.prev_fn
        self.res['fp'] = (~are_hurr_actual & are_hurr_pred).sum()
        self.res['fn'] += (are_hurr_actual & ~are_hurr_pred).sum()
        self.res['tp'] = (are_hurr_actual & are_hurr_pred).sum()
        self.res['tn'] = (~are_hurr_actual & ~are_hurr_pred).sum()
        return self.res

    def print_score(self):
        res = self.res
        self.curr_score = 1. * res['tp'] / (res['tp'] + res['fn'] + res['fp'])
        self.sensitivity = 1. * res['tp'] / (res['tp'] + res['fn'])
        self.ppv = 1. * res['tp'] / (res['tp'] + res['fp'])

        if self.curr_score >= self.highest_score:
            print('{0}score: {1}{2}, sens: {3}, ppv: {4}'.format(bcolors.FAIL, self.curr_score, bcolors.ENDC,
                                                                 self.sensitivity, self.ppv))
            self.highest_score = self.curr_score
            self.best_settings = self.settings
        else:
            print('score: {0}, sens: {1}, ppv: {2}'.format(self.curr_score, self.sensitivity, self.ppv))
        return self.curr_score

class CutoffCategoriser(Categoriser):
    def __init__(self, missed_count):
        super(CutoffCategoriser, self).__init__(missed_count)
        self.fig = 10
        self.best_so_far()

    def reset_settings(self):
        self.settings = OrderedDict()

    def best_so_far(self):
        self.settings = OrderedDict([('vort_lo', 0.000104), 
                                     # ('pwat_lo', 53), # possibly?
                                     ('t995_lo', 297.2), 
                                     ('t850_lo', 286.7), 
                                     ('maxwindspeed_lo', 16.1), 
                                     ('pambdiff_lo', 563.4)])

    def categorise(self, cat_data):
        are_hurr_pred = np.ones((len(cat_data),)).astype(bool)

        for cutoff in self.settings.keys():
            var, hilo = cutoff.split('_')
            index = SCATTER_ATTRS[var]['index']
            if hilo == 'lo':
                mask = cat_data[:, index] > self.settings[cutoff]
            elif hilo == 'hi':
                mask = cat_data[:, index] < self.settings[cutoff]

            are_hurr_pred &= mask

        return are_hurr_pred


class DACategoriser(Categoriser):
    '''Discriminant analysis categoriser'''
    def __init__(self, missed_count):
        super(DACategoriser, self).__init__(missed_count)
        self.lda = LDA()
        self.fig = 20
        max_index = SCATTER_ATTRS['lon']['index']
        self.settings['indices'] = range(max_index)

    def try_cat(self, cat_data, are_hurr_actual, **kwargs):
        self.train(cat_data, are_hurr_actual)
        super(DACategoriser, self).try_cat(cat_data, are_hurr_actual, **kwargs)

    def train(self, cat_data, are_hurr_actual):
        indices = self.settings['indices']

        self.lda.fit(cat_data[:, indices], are_hurr_actual)

    def categorise(self, cat_data):
        indices = self.settings['indices']

        are_hurr_pred = self.lda.predict(cat_data[: , indices])
        return are_hurr_pred


class SGDCategoriser(Categoriser):
    '''Stochastic Gradient Descent categoriser'''
    def __init__(self, missed_count):
        super(SGDCategoriser, self).__init__(missed_count)
        self.fig = 30
        max_index = SCATTER_ATTRS['lon']['index']
        self.settings['indices'] = range(max_index)

    def try_cat(self, cat_data, are_hurr_actual, **kwargs):
        plot = kwargs.pop('plot')
        self.sgd_clf = SGDClassifier(**kwargs)
        self.train(cat_data, are_hurr_actual)
        kwargs['plot'] = plot
        super(SGDCategoriser, self).try_cat(cat_data, are_hurr_actual, **kwargs)

    def train(self, cat_data, are_hurr_actual):
        indices = self.settings['indices']

        scaler = StandardScaler()
        scaler.fit(cat_data[:, indices])
        self.cat_data_scaled = scaler.transform(cat_data[:, indices])

        self.sgd_clf.fit(self.cat_data_scaled, are_hurr_actual)

    def categorise(self, cat_data):
        are_hurr_pred = self.sgd_clf.predict(self.cat_data_scaled)
        return are_hurr_pred

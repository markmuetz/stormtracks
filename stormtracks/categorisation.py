from collections import OrderedDict
from copy import copy

import numpy as np
import pylab as plt
try:
    import mlpy
except ImportError:
    print 'Must be on UCL computer'

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
    ('lon', {'name': 'calc', 'calc': calc_lon, 'index': 10}),
    ('lat', {'name': 'calc', 'calc': calc_lat, 'index': 11}),
])


class Categoriser(object):
    def __init__(self, missed_count):
        self.highest_score = 0
        self.missed_count = missed_count

    def try_cat(self, cat_data, are_hurr_actual, **kwargs):
        self.cat_data = cat_data
        self.are_hurr_actual = are_hurr_actual
        for k, v in kwargs.items():
            self.settings[k] = v
        self.are_hurr_pred = self.categorise(cat_data)
        self.res = self.compare(are_hurr_actual, self.are_hurr_pred)
        return self.print_score()
    
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
                 cd[:, i2][~h], 'b+', zorder=3)
        plt.plot(cd[:, i1][h], 
                 cd[:, i2][h], 'ro', zorder=2)
    
    def compare(self, are_hurr_actual, are_hurr_pred):
        res = {}
        res['fn'] = self.missed_count
        res['fp'] = (~are_hurr_actual & are_hurr_pred).sum()
        res['fn'] += (are_hurr_actual & ~are_hurr_pred).sum()
        res['tp'] = (are_hurr_actual & are_hurr_pred).sum()
        res['tn'] = (~are_hurr_actual & ~are_hurr_pred).sum()
        return res

    def print_score(self):
        res = self.res
        self.curr_score = 1. * res['tp'] / (res['tp'] + res['fn'] + res['fp'])
        self.sensitivity = 1. * res['tp'] / (res['tp'] + res['fn'])
        self.ppv = 1. * res['tp'] / (res['tp'] + res['fp'])

        if self.curr_score >= self.highest_score:
            print('{0}score: {1}{2}, sens: {3}, ppv: {4}'.format(bcolors.FAIL, self.curr_score, bcolors.ENDC,
                                                                 self.sensitivity, self.ppv))
            self.highest_score = self.curr_score
        else:
            print('score: {0}, sens: {1}, ppv: {2}'.format(self.curr_score, self.sensitivity, self.ppv))
        return self.curr_score

class CutoffCategoriser(Categoriser):
    def __init__(self, missed_count):
        super(CutoffCategoriser, self).__init__(missed_count)
        self.fig = 10
        self.best_so_far()

    def best_so_far(self):
        self.settings = OrderedDict([('vort_lo', 0.000289), 
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
    def __init__(self):
        super(DACategoriser, self).__init__(missed_count)
        self.ldac = mlpy.LDAC()
        self.fig = 20

    def try_cat(self, cat_data, are_hurr_actual, **kwargs):
        self.train(cat_data, are_hurr_actual)
        super(DACategoriser, self).try_cat(cat_data, are_hurr_actual, **kwargs)

    def train(self, cat_data, are_hurr_actual):
        self.ldac.learn(cat_data[:, 10], are_hurr_actual)

    def categorise(self, cat_data):
        are_hurr_pred = self.ldac.pred(cat_data)
        return are_hurr_pred

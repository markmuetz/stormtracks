from collections import OrderedDict
from copy import copy

import numpy as np
import mlpy

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


SCATTER_ATTRS = OrderedDict(
    # Normally x.
    vort = {'name': 'calc', 'calc': get_vort, 'index': 0},
    pmin = {'name': 'pmins', 'index': 1},
    pambdiff = {'name': 'p_ambient_diffs', 'index': 2},
    mindist = {'name': 'min_dists', 'index': 3},
    t995 = {'name': 't995s', 'index': 4},
    t850 = {'name': 't850s', 'index': 5},
    tanom = {'name': 'calc', 'calc': calc_t_anom, 'index': 6},
    maxwindspeed = {'name': 'max_windspeeds', 'index': 7},
    maxwindspeeddist = {'name': 'calc', 'calc': calc_ws_dist, 'index': 8},
    maxwindspeeddir = {'name': 'calc', 'calc': calc_ws_dir, 'index': 9},
    lon = {'name': 'calc', 'calc': calc_lon, 'index': 10},
    lat = {'name': 'calc', 'calc': calc_lat, 'index': 11},
)


class CutoffCategoriser(object):
    def __init__(self):
        self.lowest_score = 1e99
        self.reset()

    def best_so_far(self):
        self.cutoffs = OrderedDict([('vort_lo', 0.000289), 
                                    ('t995_lo', 297.2), 
                                    ('t850_lo', 287), 
                                    ('maxwindspeed_lo', 16.5), 
                                    ('pambdiff_lo', 563)])

    def reset(self):
        self.cutoffs = OrderedDict()
        self.cutoffs['vort_lo'] = 0.0003
        self.cutoffs['t995_lo'] = 297
        self.cutoffs['t850_lo'] = 287
        self.best_cutoffs = None

    def try_cat(self, cat_data, are_hurr, **kwargs):
        for k, v in kwargs.items():
            self.cutoffs[k] = v
        are_hurr_pred = self.categorise(cat_data)
        self.are_hurr_pred = are_hurr_pred
        self.res = self.print_score(are_hurr, are_hurr_pred)
    
    def categorise(self, cat_data):
        are_hurr = np.ones((len(cat_data),)).astype(bool)

        for cutoff in self.cutoffs.keys():
            var, hilo = cutoff.split('_')
            index = SCATTER_ATTRS[var]['index']
            if hilo == 'lo':
                mask = cat_data[:, index] > self.cutoffs[cutoff]
            elif hilo == 'hi':
                mask = cat_data[:, index] < self.cutoffs[cutoff]

            are_hurr &= mask

        return are_hurr

    def compare(self, are_hurr_actual, are_hurr_pred):
        res = {}
        res['fp'] = (~are_hurr_actual & are_hurr_pred).sum()
        res['fn'] = (are_hurr_actual & ~are_hurr_pred).sum()
        res['tp'] = (are_hurr_actual & are_hurr_pred).sum()
        res['tn'] = (~are_hurr_actual & ~are_hurr_pred).sum()
        return res

    def print_score(self, are_hurr_actual, are_hurr_pred):
        res = self.compare(are_hurr_actual, are_hurr_pred)
        curr_score = res['fp'] + res['fn'] - res['tp']
        if curr_score <= self.lowest_score:
            print('{0}{1}{2}'.format(bcolors.FAIL, curr_score, bcolors.ENDC))
            self.lowest_score = curr_score
            self.best_cutoffs = copy(self.cutoffs)
        else:
            print(curr_score)
        return res


class DACategoriser(object):
    '''Discriminant analysis categoriser'''
    def __init__(self):
        self.lowest_score = 1e99
        self.ldac = mlpy.LDAC()

    def train(self, cat_data, are_hurr):
        self.ldac.learn(cat_data, are_hurr)

    def try_cat(self, cat_data, are_hurr, **kwargs):
        # Make tunable?
        self.train(cat_data, are_hurr)
        are_hurr_pred = self.categorise(cat_data)
        self.are_hurr_pred = are_hurr_pred.astype(bool)
        self.res = self.print_score(are_hurr, are_hurr_pred)
    
    def categorise(self, cat_data):
        are_hurr_pred = self.ldac.pred(cat_data)
        return are_hurr_pred

    def compare(self, are_hurr_actual, are_hurr_pred):
        res = {}
        res['fp'] = (~are_hurr_actual & are_hurr_pred).sum()
        res['fn'] = (are_hurr_actual & ~are_hurr_pred).sum()
        res['tp'] = (are_hurr_actual & are_hurr_pred).sum()
        res['tn'] = (~are_hurr_actual & ~are_hurr_pred).sum()
        return res

    def print_score(self, are_hurr_actual, are_hurr_pred):
        res = self.compare(are_hurr_actual, are_hurr_pred)
        curr_score = res['fp'] + res['fn'] - res['tp']
        if curr_score <= self.lowest_score:
            print('{0}{1}{2}'.format(bcolors.FAIL, curr_score, bcolors.ENDC))
            self.lowest_score = curr_score
        else:
            print(curr_score)
        return res

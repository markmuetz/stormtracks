import os
from collections import OrderedDict
from copy import copy
from glob import glob

import simplejson
import numpy as np
import pylab as plt
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDClassifier as SkSGDClassifier
    from sklearn.lda import LDA
    from sklearn.qda import QDA
    from sklearn import tree
except ImportError:
    print 'IMPORT sklearn failed: must be on UCL computer'

from load_settings import settings
from utils.utils import geo_dist

SAVE_FILE_TPL = 'cat_{0}.json'


class ClassificationData(object):
    def __init__(self, name, data, are_hurr_actual, dates, hurr_counts, missed_count):
        self.name = name
        self.data = data
        self.are_hurr_actual = are_hurr_actual
        self.dates = dates
        self.hurr_counts = hurr_counts
        self.missed_count = missed_count


# TODO: utils.
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


class Classifier(object):
    def __init__(self):
        self.missed_count = 0
        self.settings = {}
        self.best_settings = None
        self.is_trained = False

        self.res = {}

    def list(self):
        plot_settings = []
        for fn in glob(os.path.join(settings.SETTINGS_DIR, SAVE_FILE_TPL.format('*'))):
            plot_settings.append('_'.join(os.path.basename(fn).split('.')[0].split('_')[1:]))
        return plot_settings

    def save(self, name):
        f = open(os.path.join(settings.SETTINGS_DIR,
                              SAVE_FILE_TPL.format(name)), 'w')
        simplejson.dump(self.settings, f, indent=4)

    def load(self, name):
        try:
            f = open(os.path.join(settings.SETTINGS_DIR, SAVE_FILE_TPL.format(name)), 'r')
            self_settings = simplejson.load(f)
            return self_settings
        except Exception, e:
            print('Settings {0} could not be loaded'.format(name))
            print('{0}'.format(e.message))
            raise

    def train(self, classification_data, indices=None, settings_name=None, **kwargs):
        if settings_name:
            self.settings = self.load(settings_name)
            self.classifier_kwargs = copy(self.settings)
            self.classifier_kwargs.pop('indices')
        else:
            self.settings = copy(kwargs)
            self.classifier_kwargs = copy(kwargs)

        self.is_trained = True

        if not indices:
            max_index = SCATTER_ATTRS['rh995']['index']
            self.settings['indices'] = range(max_index)
        else:
            self.settings['indices'] = indices

    def predict(self, classification_data, plot=None, var1='vort', var2='pmin', fig=None, fmt=None):
        self.missed_count = classification_data.missed_count
        self.classification_data = classification_data

        self.classify(classification_data)
        self.compare(classification_data.are_hurr_actual, self.are_hurr_pred)

        self.calc_stats()

        print(self.res)
        print(self.settings)

        if plot in ['remaining', 'all']:
            self.plot_remaining_actual(var1, var2, fig)

        if plot in ['confusion', 'all']:
            self.plot_confusion(var1, var2, fig)

        if fmt:
            if classification_data.name == 'Calibration':
                plt.figure(1)
            elif classification_data.name == 'Validation':
                plt.figure(2)
            else:
                plt.figure(0)

            plt.title(classification_data.name)
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.xlabel('sensitivity')
            plt.ylabel('ppv')
            plt.plot(self.sensitivity, self.ppv, fmt)

    def classify(self, classification_data):
        if not self.is_trained:
            raise Exception('Not yet trained')

    def plot_remaining_actual(self, var1, var2, fig=None, hurr_on_top=False):
        if not fig:
            fig = self.fig
        print('Figure {0}'.format(fig))
        plt.figure(fig)

        plt.clf()
        cd = self.classification_data.data[self.are_hurr_pred]
        h = self.classification_data.are_hurr_actual[self.are_hurr_pred]

        i1 = SCATTER_ATTRS[var1]['index']
        i2 = SCATTER_ATTRS[var2]['index']

        plt.xlabel(var1)
        plt.ylabel(var2)

        if hurr_on_top:
            plt.plot(cd[:, i1][~h],
                     cd[:, i2][~h], 'bx', zorder=1)
            plt.plot(cd[:, i1][h],
                     cd[:, i2][h], 'ko', zorder=2)
        else:
            plt.plot(cd[:, i1][~h],
                     cd[:, i2][~h], 'bx', zorder=3)
            plt.plot(cd[:, i1][h],
                     cd[:, i2][h], 'ko', zorder=2)

    def plot_confusion(self, var1, var2, fig=None, show_true_negs=False):
        if not fig:
            fig = self.fig
        print('Figure {0}'.format(fig + 1))
        plt.figure(fig + 1)
        plt.clf()

        plt.xlabel(var1)
        plt.ylabel(var2)

        i1 = SCATTER_ATTRS[var1]['index']
        i2 = SCATTER_ATTRS[var2]['index']

        cd = self.classification_data.data

        tp = self.are_hurr_pred & self.classification_data.are_hurr_actual
        tn = ~self.are_hurr_pred & ~self.classification_data.are_hurr_actual
        fp = self.are_hurr_pred & ~self.classification_data.are_hurr_actual
        fn = ~self.are_hurr_pred & self.classification_data.are_hurr_actual

        hs = ((tp, 'go', 1),
              (fp, 'ro', 3),
              (fn, 'rx', 4),
              (tn, 'gx', 2))

        if not show_true_negs:
            hs = hs[:-1]

        for h, fmt, order in hs:
            # plt.subplot(2, 2, order)
            plt.plot(cd[:, i1][h], cd[:, i2][h], fmt, zorder=order)

    def compare(self, are_hurr_actual, are_hurr_pred):
        # Missed hurrs are counted as FN.
        self.res['fn'] = self.missed_count + (are_hurr_actual & ~are_hurr_pred).sum()
        self.res['fp'] = (~are_hurr_actual & are_hurr_pred).sum()
        self.res['tp'] = (are_hurr_actual & are_hurr_pred).sum()
        self.res['tn'] = (~are_hurr_actual & ~are_hurr_pred).sum()
        return self.res

    def calc_stats(self, show=True):
        res = self.res
        try:
            self.sensitivity = 1. * res['tp'] / (res['tp'] + res['fn'])
        except:
            self.sensitivity = -1

        try:
            self.ppv = 1. * res['tp'] / (res['tp'] + res['fp'])
        except:
            self.ppv = -1

        try:
            self.fpr = 1. * res['fp'] / (res['fp'] + res['tn'])
        except:
            self.fpr = -1

        if show:
            print('sens: {0}, ppv: {1}, fpr: {2}'.format(self.sensitivity, self.ppv, self.fpr))


class ClassifierChain(Classifier):
    '''Allows classifiers to be chained together.'''
    def __init__(self, cats):
        super(ClassifierChain, self).__init__()
        self.fig = 100
        self.cats = cats

    def train(self, classification_data, indices=None, settings_name=None, **kwargs):
        '''Keyword args can be passed to e.g. the second classifier by using:
        two={'loss': 'log'}
        '''
        super(ClassifierChain, self).train(classification_data, indices, settings_name, **kwargs)
        curr_class_data = ClassificationData('curr',
                                             classification_data.data,
                                             classification_data.are_hurr_actual,
                                             classification_data.dates,
                                             classification_data.hurr_counts,
                                             classification_data.missed_count)

        args = ['one', 'two', 'three']
        for arg, classifier in zip(args, self.cats):
            classifier.train(curr_class_data,
                             **kwargs[arg]).classify(curr_class_data)

            data_mask = classifier.are_hurr_pred
            curr_class_data = ClassificationData('curr',
                                                 curr_class_data.data[data_mask],
                                                 curr_class_data.are_hurr_actual[data_mask],
                                                 curr_class_data.dates[data_mask],
                                                 curr_class_data.hurr_counts,
                                                 curr_class_data.missed_count)
        return self

    def classify(self, classification_data):
        super(ClassifierChain, self).classify(classification_data)

        curr_class_data = ClassificationData('curr',
                                             classification_data.data,
                                             classification_data.are_hurr_actual,
                                             classification_data.dates,
                                             classification_data.hurr_counts,
                                             classification_data.missed_count)
        are_hurr_pred = np.ones(len(classification_data.data)).astype(bool)
        removed_classification_data = None
        for classifier in self.cats:
            classifier.classify(curr_class_data)

            # Mask data based on what curr classifier detected as positives.
            data_mask = classifier.are_hurr_pred
            are_hurr_pred[are_hurr_pred] = data_mask
            curr_class_data = ClassificationData('curr',
                                                 curr_class_data.data[data_mask],
                                                 curr_class_data.are_hurr_actual[data_mask],
                                                 curr_class_data.dates[data_mask],
                                                 curr_class_data.hurr_counts,
                                                 curr_class_data.missed_count)

        self.are_hurr_pred = are_hurr_pred


class CutoffClassifier(Classifier):
    def __init__(self):
        super(CutoffClassifier, self).__init__()
        self.fig = 10

    def train(self, classification_data, indices=None, settings_name=None, **kwargs):
        super(CutoffClassifier, self).train(classification_data, indices, settings_name, **kwargs)
        if 'best' in self.settings and self.settings['best']:
            self.best_so_far()
        return self

    def best_so_far(self):
        self.settings = dict([('vort_lo', 0.000104),
                              # ('pwat_lo', 53), # possibly?
                              ('t995_lo', 297.2),
                              ('t850_lo', 286.7),
                              ('maxwindspeed_lo', 16.1),
                              ('pambdiff_lo', 563.4)])

    def classify(self, classification_data):
        super(CutoffClassifier, self).classify(classification_data)
        self.are_hurr_pred = np.ones((len(classification_data.data),)).astype(bool)

        for cutoff in self.settings.keys():
            if cutoff in ['indices']:
                continue
            var, hilo = cutoff.split('_')
            index = SCATTER_ATTRS[var]['index']
            if hilo == 'lo':
                mask = classification_data.data[:, index] > self.settings[cutoff]
            elif hilo == 'hi':
                mask = classification_data.data[:, index] < self.settings[cutoff]

            self.are_hurr_pred &= mask

        return self.are_hurr_pred


class LDAClassifier(Classifier):
    '''Linear Discriminant analysis classifier'''
    def __init__(self):
        super(LDAClassifier, self).__init__()
        self.fig = 20
        self.is_trainable = True
        self.is_trained = False

    def train(self, classification_data, indices=None, settings_name=None, **kwargs):
        super(LDAClassifier, self).train(classification_data, indices, settings_name, **kwargs)
        indices = self.settings['indices']

        self.lda = LDA(**self.classifier_kwargs)

        self.lda.fit(classification_data.data[:, indices], classification_data.are_hurr_actual)
        return self

    def classify(self, classification_data):
        super(LDAClassifier, self).classify(classification_data)
        indices = self.settings['indices']

        self.are_hurr_pred = self.lda.predict(classification_data.data[:, indices])
        return self.are_hurr_pred


class QDAClassifier(Classifier):
    '''Quadratic Discriminant analysis classifier'''
    def __init__(self):
        super(QDAClassifier, self).__init__()
        self.fig = 20
        self.is_trainable = True
        self.is_trained = False

    def train(self, classification_data, indices=None, settings_name=None, **kwargs):
        super(QDAClassifier, self).train(classification_data, indices, settings_name, **kwargs)
        indices = self.settings['indices']

        self.qda = QDA(**self.classifier_kwargs)

        self.qda.fit(classification_data.data[:, indices], classification_data.are_hurr_actual)
        return self

    def classify(self, classification_data):
        super(QDAClassifier, self).classify(classification_data)
        indices = self.settings['indices']

        self.are_hurr_pred = self.qda.predict(classification_data.data[:, indices])
        return self.are_hurr_pred


class DTAClassifier(Classifier):
    '''Decision tree classifier'''
    def __init__(self):
        super(DTAClassifier, self).__init__()
        self.fig = 50
        self.is_trainable = True
        self.is_trained = False

    def train(self, classification_data, indices=None, settings_name=None, **kwargs):
        super(DTAClassifier, self).train(classification_data, indices, settings_name, **kwargs)
        indices = self.settings['indices']

        self.dtc = tree.DecisionTreeClassifier(**self.classifier_kwargs)

        self.dtc.fit(classification_data.data[:, indices], classification_data.are_hurr_actual)
        return self

    def classify(self, classification_data):
        super(DTAClassifier, self).classify(classification_data)
        indices = self.settings['indices']

        self.are_hurr_pred = self.dtc.predict(classification_data.data[:, indices])
        return self.are_hurr_pred


class SGDClassifier(Classifier):
    '''Stochastic Gradient Descent classifier'''
    def __init__(self):
        super(SGDClassifier, self).__init__()
        self.fig = 30
        self.is_trainable = True
        self.is_trained = False

    def train(self, classification_data, indices=None, settings_name=None, **kwargs):
        super(SGDClassifier, self).train(classification_data, indices, settings_name, **kwargs)
        indices = self.settings['indices']
        print(self.settings)

        self.sgd_clf = SkSGDClassifier(**self.classifier_kwargs)

        self.scaler = StandardScaler()
        self.scaler.fit(classification_data.data[:, indices])
        self.classification_data_scaled =\
            self.scaler.transform(classification_data.data[:, indices])

        self.sgd_clf.fit(self.classification_data_scaled, classification_data.are_hurr_actual)
        self.is_trained = True
        return self

    def classify(self, classification_data):
        super(SGDClassifier, self).classify(classification_data)
        indices = self.settings['indices']

        self.classification_data_scaled =\
            self.scaler.transform(classification_data.data[:, indices])
        self.are_hurr_pred = self.sgd_clf.predict(self.classification_data_scaled)

        return self.are_hurr_pred

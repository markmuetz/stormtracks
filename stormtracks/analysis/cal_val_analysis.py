import sys
import datetime as dt

import pandas as pd
import numpy as np
import argh

# scikit-learn classifiers.
from sklearn.linear_model import SGDClassifier
from sklearn.qda import QDA
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler

from stormtracks.load_settings import settings
from stormtracks.results import StormtracksResultsManager, ResultNotFound

COLS = ['vort850', 'vort9950', 'max_ws', 'pmin', 't850', 't9950', 'cape', 'pwat', 'prmsl', 't_anom']
SGD_LOSSES = ['hinge',
    'modified_huber', 
    'log',
    # These don't work anyway.
    # 'squared_loss',
    # 'huber',
    # 'epsilon_insensitive'
]
SGD_PENALTY = ['l1', 'l2', 'elasticnet']


def cal_val_analysis(cols=None):
    '''Calibrates/validates classifiers from 1990 to 2009

    Calibrates on even years, validates on odd
    Trains all variants of the scikit.learn SGD classifier, as well as LDA/QDA.
    SGD pre-scales data
    Calc's FP/FN/TP/TN + sens, ppv and sens*ppv.
    '''
    # Load the data.
    if cols == None:
        cols = COLS

    cal_years = range(1990, 2010, 2)
    val_years = range(1991, 2010, 2)

    cal_cfm = get_results(cal_years, settings.RESULTS)
    val_cfm = get_results(val_years, settings.RESULTS)

    # Set up classifers.
    classifiers = []
    scalers = []
    for sgd_loss in SGD_LOSSES:
        for sgd_penalty in SGD_PENALTY:
            sgd = SGDClassifier(loss=sgd_loss, penalty=sgd_penalty)
            sgd_scaler = StandardScaler()
            classifiers.append(sgd)
            scalers.append(sgd_scaler)

    classifiers.append(LDA())
    scalers.append(None)

    classifiers.append(QDA())
    scalers.append(None)

    # Perform classification.
    for classifier, scaler in zip(classifiers, scalers):
        print('Analysing with classifier {}'.format(classifier))

        try:
            for cfm, is_cal in [(cal_cfm, True), (val_cfm, False)]:
                if is_cal:
                    print('CAL')
                else:
                    print('VAL')

                data = cfm[cols].values.astype(np.float32)
                are_hurr = ~cfm.bt_wind.isnull() & (cfm.is_hurr)

                if is_cal:
                    if scaler is not None:
                        scaler.fit(data)
                        scaled_data = scaler.transform(data)
                    else:
                        scaled_data = data

                    fit(classifier, scaled_data, are_hurr)
                else:
                    if scaler is not None:
                        scaled_data = scaler.transform(data)
                    else:
                        scaled_data = data

                print(', '.join(cols))
                predict(classifier, scaled_data, are_hurr)
            print('')
        except Exception, e:
            print('Error with classifier {}'.format(classifier))
            print(e)


def get_results(year_range, results_name):
    '''Gets the results for the given years (list)'''
    results_manager = StormtracksResultsManager(results_name)

    max_index = 0
    fields = []
    best_tracks = []
    for year in year_range:
        print(year)

        all_fields = results_manager.get_result(year, 'all_fields')
        all_fields['t_anom'] = all_fields.t9950 - all_fields.t850
        best_track_matches = results_manager.get_result(year, 'best_track_matches')

        all_fields.index += max_index
        best_track_matches.index += max_index
        max_index = all_fields.index.max() + 1

        fields.append(all_fields)
        best_tracks.append(best_track_matches)

    print('Concatenating')
    fields = pd.concat(fields)
    best_tracks = pd.concat(best_tracks)

    print('Joining')
    combined_fields_matches = fields.join(best_tracks)
    return combined_fields_matches


def predict(classifier, data, are_hurr):
    '''Predicts classes for classifier using data'''
    are_hurr_pred = classifier.predict(data)

    tp = (are_hurr & are_hurr_pred)
    tn = (~are_hurr & ~are_hurr_pred)
    fp = (~are_hurr & are_hurr_pred)
    fn = (are_hurr & ~are_hurr_pred)

    print('tp: {}'.format(tp.sum()))
    print('tn: {}'.format(tn.sum()))
    print('fp: {}'.format(fp.sum()))
    print('fn: {}'.format(fn.sum()))

    sens = 1. * tp.sum() / (tp.sum() + fn.sum())
    ppv = 1. * tp.sum() / (tp.sum() + fp.sum())
    print('sens: {}'.format(sens))
    print('ppv : {}'.format(ppv))
    print('sens*ppv : {}'.format(sens * ppv))


def fit(classifier, data, are_hurr):
    '''Fits classifier to the data'''
    classifier.fit(data, are_hurr)


if __name__ == '__main__':
    argh.dispatch_command(cal_val_analysis)

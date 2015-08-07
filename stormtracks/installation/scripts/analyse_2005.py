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

from stormtracks.results import StormtracksResultsManager, ResultNotFound

# COLS = ['vort', 'max_ws', 'pmin', 't850', 't995', 'cape', 'pwat']
# COLS = ['vort', 'max_ws', 'pmin', 't850', 't995']
# COLS = ['vort', 'max_ws', 'pmin', 't850', 't_anom']
COLS = ['vort850', 'max_ws', 'pmin', 't850']
SGD_LOSSES = ['hinge',
    'modified_huber', 
    'log',
    # 'squared_loss',
    # 'huber',
    # 'epsilon_insensitive'
]
SGD_PENALTY = ['l1', 'l2', 'elasticnet']


def analyse_year(year=2005, results_name='demo'):
    start = dt.datetime.now()

    results_manager = StormtracksResultsManager(results_name)

    try:
        all_fields = results_manager.get_result(year, 'all_fields')
        best_track_matches = results_manager.get_result(year, 'best_track_matches')
    except ResultNotFound:
	print('Could not load all_fields and best_track_matches for {}'.format(year))
	print('Run download_2005.py and process_2005.py first')
	exit(1)

    combined_fields_matches = all_fields.join(best_track_matches)

    # return combined_fields_matches
    analyse_combined(combined_fields_matches)

    end = dt.datetime.now()
    print('Analysed {} in {}'.format(year, end - start))


def analyse_combined(combined_fields_matches):
    data = combined_fields_matches[COLS].values.astype(np.float32)
    are_hurr = ~combined_fields_matches.bt_wind.isnull() & (combined_fields_matches.is_hurr)
    analyse_data(data, are_hurr)


def analyse_data(data, are_hurr):
    sgd = SGDClassifier(loss='hinge', penalty='l1')
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    fit(sgd, scaled_data, are_hurr)
    predict(sgd, scaled_data, are_hurr)


def predict(classifier, data, are_hurr):
    are_hurr_pred = classifier.predict(data)

    tp = (are_hurr & are_hurr_pred)
    tn = (~are_hurr & ~are_hurr_pred)
    fp = (~are_hurr & are_hurr_pred)
    fn = (are_hurr & ~are_hurr_pred)

    print(', '.join(COLS))

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
    classifier.fit(data, are_hurr)
    return classifier


if __name__ == '__main__':
    argh.dispatch_command(analyse_year)

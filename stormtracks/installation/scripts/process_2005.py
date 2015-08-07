import sys
import os
import datetime as dt

import argh

from stormtracks.c20data import C20Data
from stormtracks.setup_logging import get_logger
from stormtracks.load_settings import settings
import stormtracks.ibtracsdata as ibtracsdata
from stormtracks.results import StormtracksResultsManager

from stormtracks.processing.find_vortmax import VortmaxFinder
import stormtracks.processing.matching as matching

log = get_logger('st.demo')


def process_year(year=2005, results_name='demo'):
    start = dt.datetime.now()
    log.info('Processing year, results: {}, {}'.format(year, results_name))

    start_date = dt.datetime(year, 6, 1)
    end_date = dt.datetime(year, 12, 1)
    results_manager = StormtracksResultsManager(results_name)

    # Run through 20CR looking for vortmax. Collect all fields for each vortmax, save as pandas DataFrame.
    c20data = C20Data(year)
    finder = VortmaxFinder(c20data, False)
    df_year = finder.find_vort_maxima(start_date, end_date)

    results_manager.save_result(year, 'all_fields', df_year)

    # Match each best track to the corresponding vortmax from 20CR.
    ib = ibtracsdata.IbtracsData()
    ib.load_ibtracks_year(year)
    best_track_matches = matching.simple_matching(ib.best_tracks, df_year)
    results_manager.save_result(year, 'best_track_matches', best_track_matches)

    end = dt.datetime.now()
    log.info('Processed {} in {}'.format(year, end - start))

    return df_year, best_track_matches


if __name__ == '__main__':
    argh.dispatch_command(process_year)

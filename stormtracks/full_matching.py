from collections import defaultdict, OrderedDict
from copy import copy
import itertools
import time

import numpy as np
import pylab as plt

from tracking import VortMax, VortMaxTrack
from logger import get_logger
from utils.utils import geo_dist
from utils.kalman import RTSSmoother, _plot_rts_smoother
from matching import Match

log = get_logger('analysis.matching', console_level_str='INFO')

NUM_ENSEMBLE_MEMBERS = 56


def full_match_vort_tracks_by_date_to_best_tracks(all_vort_tracks_by_date, best_tracks):
    '''Takes all vorticity tracks and best tracks and matches them up

    :param vort_tracks_by_date: dict with dates as keys and lists of vort tracks as values
    :param best_tracks: list of best tracks
    :returns: array of all matches
    '''
    all_matches = []
    for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
        matches = OrderedDict()
        vort_tracks_by_date = all_vort_tracks_by_date[ensemble_member]

        for best_track in best_tracks:
            for lon, lat, date in zip(best_track.lons, best_track.lats, best_track.dates):
                if date in vort_tracks_by_date.keys():
                    vort_tracks = vort_tracks_by_date[date]
                    for vortmax in vort_tracks:
                        if (best_track, vortmax) in matches:
                            match = matches[(best_track, vortmax)]
                            match.overlap += 1
                        else:
                            match = Match(best_track, vortmax)
                            matches[(best_track, vortmax)] = match

                        match.cum_dist += geo_dist(vortmax.vortmax_by_date[date].pos, (lon, lat))

        all_matches.append(matches.values())
    return all_matches


def full_good_matches(all_matches, dist_cutoff=None, overlap_cutoff=6):
    '''Returns all matches that meet the requirements

    Requirements are that match.av_dist() < dist_cutoff and match.overlap >= overlap_cutoff
    '''
    if not dist_cutoff:
        # Defaults to distance between two grid cells (at equator) * 5.
        # dist_cutoff = geo_dist((0, 0), (2, 0)) * 5
        # Defaults to 500km
        dist_cutoff = 500
    all_good_matches = []
    for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
        matches = all_matches[ensemble_member]
        good_matches = [ma for ma in matches if ma.av_dist() < dist_cutoff and ma.overlap >= overlap_cutoff]
        all_good_matches.append(good_matches)
    return all_good_matches




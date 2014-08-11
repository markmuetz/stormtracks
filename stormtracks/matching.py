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

log = get_logger('matching', console_level_str='INFO')


class EnsembleMatch(object):
    '''Represents one match between many vorticity tracks'''
    def __init__(self, vort_track, store_all_tracks=False):
        self.av_vort_track = vort_track

        # May have to disable due to memory constraints.
        self.store_all_tracks = store_all_tracks
        if self.store_all_tracks:
            self.vort_tracks = []
            self.vort_tracks.append(vort_track)

        self.cum_dist = 0
        self.overlap = 0

        self.dates = vort_track.dates
        self.date_start = vort_track.dates[0]
        self.date_end = vort_track.dates[-1]

        index_start = np.where(vort_track.dates == self.date_start)[0][0]
        index_end = np.where(vort_track.dates == self.date_end)[0][0]

        self.tracks_added = np.ones(index_end - index_start + 1)

    def add_track(self, vort_track):
        '''Checks to see whether the track should be added, then adds it

        Compares the overlap and cumalitive distance between the track to the current average
        track, and if they are below a limit adds the track, which involves recalculating the
        average track for this match, and updating some fields.

        :param vort_track: vorticitiy track to be added
        :returns: True if it was added, otherwise False
        '''
        # Fast track most common case: no overlap in dates.
        if self.date_start > vort_track.dates[-1] or self.date_end < vort_track.dates[0]:
            return False

        # Calculate some dates and offsets.
        overlap_start = max(self.date_start, vort_track.dates[0])
        overlap_end = min(self.date_end, vort_track.dates[-1])

        index1 = np.where(self.dates == overlap_start)[0][0]
        index2 = np.where(vort_track.dates == overlap_start)[0][0]

        end1 = np.where(self.dates == overlap_end)[0][0]
        end2 = np.where(vort_track.dates == overlap_end)[0][0]

        assert (end1 - index1) == (end2 - index2)

        overlap = end1 - index1 + 1
        # Check to see whether this vort_track should be added.
        if overlap < 6:
            return False

        cum_dist = self.__calc_distance(self.av_vort_track, vort_track,
                                        index1, index2, end1, end2)

        # Check to see whether this vort_track should be added.
        dist_cutoff = geo_dist((0, 0), (2, 0)) * 6
        if cum_dist / overlap > dist_cutoff:
            return False

        if self.store_all_tracks:
            self.vort_tracks.append(vort_track)

        # import ipdb; ipdb.set_trace()
        new_tracks_added = np.zeros(len(self.dates) + len(vort_track.dates) - overlap)
        copy_start_date = self.dates[0]
        copy_length = len(self.dates)
        ones_start_date = vort_track.dates[0]
        ones_length = len(vort_track.dates)

        self.dates = np.array(sorted([d for d in set(self.dates) | set(vort_track.dates)]))
        self.date_start = self.dates[0]
        self.date_end = self.dates[-1]

        # Copy over the existing tracks added to the new tracks added.
        # new_tracks_added must be equal or greater in length than current.
        # index it using the greatest of index1, and index2.
        copy_index = np.where(self.dates == copy_start_date)[0][0]
        ones_index = np.where(self.dates == ones_start_date)[0][0]

        new_tracks_added[copy_index:copy_index + copy_length] = self.tracks_added
        new_tracks_added[ones_index:ones_index + ones_length] += np.ones(ones_length)

        self.tracks_added = new_tracks_added

        # Update some fields.
        self.overlap += overlap
        self.cum_dist += cum_dist

        vortmaxes = []
        # Update the average vort track by taking a weighted average of the new
        # track and the current average. (N.B. self.tracks_added not yet updated.)
        # import ipdb; ipdb.set_trace()
        for i, date in enumerate(self.dates):
            vm1, vm2 = None, None
            if date in self.av_vort_track.vortmax_by_date:
                vm1 = self.av_vort_track.vortmax_by_date[date]
            if date in vort_track.vortmax_by_date:
                vm2 = vort_track.vortmax_by_date[date]

            if vm1 and vm2:
                # index tracks_added using min of index1, index2
                tracks_added = self.tracks_added[i]

                # Calculate a weighted average of the averaged pos and the new pos.
                pos = tuple([((tracks_added - 1) * v1 + v2) / (tracks_added)
                             for v1, v2 in zip(vm1.pos, vm2.pos)])

                # Calculate a weighted average of the averaged vort and the new vort.
                vort = ((tracks_added - 1) * vm1.vort) / (tracks_added)

                new_vm = VortMax(date, pos, vort)
                index1 += 1
                index2 += 1
            elif vm1:
                new_vm = VortMax(vm1.date, vm1.pos, vm1.vort)
                index1 += 1
            elif vm2:
                new_vm = VortMax(vm2.date, vm2.pos, vm2.vort)
                index2 += 1
            else:
                import ipdb
                ipdb.set_trace()
                assert False, 'Should never be reached'

            vortmaxes.append(new_vm)

        next_vm = None
        for vm in vortmaxes[::-1]:
            if next_vm:
                vm.next_vortmax.append(next_vm)
            next_vm = vm
        self.av_vort_track = VortMaxTrack(vm)
        return True

    def __calc_distance(self,
                        vort_track_0, vort_track_1,
                        index1, index2, end1, end2):
        cum_dist = 0

        while index1 <= end1 and index2 <= end2:
            pos1 = vort_track_0.vortmaxes[index1].pos
            pos2 = vort_track_1.vortmaxes[index2].pos

            cum_dist += geo_dist(pos1, pos2)
            index1 += 1
            index2 += 1

        return cum_dist

    def av_dist(self):
        '''Returns the average distance between all vorticity tracks'''
        return self.cum_dist / self.overlap


class BestTrackMatch(object):
    '''Represents one match between many vorticity tracks'''
    def __init__(self, best_track, store_all_tracks=False):
        self.best_track = best_track

        # May have to disable due to memory constraints.
        self.store_all_tracks = store_all_tracks
        if self.store_all_tracks:
            self.vort_tracks = []

        self.cum_dist = 0
        self.overlap = 0

        self.dates = best_track.dates
        self.date_start = best_track.dates[0]
        self.date_end = best_track.dates[-1]

        index_start = np.where(best_track.dates == self.date_start)[0][0]
        index_end = np.where(best_track.dates == self.date_end)[0][0]

        self.tracks_added = np.ones(index_end - index_start + 1)

    def add_match(self, match):
        '''Checks to see whether the macth should be added, then adds it

        Compares the overlap and cumalitive distance between the match to the current average
        track, and if they are below a limit adds the match, which involves recalculating the
        average track for this match, and updating some fields.

        :param match: match to be added
        :returns: True if it was added, otherwise False
        '''
        vort_track = match.av_vort_track

        # Fast track most common case: no overlap in dates.
        if self.date_start > vort_track.dates[-1] or self.date_end < vort_track.dates[0]:
            return False

        # Calculate some dates and offsets.
        overlap_start = max(self.date_start, vort_track.dates[0])
        overlap_end = min(self.date_end, vort_track.dates[-1])

        index1 = np.where(self.dates == overlap_start)[0][0]
        index2 = np.where(vort_track.dates == overlap_start)[0][0]

        end1 = np.where(self.dates == overlap_end)[0][0]
        end2 = np.where(vort_track.dates == overlap_end)[0][0]

        assert (end1 - index1) == (end2 - index2)

        overlap = end1 - index1 + 1
        # Check to see whether this vort_track should be added.
        if overlap < 6:
            return False

        cum_dist = self.__calc_distance(self.best_track, vort_track,
                                        index1, index2, end1, end2)

        # Check to see whether this vort_track should be added.
        dist_cutoff = geo_dist((0, 0), (2, 0)) * 6
        if cum_dist / overlap > dist_cutoff:
            return False

        if self.store_all_tracks:
            self.vort_tracks.append(vort_track)

        if False:
            # import ipdb; ipdb.set_trace()
            new_tracks_added = np.zeros(len(self.dates) + len(vort_track.dates) - overlap)
            copy_start_date = self.dates[0]
            copy_length = len(self.dates)
            ones_start_date = vort_track.dates[0]
            ones_length = len(vort_track.dates)

            self.dates = np.array(sorted([d for d in set(self.dates) | set(vort_track.dates)]))
            self.date_start = self.dates[0]
            self.date_end = self.dates[-1]

            # Copy over the existing tracks added to the new tracks added.
            # new_tracks_added must be equal or greater in length than current.
            # index it using the greatest of index1, and index2.
            copy_index = np.where(self.dates == copy_start_date)[0][0]
            ones_index = np.where(self.dates == ones_start_date)[0][0]

            new_tracks_added[copy_index:copy_index + copy_length] = self.tracks_added
            new_tracks_added[ones_index:ones_index + ones_length] += np.ones(ones_length)

            self.tracks_added = new_tracks_added

        # Update some fields.
        self.overlap += overlap
        self.cum_dist += cum_dist

        return True

    def __calc_distance(self,
                        best_track, vort_track,
                        index1, index2, end1, end2):
        cum_dist = 0

        while index1 <= end1 and index2 <= end2:
            pos1 = (best_track.lons[index1], best_track.lats[index1])
            pos2 = vort_track.vortmaxes[index2].pos

            cum_dist += geo_dist(pos1, pos2)
            index1 += 1
            index2 += 1

        return cum_dist

    def av_dist(self):
        '''Returns the average distance between all vorticity tracks'''
        return self.cum_dist / self.overlap


class Match(object):
    '''Represents one match between a best track and a vorticity track'''
    def __init__(self, best_track, vort_track):
        self.best_track = best_track
        self.vort_track = vort_track
        self.cum_dist = 0
        self.overlap = 1
        self.overlap_start = max(best_track.dates[0], vort_track.dates[0])
        self.overlap_end = min(best_track.dates[-1], vort_track.dates[-1])

    def av_dist(self):
        '''Returns the average distance between the best and vorticity tracks'''
        return self.cum_dist / self.overlap


class CycloneMatch(object):
    '''Represents one match between a best track and a vorticity track'''
    def __init__(self, best_track, cyclone):
        self.best_track = best_track
        self.cyclone = cyclone
        self.overlap_start = max(best_track.dates[0], cyclone.dates[0])
        self.overlap_end = min(best_track.dates[-1], cyclone.dates[-1])

    def av_dist(self):
        '''Returns the average distance between the best and vorticity tracks'''
        return self.cum_dist / self.overlap


def match_vort_tracks(vort_tracks_by_dates):
    matches = OrderedDict()

    # import ipdb; ipdb.set_trace()
    vort_tracks_by_date_0 = vort_tracks_by_dates[0]
    vort_tracks_by_date_1 = vort_tracks_by_dates[1]

    all_dates = set(vort_tracks_by_date_0.keys()) | set(vort_tracks_by_date_1.keys())

    for date in all_dates:
        # This means that every vort track is represented **multiple times**, once
        # for each date where it appears. Hence the need to check for the key in matches dict.
        vort_tracks_0 = vort_tracks_by_date_0[date]
        vort_tracks_1 = vort_tracks_by_date_1[date]

        for vort_track_0 in vort_tracks_0:
            for vort_track_1 in vort_tracks_1:

                # Check that I haven't already added this match.
                if (vort_track_0, vort_track_1) not in matches:
                    ensemble_match = EnsembleMatch(vort_track_0)

                    if ensemble_match.add_track(vort_track_1):
                        matches[(vort_track_0, vort_track_1)] = ensemble_match

    return matches


def match_ensemble_vort_tracks_by_date(vort_tracks_by_date_list):
    '''Takes the normal output from a tracker and flattens them to a list'''
    start = time.time()
    vort_tracks_list = []
    for vort_tracks_by_date in vort_tracks_by_date_list:
        # Flattens values and gets rid of duplicates.
        vort_tracks = set(itertools.chain(*vort_tracks_by_date.values()))
        vort_tracks_list.append([vt for vt in vort_tracks if len(vt.dates) >= 6])

    end = time.time()
    log.info('Setup time: {0}s'.format(end - start))

    start = time.time()
    matches = match_ensemble_vort_tracks(vort_tracks_list)
    end = time.time()
    log.info('Total run time: {0}s'.format(end - start))

    return matches


def match_ensemble_vort_tracks(vort_tracks_list):
    vort_tracks = vort_tracks_list.pop()
    ensemble_matches = []
    for vort_track in vort_tracks:
        ensemble_match = EnsembleMatch(vort_track, store_all_tracks=True)
        ensemble_matches.append(ensemble_match)

    while vort_tracks_list:
        start = time.time()
        next_vort_tracks = vort_tracks_list.pop()
        unmatched_tracks = copy(next_vort_tracks)
        for next_vort_track in next_vort_tracks:
            for ensemble_match in ensemble_matches:
                if ensemble_match.add_track(next_vort_track):
                    try:
                        unmatched_tracks.remove(next_vort_track)
                    except ValueError:
                        # It's already been removed, fine.
                        pass

        for unmatched_track in unmatched_tracks:
            ensemble_match = EnsembleMatch(unmatched_track, store_all_tracks=True)
            ensemble_matches.append(ensemble_match)

        end = time.time()
        log.info('Length of matches:          {0}'.format(len(ensemble_matches)))
        log.info('Length of unmatched:        {0}'.format(len(unmatched_tracks)))
        log.info('Remaining ensemble_members: {0}'.format(len(vort_tracks_list)))
        log.info('Loop time:                  {0}s'.format(end - start))
        log.info('')

    return ensemble_matches


def match_best_track_to_ensemble_match(best_tracks, ensemble_matches):
    matches = []
    unmatched = []

    for best_track in best_tracks:
        match = BestTrackMatch(best_track)
        unmatched.append(match)

        for ensemble_match in ensemble_matches:
            if match.add_match(ensemble_match):
                matches.append(match)
                try:
                    unmatched.remove(match)
                except ValueError:
                    pass

    return matches


def match_best_tracks_to_vortmax_tracks(best_tracks, vortmax_tracks):
    matches = []
    for best_track in best_tracks:
        for vortmax_track in vortmax_tracks:
            match = match_best_track_to_vortmax_track(best_track, vortmax_track)
            if match:
                matches.append(match)
    return matches


def match_best_tracks_to_cyclones(best_tracks, cyclones):
    matches = []
    unmatched = copy(cyclones)
    for cyclone in cyclones:
        for best_track in best_tracks:
            match = match_best_track_to_cyclone(best_track, cyclone)
            if match:
                matches.append(match)
                try:
                    unmatched.remove(cyclone)
                except ValueError:
                    pass

    return matches, unmatched


def match_best_track_to_cyclone(best_track, cyclone):
    vm_match = match_best_track_to_vortmax_track(best_track, cyclone.vortmax_track)
    if vm_match:
        match = CycloneMatch(best_track, cyclone)
        match.cum_dist = vm_match.cum_dist
        match.overlap = vm_match.overlap
        return match
    else:
        return None


def match_best_track_to_vortmax_track(best_track, vortmax_track):
    if (best_track.dates[0] > vortmax_track.dates[-1] or
            best_track.dates[-1] < vortmax_track.dates[0]):
        return None

    match = Match(best_track, vortmax_track)
    for lon, lat, date in zip(best_track.lons, best_track.lats, best_track.dates):
        if date in vortmax_track.vortmax_by_date.keys():
            if not match:
                match = Match(best_track, vortmax_track)
            match.overlap += 1
            match.cum_dist += geo_dist(vortmax_track.vortmax_by_date[date].pos, (lon, lat))

    if match.overlap < 6 or match.cum_dist / match.overlap > 500:
        return None
    else:
        return match


def match_vort_tracks_by_date_to_best_tracks(vort_tracks_by_date, best_tracks):
    '''Takes all vorticity tracks and best tracks and matches them up

    :param vort_tracks_by_date: dict with dates as keys and lists of vort tracks as values
    :param best_tracks: list of best tracks
    :returns: array of all matches
    '''
    matches = OrderedDict()

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

    return matches.values()


def good_matches(matches, dist_cutoff=None, overlap_cutoff=6):
    '''Returns all matches that meet the requirements

    Requirements are that match.av_dist() < dist_cutoff and match.overlap >= overlap_cutoff
    '''
    if not dist_cutoff:
        # Defaults to distance between two grid cells (at equator) * 5.
        # dist_cutoff = geo_dist((0, 0), (2, 0)) * 5
        # Defaults to 500km
        dist_cutoff = 500
    return [ma for ma in matches if ma.av_dist() < dist_cutoff and ma.overlap >= overlap_cutoff]


def combined_match(best_tracks, all_matches):
    '''Uses all best tracks and matches to combine all matches for each best track

    :param best_tracks: list of best tracks
    :param all_matches: all matches to search through
    :returns: dict of combined_matches (key: best track, value: list of matches)
    '''

    combined_matches = {}

    for best_track in best_tracks:
        combined_matches[best_track] = []

    for matches in all_matches:
        for match in matches:
            combined_matches[match.best_track].append(match)

    return combined_matches

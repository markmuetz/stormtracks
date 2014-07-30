from collections import defaultdict, OrderedDict

import numpy as np
import pylab as plt

from tracking import VortMax, VortMaxTrack
from utils.utils import dist
from utils.kalman import RTSSmoother, _plot_rts_smoother

CUM_DIST_CUTOFF = 100


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
        # Calculate some dates and offsets.
        overlap_start = max(self.date_start, vort_track.dates[0])
        overlap_end = min(self.date_end, vort_track.dates[-1])

        index1 = np.where(self.dates == overlap_start)[0][0]
        index2 = np.where(vort_track.dates == overlap_start)[0][0]

        end1 = np.where(self.dates == overlap_end)[0][0]
        end2 = np.where(vort_track.dates == overlap_end)[0][0]

        assert (end1 - index1) == (end2 - index2)

        overlap, cum_dist = self.__calc_distance(self.av_vort_track, vort_track,
                                                 index1, index2, end1, end2)

        # Check to see whether this vort_track should be added.
        if overlap < 6 or cum_dist / overlap > 6:
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
                import ipdb; ipdb.set_trace()
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
        overlap = end1 - index1 + 1

        cum_dist = 0

        while index1 <= end1 and index2 <= end2:
            pos1 = vort_track_0.vortmaxes[index1].pos
            pos2 = vort_track_1.vortmaxes[index2].pos

            cum_dist += dist(pos1, pos2)
            index1 += 1
            index2 += 1

        return overlap, cum_dist

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
        self.is_too_far_away = False
        self.overlap_start = max(best_track.dates[0], vort_track.dates[0])
        self.overlap_end = min(best_track.dates[-1], vort_track.dates[-1])

    def av_dist(self):
        '''Returns the average distance between the best and vorticity tracks'''
        return self.cum_dist / self.overlap


def match_vort_tracks(vort_tracks_by_dates):
    all_dates = set(vort_tracks_by_date_1.keys()) | set(vort_tracks_by_date_2.keys())
    matches = OrderedDict()

    # import ipdb; ipdb.set_trace()
    vort_tracks_by_date_1 = vort_tracks_by_date[0]
    for vort_track_1 in vort_tracks_by_date_1:

    for date in all_dates:
        # This means that every vort track is represented **multiple times**, once
        # for each date where it appears. Hence the need to check for the key in matches dict.
        vort_tracks_1 = vort_tracks_by_date_1[date]
        vort_tracks_2 = vort_tracks_by_date_2[date]

        for vort_track_1 in vort_tracks_1:
            for vort_track_2 in vort_tracks_2:

                # Check that I haven't already added this match.
                if (vort_track_1, vort_track_2) not in matches:
                    ensemble_match = EnsembleMatch(vort_track_1)

                    if ensemble_match.add_track(vort_track_2):
                        matches[(vort_track_1, vort_track_2)] = ensemble_match

    return matches


def match(vort_tracks_by_date, best_tracks):
    '''Takes all vorticity tracks and best tracks and matches them up

    Uses CUM_DIST_CUTOFF to decide whether the two tracks are too far apart

    :param vort_tracks_by_date: dict with dates as keys and lists of vort tracks as values
    :param best_tracks: list of best tracks
    :returns: OrderedDict of Match objects
        * key: (best_track, vortmax) tuple
        * value: Match object
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
                        if match.is_too_far_away:
                            continue

                    match.cum_dist += dist(vortmax.vortmax_by_date[date].pos, (lon, lat))
                    if match.cum_dist > CUM_DIST_CUTOFF:
                        match.is_too_far_away = True

    return matches


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


def _cum_dist(best_track, vortmax_by_date):
    '''Calculated the cumalitive distance between a best and vortmax track'''
    d = 0
    overlap = 0
    for lon, lat, date in zip(best_track.lons, best_track.lats, best_track.dates):
        try:
            d += dist(vortmax_by_date[date].pos, (lon, lat)) ** 2
            overlap += 1
        except:
            pass
    return d, overlap


def _optimise_rts_smoothing(combined_matches):
    '''Experimental function looking at optimizing RTS smoothing'''
    F = np.matrix([[1., 0., 1., 0.],
                   [0., 1., 0., 1.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    H = np.matrix([[1., 0., 0., 0.],
                   [0., 1., 0., 0.]])
    Q = np.matrix([[2., 1., 1., 1.],
                   [1., 2., 1., 1.],
                   [1., 1., 2., 1.],
                   [1., 1., 1., 2.]])
    R = np.matrix([[2., 1.],
                   [1., 2.]])

    c = 1

    fig = None
    for best_track, vort_tracks in combined_matches.items():
        for vort_track in vort_tracks:
            if fig:
                fig.canvas.manager.window.attributes('-topmost', 0)
            fig = plt.figure(c)
            fig.canvas.manager.window.attributes('-topmost', 1)
            _optimize(best_track, vort_track)
            c += 1


def _optimize(best_track, vort_track):
    '''Experimental function looking at optimizing RTS'''
    class __Tmp(object):
        pass

    F = np.matrix([[1., 0., 1., 0.],
                   [0., 1., 0., 1.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    H = np.matrix([[1., 0., 0., 0.],
                   [0., 1., 0., 0.]])
    Q = np.matrix([[2., 1., 1., 1.],
                   [1., 2., 1., 1.],
                   [1., 1., 2., 1.],
                   [1., 1., 1., 2.]])
    R = np.matrix([[2., 1.],
                   [1., 2.]])

    baseline_dist, overlap = _cum_dist(best_track, vort_track.vortmax_by_date)

    pos = np.array([vm.pos for vm in vort_track.vortmaxes])
    rts_smoother = RTSSmoother(F, H)
    smoothed_dict = OrderedDict()
    filtered_dict = OrderedDict()

    min_d = baseline_dist
    print('baseline {0}'.format(baseline_dist))
    optimum_param = None

    x = np.matrix([pos[0, 0], pos[0, 1], 0, 0]).T
    P = np.matrix(np.eye(4)) * 10

    plt.clf()
    plt.title(str(baseline_dist))
    plt.plot(pos[:, 0], pos[:, 1], 'k+')
    plt.plot(best_track.lons, best_track.lats, 'b-')
    plt.pause(0.01)

    for q_param in np.arange(1e-4, 1e-2, 5e-4):
        for r_param in np.arange(1e-2, 1e0, 5e-2):
            xs, Ps = rts_smoother.process_data(pos, x, P, Q * q_param, R * r_param)
            smoothed_pos = np.array(xs)[:, :2, 0]
            # import ipdb; ipdb.set_trace()
            filtered_pos = np.array(rts_smoother.filtered_xs)[:, :2, 0]
            for i, date in enumerate(vort_track.vortmax_by_date.keys()):
                tmp = __Tmp()
                tmp.pos = smoothed_pos[i]
                smoothed_dict[date] = tmp
                tmp = __Tmp()
                tmp.pos = filtered_pos[i]
                filtered_dict[date] = tmp

            smoothed_new_d, overlap = _cum_dist(best_track, smoothed_dict)
            filtered_new_d, overlap = _cum_dist(best_track, filtered_dict)

            if overlap >= 6 and smoothed_new_d < min_d:
                optimum_param = (q_param, r_param, 'smoothed')
                print('new optimum param: {0}'.format(optimum_param))
                min_d = smoothed_new_d
                print('min dist {0}'.format(min_d))

                _plot_rts_smoother(rts_smoother)
                plt.title(str(optimum_param))
                plt.plot(best_track.lons, best_track.lats, 'b-')
                plt.pause(0.01)
            if overlap >= 6 and filtered_new_d < min_d:
                optimum_param = (q_param, r_param, 'filtered')
                print('new optimum param: {0}'.format(optimum_param))
                min_d = filtered_new_d

                _plot_rts_smoother(rts_smoother)
                plt.title(str(optimum_param))
                plt.plot(best_track.lons, best_track.lats, 'b-')
                plt.pause(0.01)

    print(min_d / baseline_dist)

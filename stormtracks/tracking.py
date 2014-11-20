from collections import OrderedDict

import numpy as np

from utils.kalman import KalmanFilter
from utils.utils import dist, geo_dist, pairwise, find_extrema


class CycloneTrack(object):
    def __init__(self, vortmax_track, ensemble_member):
        self.vortmax_track = vortmax_track
        self.ensemble_member = ensemble_member

        # self.local_psls = OrderedDict()
        # self.local_vorts = OrderedDict()
        self.min_dists = OrderedDict()
        self.max_windspeeds = OrderedDict()
        self.max_windspeed_positions = OrderedDict()
        self.min_dists = OrderedDict()
        self.pmins = OrderedDict()
        self.pmin_positions = OrderedDict()
        self.p_ambient_diffs = OrderedDict()
        self.t850s = OrderedDict()
        self.t995s = OrderedDict()
        self.capes = OrderedDict()
        self.pwats = OrderedDict()
        self.rh995s = OrderedDict()

        self.dates = self.vortmax_track.dates

    def get_vmax_pos(self, date):
        return self.vortmax_track.vortmax_by_date[date].pos

    def get_vort(self, date):
        return self.vortmax_track.vortmax_by_date[date].vort


class VortMaxTrack(object):
    '''
    Stores a collection of VortMax objects in a list and adds them to a
    dict that is accessible through a date for easy access.
    '''
    def __init__(self, start_vortmax, ensemble_member):
        if len(start_vortmax.prev_vortmax):
            raise Exception('start vortmax must have no previous vortmaxes')

        self.start_vortmax = start_vortmax
        self.ensemble_member = ensemble_member
        self.vortmaxes = []
        self.vortmax_by_date = OrderedDict()

        self._build_track()

    def _rebuild_track(self):
        '''Rebuilds a track, can be useful after deserialization'''
        next_vm = None
        for vm in self.vortmaxes[::-1]:
            if next_vm:
                vm.next_vortmax.append(next_vm)
            next_vm = vm

        self.vortmaxes = []
        self.vortmax_by_date = OrderedDict()

        self._build_track()

    def _build_track(self):
        self.vortmaxes.append(self.start_vortmax)
        self.vortmax_by_date[self.start_vortmax.date] = self.start_vortmax
        if len(self.start_vortmax.next_vortmax):
            vortmax = self.start_vortmax.next_vortmax[0]
            self.vortmax_by_date[vortmax.date] = vortmax

            while len(vortmax.next_vortmax) != 0:
                self.vortmaxes.append(vortmax)
                vortmax = vortmax.next_vortmax[0]
                self.vortmax_by_date[vortmax.date] = vortmax

            self.vortmaxes.append(vortmax)
            self.vortmax_by_date[vortmax.date] = vortmax

        self.lons = np.zeros(len(self.vortmaxes))
        self.lats = np.zeros(len(self.vortmaxes))
        self.dates = np.zeros(len(self.vortmaxes)).astype(object)
        for i, vortmax in enumerate(self.vortmaxes):
            self.lons[i], self.lats[i] = vortmax.pos[0], vortmax.pos[1]
            self.dates[i] = vortmax.date


class VortMax(object):
    '''
    Holds key info (date, position, vorticity value) about a vorticity
    maximum.

    To serialize this class (or any that contain objects of this class)
    you must make sure next_vortmax/prev_vortmax are None.

    :param date: date of vortmax
    :param pos: position (2 element tuple)
    :param vort: vorticity value
    '''
    def __init__(self, date, pos, vort):
        self.date = date
        self.pos = pos
        self.vort = vort
        self.next_vortmax = []
        self.prev_vortmax = []
        self.secondary_vortmax = []

    def add_next(self, vortmax):
        '''Used to make doubly linked list of vortmaxes'''
        self.next_vortmax.append(vortmax)
        vortmax.prev_vortmax.append(self)


class VortmaxFinder(object):
    '''Finds all vortmaxes for a given ensemble member

    :param gem: GlobalEnsembleMember to use
    '''
    def __init__(self, gem):
        self.gem = gem

        # Some settings to document/consider playing with.
        self.use_vort_cuttoff = True
        self.use_dist_cuttoff = True
        self.use_range_cuttoff = True
        self.use_geo_dist = True

        if self.use_geo_dist:
            self.dist = geo_dist
            self.dist_cutoff = geo_dist((0, 0), (2, 0)) * 5
        else:
            self.dist = dist
            self.dist_cutoff = 5

        # self.vort_cutoff = 5e-5 # Old value with wrong vort calc.
        self.vort_cutoff = 2.5e-5

    def find_vort_maxima(self, start_date, end_date, use_upscaled=False):
        '''Runs over the date range looking for all vorticity maxima'''
        if start_date < self.gem.dates[0]:
            raise Exception('Start date is out of date range, try setting the year appropriately')
        elif end_date > self.gem.dates[-1]:
            raise Exception('End date is out of date range, try setting the year appropriately')

        index = np.where(self.gem.dates == start_date)[0][0]
        end_index = np.where(self.gem.dates == end_date)[0][0]

        self.vortmax_time_series = OrderedDict()

        while index <= end_index:
            date = self.gem.dates[index]
            self.gem.set_date(date)

            vortmaxes = []

            if use_upscaled:
                vmaxs = self.gem.c20data.up_vmaxs
            else:
                vmaxs = self.gem.c20data.vmaxs

            for vmax in vmaxs:
                if self.use_range_cuttoff and not (260 < vmax[1][0] < 340 and
                   0 < vmax[1][1] < 60):
                    continue

                if self.use_vort_cuttoff and vmax[0] < self.vort_cutoff:
                    continue

                vortmax = VortMax(date, vmax[1], vmax[0])
                vortmaxes.append(vortmax)

            if self.use_dist_cuttoff:
                secondary_vortmaxes = []
                for i in range(len(vortmaxes)):
                    v1 = vortmaxes[i]
                    for j in range(i + 1, len(vortmaxes)):
                        v2 = vortmaxes[j]
                        if self.dist(v1.pos, v2.pos) < self.dist_cutoff:
                            if v1.vort > v2.vort:
                                v1.secondary_vortmax.append(v2)
                                secondary_vortmaxes.append(v2)
                            elif v1.vort <= v2.vort:
                                v2.secondary_vortmax.append(v1)
                                secondary_vortmaxes.append(v1)

                for v in secondary_vortmaxes:
                    if v in vortmaxes:
                        vortmaxes.remove(v)

            for i, v in enumerate(vortmaxes):
                v.index = i

            self.vortmax_time_series[date] = vortmaxes

            index += 1

        return self.vortmax_time_series


class VortmaxKalmanFilterTracker(object):
    '''Uses a Kalman Filter to try to track vorticity maxima

    The main idea is to use the innovation parameter from the Kalman Filter estimation process
    as a measure of how likely it is that a vorticity maxima in a subsequent timestep belongs to
    the same track as a vorticity maxima in the current timestep'''
    def __init__(self, ensemble_member, Q_mult=0.001, R_mult=0.1):
        self.ensemble_member = ensemble_member
        self.dist_cutoff = 10

        F = np.matrix([[1., 0., 1., 0.],
                       [0., 1., 0., 1.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])

        H = np.matrix([[1., 0., 0., 0.],
                       [0., 1., 0., 0.]])

        self.kf = KalmanFilter(F, H)

        self.Q = np.matrix([[2., 1., 1., 1.],
                            [1., 2., 1., 1.],
                            [1., 1., 2., 1.],
                            [1., 1., 1., 2.]]) * Q_mult

        self.R = np.matrix([[2., 1.],
                            [1., 2.]]) * R_mult

    def _construct_vortmax_tracks_by_date(self, vortmax_tracks):
        self.vort_tracks_by_date = OrderedDict()
        for vortmax_track in vortmax_tracks.values():
            for date in vortmax_track.vortmax_by_date.keys():
                if date not in self.vort_tracks_by_date:
                    self.vort_tracks_by_date[date] = []
                self.vort_tracks_by_date[date].append(vortmax_track)

            for vortmax in vortmax_track.vortmaxes:
                vortmax.prev_vortmax = []
                vortmax.next_vortmax = []

        return self.vort_tracks_by_date

    def track_vort_maxima(self, vortmax_time_series):
        '''Uses a generated list of vortmaxes to track them from one timestep to another

        :param vortmax_time_series: dict of vortmaxes
        :returns: self.vort_tracks_by_date (OrderedDict)
        '''
        vortmax_tracks = {}
        # Loops over 2 lists of vortmaxes
        # import ipdb; ipdb.set_trace()
        for vs1, vs2 in pairwise(vortmax_time_series.values()):
            for v1 in vs1:
                if v1 not in vortmax_tracks:
                    vortmax_track = VortMaxTrack(v1, self.ensemble_member)
                    vortmax_track.xs = [np.matrix([v1.pos[0], v1.pos[1], 0, 0]).T]
                    vortmax_track.Ps = [np.matrix(np.eye(4)) * 10]
                else:
                    vortmax_track = vortmax_tracks[v1]

                min_innov2 = 1e10
                v2next = None

                for v2 in vs2:
                    d = geo_dist(v1.pos, v2.pos)
                    if d < self.dist_cutoff:
                        x = vortmax_track.xs[-1]
                        P = vortmax_track.Ps[-1]

                        # Calc the innovation using a Kalman filter.
                        x, P = self.kf.estimate(x, P, np.matrix(v2.pos).T, self.Q, self.R)

                        innov2 = np.sum(self.kf.y.getA() ** 2)
                        if innov2 < min_innov2:
                            min_innov2 = innov2
                            v2next = v2
                            x_next = x
                            P_next = P

                # Add the vortmax with the smallest innov in the next timestep.
                if v2next:
                    v1.add_next(v2next)
                    vortmax_track.vortmaxes.append(v2next)
                    vortmax_track.xs.append(x_next)
                    vortmax_track.Ps.append(P_next)
                    vortmax_tracks[v2next] = vortmax_track

        if False:
            # Some vortmaxes may have more than one previous vortmax.
            # Find these and choose the nearest one as the actual previous.
            for vs in vortmax_time_series.values():
                for v in vs:
                    if len(v.prev_vortmax) > 1:
                        min_dist = 8
                        vprev = None
                        for pv in v.prev_vortmax:
                            d = geo_dist(pv.pos, v.pos)
                            if d < min_dist:
                                min_dist = d
                                vprev = pv

                        for pv in v.prev_vortmax:
                            if pv != vprev:
                                pv.next_vortmax.remove(v)

                        v.prev_vortmax = [vprev]

            return self._construct_vortmax_tracks_by_date(vortmax_time_series)

        return self._construct_vortmax_tracks_by_date(vortmax_tracks)


class VortmaxNearestNeighbourTracker(object):
    '''Simple nearest neighbour tracker

    Assumes that the two nearest points from one timestep to another belong to the same track.
    '''
    def __init__(self, ensemble_member):
        self.use_geo_dist = True
        self.ensemble_member = ensemble_member

        if self.use_geo_dist:
            self.dist = geo_dist
            self.dist_cutoff = geo_dist((0, 0), (2, 0)) * 8
        else:
            self.dist = dist
            self.dist_cutoff = 5

    def _construct_vortmax_tracks_by_date(self, vortmax_time_series):
        self.vortmax_tracks = []
        self.vort_tracks_by_date = OrderedDict()
        # Find all start vortmaxes (those that have no previous vortmaxes)
        # and use these to generate a track.
        for vortmaxes in vortmax_time_series.values():
            for vortmax in vortmaxes:
                if len(vortmax.prev_vortmax) == 0:
                    vortmax_track = VortMaxTrack(vortmax, self.ensemble_member)
                    if len(vortmax_track.vortmaxes) >= 6:
                        self.vortmax_tracks.append(vortmax_track)

                        for date in vortmax_track.vortmax_by_date.keys():
                            if date not in self.vort_tracks_by_date:
                                self.vort_tracks_by_date[date] = []
                            self.vort_tracks_by_date[date].append(vortmax_track)

                # Allows vort_tracks_by_date to be serialized.
                vortmax.next_vortmax = []
                vortmax.prev_vortmax = []

        return self.vort_tracks_by_date

    def track_vort_maxima(self, vortmax_time_series):
        '''Uses a generated list of vortmaxes to track them from one timestep to another

        :param vortmax_time_series: dict of vortmaxes
        :returns: self.vort_tracks_by_date (OrderedDict)
        '''
        # Loops over 2 lists of vortmaxes
        for vs1, vs2 in pairwise(vortmax_time_series.values()):
            for v1 in vs1:
                min_dist = self.dist_cutoff
                v2next = None

                # Find the nearest vortmax in the next timestep.
                for v2 in vs2:
                    d = self.dist(v1.pos, v2.pos)
                    if d < min_dist:
                        min_dist = d
                        v2next = v2

                # Add the nearest vortmax in the next timestep.
                if v2next:
                    v1.add_next(v2next)
                    if len(v1.next_vortmax) != 1:
                        raise Exception('There should only ever be one next_vortmax')

        # Some vortmaxes may have more than one previous vortmax.
        # Find these and choose the nearest one as the actual previous.
        for vs in vortmax_time_series.values():
            for v in vs:
                if len(v.prev_vortmax) > 1:
                    min_dist = self.dist_cutoff
                    vprev = None
                    for pv in v.prev_vortmax:
                        d = self.dist(pv.pos, v.pos)
                        if d < min_dist:
                            min_dist = d
                            vprev = pv

                    for pv in v.prev_vortmax:
                        if pv != vprev:
                            pv.next_vortmax.remove(v)

                    v.prev_vortmax = [vprev]

        return self._construct_vortmax_tracks_by_date(vortmax_time_series)


class FieldFinder(object):
    def __init__(self, c20data, vort_tracks_by_date, ensemble_member):
        self.c20data = c20data
        self.vort_tracks_by_date = vort_tracks_by_date
        self.ensemble_member = ensemble_member
        self.cyclone_tracks = {}

    def collect_fields(self):
        # import ipdb; ipdb.set_trace()
        for date in self.vort_tracks_by_date.keys():
            self.c20data.set_date(date, self.ensemble_member)

            vort_tracks = self.vort_tracks_by_date[date]
            for vort_track in vort_tracks:
                cyclone_track = None
                if vort_track not in self.cyclone_tracks:
                    # TODO: reenable.
                    # if vort_track.ensemble_member != self.ensemble_member:
                        # raise Exception('Vort track is from a different ensemble member')
                    if len(vort_track.vortmaxes) >= 6:
                        cyclone_track = CycloneTrack(vort_track, self.ensemble_member)
                        self.cyclone_tracks[vort_track] = cyclone_track
                else:
                    cyclone_track = self.cyclone_tracks[vort_track]

                if cyclone_track:
                    self.add_fields_to_track(date, cyclone_track)

    def add_fields_to_track(self, date, cyclone_track):
        actual_vmax_pos = cyclone_track.get_vmax_pos(date)
        # Round values to the nearest multiple of 2 (vmax_pos can come from an interpolated field)
        vmax_pos = tuple([int(round(p / 2.)) * 2 for p in actual_vmax_pos])
        lon_index = np.where(self.c20data.lons == vmax_pos[0])[0][0]
        lat_index = np.where(self.c20data.lats == vmax_pos[1])[0][0]

        min_lon, max_lon = lon_index - 5, lon_index + 6
        min_lat, max_lat = lat_index - 5, lat_index + 6
        local_slice = (slice(min_lat, max_lat), slice(min_lon, max_lon))

        local_psl = self.c20data.psl[local_slice].copy()
        local_vort = self.c20data.vort[local_slice].copy()
        local_u = self.c20data.u[local_slice].copy()
        local_v = self.c20data.v[local_slice].copy()

        # this should save a lot of hd space by not saving these.

        # cyclone_track.local_psls[date] = local_psl
        # cyclone_track.local_vorts[date] = local_vort

        local_windspeed = np.sqrt(local_u ** 2 + local_v ** 2)
        max_windspeed_pos = np.unravel_index(np.argmax(local_windspeed), (11, 11))
        max_windspeed = local_windspeed[max_windspeed_pos]

        lon = self.c20data.lons[min_lon + max_windspeed_pos[1]]
        lat = self.c20data.lats[min_lat + max_windspeed_pos[0]]

        cyclone_track.max_windspeeds[date] = max_windspeed
        cyclone_track.max_windspeed_positions[date] = (lon, lat)

        e, index_pmaxs, index_pmins = find_extrema(local_psl)
        min_dist = 1000
        pmin = None
        pmin_pos = None
        for index_pmin in index_pmins:
            lon = self.c20data.lons[min_lon + index_pmin[1]]
            lat = self.c20data.lats[min_lat + index_pmin[0]]

            local_pmin = local_psl[index_pmin[0], index_pmin[1]]
            local_pmin_pos = (lon, lat)
            dist = geo_dist(actual_vmax_pos, local_pmin_pos)
            if dist < min_dist:
                min_dist = dist
                pmin = local_pmin
                pmin_pos = local_pmin_pos

        cyclone_track.min_dists[date] = min_dist
        cyclone_track.pmins[date] = pmin
        cyclone_track.pmin_positions[date] = pmin_pos

        if pmin:
            cyclone_track.p_ambient_diffs[date] = local_psl.mean() - pmin
        else:
            cyclone_track.p_ambient_diffs[date] = None

        cyclone_track.t850s[date] = self.c20data.t850[lat_index, lon_index]
        cyclone_track.t995s[date] = self.c20data.t995[lat_index, lon_index]
        cyclone_track.capes[date] = self.c20data.cape[lat_index, lon_index]
        cyclone_track.pwats[date] = self.c20data.pwat[lat_index, lon_index]
        # No longer using due to it not having# No longer using due to it not having much
        # discriminatory power and space constraints.
        cyclone_track.rh995s[date] = 0

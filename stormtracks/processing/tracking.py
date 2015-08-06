from collections import OrderedDict, namedtuple
import datetime as dt

import numpy as np

from .. import setup_logging
from ..utils.utils import dist, geo_dist, pairwise, find_extrema

log = setup_logging.get_logger('st.tracking')

NUM_ENSEMBLE_MEMBERS = 56


class VortmaxNearestNeighbourTracker(object):
    '''Simple nearest neighbour tracker

    Assumes that the two nearest points from one timestep to another belong to the same track.
    '''
    def __init__(self):
        self.use_geo_dist = True

        if self.use_geo_dist:
            self.dist = geo_dist
            self.dist_cutoff = geo_dist((0, 0), (2, 0)) * 8
        else:
            self.dist = dist
            self.dist_cutoff = 5

    def _construct_vortmax_tracks_by_date(self, all_vortmax_time_series):
        self.all_vort_tracks_by_date = []
        for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
            vortmax_time_series = all_vortmax_time_series[ensemble_member]
            vortmax_tracks = []
            vort_tracks_by_date = OrderedDict()
            # Find all start vortmaxes (those that have no previous vortmaxes)
            # and use these to generate a track.
            for vortmaxes in vortmax_time_series.values():
                for vortmax in vortmaxes:
                    if len(vortmax.prev_vortmax) == 0:
                        vortmax_track = VortMaxTrack(vortmax, ensemble_member)
                        if len(vortmax_track.vortmaxes) >= 6:
                            vortmax_tracks.append(vortmax_track)

                            for date in vortmax_track.vortmax_by_date.keys():
                                if date not in vort_tracks_by_date:
                                    vort_tracks_by_date[date] = []
                                vort_tracks_by_date[date].append(vortmax_track)

                    # Allows vort_tracks_by_date to be serialized.
                    vortmax.next_vortmax = []
                    vortmax.prev_vortmax = []

            self.all_vort_tracks_by_date.append(vort_tracks_by_date)
        return self.all_vort_tracks_by_date


    def track(self, df):
        all_vortmax_time_series = []
        for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
            vortmax_time_series = OrderedDict()
            all_vortmax_time_series.append(vortmax_time_series)
            for i in df[df.em == ensemble_member].index:
                row = df.loc[i]
                if row.date not in vortmax_time_series:
                    vortmax_time_series[row.date] = []
                vortmax = VortMax(row.date, (row.lon, row.lat), row.vort)
                vortmax_time_series[vortmax.date].append(vortmax)
        print('Made objects')
        return self.track_vort_maxima(all_vortmax_time_series)


    def track_vort_maxima(self, all_vortmax_time_series):
        '''Uses a generated list of vortmaxes to track them from one timestep to another

        :param vortmax_time_series: dict of vortmaxes
        :returns: self.vort_tracks_by_date (OrderedDict)
        '''
        for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
            log.info('Tracking vortmaxima: {0}'.format(ensemble_member))
            vortmax_time_series = all_vortmax_time_series[ensemble_member]
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

        return self._construct_vortmax_tracks_by_date(all_vortmax_time_series)


class FieldFinder(object):
    def __init__(self, c20data, all_vort_tracks_by_date):
        self.c20data = c20data
        self.all_vort_tracks_by_date = all_vort_tracks_by_date
        self.all_cyclone_tracks = []
        for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
            self.all_cyclone_tracks.append({})

    def collect_fields(self, start_date, end_date):
        # import ipdb; ipdb.set_trace()
        index = np.where(self.c20data.dates == start_date)[0][0]
        end_index = np.where(self.c20data.dates == end_date)[0][0]

        while index <= end_index:
            date = self.c20data.dates[index]
            log.info('Collecting fields for: {0}'.format(date))
            self.c20data.set_date(date)
            for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
                cyclone_tracks = self.all_cyclone_tracks[ensemble_member]

                if date not in self.all_vort_tracks_by_date[ensemble_member]:
                    continue

                vort_tracks = self.all_vort_tracks_by_date[ensemble_member][date]
                for vort_track in vort_tracks:
                    cyclone_track = None
                    if vort_track not in cyclone_tracks:
                        if len(vort_track.vortmaxes) >= 6:
                            cyclone_track = CycloneTrack(vort_track, ensemble_member)
                            cyclone_tracks[vort_track] = cyclone_track
                    else:
                        cyclone_track = cyclone_tracks[vort_track]

                    if cyclone_track:
                        self.add_fields_to_track(date, cyclone_track, ensemble_member)
            index += 1

    def add_fields_to_track(self, date, cyclone_track, ensemble_member):
        actual_vmax_pos = cyclone_track.get_vmax_pos(date)
        # Round values to the nearest multiple of 2 (vmax_pos can come from an interpolated field)
        vmax_pos = tuple([int(round(p / 2.)) * 2 for p in actual_vmax_pos])
        lon_index = np.where(self.c20data.lons == vmax_pos[0])[0][0]
        lat_index = np.where(self.c20data.lats == vmax_pos[1])[0][0]

        min_lon, max_lon = lon_index - 5, lon_index + 6
        min_lat, max_lat = lat_index - 5, lat_index + 6
        local_slice = (slice(min_lat, max_lat), slice(min_lon, max_lon))

        local_prmsl = self.c20data.prmsl[ensemble_member][local_slice].copy()

	# 9950 pressure level.
        local_windspeed = np.sqrt(self.c20data.u9950[ensemble_member][local_slice] ** 2 +
                                  self.c20data.v9950[ensemble_member][local_slice] ** 2)
        max_windspeed_pos = np.unravel_index(np.argmax(local_windspeed), (11, 11))
        max_windspeed = local_windspeed[max_windspeed_pos]

        lon = self.c20data.lons[min_lon + max_windspeed_pos[1]]
        lat = self.c20data.lats[min_lat + max_windspeed_pos[0]]

        cyclone_track.max_windspeeds[date] = max_windspeed
        cyclone_track.max_windspeed_positions[date] = (lon, lat)

        e, index_pmaxs, index_pmins = find_extrema(local_prmsl)
        min_dist = 1000
        pmin = None
        pmin_pos = None
        for index_pmin in index_pmins:
            lon = self.c20data.lons[min_lon + index_pmin[1]]
            lat = self.c20data.lats[min_lat + index_pmin[0]]

            local_pmin = local_prmsl[index_pmin[0], index_pmin[1]]
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
            cyclone_track.p_ambient_diffs[date] = local_prmsl.mean() - pmin
        else:
            cyclone_track.p_ambient_diffs[date] = None

        cyclone_track.t850s[date] = self.c20data.t850[ensemble_member][lat_index, lon_index]
        cyclone_track.t995s[date] = self.c20data.t995[ensemble_member][lat_index, lon_index]
        cyclone_track.capes[date] = self.c20data.cape[ensemble_member][lat_index, lon_index]
        cyclone_track.pwats[date] = self.c20data.pwat[ensemble_member][lat_index, lon_index]
        # No longer using due to it not having much
        # discriminatory power and space constraints.
        cyclone_track.rh995s[date] = 0




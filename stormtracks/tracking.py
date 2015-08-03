from collections import OrderedDict
import datetime as dt

import numpy as np
import pandas as pd

from logger import get_logger
from utils.utils import dist, geo_dist, pairwise, find_extrema
# from tracking import VortMax, VortMaxTrack, CycloneTrack

log = get_logger('analysis.tracking', console_level_str='INFO')

NUM_ENSEMBLE_MEMBERS = 56


class FullVortmaxFinder(object):
    '''Finds all vortmaxes across ensemble members'''
    def __init__(self, fc20data, use_dist_cutoff=True):
        self.fc20data = fc20data

        # Some settings to document/consider playing with.
        self.use_vort_cutoff = True
        self.use_dist_cutoff = use_dist_cutoff
        self.use_range_cutoff = True
        self.use_geo_dist = True

        if self.use_geo_dist:
            self.dist = geo_dist
            self.dist_cutoff = geo_dist((0, 0), (2, 0)) * 5
        else:
            self.dist = dist
            self.dist_cutoff = 5

        # self.vort_cutoff = 5e-5 # Old value with wrong vort calc.
        # self.vort_cutoff = 2.5e-5
        self.vort_cutoff = 1e-5

    def find_vort_maxima(self, start_date, end_date):
        '''Runs over the date range looking for all vorticity maxima'''
        if start_date < self.fc20data.dates[0]:
            raise Exception('Start date is out of date range, try setting the year appropriately')
        elif end_date > self.fc20data.dates[-1]:
            raise Exception('End date is out of date range, try setting the year appropriately')
        index = np.where(self.fc20data.dates == start_date)[0][0]
        end_index = np.where(self.fc20data.dates == end_date)[0][0]

        self.all_vortmax_time_series = []
        results = []

        for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
            self.all_vortmax_time_series.append(OrderedDict())

        while index <= end_index:
            date = self.fc20data.dates[index]
            self.fc20data.set_date(date)
            start = dt.datetime.now()

            log.info('Finding vortmaxima: {0}'.format(date))

            for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
                vortmax_time_series = self.all_vortmax_time_series[ensemble_member]
                vortmaxes = []
                vmaxs = self.fc20data.vmaxs[ensemble_member]

                for vmax in vmaxs:
                    if self.use_range_cutoff and not (260 < vmax[1][0] < 340 and
                       0 < vmax[1][1] < 60):
                        continue

                    if self.use_vort_cutoff and vmax[0] < self.vort_cutoff:
                        continue

                    vortmax = VortMax(date, vmax[1], vmax[0])
                    vortmaxes.append(vortmax)

                if self.use_dist_cutoff:
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

                vortmax_time_series[date] = vortmaxes

                for vortmax in vortmaxes:
                    row = {'date': date,
                           'em': ensemble_member,
                           'lon': vortmax.pos[0],
                           'lat': vortmax.pos[1],
                           'vort': vortmax.vort}
                    res = self.get_other_fields(ensemble_member, vortmax)
                    row.update(res)
                    results.append(row)

            end = dt.datetime.now()
            print('  Found vortmaxima and fields in {}'.format(end - start))
            index += 1

        columns = ['date', 'em', 
                   'lon', 
                   'lat', 
                   'max_ws_lon',
                   'max_ws_lat',
                   'pmin_lon',
                   'pmin_lat',
                   'vort', 
                   'max_ws',
                   'pmin_dist',
                   'pmin',
                   'p_ambient_diff',
                   't850s',
                   't995s',
                   'capes',
                   'pwats']
        df = pd.DataFrame(results, columns=columns)
        end = dt.datetime.now()
        return df

    def get_other_fields(self, ensemble_member, vortmax):
        res = {}
        vmax_pos = vortmax.pos
        # Round values to the nearest multiple of 2 (vmax_pos can come from an interpolated field)
        # vmax_pos = tuple([int(round(p / 2.)) * 2 for p in actual_vmax_pos])
        lon_index = np.where(self.fc20data.lons == vmax_pos[0])[0][0]
        lat_index = np.where(self.fc20data.lats == vmax_pos[1])[0][0]

        min_lon, max_lon = lon_index - 5, lon_index + 6
        min_lat, max_lat = lat_index - 5, lat_index + 6
        local_slice = (slice(min_lat, max_lat), slice(min_lon, max_lon))

        local_prmsl = self.fc20data.prmsl[ensemble_member][local_slice].copy()

        local_windspeed = np.sqrt(self.fc20data.u[ensemble_member][local_slice] ** 2 +
                                  self.fc20data.v[ensemble_member][local_slice] ** 2)
        max_windspeed_pos = np.unravel_index(np.argmax(local_windspeed), (11, 11))
        max_windspeed = local_windspeed[max_windspeed_pos]

        lon = self.fc20data.lons[min_lon + max_windspeed_pos[1]]
        lat = self.fc20data.lats[min_lat + max_windspeed_pos[0]]

        res['max_ws'] = max_windspeed
        res['max_ws_lon'] = lon
        res['max_ws_lat'] = lat

        e, index_pmaxs, index_pmins = find_extrema(local_prmsl)
        min_dist = 1000
        pmin = None
        pmin_pos = None
        for index_pmin in index_pmins:
            lon = self.fc20data.lons[min_lon + index_pmin[1]]
            lat = self.fc20data.lats[min_lat + index_pmin[0]]

            local_pmin = local_prmsl[index_pmin[0], index_pmin[1]]
            local_pmin_pos = (lon, lat)
            dist = geo_dist(vmax_pos, local_pmin_pos)
            if dist < min_dist:
                min_dist = dist
                pmin = local_pmin
                pmin_pos = local_pmin_pos

        res['pmin_dist'] = min_dist
        if pmin:
            res['pmin'] = pmin
            res['pmin_lon'] = pmin_pos[0]
            res['pmin_lat'] = pmin_pos[1]
            res['p_ambient_diff'] = local_prmsl.mean() - pmin
        else:
            res['pmin'] = local_prmsl.min()
            res['pmin_lon'] = None
            res['pmin_lat'] = None
            res['p_ambient_diff'] = local_prmsl.mean() - local_prmsl.min()

        res['t850s'] = self.fc20data.t850[ensemble_member][lat_index, lon_index]
        res['t995s'] = self.fc20data.t995[ensemble_member][lat_index, lon_index]
        res['capes'] = self.fc20data.cape[ensemble_member][lat_index, lon_index]
        res['pwats'] = self.fc20data.pwat[ensemble_member][lat_index, lon_index]
        # No longer using due to it not having much
        # discriminatory power and space constraints.
        # cyclone_track.rh995s[date] = 0
        return res



class FullVortmaxNearestNeighbourTracker(object):
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


class FullFieldFinder(object):
    def __init__(self, fc20data, all_vort_tracks_by_date):
        self.fc20data = fc20data
        self.all_vort_tracks_by_date = all_vort_tracks_by_date
        self.all_cyclone_tracks = []
        for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
            self.all_cyclone_tracks.append({})

    def collect_fields(self, start_date, end_date):
        # import ipdb; ipdb.set_trace()
        index = np.where(self.fc20data.dates == start_date)[0][0]
        end_index = np.where(self.fc20data.dates == end_date)[0][0]

        while index <= end_index:
            date = self.fc20data.dates[index]
            log.info('Collecting fields for: {0}'.format(date))
            self.fc20data.set_date(date)
            for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
                cyclone_tracks = self.all_cyclone_tracks[ensemble_member]

                if date not in self.all_vort_tracks_by_date[ensemble_member]:
                    continue

                vort_tracks = self.all_vort_tracks_by_date[ensemble_member][date]
                for vort_track in vort_tracks:
                    cyclone_track = None
                    if vort_track not in cyclone_tracks:
                        # TODO: reenable.
                        # if vort_track.ensemble_member != self.ensemble_member:
                            # raise Exception('Vort track is from a different ensemble member')
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
        lon_index = np.where(self.fc20data.lons == vmax_pos[0])[0][0]
        lat_index = np.where(self.fc20data.lats == vmax_pos[1])[0][0]

        min_lon, max_lon = lon_index - 5, lon_index + 6
        min_lat, max_lat = lat_index - 5, lat_index + 6
        local_slice = (slice(min_lat, max_lat), slice(min_lon, max_lon))

        local_prmsl = self.fc20data.prmsl[ensemble_member][local_slice].copy()
        # local_vort = self.c20data.vort[local_slice].copy()
        # local_u = .copy()
        # local_v = .copy()

        # this should save a lot of hd space by not saving these.

        # cyclone_track.local_prmsl[date] = local_prmsl
        # cyclone_track.local_vorts[date] = local_vort

        local_windspeed = np.sqrt(self.fc20data.u[ensemble_member][local_slice] ** 2 +
                                  self.fc20data.v[ensemble_member][local_slice] ** 2)
        max_windspeed_pos = np.unravel_index(np.argmax(local_windspeed), (11, 11))
        max_windspeed = local_windspeed[max_windspeed_pos]

        lon = self.fc20data.lons[min_lon + max_windspeed_pos[1]]
        lat = self.fc20data.lats[min_lat + max_windspeed_pos[0]]

        cyclone_track.max_windspeeds[date] = max_windspeed
        cyclone_track.max_windspeed_positions[date] = (lon, lat)

        e, index_pmaxs, index_pmins = find_extrema(local_prmsl)
        min_dist = 1000
        pmin = None
        pmin_pos = None
        for index_pmin in index_pmins:
            lon = self.fc20data.lons[min_lon + index_pmin[1]]
            lat = self.fc20data.lats[min_lat + index_pmin[0]]

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

        cyclone_track.t850s[date] = self.fc20data.t850[ensemble_member][lat_index, lon_index]
        cyclone_track.t995s[date] = self.fc20data.t995[ensemble_member][lat_index, lon_index]
        cyclone_track.capes[date] = self.fc20data.cape[ensemble_member][lat_index, lon_index]
        cyclone_track.pwats[date] = self.fc20data.pwat[ensemble_member][lat_index, lon_index]
        # No longer using due to it not having much
        # discriminatory power and space constraints.
        cyclone_track.rh995s[date] = 0




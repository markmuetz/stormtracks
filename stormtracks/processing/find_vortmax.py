from collections import OrderedDict, namedtuple
import datetime as dt

import numpy as np
import pandas as pd

from .. import setup_logging
from ..utils.utils import dist, geo_dist, find_extrema

log = setup_logging.get_logger('st.find_vortmax')

VortMax = namedtuple('VortMax', ['date', 'pos', 'vort'])

NUM_ENSEMBLE_MEMBERS = 56


class VortmaxFinder(object):
    '''Finds all vortmaxes across ensemble members'''
    def __init__(self, c20data, use_dist_cutoff=True):
        self.c20data = c20data

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
	log.info('VortmaxFinder setup:')
	for setting in ['use_vort_cutoff',
		        'use_dist_cutoff',
			'use_range_cutoff',
			'use_geo_dist',
			'vort_cutoff']:
	    log.info('{}: {}'.format(setting, getattr(self, setting)))

    def find_vort_maxima(self, start_date, end_date):
        '''Runs over the date range looking for all vorticity maxima'''
        if start_date < self.c20data.dates[0]:
            raise Exception('Start date is out of date range, try setting the year appropriately')
        elif end_date > self.c20data.dates[-1]:
            raise Exception('End date is out of date range, try setting the year appropriately')
	log.info('finding vortmaxima in range {}-{}'.format(start_date, end_date))
        index = np.where(self.c20data.dates == start_date)[0][0]
        end_index = np.where(self.c20data.dates == end_date)[0][0]

        self.all_vortmax_time_series = []
        results = []

        start = dt.datetime.now()

        for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
            self.all_vortmax_time_series.append(OrderedDict())

        while index <= end_index:
            date = self.c20data.dates[index]
            self.c20data.set_date(date)

            print('Finding vortmaxima: {0}'.format(date))
            log.debug('Finding vortmaxima: {0}'.format(date))

            for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
                vortmax_time_series = self.all_vortmax_time_series[ensemble_member]
                vortmaxes = []
                vmaxs = self.c20data.vmaxs850[ensemble_member]

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
                           'vort850': vortmax.vort}
                    res = self.get_other_fields(ensemble_member, vortmax)
                    row.update(res)
                    results.append(row)

            index += 1

	end = dt.datetime.now()
	log.info('Found vortmaxima and fields in {}'.format(end - start))

        columns = ['date', 'em', 
                   'lon', 
                   'lat', 
                   'max_ws_lon',
                   'max_ws_lat',
                   'pmin_lon',
                   'pmin_lat',
                   'vort9950', 
                   'vort850', 
                   'max_ws',
                   'prmsl',
                   'pmin_dist',
                   'pmin',
                   'p_ambient_diff',
                   't850',
                   't9950',
                   'cape',
                   'pwat']
        df = pd.DataFrame(results, columns=columns)
        end = dt.datetime.now()
        return df

    def get_other_fields(self, ensemble_member, vortmax):
        res = {}
        vmax_pos = vortmax.pos
        # Round values to the nearest multiple of 2 (vmax_pos can come from an interpolated field)
        # vmax_pos = tuple([int(round(p / 2.)) * 2 for p in actual_vmax_pos])
        lon_index = np.where(self.c20data.lons == vmax_pos[0])[0][0]
        lat_index = np.where(self.c20data.lats == vmax_pos[1])[0][0]

        min_lon, max_lon = lon_index - 5, lon_index + 6
        min_lat, max_lat = lat_index - 5, lat_index + 6
        local_slice = (slice(min_lat, max_lat), slice(min_lon, max_lon))

        local_prmsl = self.c20data.prmsl[ensemble_member][local_slice].copy()

        local_windspeed = np.sqrt(self.c20data.u9950[ensemble_member][local_slice] ** 2 +
                                  self.c20data.v9950[ensemble_member][local_slice] ** 2)
        max_windspeed_pos = np.unravel_index(np.argmax(local_windspeed), (11, 11))
        max_windspeed = local_windspeed[max_windspeed_pos]

        lon = self.c20data.lons[min_lon + max_windspeed_pos[1]]
        lat = self.c20data.lats[min_lat + max_windspeed_pos[0]]

        res['max_ws'] = max_windspeed
        res['max_ws_lon'] = lon
        res['max_ws_lat'] = lat

        res['prmsl'] = self.c20data.prmsl[ensemble_member][lat_index, lon_index]
        e, index_pmaxs, index_pmins = find_extrema(local_prmsl)
        min_dist = 1000
        pmin = None
        pmin_pos = None
        for index_pmin in index_pmins:
            lon = self.c20data.lons[min_lon + index_pmin[1]]
            lat = self.c20data.lats[min_lat + index_pmin[0]]

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

        res['vort9950'] = self.c20data.vort9950[ensemble_member][lat_index, lon_index]
        res['t850'] = self.c20data.t850[ensemble_member][lat_index, lon_index]
        res['t9950'] = self.c20data.t9950[ensemble_member][lat_index, lon_index]
        res['cape'] = self.c20data.cape[ensemble_member][lat_index, lon_index]
        res['pwat'] = self.c20data.pwat[ensemble_member][lat_index, lon_index]
        # No longer using due to it not having much
        # discriminatory power and space constraints.
        # cyclone_track.rh995s[date] = 0
        return res

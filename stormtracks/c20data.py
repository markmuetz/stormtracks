import os
import time
import datetime as dt

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from scipy.ndimage.filters import maximum_filter, minimum_filter

from utils.c_wrapper import cvort, cvort4
from settings import settings

DATA_DIR = os.path.join(settings.DATA_DIR, 'c20_full')

EARTH_RADIUS = 6371
EARTH_CIRC = EARTH_RADIUS * 2 * np.pi

class C20Data(object):
    def __init__(self, 
	        start_year, ensemble=True, smoothing=False, verbose=True,
		upscaling=False):
        self._year = start_year
        self.dx = None
        self.date = None
        self.smoothing = smoothing
        self.ensemble = ensemble
        self.verbose = verbose
        self.upscaling = upscaling

        self.debug = False

        self.load_datasets(self._year)

    def __say(self, text):
        if self.verbose:
            print(text)

    def set_year(self, year):
        self._year = year
        self.close_datasets()
        self.load_datasets(self._year)

    def close_datasets(self):
        self.nc_prmsl.close()
        self.nc_u.close()
        self.nc_v.close()

    def load_datasets(self, year):
        y = str(year)
        if not self.ensemble:
            self.nc_prmsl = Dataset('{0}/{0}/prmsl.{0}.nc'.format(y))
            self.nc_u = Dataset('data/c20/{0}/uwnd.sig995.{0}.nc'.format(y))
            self.nc_v = Dataset('data/c20/{0}/vwnd.sig995.{0}.nc'.format(y))
            start_date = dt.datetime(1800, 1, 1)
            hours_since_1800 = self.nc_prmsl.variables['time'][:]
            self.dates = np.array([start_date + dt.timedelta(hs / 24.) for hs in hours_since_1800])
        else:
            self.nc_prmsl = Dataset('{0}/{1}/prmsl_{1}.nc'.format(DATA_DIR, y))
            self.nc_u = Dataset('{0}/{1}/u9950_{1}.nc'.format(DATA_DIR, y))
            self.nc_v = Dataset('{0}/{1}/v9950_{1}.nc'.format(DATA_DIR, y))
            start_date = dt.datetime(1, 1, 1)
            hours_since_JC = self.nc_prmsl.variables['time'][:]
            self.dates = np.array([start_date + dt.timedelta(hs / 24.) - dt.timedelta(2) for hs in hours_since_JC])
            self.number_enseble_members = self.nc_prmsl.variables['prmsl'].shape[1]

        self.lon = self.nc_prmsl.variables['lon'][:]
        self.lat = self.nc_prmsl.variables['lat'][:]

        dlon = self.lon[2] - self.lon[0]

        # N.B. array as dx varies with lat.
        self.dx = (dlon * np.cos(self.lat * np.pi / 180) * EARTH_CIRC)
        self.dy = (self.lat[0] - self.lat[2]) * EARTH_CIRC

        self.f_lon = interp1d(np.arange(0, 180), self.lon)
        self.f_lat = interp1d(np.arange(0, 91), self.lat)

    def first_date(self, ensemble_member=0, ensemble_mode='member'):
        return self.set_date(self.dates[0], ensemble_member, ensemble_mode)

    def next_date(self, ensemble_member=0, ensemble_mode='member'):
        index = np.where(self.dates == self.date)[0][0]
        if index < len(self.dates):
            date = self.dates[index + 1]
            return self.set_date(date, ensemble_member, ensemble_mode)
        else:
            return None

    def prev_date(self, ensemble_member=0, ensemble_mode='member'):
        index = np.where(self.dates == self.date)[0][0]
        if index > 0:
            date = self.dates[index - 1]
            return self.set_date(date, ensemble_member, ensemble_mode)
        else:
            return None

    def set_date(self, date, ensemble_member=0, ensemble_mode='member'):
        if date != self.date or ensemble_member != self.ensemble_member:
            try:
                self.__say("Setting date to {0}".format(date))
                index = np.where(self.dates == date)[0][0]
                self.date = date
                self.ensemble_member = ensemble_member
                self.ensemble_mode = ensemble_mode
                if not self.ensemble:
                    self.__process_data(index)
                else:
                    self.__process_ensemble_data(index, ensemble_member, ensemble_mode)
            except:
                self.date = None
                self.ensemble_member = None
                self.ensemble_mode = None
                raise
        return date

    def cvorticity(self, u, v):
        vort = np.zeros_like(u)
        cvort(u, v, u.shape[0], u.shape[1], self.dx, self.dy, vort)
        return vort

    def cvorticity4(self, u, v):
        '''Taken from Walsh's Algorithm'''
        vort = np.zeros_like(u)
        cvort4(u, v, u.shape[0], u.shape[1], self.dx, self.dy, vort)
        return vort

    def vorticity(self, u, v):
        vort = np.zeros_like(u)

        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                du_dy = (u[i + 1, j] - u[i - 1, j])/ self.dy
                dv_dx = (v[i, j + 1] - v[i, j - 1])/ self.dx[i]

                vort[i, j] = dv_dx - du_dy
        return vort

    def fourth_order_vorticity(self, u, v):
        '''Taken from Walsh's Algorithm'''
        vort = np.zeros_like(u)

        for i in range(2, u.shape[0] - 2):
            for j in range(2, u.shape[1] - 2):
                du_dy1 = 2 * (u[i + 1, j] - u[i - 1, j]) / (3 * self.dy)
                du_dy2 = (u[i + 2, j] - u[i - 2, j]) / (12 * self.dy)
                du_dy = du_dy1 - du_dy2

                dv_dx1 = 2 * (v[i, j + 1] - v[i, j - 1]) / (3 * self.dx[i])
                dv_dx2 = (v[i, j + 2] - v[i, j - 2]) / (12 * self.dx[i])
                dv_dx = dv_dx1 - dv_dx2

                vort[i, j] = dv_dx - du_dy
        return vort

    def __process_data(self, i):
        start = time.time()
        self.psl = self.nc_prmsl.variables['prmsl'][i]

        # TODO: Why minus sign?
        self.u = - self.nc_u.variables['uwnd'][i]
        self.v = self.nc_v.variables['vwnd'][i]

        end = time.time()
        self.__say('  Loaded psl, u, v in {0}'.format(end - start))

        start = time.time()
        self.vort  = self.vorticity(self.u, self.v, self.lon, self.lat)
        self.vort4 = self.fourth_order_vorticity(self.u, self.v, self.lon, self.lat)
        end = time.time()
        self.__say("  Calc'd vorticity in {0}".format(end - start))

        start = time.time()
        e, index_pmaxs, index_pmins = find_extrema2(self.psl)
        self.pmins = [(self.psl[pmin[0], pmin[1]], (self.lon[pmin[1]], self.lat[pmin[0]])) for pmin in index_pmins]
        e, index_vmaxs, index_vmins = find_extrema2(self.vort)
        self.vmaxs = [(self.vort[vmax[0], vmax[1]], (self.lon[vmax[1]], self.lat[vmax[0]])) for vmax in index_vmaxs]

        end = time.time()
        self.__say('  Found maxima/minima in {0}'.format(end - start))

        if self.smoothing:
            start = time.time()
            self.smoothed_vort = ndimage.filters.gaussian_filter(self.vort, 1, mode='nearest')
            e, index_svmaxs, index_svmins = find_extrema2(self.smoothed_vort)
            self.smoothed_vmaxs = [(self.smoothed_vort[svmax[0], svmax[1]], (self.lon[svmax[1]], self.lat[svmax[0]])) for svmax in index_svmaxs]
            end = time.time()
            self.__say('  Smoothed vorticity in {0}'.format(end - start))
	

    def __process_ensemble_data(self, i, ensemble_member, ensemble_mode):
        if ensemble_mode not in ['member', 'mean', 'full', 'diff']:
            raise Exception('ensemble_mode should be one of member, mean, diff or full')

        if ensemble_mode == 'member':
            if ensemble_member < 0 or ensemble_member >= self.number_enseble_members:
                raise Exception('Ensemble member must be be between 0 and {0}'.format(self.number_enseble_members))

        start = time.time()
        if ensemble_mode == 'member':
            self.psl = self.nc_prmsl.variables['prmsl'][i, ensemble_member]
        elif ensemble_mode == 'mean':
            self.psl = self.nc_prmsl.variables['prmsl'][i].mean(axis=0)
        elif ensemble_mode == 'diff':
            self.psl = self.nc_prmsl.variables['prmsl'][i].max(axis=0) - self.nc_prmsl.variables['prmsl'][i].min(axis=0)
        elif ensemble_mode == 'full':
            self.psl = self.nc_prmsl.variables['prmsl'][i]

        # TODO: Why minus sign?
        if ensemble_mode == 'member':
            self.u = - self.nc_u.variables['u9950'][i, ensemble_member]
            self.v = self.nc_v.variables['v9950'][i, ensemble_member]
        elif ensemble_mode == 'mean':
            self.u = - self.nc_u.variables['u9950'][i].mean(axis=0)
            self.v = self.nc_v.variables['v9950'][i].mean(axis=0)
        elif ensemble_mode == 'diff':
            self.u =  - self.nc_u.variables['u9950'][i].max(axis=0) - self.nc_u.variables['u9950'][i].min(axis=0) 
            self.v =  self.nc_v.variables['v9950'][i].max(axis=0) - self.nc_v.variables['v9950'][i].min(axis=0)
        elif ensemble_mode == 'full':
            self.u = - self.nc_u.variables['u9950'][i]
            self.v = self.nc_v.variables['v9950'][i]

        end = time.time()
        self.__say('  Loaded psl, u, v in {0}'.format(end - start))

        start = time.time()
        if ensemble_mode in ['member', 'mean', 'diff']:
            self.vort  = self.cvorticity(self.u, self.v)
            self.vort4 = self.cvorticity4(self.u, self.v)
        else:
            vort = []
            vort4 = []
            for i in range(self.number_enseble_members):
                vort.append(self.cvorticity(self.u[i], self.v[i]))
                vort4.append(self.cvorticity4(self.u[i], self.v[i]))
            self.vort = np.array(vort)
            self.vort4 = np.array(vort4)

        end = time.time()
        self.__say("  Calc'd c vorticity in {0}".format(end - start))

        if self.debug:
            start = time.time()
            vort  = self.vorticity(self.u, self.v)
            vort4 = self.fourth_order_vorticity(self.u, self.v)
            end = time.time()
            self.__say("  Calc'd vorticity in {0}".format(end - start))

            if abs((self.vort - vort).max()) > 1e-10:
                raise Exception('Difference between python/c vort calc')

            if abs((self.vort4 - vort4).max()) > 1e-10:
                raise Exception('Difference between python/c vort4 calc')

        if ensemble_mode in ['member', 'mean', 'diff']:
            e, index_pmaxs, index_pmins = find_extrema2(self.psl)
            self.pmins = [(self.psl[pmin[0], pmin[1]], (self.lon[pmin[1]], self.lat[pmin[0]])) for pmin in index_pmins]

            e, index_vmaxs, index_vmins = find_extrema2(self.vort)
            self.vmaxs = [(self.vort[vmax[0], vmax[1]], (self.lon[vmax[1]], self.lat[vmax[0]])) for vmax in index_vmaxs]

            e, index_v4maxs, index_v4mins = find_extrema2(self.vort4)
            self.v4maxs = [(self.vort4[v4max[0], v4max[1]], (self.lon[v4max[1]], self.lat[v4max[0]])) for v4max in index_v4maxs]
        else:
            self.pmins = []
            self.vmaxs = []
            self.v4maxs = []

            for i in range(self.number_enseble_members):
                e, index_pmaxs, index_pmins = find_extrema2(self.psl[i])
                self.pmins.append([(self.psl[i, pmin[0], pmin[1]], (self.lon[pmin[1]], self.lat[pmin[0]])) for pmin in index_pmins])

                e, index_vmaxs, index_vmins = find_extrema2(self.vort[i])
                self.vmaxs.append([(self.vort[i, vmax[0], vmax[1]], (self.lon[vmax[1]], self.lat[vmax[0]])) for vmax in index_vmaxs])

                e, index_v4maxs, index_v4mins = find_extrema2(self.vort4[i])
                self.v4maxs.append([(self.vort4[i, v4max[0], v4max[1]], (self.lon[v4max[1]], self.lat[v4max[0]])) for v4max in index_v4maxs])

        end = time.time()
        self.__say('  Found maxima/minima in {0}'.format(end - start))
        if self.smoothing:
            start = time.time()
            self.smoothed_vort = ndimage.filters.gaussian_filter(self.vort, 1, mode='nearest')
            e, index_svmaxs, index_svmins = find_extrema2(self.smoothed_vort)
            self.smoothed_vmaxs = [(self.smoothed_vort[svmax[0], svmax[1]], (self.lon[svmax[1]], self.lat[svmax[0]])) for svmax in index_svmaxs]
            end = time.time()
            self.__say('  Smoothed vorticity in {0}'.format(end - start))

	if self.upscaling:
            start = time.time()
            self.up_lon, self.up_lat, self.up_vort  = upscale_field(self.lon, self.lat, self.vort)
            e, index_upvmaxs, index_upvmins = find_extrema2(self.up_vort)
            self.up_vmaxs = [(self.up_vort[upvmax[0], upvmax[1]], (self.up_lon[upvmax[1]], self.up_lat[upvmax[0]])) for upvmax in index_upvmaxs]
            end = time.time()
            self.__say('  Upscaled vorticity in {0}'.format(end - start))


# TODO: utils.
def find_extrema(array):
    extrema = np.zeros_like(array)
    maximums = []
    minimums = []
    for i in range(1, array.shape[0] - 1):
        for j in range(0, array.shape[1]):
            val = array[i, j]

            is_max, is_min = True, True
            for ii in range(i - 1, i + 2):
                for jj in range(j - 1, j + 2):
                    if val < array[ii, jj % array.shape[1]]: 
                        is_max = False
                    elif val > array[ii, jj % array.shape[1]]: 
                        is_min = False
            if is_max:
                extrema[i, j] = 1
                maximums.append((i, j))
            elif is_min:
                extrema[i, j] = -1
                minimums.append((i, j))
    return extrema, maximums, minimums

# TODO: utils.
def find_extrema2(array):
    extrema = np.zeros_like(array)
    maximums = []
    minimums = []

    local_max = maximum_filter(array, size=(3, 3)) == array
    local_min = minimum_filter(array, size=(3, 3)) == array
    extrema += local_max
    extrema -= local_min

    where_max = np.where(local_max)
    where_min = np.where(local_min)

    for max_point in zip(where_max[0], where_max[1]):
        if max_point[0] != 0 and max_point[0] != array.shape[0] - 1:
            maximums.append(max_point)

    for min_point in zip(where_min[0], where_min[1]):
        if min_point[0] != 0 and min_point[0] != array.shape[0] - 1:
            minimums.append(min_point)

    return extrema, maximums, minimums




import os
import time
import datetime as dt

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage

from utils.c_wrapper import cvort, cvort4
from utils.utils import find_extrema, upscale_field
from load_settings import settings

DATA_DIR = settings.C20_FULL_DATA_DIR

EARTH_RADIUS = 6371000
EARTH_CIRC = EARTH_RADIUS * 2 * np.pi


class C20Data(object):
    '''Class used for accessing data from C20 Reanalysis project.

    This acts as a wrapper around netCDF4.Datasets and makes it easy to view data.
    Typically it exposes the psl, vort, and vort4 fields for one ensemble member.
    (vort4 is calculated using a 4th order vorticity calculation.) It will load these
    fields, along with corresponding maxima (vorticity) and minima (pressure) each time
    a new date is set for an object.

    :param start_year: Year from which to take data
    :param smoothing: Apply smoothing to data fields
    :param upscaling: Upscale data using cubic splines
    :param verbose: Prints lots of output
    :param pressure_level: Which pressure level (850/995 hPa) to use
    :param scale_factor: how much to scale data by
    '''

    def __init__(self, start_year, fields='all',
                 smoothing=False, upscaling=False, verbose=True,
                 pressure_level=850, scale_factor=2):
        self._year = start_year
        self.dx = None
        self.date = None
        self.smoothing = smoothing
        self.verbose = verbose
        self.upscaling = upscaling
        self.pressure_level = pressure_level
        self.scale_factor = scale_factor

        if fields == 'all':
            # self.fields = ['u', 'v', 'psl', 't995', 't850', 'cape', 'pwat', 'rh995']
            # rh995 has been removed.
            self.fields = ['u', 'v', 'psl', 't995', 't850', 'cape', 'pwat']
        else:
            self.fields = fields

        if self.pressure_level == 250:
            self.u_nc_field = 'u250'
            self.v_nc_field = 'v250'
        elif self.pressure_level == 850:
            self.u_nc_field = 'u850'
            self.v_nc_field = 'v850'
        elif self.pressure_level == 995:
            self.u_nc_field = 'u9950'
            self.v_nc_field = 'v9950'

        self.debug = False

        self.load_datasets(self._year)

    def __say(self, message):
        '''Prints a message if ``self.verbose == True``'''
        if self.verbose:
            print(message)

    def set_year(self, year):
        '''Sets a year and loads the relevant dataset'''
        self._year = year
        self.close_datasets()
        self.load_datasets(self._year)

    def close_datasets(self):
        '''Closes all open datasets'''
        self.nc_prmsl.close()
        self.nc_u.close()
        self.nc_v.close()

    def load_datasets(self, year):
        '''Loads datasets for a given year

        Just sets up the NetCDF4 objects, doesn't actually load any data apart from
        lons/lats and dates.
        '''
        any_dataset = None
        dataset_fieldname = None
        if 'psl' in self.fields:
            self.nc_prmsl = Dataset('{0}/{1}/prmsl_{1}.nc'.format(DATA_DIR, year))
            any_dataset = self.nc_prmsl
            dataset_fieldname = 'prmsl'
        if 'u' in self.fields:
            self.nc_u = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, self.u_nc_field))
            any_dataset = self.nc_u
            dataset_fieldname = self.u_nc_field
        if 'v' in self.fields:
            self.nc_v = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, self.v_nc_field))
            any_dataset = self.nc_v
            dataset_fieldname = self.v_nc_field
        if 't850' in self.fields:
            self.nc_t850 = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, 't850'))
            any_dataset = self.nc_t850
            dataset_fieldname = 't850'
        if 't995' in self.fields:
            self.nc_t995 = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, 't9950'))
            any_dataset = self.nc_t995
            dataset_fieldname = 't9950'
        if 'cape' in self.fields:
            self.nc_cape = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, 'cape'))
            any_dataset = self.nc_cape
            dataset_fieldname = 'cape'
        if 'pwat' in self.fields:
            self.nc_pwat = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, 'pwat'))
            any_dataset = self.nc_pwat
            dataset_fieldname = 'pwat'
        if 'rh995' in self.fields:
            self.nc_rh995 = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, 'rh9950'))
            any_dataset = self.nc_rh995
            dataset_fieldname = 'rh9950'

        start_date = dt.datetime(1, 1, 1)
        hours_since_JC = any_dataset.variables['time'][:]
        self.dates = np.array([start_date + dt.timedelta(hs / 24.) -
                              dt.timedelta(2) for hs in hours_since_JC])
        self.number_enseble_members = any_dataset.variables[dataset_fieldname].shape[1]

        self.lons = any_dataset.variables['lon'][:]
        self.lats = any_dataset.variables['lat'][:]

        dlon = self.lons[2] - self.lons[0]

        # N.B. array as dx varies with lat.
        # lons, lats are in degres.
        self.dx = (dlon * np.cos(self.lats * np.pi / 180) * EARTH_CIRC) / 360.
        self.dy = (self.lats[0] - self.lats[2]) * EARTH_CIRC / 360.

        # Interpolation functions.
        self.f_lon = interp1d(np.arange(0, 180), self.lons)
        self.f_lat = interp1d(np.arange(0, 91), self.lats)

    def first_date(self, ensemble_member=0):
        '''Sets date to the first date of the year (i.e. Jan the 1st)'''
        return self.set_date(self.dates[0], ensemble_member)

    def next_date(self, ensemble_member=0):
        '''Moves date on by one timestep (6hr)'''
        index = np.where(self.dates == self.date)[0][0]
        if index < len(self.dates):
            date = self.dates[index + 1]
            return self.set_date(date, ensemble_member)
        else:
            return None

    def prev_date(self, ensemble_member=0):
        '''Moves date back by one timestep (6hr)'''
        index = np.where(self.dates == self.date)[0][0]
        if index > 0:
            date = self.dates[index - 1]
            return self.set_date(date, ensemble_member)
        else:
            return None

    def set_date(self, date, ensemble_member=0):
        '''Sets date and loads all data for that date

        Will have no effect if there is no difference in date or ensemble_member.

        :param date: date to load
        :param ensemble_member: ensemble member to load

        :returns: date if successful, otherwise None
        '''
        if date != self.date or ensemble_member != self.ensemble_member:
            try:
                self.__say("Setting date to {0}".format(date))
                index = np.where(self.dates == date)[0][0]
                self.date = date
                self.ensemble_member = ensemble_member
                self.__process_ensemble_data(index, ensemble_member)
            except:
                self.date = None
                self.ensemble_member = None
                raise
        return date

    def cvorticity(self, u, v):
        '''Calculates the (2nd order) vorticity by calling into a c function'''
        vort = np.zeros_like(u)
        cvort(u, v, u.shape[0], u.shape[1], self.dx, self.dy, vort)
        return vort

    def cvorticity4(self, u, v):
        '''Calculates the (4th order) vorticity by calling into a c function

        Algorithm was taken from Walsh's code'''
        vort = np.zeros_like(u)
        cvort4(u, v, u.shape[0], u.shape[1], self.dx, self.dy, vort)
        return vort

    def vorticity(self, u, v):
        '''Calculates the (2nd order) vorticity using python'''
        # TODO: Move to utils.
        vort = np.zeros_like(u)

        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                dv_dx = (v[i, j + 1] - v[i, j - 1]) / self.dx[i]
                du_dy = (u[i + 1, j] - u[i - 1, j]) / self.dy

                vort[i, j] = dv_dx - du_dy
        return vort

    def fourth_order_vorticity(self, u, v):
        '''Calculates the (4th order) vorticity using python

        Algorithm was taken from Walsh's code'''
        # TODO: Move to utils.
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

    def __process_ensemble_data(self, index, ensemble_member):
        '''
        Processes data for one ensemble member

        Loads the relevant data and then performs a variety of calculations on it.
        At a minimum, psl, vort and vort4 will be calculated for the current date, as well
        as their maxima/minima as appropriate. Additionally (depending on how class is configured),
        smoothed_vort and up_vort (upscaled_vorticity) can be calculated.

        Rough times for each step are recorded.

        :param index: index of timestep in C20 data
        :param ensemble_member: which ensemble member to process
        '''
        if ensemble_member < 0 or ensemble_member >= self.number_enseble_members:
            raise Exception('Ensemble member must be be between 0 and {0}'.format(
                self.number_enseble_members))

        start = time.time()
        self.__load_ensemble_data(index, ensemble_member)
        end = time.time()
        self.__say('  Loaded psl, u, v in {0}'.format(end - start))

        if 'u' in self.fields and 'v' in self.fields:
            start = time.time()
            self.__calculate_vorticities()
            end = time.time()
            self.__say('  Calculated vorticity in {0}'.format(end - start))

        start = time.time()
        self.__find_min_max_from_fields()
        end = time.time()
        self.__say('  Found maxima/minima in {0}'.format(end - start))

        if self.smoothing:
            start = time.time()
            self.__calc_smoothed_fields()
            end = time.time()
            self.__say('  Smoothed vorticity in {0}'.format(end - start))

        if self.upscaling:
            start = time.time()
            self.__upscale_fields()
            end = time.time()
            self.__say('  Upscaled vorticity in {0}'.format(end - start))

    def __load_ensemble_data(self, index, ensemble_member):
        '''Loads the raw data from the NetCDF4 files'''
        if 'psl' in self.fields:
            self.psl = self.nc_prmsl.variables['prmsl'][index, ensemble_member]
        if 'u' in self.fields:
            # TODO: Why minus sign?
            self.u = - self.nc_u.variables[self.u_nc_field][index, ensemble_member]
        if 'v' in self.fields:
            self.v = self.nc_v.variables[self.v_nc_field][index, ensemble_member]
        if 't850' in self.fields:
            self.t850 = self.nc_t850.variables['t850'][index, ensemble_member]
        if 't995' in self.fields:
            self.t995 = self.nc_t995.variables['t9950'][index, ensemble_member]
        if 'cape' in self.fields:
            self.cape = self.nc_cape.variables['cape'][index, ensemble_member]
        if 'pwat' in self.fields:
            self.pwat = self.nc_pwat.variables['pwat'][index, ensemble_member]
        if 'rh995' in self.fields:
            self.rh995 = self.nc_rh995.variables['rh9950'][index, ensemble_member]

    def __calculate_vorticities(self):
        '''Calculates vort (2nd order) and vort4 (4th order)

        Uses c functions for speed. The accuracy of this can be tested
        by setting ``self.debug == True``.'''
        self.vort = self.cvorticity(self.u, self.v)
        self.vort4 = self.cvorticity4(self.u, self.v)

        if self.debug:
            start = time.time()
            vort = self.vorticity(self.u, self.v)
            vort4 = self.fourth_order_vorticity(self.u, self.v)
            end = time.time()
            self.__say("  Calc'd vorticity in {0}".format(end - start))

            if abs((self.vort - vort).max()) > 1e-10:
                raise Exception('Difference between python/c vort calc')

            if abs((self.vort4 - vort4).max()) > 1e-10:
                raise Exception('Difference between python/c vort4 calc')

    def __find_min_max_from_fields(self):
        '''Finds the minima (psl) and maxima (vort/vort4)'''

        if 'psl' in self.fields:
            e, index_pmaxs, index_pmins = find_extrema(self.psl)
            self.pmins = [(self.psl[pmin[0], pmin[1]], (self.lons[pmin[1]], self.lats[pmin[0]]))
                          for pmin in index_pmins]
            self.pmaxs = [(self.psl[pmax[0], pmax[1]], (self.lons[pmax[1]], self.lats[pmax[0]]))
                          for pmax in index_pmaxs]

        if 'u' in self.fields and 'v' in self.fields:
            e, index_vmaxs, index_vmins = find_extrema(self.vort)
            self.vmaxs = [
                (self.vort[vmax[0], vmax[1]], (self.lons[vmax[1]], self.lats[vmax[0]]))
                for vmax in index_vmaxs]
            self.vmins = [
                (self.vort[vmin[0], vmin[1]], (self.lons[vmin[1]], self.lats[vmin[0]]))
                for vmin in index_vmins]

            e, index_v4maxs, index_v4mins = find_extrema(self.vort4)
            self.v4maxs = [
                (self.vort4[v4max[0], v4max[1]], (self.lons[v4max[1]], self.lats[v4max[0]]))
                for v4max in index_v4maxs]

    def __calc_smoothed_fields(self):
        '''Calculated a smoothed vorticity using a guassian filter'''
        self.smoothed_vort = ndimage.filters.gaussian_filter(self.vort, 1, mode='nearest')
        e, index_svmaxs, index_svmins = find_extrema(self.smoothed_vort)
        self.smoothed_vmaxs = [
            (self.smoothed_vort[svmax[0], svmax[1]],
                (self.lons[svmax[1]], self.lats[svmax[0]])) for svmax in index_svmaxs]

    def __upscale_fields(self):
        '''Upscales the vorticity field using cubic interpolation'''
        # TODO: I'm upscaling the vorticity field directly. Does it make a difference
        # if I upscale the u/v fields first then calc vorticity?
        self.up_lons, self.up_lats, self.up_vort = \
            upscale_field(self.lons, self.lats, self.vort,
                          x_scale=self.scale_factor, y_scale=self.scale_factor)
        e, index_upvmaxs, index_upvmins = find_extrema(self.up_vort)
        self.up_vmaxs = [(self.up_vort[upvmax[0], upvmax[1]],
                         (self.up_lons[upvmax[1]], self.up_lats[upvmax[0]]))
                         for upvmax in index_upvmaxs]


class GlobalEnsembleMember(object):
    ''' Wrapper around a C20Data object

    holds state of which ensemble member is currently being analysed
    '''
    def __init__(self, c20data, ensemble_member=0):
        '''
        :param c20data: C20Data object to use
        :param ensemble_member: which ensemble member this object will access
        '''
        self.c20data = c20data
        self.dates = c20data.dates
        self.lons = c20data.lons
        self.lats = c20data.lats

        self.date = None
        self.cyclones_by_date = {}
        self.ensemble_member = ensemble_member

    def set_year(self, year):
        '''Change the year of the stored C20Data object'''
        self.year = year
        self.c20data.set_year(year)
        self.dates = self.c20data.dates

    def set_date(self, date):
        '''Sets the date, does nothing if it is the same as the stored date'''
        if date != self.date:
            self.date = date
            self.c20data.set_date(date, self.ensemble_member)

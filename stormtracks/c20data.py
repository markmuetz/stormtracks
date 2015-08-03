import os
import time
import datetime as dt

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage

from utils.c_wrapper import cvort, cvort4
from utils.utils import cfind_extrema, upscale_field
from load_settings import settings

DATA_DIR = settings.C20_FULL_DATA_DIR

EARTH_RADIUS = 6371000
EARTH_CIRC = EARTH_RADIUS * 2 * np.pi
NUM_ENSEMBLE_MEMBERS = 56


class FullC20Data(object):
    '''Class used for accessing data from C20 Reanalysis project.

    This acts as a wrapper around netCDF4.Datasets and makes it easy to view data.
    Typically it exposes the prmsl, vort, and vort4 fields for one ensemble member.
    (vort4 is calculated using a 4th order vorticity calculation.) It will load these
    fields, along with corresponding maxima (vorticity) and minima (pressure) each time
    a new date is set for an object.

    :param start_year: Year from which to take data
    :param verbose: Prints lots of output
    :param pressure_level: Which pressure level (850/995 hPa) to use
    '''

    def __init__(self, start_year, fields='all', verbose=True,
                 pressure_level=850):
        self._year = start_year
        self.dx = None
        self.date = None
        self.verbose = verbose
        self.pressure_level = pressure_level

        if fields == 'all':
            # self.fields = ['u', 'v', 'prmsl', 't995', 't850', 'cape', 'pwat', 'rh995']
            # rh995 has been removed.
            self.fields = ['u', 'v', 'prmsl', 't995', 't850', 'cape', 'pwat']
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
        for dataset in self.all_datasets:
            dataset.close()

    def load_datasets(self, year):
        '''Loads datasets for a given year

        Just sets up the NetCDF4 objects, doesn't actually load any data apart from
        lons/lats and dates.
        '''
        any_dataset = None
        dataset_fieldname = None
        self.all_datasets = []
        if 'prmsl' in self.fields:
            self.nc_prmsl = Dataset('{0}/{1}/prmsl_{1}.nc'.format(DATA_DIR, year))
            self.all_datasets.append(self.nc_prmsl)
            any_dataset = self.nc_prmsl
            dataset_fieldname = 'prmsl'
        if 'u' in self.fields:
            self.nc_u = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, self.u_nc_field))
            self.all_datasets.append(self.nc_u)
            any_dataset = self.nc_u
            dataset_fieldname = self.u_nc_field
        if 'v' in self.fields:
            self.nc_v = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, self.v_nc_field))
            self.all_datasets.append(self.nc_v)
            any_dataset = self.nc_v
            dataset_fieldname = self.v_nc_field
        if 't850' in self.fields:
            self.nc_t850 = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, 't850'))
            any_dataset = self.nc_t850
            self.all_datasets.append(self.nc_t850)
            dataset_fieldname = 't850'
        if 't995' in self.fields:
            self.nc_t995 = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, 't9950'))
            any_dataset = self.nc_t995
            self.all_datasets.append(self.nc_t995)
            dataset_fieldname = 't9950'
        if 'cape' in self.fields:
            self.nc_cape = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, 'cape'))
            self.all_datasets.append(self.nc_cape)
            any_dataset = self.nc_cape
            dataset_fieldname = 'cape'
        if 'pwat' in self.fields:
            self.nc_pwat = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, 'pwat'))
            self.all_datasets.append(self.nc_pwat)
            any_dataset = self.nc_pwat
            dataset_fieldname = 'pwat'
        if 'rh995' in self.fields:
            self.nc_rh995 = Dataset('{0}/{1}/{2}_{1}.nc'.format(DATA_DIR, year, 'rh9950'))
            self.all_datasets.append(self.nc_rh995)
            any_dataset = self.nc_rh995
            dataset_fieldname = 'rh9950'

        start_date = dt.datetime(1, 1, 1)
        hours_since_JC = any_dataset.variables['time'][:]
        self.number_enseble_members = any_dataset.variables[dataset_fieldname].shape[1]

        self.lons = any_dataset.variables['lon'][:]
        self.lats = any_dataset.variables['lat'][:]

        self.dates = np.array([start_date + dt.timedelta(hs / 24.) -
                              dt.timedelta(2) for hs in hours_since_JC])

        dlon = self.lons[2] - self.lons[0]

        # N.B. array as dx varies with lat.
        # lons, lats are in degres.
        self.dx = (dlon * np.cos(self.lats * np.pi / 180) * EARTH_CIRC) / 360.
        self.dy = (self.lats[0] - self.lats[2]) * EARTH_CIRC / 360.

        # Interpolation functions.
        self.f_lon = interp1d(np.arange(0, 180), self.lons)
        self.f_lat = interp1d(np.arange(0, 91), self.lats)

    def first_date(self):
        '''Sets date to the first date of the year (i.e. Jan the 1st)'''
        return self.set_date(self.dates[0])

    def next_date(self):
        '''Moves date on by one timestep (6hr)'''
        index = np.where(self.dates == self.date)[0][0]
        if index < len(self.dates):
            date = self.dates[index + 1]
            return self.set_date(date)
        else:
            return None

    def prev_date(self):
        '''Moves date back by one timestep (6hr)'''
        index = np.where(self.dates == self.date)[0][0]
        if index > 0:
            date = self.dates[index - 1]
            return self.set_date(date)
        else:
            return None

    def set_date(self, date):
        '''Sets date and loads all data for that date

        Will have no effect if there is no difference in date.

        :param date: date to load

        :returns: date if successful, otherwise None
        '''
        if date != self.date:
            try:
                self.__say("Setting date to {0}".format(date))
                index = np.where(self.dates == date)[0][0]
                self.date = date
                self.__process_ensemble_data(index)
            except:
                self.date = None
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

    def __process_ensemble_data(self, index):
        '''
        Processes data for one ensemble member

        Loads the relevant data and then performs a variety of calculations on it.
        At a minimum, prmsl, vort and vort4 will be calculated for the current date, as well
        as their maxima/minima as appropriate. Additionally (depending on how class is configured),
        smoothed_vort and up_vort (upscaled_vorticity) can be calculated.

        Rough times for each step are recorded.

        :param index: index of timestep in C20 data
        '''
        start = time.time()
        self.__load_ensemble_data(index)
        end = time.time()
        fields = ', '.join(self.fields)
        self.__say('  Loaded {0} in {1}'.format(fields, end - start))

        if 'u' in self.fields and 'v' in self.fields:
            start = time.time()
            self.__calculate_vorticities()
            end = time.time()
            self.__say('  Calculated vorticity in {0}'.format(end - start))

        start = time.time()
        self.__find_min_max_from_fields()
        end = time.time()
        self.__say('  Found maxima/minima in {0}'.format(end - start))

    def __load_ensemble_data(self, index):
        '''Loads the raw data from the NetCDF4 files'''
        if 'prmsl' in self.fields:
            self.prmsl = self.nc_prmsl.variables['prmsl'][index]
        if 'u' in self.fields:
            # TODO: Why minus sign?
            self.u = - self.nc_u.variables[self.u_nc_field][index]
        if 'v' in self.fields:
            self.v = self.nc_v.variables[self.v_nc_field][index]
        if 't850' in self.fields:
            self.t850 = self.nc_t850.variables['t850'][index]
        if 't995' in self.fields:
            self.t995 = self.nc_t995.variables['t9950'][index]
        if 'cape' in self.fields:
            self.cape = self.nc_cape.variables['cape'][index]
        if 'pwat' in self.fields:
            self.pwat = self.nc_pwat.variables['pwat'][index]
        if 'rh995' in self.fields:
            self.rh995 = self.nc_rh995.variables['rh9950'][index]

    def __calculate_vorticities(self):
        '''Calculates vort (2nd order) and vort4 (4th order)

        Uses c functions for speed.'''
        self.vort = []
        # self.vort4 = []
        for em in range(NUM_ENSEMBLE_MEMBERS):
            self.vort.append(self.cvorticity(self.u[em], self.v[em]))
            # self.vort4.append(self.cvorticity4(self.u[em], self.v[em]))

    def __find_min_max_from_fields(self):
        '''Finds the minima (prmsl) and maxima (vort/vort4)'''
        if 'prmsl' in self.fields:
            self.pmins, self.pmaxs = [], []
            for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
                e, index_pmaxs, index_pmins = cfind_extrema(self.prmsl[ensemble_member])
                self.pmins.append([(self.prmsl[ensemble_member][pmin[0], pmin[1]], (self.lons[pmin[1]], self.lats[pmin[0]]))
                              for pmin in index_pmins])
                self.pmaxs.append([(self.prmsl[ensemble_member][pmax[0], pmax[1]], (self.lons[pmax[1]], self.lats[pmax[0]]))
                              for pmax in index_pmaxs])

        if 'u' in self.fields and 'v' in self.fields:
            self.vmins, self.vmaxs = [], []
            for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
                e, index_vmaxs, index_vmins = cfind_extrema(self.vort[ensemble_member])
                self.vmaxs.append([
                    (self.vort[ensemble_member][vmax[0], vmax[1]], (self.lons[vmax[1]], self.lats[vmax[0]]))
                    for vmax in index_vmaxs])
                self.vmins.append([
                    (self.vort[ensemble_member][vmin[0], vmin[1]], (self.lons[vmin[1]], self.lats[vmin[0]]))
                    for vmin in index_vmins])


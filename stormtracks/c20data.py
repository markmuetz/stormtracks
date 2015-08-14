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
import setup_logging

C20_DATA_DIR = os.path.join(settings.DATA_DIR, 'c20_full')

EARTH_RADIUS = 6371000
EARTH_CIRC = EARTH_RADIUS * 2 * np.pi
NUM_ENSEMBLE_MEMBERS = 56

log = setup_logging.get_logger('st.find_vortmax')


class C20Data(object):
    '''Class used for accessing data from C20 Reanalysis project.

    This acts as a wrapper around netCDF4.Datasets and makes it easy to view data.
    Typically it exposes the prmsl and vort850/vort9950 fields for all ensemble members.
    It will load these fields, along with corresponding maxima (vorticity) and minima (pressure)
    each time a new date is set.

    :param year: Year from which to take data
    :param fields: List of C20 fields that are to be loaded, or use 'all' for complete set
    :param version: Version of C20 data to use
    '''

    def __init__(self, year, fields='all', version='v1'):
        self._year = year
        self.dx = None
        self.date = None
	self.version = version
	log.info('C20Data: year={}, version={}'.format(year, version))

        if fields == 'all':
            # rh995 has been removed.
            self.fields = ['u9950', 'v9950', 'u850', 'v850', 'prmsl', 't9950', 't850', 'cape', 'pwat']
        else:
            self.fields = fields

        if 'u9950' in self.fields and 'v9950' in self.fields:
            self.calc_9950_vorticity = True
        else:
            self.calc_9950_vorticity = False

        if 'u850' in self.fields and 'v850' in self.fields:
            self.calc_850_vorticity = True
        else:
            self.calc_850_vorticity = False

        fields = ', '.join(self.fields)
	log.info('Using: {}'.format(fields))
        self._load_datasets(self._year)

    def set_year(self, year):
        '''Sets a year and loads the relevant dataset'''
        self._year = year
        self.close_datasets()
        self._load_datasets(self._year)

    def close_datasets(self):
        '''Closes all open datasets'''
        for dataset in self.nc_datasets.values():
            dataset.close()

    def _load_datasets(self, year):
        '''Loads datasets for a given year

        Just sets up the NetCDF4 objects, doesn't actually load any data apart from
        lons/lats and dates.
        '''
        # All datasets have lon/lat/time info in them, so any will do.
        any_dataset = None
        dataset_fieldname = None
        self.nc_datasets = {}

        for field in self.fields:
            # e.g. ~/stormtracks_data/data/c20_full/2005/prmsl_2005.nc
	    path = os.path.join(C20_DATA_DIR, self.version, str(year), '{}_{}.nc'.format(field, year))
	    if not os.path.exists(path):
		msg = 'File does not exist: {}'.format(path)
		log.error(msg)
		raise RuntimeError(msg)
	    log.debug('Loading {} from {}'.format(field, path))
            dataset = Dataset(path)
            dataset_fieldname = field
            any_dataset = dataset
            self.nc_datasets[field] = dataset

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
	self.first_date()

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
	    log.warn('Trying to set date beyond date range')
            return None

    def prev_date(self):
        '''Moves date back by one timestep (6hr)'''
        index = np.where(self.dates == self.date)[0][0]
        if index > 0:
            date = self.dates[index - 1]
            return self.set_date(date)
        else:
	    log.warn('Trying to set date beyond date range')
            return None

    def set_date(self, date):
        '''Sets date and loads all data for that date

        Will have no effect if there is no difference in date.

        :param date: date to load

        :returns: date if successful, otherwise None
        '''
        if date != self.date:
            try:
                log.debug("Setting date to {0}".format(date))
                index = np.where(self.dates == date)[0][0]
                self.date = date
                self._process_ensemble_data(index)
            except:
                self.date = None
		log.exception('Problem loading date {}'.format(date))
                raise
        return date

    def _cvorticity(self, u, v):
        '''Calculates the (2nd order) vorticity by calling into a c function'''
        vort = np.zeros_like(u)
        cvort(u, v, u.shape[0], u.shape[1], self.dx, self.dy, vort)
        return vort

    def _cvorticity4(self, u, v):
        '''Calculates the (4th order) vorticity by calling into a c function

        Algorithm was taken from Walsh's code'''
        vort = np.zeros_like(u)
        cvort4(u, v, u.shape[0], u.shape[1], self.dx, self.dy, vort)
        return vort

    def _process_ensemble_data(self, index):
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
        self._load_ensemble_data(index)
        end = time.time()
        fields = ', '.join(self.fields)
        log.debug('  Loaded {0} in {1}'.format(fields, end - start))

        if self.calc_9950_vorticity:
            start = time.time()
            self._calculate_vorticities('9950')
            end = time.time()
            log.debug('  Calculated 9950 vorticity in {0}'.format(end - start))
        if self.calc_850_vorticity:
            start = time.time()
            self._calculate_vorticities('850')
            end = time.time()
            log.debug('  Calculated 850 vorticity in {0}'.format(end - start))

        start = time.time()
        self._find_min_max_from_fields()
        end = time.time()
        log.debug('  Found maxima/minima in {0}'.format(end - start))

    def _load_ensemble_data(self, index):
        '''Loads the raw data from the NetCDF4 files'''
        # N.B. it is very important how the data is loaded. The data is stored in NetCDF4 files,
        # which in turn uses HDF5 as a storage medium. HDF5 allows for compression of particular
        # subsets of data ('chunks'). If you access the data in terms of these chunks, it will be
        # **much** faster, which is why all data for one date is loaded at a time, i.e. 56x91x180
        # cells, or num_ensemble_members x lat x lon.
        # This can be seen by looking at e.g. c20data.prmsl.shape, which will be (56, 91, 180).
        for field in self.fields:
            if field in ['u9950', 'u850', 'u250']:
                setattr(self, field, - self.nc_datasets[field].variables[field][index])
            else:
                setattr(self, field, self.nc_datasets[field].variables[field][index])

    def _calculate_vorticities(self, pressure_level):
        '''Calculates vort (2nd order) and vort4 (4th order)

        Uses c functions for speed.'''
        vort = []
        # self.vort4 = []
        if pressure_level == '9950':
            for em in range(NUM_ENSEMBLE_MEMBERS):
                vort.append(self._cvorticity(self.u9950[em], self.v9950[em]))
                # vort4.append(self._cvorticity4(self.u[em], self.v[em]))
        elif pressure_level == '850':
            for em in range(NUM_ENSEMBLE_MEMBERS):
                vort.append(self._cvorticity(self.u850[em], self.v850[em]))
                # vort4.append(self._cvorticity4(self.u[em], self.v[em]))
        setattr(self, 'vort{}'.format(pressure_level), vort)

    def _find_min_max_from_fields(self):
        '''Finds the minima (prmsl) and maxima (vort/vort4)'''
        if 'prmsl' in self.fields:
            self.pmins, self.pmaxs = [], []
            for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
                e, index_pmaxs, index_pmins = cfind_extrema(self.prmsl[ensemble_member])
                self.pmins.append([(self.prmsl[ensemble_member][pmin[0], pmin[1]], (self.lons[pmin[1]], self.lats[pmin[0]]))
                              for pmin in index_pmins])

        if 'u9950' in self.fields and 'v9950' in self.fields:
            self.vmaxs9950 = []
            for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
                e, index_vmaxs, index_vmins = cfind_extrema(self.vort9950[ensemble_member])
                self.vmaxs9950.append([
                    (self.vort9950[ensemble_member][vmax[0], vmax[1]], (self.lons[vmax[1]], self.lats[vmax[0]]))
                    for vmax in index_vmaxs])

        if 'u850' in self.fields and 'v850' in self.fields:
            self.vmaxs850 = []
            for ensemble_member in range(NUM_ENSEMBLE_MEMBERS):
                e, index_vmaxs, index_vmins = cfind_extrema(self.vort850[ensemble_member])
                self.vmaxs850.append([
                    (self.vort850[ensemble_member][vmax[0], vmax[1]], (self.lons[vmax[1]], self.lats[vmax[0]]))
                    for vmax in index_vmaxs])

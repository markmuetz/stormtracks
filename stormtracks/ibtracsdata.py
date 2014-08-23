import os
from glob import glob
import datetime as dt
from collections import Counter

import numpy as np
import netCDF4 as nc

from load_settings import settings

DATA_DIR = settings.IBTRACS_DATA_DIR


class IbtracsData(object):
    '''Class used for accessing IBTrACS data

    Wraps the underlying NetCDF4 files and extracts the information required from them.

    :param data_dir: directory where IBTrACS NetCDF4 files are held
    :param verbose: whether to pring lots of output
    '''
    def __init__(self, data_dir=None, verbose=True):
        if data_dir:
            self.data_dir = data_dir
            self.path_tpl = os.path.join(data_dir, '{0}*.nc')
        else:
            self.data_dir = DATA_DIR
            self.path_tpl = os.path.join(DATA_DIR, '{0}*.nc')
        self.verbose = verbose

    def __say(self, message):
        '''Prints a message if ``self.verbose == True``'''
        if self.verbose:
            print(message)

    def load_ibtracks_year(self, year, basin='NA'):
        '''Loads a given year's worth of data

        :param year: year to load
        :param basin: which basins to load ('all' for all of them)
        :returns: list of loaded best tracks
        '''
        y = str(year)

        filenames = glob(self.path_tpl.format(y))
        self.best_tracks = self._load_ibtracks_filenames(year, basin, filenames)
        return self.best_tracks

    def _load_ibtracks_filenames(self, year, basin, filenames):
        basins = Counter()
        best_tracks = []
        for filename in filenames:
            try:
                best_track = self._load_ibtracks_data(year, filename)
                if basin == 'all' or basin == best_track.basin:
                    best_track.index = len(best_tracks)
                    best_tracks.append(best_track)
                basins[best_track.basin] += 1
            except Exception, e:
                self.__say('Could not load data for {0}'.format(filename))
                self.__say(e.message)
        return best_tracks

    def _load_ibtracks_data(self, year, filename):
        self.__say(filename.split('/')[-1])
        dataset = nc.Dataset(filename)
        best_track = IbStormtrack(year, filename.split('/')[-1].split('.')[0])
        best_track.basin = _convert_ib_field(dataset.variables['genesis_basin'])

        dates = []
        for i in range(dataset.variables['nobs'].getValue()):
            time_string = _convert_ib_field(dataset.variables['isotime'][i])
            date = dt.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
            dates.append(date)
        best_track.dates = np.array(dates)

        best_track.pressures = dataset.variables['pres_for_mapping'][:]
        best_track.winds = dataset.variables['wind_for_mapping'][:]

        # Convert lons to 0 to 360. (They start off -180 to 180).
        ib_lons = dataset.variables['lon_for_mapping'][:]
        best_track.lons = np.zeros_like(ib_lons)
        for i, lon in enumerate(ib_lons):
            best_track.lons[i] = lon if lon > 0 else lon + 360

        best_track.lats = dataset.variables['lat_for_mapping'][:]
        if best_track.basin == 'NA':
            best_track.cls = []
            best_track.is_hurricane = False
            for i in range(dataset.variables['nobs'].getValue()):
                cls = _convert_ib_field(dataset.variables['atcf_class'][i])
                if cls == 'HU':
                    best_track.is_hurricane = True
                best_track.cls.append(cls)

        return best_track

    def load_wilma_katrina(self):
        '''Loads only best tracks corresponding to Wilma and Katrina (2005)'''
        wilma_fn = '2005289N18282.ibtracs.v03r05.nc'
        katrina_fn = '2005236N23285.ibtracs.v03r05.nc'
        wilma_path = os.path.join(self.data_dir, wilma_fn)
        katrina_path = os.path.join(self.data_dir, katrina_fn)
        return self._load_ibtracks_filenames(2005, 'NA', [wilma_path, katrina_path])


class IbStormtrack(object):
    '''Holds info about an IBTrACS best track'''
    def __init__(self, year, name):
        self.year = year
        self.name = name
        self.is_matched = False


def _convert_ib_field(array):
    '''Utility function for converting IBTrACS field to string'''
    return ''.join(array)

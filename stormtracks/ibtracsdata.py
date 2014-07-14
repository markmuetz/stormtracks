import os
from glob import glob
import datetime as dt
from collections import Counter

import numpy as np
import netCDF4 as nc

from load_settings import settings

DATA_DIR = settings.IBTRACS_DATA_DIR

class IbStormtrack(object):
    def __init__(self, year, name):
	self.year  = year
	self.name  = name
	self.is_matched = False

def _convert_ib_field(array):
    return ''.join(array)

class IbtracsData(object):
    def __init__(self, data_dir=None):
	if data_dir:
	    self.path_tpl = os.path.join(data_dir, '{0}*.nc')
	else:
	    self.path_tpl = os.path.join(DATA_DIR, '{0}*.nc')

    def load_ibtracks_year(self, year):
	y = str(year)

	filenames = glob(self.path_tpl.format(y))
	self.best_tracks = self._load_ibtracks_filenames(year, filenames)
	return self.best_tracks

    def _load_ibtracks_filenames(self, year, filenames):
	basins = Counter()
	stormtracks = []
	for filename in filenames:
	    try:
		s = self._load_ibtracks_data(year, filename)
		if s.basin == 'NA':
		    s.index = len(stormtracks)
		    stormtracks.append(s)
		basins[s.basin] += 1
	    except Exception, e:
		print('Could not load data for %s'%filename)
		print(e.message)
	return stormtracks

    def _load_ibtracks_data(self, year, filename):
	print(filename.split('/')[-1])
	dataset = nc.Dataset(filename)
	s = IbStormtrack(year, filename.split('/')[-1].split('.')[0])
	s.basin = _convert_ib_field(dataset.variables['genesis_basin'])

	dates = []
	for i in range(dataset.variables['nobs'].getValue()):
	    date = dt.datetime.strptime(_convert_ib_field(dataset.variables['isotime'][i]), '%Y-%m-%d %H:%M:%S')
	    dates.append(date)

	# Convert lons to 0 to 360. (They start off -180 to 180).
        ib_lon = dataset.variables['lon_for_mapping'][:]
        s.lons = np.zeros_like(ib_lon)
        for i, lons in enumerate(ib_lon):
            s.lons[i] = lons if lons > 0 else lons + 360

	s.lats = dataset.variables['lat_for_mapping'][:]
	if s.basin == 'NA':
	    s.cls = []
	    s.is_hurricane = False
	    for i in range(dataset.variables['nobs'].getValue()):
		cls = _convert_ib_field(dataset.variables['atcf_class'][i])
		if cls == 'HU':
		    s.is_hurricane = True
		s.cls.append(cls)


	s.dates = np.array(dates)
	return s

    def load_wilma_katrina(self):
	wilma_fn = 'data/ibtracs/2005289N18282.ibtracs.v03r05.nc'
	katrina_fn = 'data/ibtracs/2005236N23285.ibtracs.v03r05.nc'
	return self._load_ibtracks_filenames(2005, [wilma_fn, katrina_fn])

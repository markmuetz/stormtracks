from glob import glob
import datetime as dt
from collections import Counter
import numpy as np
import netCDF4 as nc

class Stormtrack(object):
    def __init__(self, region, year, name):
	self.region  = region
	self.year  = year
	self.name  = name

class IbStormtrack(object):
    def __init__(self, year, name, ds):
	self.year  = year
	self.name  = name
	self.ds = ds

def load_stormtracks_data():
    regions = ['atlantic', 'e_pacific', 'w_pacific', 's_pacific', 's_indian', 'n_indian']
    start_year, end_year = 2000, 2013
    stormtracks = []
    cats = Counter()
    for region in regions:
	for year in range(start_year, end_year):
	    print(region, year)
	    y = str(year)
	    filenames = glob('data/stormtracks/%s/%s/*'%(region, y))

	    for fn in filenames:
		try:
		    s = load_stromtrack_data(region, year, fn)
		    stormtracks.append(s)
		    for cat in s.categories:
			cats[cat] += 1
		except Exception, e:
		    print('Could not load data for %s'%fn)
		    print(e.message)

    return stormtracks, cats

def load_ibtracks_year(year):
    y = str(year)

    filenames = glob('data/ibtracs/%s*.nc'%(y))
    return load_ibtracks_filenames(year, filenames)

def load_ibtracks_filenames(year, filenames):
    basins = Counter()
    stormtracks = []
    for fn in filenames:
	try:
	    s = load_ibtracks_data(year, fn)
	    if s.basin == 'NA':
		stormtracks.append(s)
	    basins[s.basin] += 1
	except Exception, e:
	    print('Could not load data for %s'%fn)
	    print(e.message)
    return stormtracks, basins

def ibs(array):
    return ''.join(array)

def load_ibtracks_data(year, fn):
    print(fn.split('/')[-1])
    dataset = nc.Dataset(fn)
    s = IbStormtrack(year, fn.split('/')[-1].split('.')[0], dataset)
    s.basin = ibs(dataset.variables['genesis_basin'])

    dates = []
    cats  = []
    lons  = []
    lats  = []
    for i in range(dataset.variables['nobs'].getValue()):
	date = dt.datetime.strptime(ibs(dataset.variables['isotime'][i]), '%Y-%m-%d %H:%M:%S')
	dates.append(date)

    s.lon = dataset.variables['lon_for_mapping'][:]
    s.lat = dataset.variables['lat_for_mapping'][:]
    if s.basin == 'NA':
	s.cls = []
	s.is_hurricane = False
	for i in range(dataset.variables['nobs'].getValue()):
	    cls = ibs(dataset.variables['atcf_class'][i])
	    if cls == 'HU':
		s.is_hurricane = True
	    s.cls.append(cls)


    s.lon_keys = [k for k in dataset.variables.keys() if k.find('lon') != -1]

    s.dates = np.array(dates)
    return s

def load_wilma_katrina():
    wilma_fn = 'data/ibtracs/2005289N18282.ibtracs.v03r05.nc'
    katrina_fn = 'data/ibtracs/2005236N23285.ibtracs.v03r05.nc'
    return load_ibtracks_filenames(2005, [wilma_fn, katrina_fn])

def load_ibtracks_stormtrack_data(fn):
    pass

def load_stromtrack_data(region, year, fn):
    #print(fn)
    with open(fn, 'r') as f:
	lines = f.readlines()

    track = []
    date_strs = []
    dates = []
    categories = []
    winds = []
    pressures = []

    if lines[0] == 'SKIP\n':
	raise Exception('skipping due to first line of file')

    for line in lines[3:]:
	split_line = line.split()
	lat = float(split_line[1])
	lon = float(split_line[2])
	date_str = split_line[3]
        date = dt.datetime.strptime('%s/%s'%(str(year), date_str), '%Y/%m/%d/%HZ')
	#wind = float(split_line[4])
	wind = split_line[4]
	#pressure = float(split_line[5])
	pressure = split_line[5]
	category = "_".join(split_line[6:])

	track.append((lon, lat))
	date_strs.append(date_str)
	dates.append(date)
	categories.append(category)
	winds.append(wind)
	pressures.append(pressure)

    s = Stormtrack(region, year, fn.split('/')[-1].split('.')[0])
    s.track = np.array(track)
    s.dates = np.array(dates)
    s.categories = np.array(categories)
    s.winds = np.array(winds)
    s.pressures = np.array(pressures)
    return s

if __name__ == '__main__':
    load_stormtracks_data()

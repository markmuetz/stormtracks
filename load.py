from glob import glob
import datetime as dt
from collections import Counter
import numpy as np
import pylab as plt
import hashlib
import netCDF4 as nc

class Stormtrack(object):
    def __init__(self, region, year, name):
	self.region  = region
	self.year  = year
	self.name  = name
	pass

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

def load_ibtracks_stormtracks_data():
    wilma = nc.Dataset('data/ibtracs/2005289N18282.ibtracs.v03r05.nc')
    lons = [wilma.variables[k][:] for k in wilma.variables.keys() if k.find('lon') != -1]
    lats = [wilma.variables[k][:] for k in wilma.variables.keys() if k.find('lat') != -1]
    #plt.plot(lons[0] + 360, lats[0])
    katrina = nc.Dataset('data/ibtracs/2005236N23285.ibtracs.v03r05.nc')
    return wilma, katrina



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

def plot_stormtracks(stormtracks, region=None, category=None, fmt='b-', start_date=None, end_date=None):
    for s in stormtracks:
	if region and s.region != region:
	    continue
	if s.track[:, 0].max() > 0 and s.track[:, 0].min() < 0:
	    continue

        if start_date and end_date:
            mask = (np.array(s.dates) > start_date) & (np.array(s.dates) < end_date)
	    plt.plot(s.track[:, 0][mask], s.track[:, 1][mask], fmt)
            continue

	
	if category:
	    mask = np.array(s.categories, dtype=object) == category
	    plt.plot(s.track[:, 0][mask], s.track[:, 1][mask], fmt)
	else:
	    plt.plot(s.track[:, 0], s.track[:, 1], fmt)

if __name__ == '__main__':
    load_stormtracks_data()

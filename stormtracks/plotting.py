from __future__ import print_function

import os
import datetime as dt
import pickle
from glob import glob

import pylab as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

import detect
from load_settings import settings

class PlotterLayout(object):
    def __init__(self, date, plots):
	self.version = '0.1'
	self.date = date
        self.name = ''
	self.plots = plots

	self.ensemble_member = 0
	self.ensemble_mode = 'member'


class PlotterSettings(object):
    def __init__(self):
	self.figure = 1
	self.subplot = '111'

	#self.field = None
	self.field = 'psl'
	self.points = 'pmins'
	self.loc = 'wa'

	self.vmax = None
	self.vmin = None

	self.best_track_index = -1
	self.vort_max_track_index = -1
	self.pressure_min_track_index = -1

	self.match_index = -1


class Plotter(object):
    def __init__(self, best_tracks, ncdata, gdatas, all_matches):
        self.best_tracks = best_tracks

        self.ncdata = ncdata
        self.gdatas = gdatas
        self.all_matches = all_matches
	self.date = self.ncdata.first_date()
        self.layout = PlotterLayout(self.date, [PlotterSettings()])

	self.date = None
	self.field_names = ['psl', 'vort', 'vort4', 'windspeed']
	self.track_names = ['best', 'vort_max', 'pressure_min']
	self.point_names = ['vort_max', 'pressure_min']


    def plot(self):
	ensemble_member = self.layout.ensemble_member
	date = self.layout.date

	plt.clf()
	for settings in self.layout.plots:
	    plt.figure(settings.figure)
	    plt.subplot(settings.subplot)

	    title = [str(date), str(ensemble_member)]

	    if settings.field:
		title.append(settings.field)
		field = getattr(self.ncdata, settings.field)
		plot_on_earth(self.ncdata.lon, self.ncdata.lat, field, settings.vmin, settings.vmax, settings.loc)
	    else:
		plot_on_earth(self.ncdata.lon, self.ncdata.lat, None, None, None, settings.loc)
	    
	    if settings.points:
		title.append(settings.points)
		points = getattr(self.ncdata, settings.points)
		for p_val, p_loc in points:
		    plt.plot(convert(p_loc[0]), p_loc[1], 'kx')
	    
	    if settings.best_track_index != -1:
		plot_ibtrack_with_date(self.best_tracks[settings.best_track_index], date)

	    if settings.vort_max_track_index != -1:
		plot_ibtrack_with_date(self.best_tracks[ensemble_member][settings.best_track_index], date)

	    if settings.pressure_min_track_index != -1:
		plot_ibtrack_with_date(self.best_tracks[ensemble_member][settings.best_track_index], date)

	    if settings.match_index != -1:
		plot_match(self.all_matches[ensemble_member][settings.match_index], date)

	    plt.title(' '.join(title))


    def save(self, name):
        self.layout.name = name
        f = open(os.join(settings.SETTINGS_DIR, 'plot_{0}.pkl'.format(self.layout.name)), 'w')
        pickle.dump(self.layout, f)

    def load(self, name):
        try:
            f = open(os.join(settings.SETTINGS_DIR, 'plot_{0}.pkl'.format(self.layout.name)), 'r')
            layout = pickle.load(f)
	    if layout.version != self.layout.version:
		print('version mismatch, may not work! Press c to continue anyway.')
		r = raw_input()
		if r != 'c':
		    raise Exception('Version mismatch (user cancelled)')

            self.layout = layout
	    self.set_date(self.layout.date)
	except Exception, e:
            print('Settings {0} could not be loaded'.format(name))
            print('{0}'.format(e.message))

    def list(self):
        for fn in glob('settings/plot_*.dat'):
            print('_'.join(os.path.basename(fn).split('.')[0].split('_')[1:]))
    
    def set_date(self, date):
	self.layout.date = self.ncdata.set_date(date, self.layout.ensemble_member, self.layout.ensemble_mode)
	self.plot()
    
    def next_date(self):
	self.layout.date = self.ncdata.next_date(self.layout.ensemble_member, self.layout.ensemble_mode)
	self.plot()
    
    def prev_date(self):
	self.layout.date = self.ncdata.prev_date(self.layout.ensemble_member, self.layout.ensemble_mode)
	self.plot()
    
    def set_ensemble_member(self, ensemble_member):
	self.layout.ensemble_member = ensemble_member
	self.ncdata.set_date(self.layout.date, self.layout.ensemble_member, self.layout.ensemble_mode)
	self.plot()

    def next_ensemble_member(self):
	self.set_ensemble_member(self.layout.ensemble_member + 1)

    def prev_ensemble_member(self):
	self.set_ensemble_member(self.layout.ensemble_member - 1)

    def set_match(self, match_index):
	self.match_index = match_index
	for settings in self.layout.plots:
	    settings.match_index = match_index
	self.plot()

    def next_match(self):
	self.set_match(self.match_index + 1)
    
    def prev_match(self):
	self.set_match(self.match_index - 1)
    
    def interactive_plot(self):
	cmd = ''
	args = []
	while True:
	    try:
		prev_cmd = cmd
		prev_args = args

		if cmd not in ['c', 'cr']:
		    print('# ', end='')
		    r = raw_input()

		if r == '':
		    cmd = prev_cmd
		    args = prev_args
		else:
		    try:
			cmd = r.split(' ')[0]
			args = r.split(' ')[1:]
		    except:
			cmd = r
			args = None

		if cmd == 'q':
		    break
		elif cmd == 'pl':
		    print('plot')
		    self.next_date()
		    self.plot()
		elif cmd == 'n':
		    print('next')
		    self.next_date()
		    self.plot()
		elif cmd == 'p':
		    print('prev')
		    self.prev_date()
		    self.plot()
		elif cmd == 'c':
		    print('continuing')
		    self.ncdata.next_date()
		    plt.pause(0.01)
		    self.plot()
		elif cmd == 'cr':
		    print('continuing backwards')
		    self.ncdata.prev_date()
		    plt.pause(0.01)
		    self.plot()
		elif cmd == 's':
		    self.save(args[0])
		elif cmd == 'l':
		    self.load(args[0])
		elif cmd == 'ls':
		    self.list()
		elif cmd == 'em':
		    if args[0] == 'n':
			self.next_ensemble_member()
		    elif args[0] == 'p':
			self.prev_ensemble_member()
		    else:
			try:
			    self.set_ensemble_member(int(args[0]))
			except:
			    pass


	    except KeyboardInterrupt, ki:
		# Handle ctrl+c
		# deletes ^C from terminal:
		print('\r', end='')

		cmd = ''
		print('ctrl+c pressed')




def plot_month(gdata, tracks, year, month):
    plt.clf()
    plot_all_vmax(gdata, dt.datetime(year, month, 1), dt.datetime(year, month + 1, 1))
    plot_ibtracks(tracks, dt.datetime(year, month, 1), dt.datetime(year, month + 1, 1))


def plot_ibtrack_with_date(track, date):
    plot_ibtrack(track)
    try:
	index = np.where(track.dates == date)[0][0]
	if track.cls[index] == 'HU':
	    plt.plot(track.lon[index], track.lat[index], 'ro')
	else:
	    plt.plot(track.lon[index], track.lat[index], 'ko')
    except:
	print("Couldn't plot track at date {0} (start {1}, end {2})".format(date, track.dates[0], track.dates[-1]))

def time_plot_match(ncdata, match):
    track = match.track
    vort_track = match.vort_track
    for i, d in enumerate(track.dates):
	ncdata.set_date(d)
	plt.clf()

	plt.title(d)

	plot_ibtrack_with_date(d, i, track)

	plot_on_earth(ncdata.lon, ncdata.lat, ncdata.vort, -6e-5, 5e-4, 'wa')
	plot_vmax_tree(None, vort_track.start_vortmax, 0)
	plot_ibtrack(track)
	plot_match_dist(track, vort_track.start_vortmax, d)

	#plt.pause(0.1)
	raw_input()

def time_plot_ibtrack(ncdata, track):
    for i, d in enumerate(track.dates):
	ncdata.set_date(d)
	plt.clf()

	#plt.subplot(2, 1, 1)
	#plt.title(d)
	#plot_ibtrack_with_date(d, i, track)
	#plot_on_earth(ncdata.lon, ncdata.lat, ncdata.psl, vmin=99500, vmax=103000, loc='wa')

	#plt.subplot(2, 1, 2)
	plot_ibtrack_with_date(d, i, track)
	plot_on_earth(ncdata.lon, ncdata.lat, ncdata.vort, -6e-5, 5e-4)

	#plt.pause(0.1)
	raw_input()

def convert(n):
    return n if n <= 180 else n - 360

def track_length(vortmax):
    length = 0
    while len(vortmax.next_vortmax) != 0:
	length += 1
	if len(vortmax.next_vortmax) != 1:
	    print('len(vortmax.next_vortmax) != 1')
	vortmax = vortmax.next_vortmax[0]
    return length


def time_plot_vmax(gdata):
    for vs in gdata.vortmax_time_series:
	plt.clf()
	plot_on_earth(gdata.ncdata.lon, gdata.ncdata.lat, None, 0, 0, 'wa')
	for vmax in vs:
	    plt.plot(convert(vmax.pos[0]), vmax.pos[1], 'ko')
	    plt.annotate(str(vmax.pos), (convert(vmax.pos[0]), vmax.pos[1] + 0.2))
	raw_input()


def plot_all_vmax(gdata, start_date, end_date):
    plot_on_earth(gdata.ncdata.lon, gdata.ncdata.lat, None, 0, 0, 'wa')
    for vs in gdata.vortmax_time_series.values():
	for vm in vs:
	    if len(vm.prev_vortmax) == 0:
		if start_date < vm.date < end_date:
		    if track_length(vm) >= 6:
			plot_vmax_tree(None, vm, 0)

def plot_matches(ncdata, matches, clear=False):
    for match in matches:
	if clear:
	    plt.clf()
	    plot_on_earth(ncdata.lon, ncdata.lat, None, 0, 0, 'wa')
	plot_match(match)
	
	raw_input()

def plot_match(match, date):
    plot_vmax_tree(date, None, match.vort_track.start_vortmax, 0)
    plot_ibtrack_with_date(match.track, date)
    plot_match_dist(match.track, match.vort_track.start_vortmax, date)
    print('Overlap: {0}, cum dist: {1}, av dist: {2}'.format(match.overlap, match.cum_dist, match.av_dist()))

def plot_match_dist(track, vortmax, date):
    track_index = 0
    while track.dates[track_index] != vortmax.date:
	if track.dates[track_index] < vortmax.date:
	    track_index += 1
	    if track_index == len(track.dates):
		break
	else:
	    if len(vortmax.next_vortmax) == 0:
		break
	    vortmax = vortmax.next_vortmax[0]
    
    if track.dates[track_index] == vortmax.date:
	while True:
	    if date and date == track.dates[track_index]:
		plt.plot((track.lon[track_index], convert(vortmax.pos[0])),
			 (track.lat[track_index], vortmax.pos[1]), 'r-')
	    else:
		plt.plot((track.lon[track_index], convert(vortmax.pos[0])),
			 (track.lat[track_index], vortmax.pos[1]), 'y-')
	    track_index += 1
	    if len(vortmax.next_vortmax):
		vortmax = vortmax.next_vortmax[0]

		if track_index == len(track.dates):
		    break
		elif len(vortmax.next_vortmax) == 0:
		    break
	    else:
		break


def plot_ensemble_matches(ncdata, combined_matches):
    for track, vortmaxes in combined_matches.items():
	plt.clf()
	plot_on_earth(ncdata.lon, ncdata.lat, None, None, None, 'wa')
	plot_ibtrack(track)
	for vortmax in vortmaxes:
	    plot_vmax_tree(None, None, vortmax.vortmaxes[0], 0)

	if raw_input() == 'q':
	    return



def plot_vmax_tree(date, last_vmax, vmax, level, max_level=60):
    #print('{0}: Plotting {1}'.format(level, vmax.pos))
    #if level == 21:
	#import ipdb; ipdb.set_trace()
    if level == 0:
        plt.annotate('{0}: {1}'.format(vmax.date, vmax.index), (convert(vmax.pos[0]), vmax.pos[1] + 0.2))

    if vmax.date == date:
	plt.plot(convert(vmax.pos[0]), vmax.pos[1], 'ko')

    #else:
	#plt.plot(convert(vmax.pos[0]), vmax.pos[1], 'ko')
    #plt.annotate(level, (convert(vmax.pos[0]), vmax.pos[1] + 0.2))

    if last_vmax:
	if abs(convert(last_vmax.pos[0]) - convert(vmax.pos[0])) < 90:
	    plt.plot([convert(last_vmax.pos[0]), convert(vmax.pos[0])],  [last_vmax.pos[1], vmax.pos[1]], 'y-')

    if level > max_level:
	return

    for nvm in vmax.next_vortmax:
	plot_vmax_tree(date, vmax, nvm, level + 1)


def plot_vort_vort4(ncdata, date):
    ncdata.set_date(date)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.title(date)
    plot_on_earth(ncdata.lon, ncdata.lat, ncdata.vort, vmin=-5e-6, vmax=2e-4)

    plt.subplot(2, 2, 3)
    plot_on_earth(ncdata.lon, ncdata.lat, ncdata.vort4, vmin=-5e-6, vmax=2e-4)

    plt.subplot(2, 2, 2)
    plot_on_earth(ncdata.lon, ncdata.lat, None)
    print(len(ncdata.vmaxs))
    for v, vmax in ncdata.vmaxs:
	if v > 1e-4:
	    plt.plot(convert(vmax[0]), vmax[1], 'go')
	else:
	    plt.plot(convert(vmax[0]), vmax[1], 'kx')

    plt.subplot(2, 2, 4)
    plot_on_earth(ncdata.lon, ncdata.lat, None)
	
    print(len(ncdata.v4maxs))
    for v, vmax in ncdata.v4maxs:
	if v > 5e-5:
	    plt.plot(convert(vmax[0]), vmax[1], 'go')
	else:
	    plt.plot(convert(vmax[0]), vmax[1], 'kx')


def time_plot_ibtracks_pressure_vort(ncdata, tracks, dates):
    for i, d in enumerate(dates):
	plt.clf()
	plt.subplot(2, 2, 1)
	plt.title(d)
	plot_on_earth(ncdata.lon, ncdata.lat, ncdata.get_pressure_from_date(d), vmin=99500, vmax=103000)

	plt.subplot(2, 2, 3)
	plot_on_earth(ncdata.lon, ncdata.lat, ncdata.get_vort_from_date(d), vmin=-5e-6, vmax=2e-4)

	plt.subplot(2, 2, 2)
	plot_on_earth(ncdata.lon, ncdata.lat, None)
	for p, pmin in ncdata.pmins:
	    plt.plot(convert(pmin[0]), pmin[1], 'kx')

	plt.subplot(2, 2, 4)
	plot_on_earth(ncdata.lon, ncdata.lat, None)
	    
	for v, vmax in ncdata.vmaxs:
	    if v > 10:
		plt.plot(convert(vmax[0]), vmax[1], 'go')
	    else:
		plt.plot(convert(vmax[0]), vmax[1], 'kx')

	for track in tracks:
	    #import ipdb; ipdb.set_trace()
	    
	    if track.dates[0] > d or track.dates[-1] < d:
		continue

	    index = 0
	    for j, date in enumerate(track.dates):
		if date == d:
		    index = j
		    break
	    for i in range(4):
		plt.subplot(2, 2, i + 1)
		plot_ibtrack_with_date(d, index, track)


	plt.pause(0.1)
	print(d)
	raw_input()

def time_plot_ibtracks_vort_smoothedvort(ncdata, tracks, dates):
    for i, d in enumerate(dates):
	ncdata.set_date(d)
	plt.clf()
	plt.subplot(2, 2, 1)
	plt.title(d)
	plot_on_earth(ncdata.lon, ncdata.lat, ncdata.vort, vmin=-5, vmax=15)

	plt.subplot(2, 2, 3)
	plot_on_earth(ncdata.lon, ncdata.lat, ncdata.smoothed_vort, vmin=-5, vmax=15)

	plt.subplot(2, 2, 2)
	plot_on_earth(ncdata.lon, ncdata.lat, None)
	for v, vmax in ncdata.vmaxs:
	    if v > 10:
		plt.plot(convert(vmax[0]), vmax[1], 'go')
	    elif v > 4:
		plt.plot(convert(vmax[0]), vmax[1], 'kx')
	    else:
		plt.plot(convert(vmax[0]), vmax[1], 'y+')

	plt.subplot(2, 2, 4)
	plot_on_earth(ncdata.lon, ncdata.lat, None)
	for v, vmax in ncdata.smoothed_vmaxs:
	    if v > 10:
		plt.plot(convert(vmax[0]), vmax[1], 'go')
	    elif v > 4:
		plt.plot(convert(vmax[0]), vmax[1], 'kx')
	    else:
		plt.plot(convert(vmax[0]), vmax[1], 'y+')

	for track in tracks:
	    #import ipdb; ipdb.set_trace()
	    
	    if track.dates[0] > d or track.dates[-1] < d:
		continue

	    index = 0
	    for j, date in enumerate(track.dates):
		if date == d:
		    index = j
		    break
	    for i in range(4):
		plt.subplot(2, 2, i + 1)
		plot_ibtrack_with_date(d, index, track)


	plt.pause(0.1)
	print(d)
	raw_input()

def plot_extrema(lon, lat, maxs, mins):
    if maxs != None:
	for mx in maxs:
	    plt.plot(convert(lon[mx[1]]), lat[mx[0]], 'rx')
    if mins != None:
	for mn in mins:
	    plt.plot(convert(lon[mn[1]]), lat[mn[0]], 'b+')


def plot_isobar(isobar, point):
    plt.clf()
    plt.figure(10)
    plt.plot(isobar.path[:, 0], isobar.path[:, 1])
    plt.plot(point.x, point.y, 'kx')
    print(isobar.contains(point))

def plot_wilma_track():
    w = load_wilma()
    plot_cyclone_track(w)
    return w

def plot_cyclone_vort(cyclone):
    plt.imshow(cyclone.vort[::-1, :], interpolation='nearest')

def plot_cyclone_psl(cyclone):
    plt.imshow(cyclone.psl[::-1, :], interpolation='nearest')

def plot_cyclone_windspeed(cyclone):
    plt.imshow(cyclone.wind_speed[::-1, :], interpolation='nearest')

def plot_cyclone_wind(cyclone):
    plt.quiver(cyclone.u, cyclone.v)

def plot_cyclone_chain(cyclone_set):
    for cyclone in cyclone_set.cyclones:
        plot_cyclone(cyclone)

def plot_all_tracks(tc_sets):
    for c_sets in tc_sets:
	for c_set in c_sets:
	    plot_cyclone_track(c_set)

def plot_all_stats(all_cyclone_sets):
    plt.figure(1)
    plt.cla()

    plt.figure(2)
    plt.cla()

    plt.figure(3)
    plt.cla()
    for cyclone_set in all_cyclone_sets:
        for cyclone in cyclone_set.cyclones:
	    for c in cyclone.cyclones:
		plt.figure(1)
		plt.plot(c.max_vort, c.min_psl, 'kx')

		plt.figure(2)
		plt.plot(c.max_wind_speed, c.min_psl, 'kx')

		plt.figure(3)
		plt.plot(c.max_vort, c.max_wind_speed, 'kx')

def plot_cyclone_stats(c_set, curr_c, min_length=5):
    if len(c_set.cyclones) < min_length:
        return
    min_psls  = []
    max_vorts = []
    max_winds = []

    for i, c in enumerate(c_set.cyclones):
        min_psls.append(c.psl.min())
        max_vorts.append(c.vort.max())
        max_winds.append(c.wind_speed.max())
        if c == curr_c:
            plt.subplot(2, 1, 1)
            plt.plot(i, c.psl.min(), 'ro')

            plt.subplot(2, 1, 2)
            plt.plot(i, c.wind_speed.max(), 'ro')

    plt.subplot(2, 1, 1)
    plt.plot(min_psls)

    plt.subplot(2, 1, 2)
    plt.plot(max_winds)


def plot_cyclone_track(c_set, min_length=2):
    if len(c_set.cyclones) < min_length:
        return
    coords = []

    for cyclone in c_set.cyclones:
        coords.append((cyclone.cell_pos[0], cyclone.cell_pos[1]))
    coords = np.array(coords)
    plt.plot(coords[:, 0], coords[:, 1], 'g-')

def plot_all_cyclones(cyclones):
    plt.figure(1)
    plt.cla()
    for cyclone in cyclones:
        plot_cyclone(cyclone)

def plot_cyclone(cyclone):
    plt.plot(cyclone.cell_pos[0], cyclone.cell_pos[1], 'k+')
    for isobar in cyclone.isobars:
        plt.xlim((0, 360))
        plt.ylim((-90, 90))
        plt.plot(isobar.contour[:, 0], isobar.contour[:, 1])

def plot_raster(c):
    i = c.isobars[-1]
    a = path_to_raster(i.path)
    b, d = fill_raster(a)

    plt.figure(1)
    plt.clf()
    plt.plot(i.path[:, 0], i.path[:, 1], 'b-')

    plt.figure(2)
    plt.clf()
    plt.imshow(a[::-1, :], interpolation='nearest')

    plt.figure(3)
    plt.clf()
    plt.imshow(b[::-1, :], interpolation='nearest')

    plt.figure(4)
    plt.clf()
    plt.imshow(c.psl[::-1, :], interpolation='nearest')

def plot_rasters(cs, index):
    for c in cs[index]:
	plot_raster(c)
	raw_input()

def plot_problems():
    args = create_args()
    args.start = 0
    args.end = 3
    cs, pt = main(args)
    plot_raster(cs[1][3])
    raw_input()

    plot_raster(cs[3][11])
    raw_input()

def plot_cyclone_progression(c_set):
    for c in c_set.cyclones:
        plt.figure(1)
        plt.clf()
        plt.title(' %s vorticity'%str(c.date))
        plot_cyclone_vort(c)
        plt.colorbar()

        plt.figure(2)
        plt.clf()
        plt.title(' %s pressure'%str(c.date))
        plot_cyclone_psl(c)
        plt.colorbar()

        plt.figure(3)
        plt.clf()
        plt.title(' %s windspeed'%str(c.date))
        plot_cyclone_windspeed(c)
        plt.colorbar()

        plt.figure(4)
        plt.clf()
        plt.title(' %s wind'%str(c.date))
        plot_cyclone_wind(c)

        plt.figure(5)
        plt.clf()
        plot_cyclone_track(c_set)
        coord = (c.cell_pos[0], c.cell_pos[1])
        plt.plot(coord[0], coord[1], 'ro')

        plt.figure(6)
        plt.clf()
        plot_cyclone_stats(c_set, c)

        raw_input()

def plot_ibtracks(ss, start_date, end_date):
    #plt.xlim((-180, 180))
    #plt.ylim((-90, 90))
    for s in ss:
	if s.dates[0] >= start_date and s.dates[0] <= end_date:
	    plot_ibtrack(s)

def plot_ibtrack(s, offset=0):
    plt.plot(s.lon + offset, s.lat, 'r-')
    plt.annotate(str(s.index), (s.lon[0] + 0.2, s.lat[0] + 0.2))

def plot_track(nc_dataset):
    plt.plot(lons[0] + 360, lats[0])

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

def plot_wilma(ncdata):
    plot_between_dates(ncdata, dt.datetime(2005, 10, 18), dt.datetime(2005, 10, 28))

def plot_between_dates(ncdata, start_date, end_date):
    date = ncdata.set_date(start_date)
    while date < end_date:
	plt.clf()

	plt.subplot(2, 2, 1)
	plt.title(date)
	#plot_on_earth(ncdata.lon, ncdata.lat, ncdata.psl, 97000, 106000, 'wa')
	plot_on_earth(ncdata.lon, ncdata.lat, ncdata.vort, -6e-5, 5e-4, 'wa')

	plt.subplot(2, 2, 3)
	plot_on_earth(ncdata.lon, ncdata.lat, ncdata.vort4, -6e-5, 5e-4, 'wa')

	plt.subplot(2, 2, 2)
	plot_on_earth(ncdata.lon, ncdata.lat, None, 0, 0, 'wa')
	for v, vmax in ncdata.vmaxs:
	    if v > 3e-4:
		plt.plot(convert(vmax[0]), vmax[1], 'ro')
		plt.annotate('{0:2.1f}'.format(v * 1e4), (convert(vmax[0]), vmax[1] + 0.2))
	    elif v > 2e-4:
		plt.plot(convert(vmax[0]), vmax[1], 'yo')
		plt.annotate('{0:2.1f}'.format(v * 1e4), (convert(vmax[0]), vmax[1] + 0.2))
	    elif v > 1e-4:
		plt.plot(convert(vmax[0]), vmax[1], 'go')
	    else:
		#plt.plot(convert(vmax[0]), vmax[1], 'kx')
		pass

	plt.subplot(2, 2, 4)
	plot_on_earth(ncdata.lon, ncdata.lat, None, 0, 0, 'wa')
	for v, vmax in ncdata.v4maxs:
	    if v > 3e-4:
		plt.plot(convert(vmax[0]), vmax[1], 'ro')
		plt.annotate('{0:2.1f}'.format(v * 1e4), (convert(vmax[0]), vmax[1] + 0.2))
	    elif v > 2e-4:
		plt.plot(convert(vmax[0]), vmax[1], 'yo')
		plt.annotate('{0:2.1f}'.format(v * 1e4), (convert(vmax[0]), vmax[1] + 0.2))
	    elif v > 1e-4:
		plt.plot(convert(vmax[0]), vmax[1], 'go')
	    else:
		#plt.plot(convert(vmax[0]), vmax[1], 'kx')
		pass

	raw_input()
	date = ncdata.next_date()
	#plt.pause(0.1)


def plot_on_earth(lons, lats, data, vmin=None, vmax=None, loc='earth'):
    if loc == 'earth':
	m = Basemap(projection='cyl', resolution='c', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
    elif loc == 'wa':
	m = Basemap(projection='cyl', resolution='c', llcrnrlat=0, urcrnrlat=60, llcrnrlon=-120, urcrnrlon=-30)

    if data != None:
	plot_lons, plot_data = extend_data(lons, lats, data)
	lons, lats = np.meshgrid(plot_lons, lats)
	x, y = m(lons, lats)
	if vmin:
	    m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax)
	else:
	    m.pcolormesh(x, y, plot_data)

    m.drawcoastlines()

    p_labels = [0, 1, 0, 0]

    m.drawparallels(np.arange(-90.,90.1,45.), labels=p_labels, fontsize=10)
    m.drawmeridians(np.arange(-180.,180.,60.), labels=[0, 0, 0, 1], fontsize=10)


def extend_data(lons, lats, data):
    if False:
        # Adds extra data at the end.
        plot_offset = 2
        plot_lons = np.zeros((lons.shape[0] + plot_offset,))
        plot_lons[:-plot_offset] = lons
        plot_lons[-plot_offset:] = lons[-plot_offset:] + 3.75 * plot_offset

        plot_data = np.zeros((data.shape[0], data.shape[1] + plot_offset))
        plot_data[:, :-plot_offset] = data
        plot_data[:, -plot_offset:] = data[:, :plot_offset]
    else:
        # Adds extra data before the start.
	#import ipdb; ipdb.set_trace()
	delta = lons[1] - lons[0]
        plot_offset = 180
        plot_lons = np.ma.zeros((lons.shape[0] + plot_offset,))
        plot_lons[plot_offset:] = lons
        plot_lons[:plot_offset] = lons[-plot_offset:] - delta * (lons.shape[0])

        plot_data = np.ma.zeros((data.shape[0], data.shape[1] + plot_offset))
        plot_data[:, plot_offset:] = data
        plot_data[:, :plot_offset] = data[:, -plot_offset:]

    return plot_lons, plot_data




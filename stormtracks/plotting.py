from __future__ import print_function

import os
import datetime as dt
import pickle
from glob import glob

import pylab as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

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

        # self.field = None
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
    def __init__(self, best_tracks, c20data, gdatas, all_matches):
        self.best_tracks = best_tracks

        self.c20data = c20data
        self.gdatas = gdatas
        self.all_matches = all_matches
        self.date = self.c20data.first_date()
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
                field = getattr(self.c20data, settings.field)
                raster_on_earth(self.c20data.lons, self.c20data.lats, field,
                                settings.vmin, settings.vmax, settings.loc)
            else:
                raster_on_earth(self.c20data.lons, self.c20data.lats,
                                None, None, None, settings.loc)

            if settings.points:
                title.append(settings.points)
                points = getattr(self.c20data, settings.points)
                for p_val, p_loc in points:
                    plot_point_on_earth(p_loc[0], p_loc[1], 'kx')

            if settings.best_track_index != -1:
                plot_ibtrack_with_date(self.best_tracks[settings.best_track_index], date)

            if settings.vort_max_track_index != -1:
                pass

            if settings.pressure_min_track_index != -1:
                pass

            if settings.match_index != -1:
                plot_match_with_date(self.all_matches[ensemble_member][settings.match_index], date)

            plt.title(' '.join(title))

    def save(self, name):
        self.layout.name = name
        f = open(os.path.join(settings.SETTINGS_DIR, 'plot_{0}.pkl'.format(self.layout.name)), 'w')
        pickle.dump(self.layout, f)

    def load(self, name, is_plot=True):
        try:
            f = open(os.path.join(settings.SETTINGS_DIR, 'plot_{0}.pkl'.format(name)), 'r')
            layout = pickle.load(f)
            if layout.version != self.layout.version:
                r = raw_input('version mismatch, may not work! Press c to continue anyway: ')
                if r != 'c':
                    raise Exception('Version mismatch (user cancelled)')

            self.layout.name = name
            self.layout = layout
            self.set_date(self.layout.date, is_plot)
        except Exception, e:
            print('Settings {0} could not be loaded'.format(name))
            print('{0}'.format(e.message))
            raise

    def print_list(self):
        for settings_name in self.list():
            print(settings_name)

    def delete(self, name):
        file_name = os.path.join(settings.SETTINGS_DIR, 'plot_{0}.pkl'.format(name))
        os.remove(file_name)

    def list(self):
        plotting_settings = []
        for fn in glob(os.path.join(settings.SETTINGS_DIR, 'plot_*.pkl')):
            plotting_settings.append('_'.join(os.path.basename(fn).split('.')[0].split('_')[1:]))
        return plotting_settings

    def set_date(self, date, is_plot=True):
        self.layout.date = self.c20data.set_date(
            date, self.layout.ensemble_member, self.layout.ensemble_mode)
        if is_plot:
            self.plot()

    def next_date(self):
        self.layout.date = self.c20data.next_date(
            self.layout.ensemble_member, self.layout.ensemble_mode)
        self.plot()

    def prev_date(self):
        self.layout.date = self.c20data.prev_date(
            self.layout.ensemble_member, self.layout.ensemble_mode)
        self.plot()

    def set_ensemble_member(self, ensemble_member):
        self.layout.ensemble_member = ensemble_member
        self.c20data.set_date(
            self.layout.date, self.layout.ensemble_member, self.layout.ensemble_mode)
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
                    r = raw_input('# ')

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
                    self.c20data.next_date()
                    plt.pause(0.01)
                    self.plot()
                elif cmd == 'cr':
                    print('continuing backwards')
                    self.c20data.prev_date()
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


def plot_vortmax_track_with_date(vortmax_track, date=None):
    plot_track(vortmax_track)
    if date:
        try:
            index = np.where(vortmax_track.dates == date)[0][0]
            plot_point_on_earth(vortmax_track.lons[index], vortmax_track.lats[index], 'ko')
        except:
            print("Couldn't plot track at date {0} (start {1}, end {2})".format(
                date, vortmax_track.dates[0], vortmax_track.dates[-1]))


def plot_ibtrack_with_date(best_track, date=None):
    plot_track(best_track)
    if date:
        try:
            index = np.where(best_track.dates == date)[0][0]
            if best_track.cls[index] == 'HU':
                plot_point_on_earth(best_track.lons[index], best_track.lats[index], 'ro')
            else:
                plot_point_on_earth(best_track.lons[index], best_track.lats[index], 'ko')
        except:
            print("Couldn't plot best_track at date {0} (start {1}, end {2})".format(
                date, best_track.dates[0], best_track.dates[-1]))


# TODO: integrate with Plotting class.
def time_plot_match(c20data, match):
    best_track = match.best_track
    vort_track = match.vort_track
    for date in best_track.dates:
        c20data.set_date(date)
        plt.clf()

        plt.title(date)

        raster_on_earth(c20data.lons, c20data.lats, c20data.vort, -6e-5, 5e-4, 'wa')
        plot_vortmax_track_with_date(vort_track, date)
        plot_ibtrack_with_date(best_track, date)

        plot_match_dist_with_date(match, date)

        raw_input()


# TODO: integrate with Plotting class.
def time_plot_ibtrack(c20data, best_track):
    for date in best_track.dates:
        c20data.set_date(date)
        plt.clf()

        plt.subplot(2, 1, 1)
        plt.title(date)
        plot_ibtrack_with_date(best_track, date)
        raster_on_earth(c20data.lons, c20data.lats, c20data.psl, vmin=99500, vmax=103000, loc='wa')

        plt.subplot(2, 1, 2)
        plot_ibtrack_with_date(best_track, date)
        raster_on_earth(c20data.lons, c20data.lats, c20data.vort, -6e-5, 5e-4, loc='wa')

        raw_input()


# TODO: integrate with Plotting class.
def time_plot_vmax(gdata):
    for date in gdata.vortmax_time_series.keys():
        vortmaxes = gdata.vortmax_time_series[date]
        plt.clf()
        plt.title(date)
        raster_on_earth(gdata.c20data.lons, gdata.c20data.lats, None, 0, 0, 'wa')
        for vmax in vortmaxes:
            plot_point_on_earth(vmax.pos[0], vmax.pos[1], 'ko')
        raw_input()


# TODO: integrate with Plotting class.
def plot_matches(c20data, matches, clear=False):
    for match in matches:
        if clear:
            plt.clf()
            raster_on_earth(c20data.lons, c20data.lats, None, 0, 0, 'wa')
        plot_match_with_date(match)

        raw_input()


def plot_match_with_date(match, date=None):
    plot_vortmax_track_with_date(match.vort_track, date)
    plot_ibtrack_with_date(match.best_track, date)
    plot_match_dist_with_date(match, date)
    print('Overlap: {0}, cum dist: {1}, av dist: {2}'.format(
        match.overlap, match.cum_dist, match.av_dist()))


def plot_match_dist_with_date(match, date):
    best_track = match.best_track
    vortmax_track = match.vort_track

    track_index = np.where(best_track.dates == match.overlap_start)[0][0]
    vortmax_track_index = np.where(vortmax_track.dates == match.overlap_start)[0][0]

    while True:
        vortmax = vortmax_track.vortmaxes[vortmax_track_index]
        if date and date == best_track.dates[track_index]:
            plot_path_on_earth(np.array((best_track.lons[track_index], vortmax.pos[0])),
                               np.array((best_track.lats[track_index], vortmax.pos[1])), 'r-')
        else:
            plot_path_on_earth(np.array((best_track.lons[track_index], vortmax.pos[0])),
                               np.array((best_track.lats[track_index], vortmax.pos[1])), 'y-')

        track_index += 1
        vortmax_track_index += 1
        print(track_index, vortmax_track_index)

        if track_index >= len(best_track.lons) or vortmax_track_index >= len(vortmax_track.lons):
            break


# TODO: integrate with Plotting class.
def plot_ensemble_matches(c20data, combined_matches):
    for best_track, matches in combined_matches.items():
        plt.clf()
        raster_on_earth(c20data.lons, c20data.lats, None, None, None, 'wa')
        plot_track(best_track)
        for match in matches:
            plot_vortmax_track_with_date(match.vort_track)

        if raw_input() == 'q':
            return


def plot_vort_vort4(c20data, date):
    c20data.set_date(date)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.title(date)
    raster_on_earth(c20data.lons, c20data.lats, c20data.vort, vmin=-5e-6, vmax=2e-4)

    plt.subplot(2, 2, 3)
    raster_on_earth(c20data.lons, c20data.lats, c20data.vort4, vmin=-5e-6, vmax=2e-4)

    plt.subplot(2, 2, 2)
    raster_on_earth(c20data.lons, c20data.lats, None)
    print(len(c20data.vmaxs))
    for v, vmax in c20data.vmaxs:
        if v > 1e-4:
            plot_point_on_earth(vmax[0], vmax[1], 'go')
        else:
            plot_point_on_earth(vmax[0], vmax[1], 'kx')

    plt.subplot(2, 2, 4)
    raster_on_earth(c20data.lons, c20data.lats, None)

    print(len(c20data.v4maxs))
    for v, vmax in c20data.v4maxs:
        if v > 5e-5:
            plot_point_on_earth(vmax[0], vmax[1], 'go')
        else:
            plot_point_on_earth(vmax[0], vmax[1], 'kx')


def time_plot_ibtracks_pressure_vort(c20data, best_tracks, dates):
    for i, date in enumerate(dates):
        c20data.set_date(date)
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.title(date)
        raster_on_earth(c20data.lons, c20data.lats, c20data.psl, vmin=99500, vmax=103000)

        plt.subplot(2, 2, 3)
        raster_on_earth(c20data.lons, c20data.lats, c20data.vort, vmin=-5e-6, vmax=2e-4)

        plt.subplot(2, 2, 2)
        raster_on_earth(c20data.lons, c20data.lats, None)
        for p, pmin in c20data.pmins:
            plot_point_on_earth(pmin[0], pmin[1], 'kx')

        plt.subplot(2, 2, 4)
        raster_on_earth(c20data.lons, c20data.lats, None)

        for v, vmax in c20data.vmaxs:
            if v > 10:
                plot_point_on_earth(vmax[0], vmax[1], 'go')
            else:
                plot_point_on_earth(vmax[0], vmax[1], 'kx')

        for best_track in best_tracks:
            if best_track.dates[0] > date or best_track.dates[-1] < date:
                continue

            index = 0
            for j, track_date in enumerate(best_track.dates):
                if track_date == date:
                    index = j
                    break
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plot_ibtrack_with_date(best_track, date)

        plt.pause(0.1)
        print(date)
        raw_input()


def time_plot_ibtracks_vort_smoothedvort(c20data, best_tracks, dates):
    if not c20data.smoothing:
        print('Turning on smoothing for c20 data')
        c20data.smoothing = True

    for i, date in enumerate(dates):
        c20data.set_date(date)
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.title(date)
        raster_on_earth(c20data.lons, c20data.lats, c20data.vort)

        plt.subplot(2, 2, 3)
        raster_on_earth(c20data.lons, c20data.lats, c20data.smoothed_vort)

        plt.subplot(2, 2, 2)
        raster_on_earth(c20data.lons, c20data.lats, None)
        for v, vmax in c20data.vmaxs:
            if v > 10:
                plot_point_on_earth(vmax[0], vmax[1], 'go')
            elif v > 4:
                plot_point_on_earth(vmax[0], vmax[1], 'kx')
            else:
                plot_point_on_earth(vmax[0], vmax[1], 'y+')

        plt.subplot(2, 2, 4)
        raster_on_earth(c20data.lons, c20data.lats, None)
        for v, vmax in c20data.smoothed_vmaxs:
            if v > 10:
                plot_point_on_earth(vmax[0], vmax[1], 'go')
            elif v > 4:
                plot_point_on_earth(vmax[0], vmax[1], 'kx')
            else:
                plot_point_on_earth(vmax[0], vmax[1], 'y+')

        for best_track in best_tracks:
            if best_track.dates[0] > date or best_track.dates[-1] < date:
                continue

            index = 0
            for j, track_date in enumerate(best_track.dates):
                if track_date == date:
                    index = j
                    break
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plot_ibtrack_with_date(best_track, date)

        plt.pause(0.1)
        print(date)
        raw_input()


def plot_ibtracks(best_tracks, start_date, end_date):
    for best_track in best_tracks:
        if best_track.dates[0] >= start_date and best_track.dates[0] <= end_date:
            plot_track(best_track)


def plot_track(best_track, plt_fmt=None):
    if plt_fmt:
        plot_path_on_earth(best_track.lons, best_track.lats, plt_fmt)
    else:
        plot_path_on_earth(best_track.lons, best_track.lats, 'r-')


# TODO: integrate with Plotting class.
def plot_wilma(c20data):
    plot_between_dates(c20data, dt.datetime(2005, 10, 18), dt.datetime(2005, 10, 28))


# TODO: integrate with Plotting class.
def plot_between_dates(c20data, start_date, end_date):
    date = c20data.set_date(start_date)
    while date < end_date:
        plt.clf()

        plt.subplot(2, 2, 1)
        plt.title(date)
        # raster_on_earth(c20data.lons, c20data.lats, c20data.psl, 97000, 106000, 'wa')
        raster_on_earth(c20data.lons, c20data.lats, c20data.vort, -6e-5, 5e-4, 'wa')

        plt.subplot(2, 2, 3)
        raster_on_earth(c20data.lons, c20data.lats, c20data.vort4, -6e-5, 5e-4, 'wa')

        plt.subplot(2, 2, 2)
        raster_on_earth(c20data.lons, c20data.lats, None, 0, 0, 'wa')
        for v, vmax in c20data.vmaxs:
            if v > 3e-4:
                plot_point_on_earth(vmax[0], vmax[1], 'ro')
                plt.annotate('{0:2.1f}'.format(v * 1e4), (vmax[0], vmax[1] + 0.2))
            elif v > 2e-4:
                plot_point_on_earth(vmax[0], vmax[1], 'yo')
                plt.annotate('{0:2.1f}'.format(v * 1e4), (vmax[0], vmax[1] + 0.2))
            elif v > 1e-4:
                plot_point_on_earth(vmax[0], vmax[1], 'go')
            else:
                # plot_point_on_earth(vmax[0], vmax[1], 'kx')
                pass

        plt.subplot(2, 2, 4)
        raster_on_earth(c20data.lons, c20data.lats, None, 0, 0, 'wa')
        for v, vmax in c20data.v4maxs:
            if v > 3e-4:
                plot_point_on_earth(vmax[0], vmax[1], 'ro')
                plt.annotate('{0:2.1f}'.format(v * 1e4), (vmax[0], vmax[1] + 0.2))
            elif v > 2e-4:
                plot_point_on_earth(vmax[0], vmax[1], 'yo')
                plt.annotate('{0:2.1f}'.format(v * 1e4), (vmax[0], vmax[1] + 0.2))
            elif v > 1e-4:
                plot_point_on_earth(vmax[0], vmax[1], 'go')
            else:
                # plot_point_on_earth(vmax[0], vmax[1], 'kx')
                pass

        raw_input()
        date = c20data.next_date()
        # plt.pause(0.1)


def lons_convert(lons):
    m = lons > 180
    lons[m] = lons[m] - 360
    return lons


def lon_convert(lon):
    return lon if lon <= 180 else lon - 360


def plot_path_on_earth(lons, lats, plot_fmt=None):
    if plot_fmt:
        plt.plot(lons_convert(lons), lats, plot_fmt)
    else:
        plt.plot(lons_convert(lons), lats)


def plot_point_on_earth(lon, lat, plot_fmt=None):
    if plot_fmt:
        plt.plot(lon_convert(lon), lat, plot_fmt)
    else:
        plt.plot(lon_convert(lon), lat)


def raster_on_earth(lons, lats, data, vmin=None, vmax=None, loc='earth'):
    if loc == 'earth':
        m = Basemap(projection='cyl', resolution='c',
                    llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
    elif loc == 'wa':
        m = Basemap(projection='cyl', resolution='c',
                    llcrnrlat=0, urcrnrlat=60, llcrnrlon=-120, urcrnrlon=-30)

    if data is not None:
        plot_lons, plot_data = extend_data(lons, lats, data)
        lons, lats = np.meshgrid(plot_lons, lats)
        x, y = m(lons, lats)
        if vmin:
            m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax)
        else:
            m.pcolormesh(x, y, plot_data)

    m.drawcoastlines()

    p_labels = [0, 1, 0, 0]

    m.drawparallels(np.arange(-90., 90.1, 45.), labels=p_labels, fontsize=10)
    m.drawmeridians(np.arange(-180., 180., 60.), labels=[0, 0, 0, 1], fontsize=10)


def extend_data(lons, lats, data):
    if False:
        # TODO: probably doesn't work!
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
        delta = lons[1] - lons[0]
        plot_offset = 180
        plot_lons = np.ma.zeros((lons.shape[0] + plot_offset,))
        plot_lons[plot_offset:] = lons
        plot_lons[:plot_offset] = lons[-plot_offset:] - delta * (lons.shape[0])

        plot_data = np.ma.zeros((data.shape[0], data.shape[1] + plot_offset))
        plot_data[:, plot_offset:] = data
        plot_data[:, :plot_offset] = data[:, -plot_offset:]

    return plot_lons, plot_data


if False:
    # Possibly defunct functions, but may be useful in the future.
    # Not checked
    def plot_isobar(isobar, point):
        plt.clf()
        plt.figure(10)
        plot_path_on_earth(isobar.path[:, 0], isobar.path[:, 1])
        plot_path_on_earth(point.x, point.y, 'kx')
        print(isobar.contains(point))

    # Not checked
    def plot_cyclone_vort(cyclone):
        plt.imshow(cyclone.vort[::-1, :], interpolation='nearest')

    # Not checked
    def plot_cyclone_psl(cyclone):
        plt.imshow(cyclone.psl[::-1, :], interpolation='nearest')

    # Not checked
    def plot_cyclone_windspeed(cyclone):
        plt.imshow(cyclone.wind_speed[::-1, :], interpolation='nearest')

    # Not checked
    def plot_cyclone_wind(cyclone):
        plt.quiver(cyclone.u, cyclone.v)

    # Not checked
    def plot_cyclone_chain(cyclone_set):
        for cyclone in cyclone_set.cyclones:
            plot_cyclone(cyclone)

    # Not checked
    def plot_all_tracks(tc_sets):
        for c_sets in tc_sets:
            for c_set in c_sets:
                plot_cyclone_track(c_set)

    # Not checked
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

    # Not checked
    def plot_cyclone_stats(c_set, curr_c, min_length=5):
        if len(c_set.cyclones) < min_length:
            return
        min_psls = []
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

    # Not checked
    def plot_cyclone_track(c_set, min_length=2):
        if len(c_set.cyclones) < min_length:
            return
        coords = []

        for cyclone in c_set.cyclones:
            coords.append((cyclone.cell_pos[0], cyclone.cell_pos[1]))
        coords = np.array(coords)
        plot_path_on_earth(coords[:, 0], coords[:, 1], 'g-')

    # Not checked
    def plot_all_cyclones(cyclones):
        plt.figure(1)
        plt.cla()
        for cyclone in cyclones:
            plot_cyclone(cyclone)

    # Not checked
    def plot_cyclone(cyclone):
        plot_point_on_earth(cyclone.cell_pos[0], cyclone.cell_pos[1], 'k+')
        for isobar in cyclone.isobars:
            plt.xlim((0, 360))
            plt.ylim((-90, 90))
            plot_path_on_earth(isobar.contour[:, 0], isobar.contour[:, 1])

    # Not checked
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

    # Not checked
    def plot_rasters(cs, index):
        for c in cs[index]:
            plot_raster(c)
            raw_input()

    # Not checked
    def plot_problems():
        args = create_args()
        args.start = 0
        args.end = 3
        cs, pt = main(args)
        plot_raster(cs[1][3])
        raw_input()

        plot_raster(cs[3][11])
        raw_input()

    # Not checked
    def plot_cyclone_progression(c_set):
        for c in c_set.cyclones:
            plt.figure(1)
            plt.clf()
            plt.title(' {0} vorticity'.format(c.date))
            plot_cyclone_vort(c)
            plt.colorbar()

            plt.figure(2)
            plt.clf()
            plt.title(' {0} pressure'.format(c.date))
            plot_cyclone_psl(c)
            plt.colorbar()

            plt.figure(3)
            plt.clf()
            plt.title(' {0} windspeed'.format(c.date))
            plot_cyclone_windspeed(c)
            plt.colorbar()

            plt.figure(4)
            plt.clf()
            plt.title(' {0} wind'.format(c.date))
            plot_cyclone_wind(c)

            plt.figure(5)
            plt.clf()
            plot_cyclone_track(c_set)
            coord = (c.cell_pos[0], c.cell_pos[1])
            plot_point_on_earth(coord[0], coord[1], 'ro')

            plt.figure(6)
            plt.clf()
            plot_cyclone_stats(c_set, c)

            raw_input()

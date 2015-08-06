from __future__ import print_function

import copy
import os
import datetime as dt
from glob import glob

import simplejson
import pylab as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

from ..utils.c_wrapper import cvort, cvort4
from ..load_settings import settings

SAVE_FILE_TPL = 'plot_{0}.json'

DEFAULT_PLOTTER_LAYOUT = {
    'version': 0.1,
    'date': None,
    'name': None,
    'figure': 1,
    'plot_settings': [],
    'ensemble_member': 0,
    'ensemble_mode': 'member',
    }

DEFAULT_PLOT_SETTINGS = {
    'subplot': '111',
    'field': 'prmsl',
    'points': 'pmins',
    'loc': 'wa',
    'vmax': None,
    'vmin': None,
    'best_track_index': -1,
    'vort_max_track_index': -1,
    'pressure_min_track_index': -1,
    'match_index': -1,
    }


def plot_pandas_matched_unmatched(df_year, bt_matches, f1='vort', f2='t850s', flip=False, alpha=0.5):
    plt.clf()
    comb = df_year.join(bt_matches)
    matched = comb[~comb.bt_name.isnull()]
    unmatched = comb[comb.bt_name.isnull()]
    if not flip:
        zo1 = 1
        zo2 = 2
        alpha1=1
        alpha2=alpha
    else:
        zo1 = 2
        zo2 = 1
        alpha1=alpha
        alpha2=1
    uf1 = getattr(unmatched, f1)
    uf2 = getattr(unmatched, f2)
    mf1 = getattr(matched, f1)
    mf2 = getattr(matched, f2)
    plt.scatter(uf1, uf2, c='k', marker='s', zorder=zo1, alpha=alpha1)
    plt.scatter(mf1, mf2, c=matched.bt_wind, marker='s', zorder=zo2, alpha=alpha2)
    plt.xlim((uf1.min(), uf1.max()))
    plt.ylim((uf2.min(), uf2.max()))


def plot_pandas_tracks(ib, matched, mode='all'):
    for bt in ib.best_tracks:
        plt.plot(bt.lons, bt.lats, 'r-', zorder=1)
        plt.scatter(x=bt.lons, y=bt.lats, c=bt.winds, marker='o', zorder=2)
        if mode == 'all':
            lon_lat = matched[matched.bt_name == bt.name]
            plt.scatter(lon_lat.lon, lon_lat.lat, marker='x', zorder=0)
        elif mode == 'mean':
            lon_lat = matched[matched.bt_name == bt.name].groupby('date')['lon', 'lat'].mean()
            lon_lat_std = matched[matched.bt_name == bt.name].groupby('date')['lon', 'lat'].std()
            plt.errorbar(lon_lat.lon, lon_lat.lat, xerr=lon_lat_std.lon, yerr=lon_lat_std.lat, marker='x', zorder=0)
            plt.clim(0, 150)
        elif mode == 'median':
            lon_lat = matched[matched.bt_name == bt.name].groupby('date')['lon', 'lat'].median()
            lon_lat_std = matched[matched.bt_name == bt.name].groupby('date')['lon', 'lat'].std()
            plt.errorbar(lon_lat.lon, lon_lat.lat, xerr=lon_lat_std.lon, yerr=lon_lat_std.lat, marker='x', zorder=0)
            plt.clim(0, 150)
        plt.title('Best track {}: {} - {}'.format(bt.name, bt.dates[0], bt.dates[-1]))

        plt.pause(.1)
        raw_input()
        plt.clf()



def datetime_encode_handler(obj):
    if hasattr(obj, 'isoformat'):
        return {'__isotime__': obj.isoformat()}
    else:
        raise TypeError('Object of type %s with value of {0} is not JSON serializable'.format(
            type(obj), repr(obj)))


def datetime_decode_hook(dct):
    if '__isotime__' in dct:
        return dt.datetime.strptime(dct['__isotime__'], '%Y-%m-%dT%H:%M:%S')
    return dct


class Plotter(object):
    def __init__(self, title, best_tracks, c20data, all_matches):
        self.title = title

        self.best_tracks = best_tracks

        self.c20data = c20data
        self.all_matches = all_matches
        self.date = self.c20data.first_date()

        self.layout = copy.copy(DEFAULT_PLOTTER_LAYOUT)
        self.layout['date'] = self.date
        self.layout['plot_settings'].append(DEFAULT_PLOT_SETTINGS)

        self.date = None
        self.field_names = ['prmsl', 'vort', 'vort4', 'windspeed']
        self.track_names = ['best', 'vort_max', 'pressure_min']
        self.point_names = ['vort_max', 'pressure_min']

    def plot_match_from_best_track(self, best_track):
        for matches in self.all_matches:
            for match in matches:
                if best_track.name == match.best_track.name:
                    for settings in self.layout['plot_settings']:
                        settings['match_index'] = matches.index(match)
                    self.plot()
                    return
        print('Match not found for track')

    def plot(self):
        ensemble_member = self.layout['ensemble_member']
        date = self.layout['date']

        plt.figure(self.layout['figure'])
        plt.clf()
        for plot_settings in self.layout['plot_settings']:
            plt.subplot(plot_settings['subplot'])

            title = [self.title, str(date), str(ensemble_member)]

            if plot_settings['field']:
                title.append(plot_settings['field'])
                field = getattr(self.c20data, plot_settings['field'])
                raster_on_earth(self.c20data.lons, self.c20data.lats, field,
                                plot_settings['vmin'], plot_settings['vmax'], plot_settings['loc'])
            else:
                raster_on_earth(self.c20data.lons, self.c20data.lats,
                                None, None, None, plot_settings['loc'])

            if plot_settings['points']:
                title.append(plot_settings['points'])
                points = getattr(self.c20data, plot_settings['points'])
                for p_val, p_loc in points:
                    plot_point_on_earth(p_loc[0], p_loc[1], 'kx')

            if plot_settings['best_track_index'] != -1:
                plot_ibtrack_with_date(self.best_tracks[plot_settings['best_track_index']], date)

            if plot_settings['vort_max_track_index'] != -1:
                pass

            if plot_settings['pressure_min_track_index'] != -1:
                pass

            if plot_settings['match_index'] != -1:
                plot_match_with_date(
                    self.all_matches[ensemble_member][plot_settings['match_index']], date)

            plt.title(' '.join(title))

    def save(self, name):
        self.layout['name'] = name
        f = open(os.path.join(settings.SETTINGS_DIR, 'plots',
                              SAVE_FILE_TPL.format(self.layout['name'])), 'w')
        simplejson.dump(self.layout, f, default=datetime_encode_handler, indent=4)

    def load(self, name, is_plot=True):
        try:
            f = open(os.path.join(settings.SETTINGS_DIR, 'plots', SAVE_FILE_TPL.format(name)), 'r')
            layout = simplejson.load(f, object_hook=datetime_decode_hook)
            print(layout)
            if layout['version'] != self.layout['version']:
                r = raw_input('version mismatch, may not work! Press c to continue anyway: ')
                if r != 'c':
                    raise Exception('Version mismatch (user cancelled)')

            self.layout['name'] = name
            self.layout = layout
            self.set_date(self.layout['date'], is_plot)
        except Exception, e:
            print('Settings {0} could not be loaded'.format(name))
            print('{0}'.format(e.message))
            raise

    def print_list(self):
        for plot_settings_name in self.list():
            print(plot_settings_name)

    def delete(self, name):
        file_name = os.path.join(settings.SETTINGS_DIR, 'plots', SAVE_FILE_TPL.format(name))
        os.remove(file_name)

    def list(self):
        plot_settings = []
        for fn in glob(os.path.join(settings.SETTINGS_DIR, 'plots', SAVE_FILE_TPL.format('*'))):
            plot_settings.append('_'.join(os.path.basename(fn).split('.')[0].split('_')[1:]))
        return plot_settings

    def set_date(self, date, is_plot=True):
        self.layout['date'] = self.c20data.set_date(
            date, self.layout['ensemble_member'], self.layout['ensemble_mode'])
        if is_plot:
            self.plot()

    def next_date(self):
        self.layout['date'] = self.c20data.next_date(
            self.layout['ensemble_member'], self.layout['ensemble_mode'])
        self.plot()

    def prev_date(self):
        self.layout['date'] = self.c20data.prev_date(
            self.layout['ensemble_member'], self.layout['ensemble_mode'])
        self.plot()

    def set_ensemble_member(self, ensemble_member):
        self.layout['ensemble_member'] = ensemble_member
        self.c20data.set_date(
            self.layout['date'], self.layout['ensemble_member'], self.layout['ensemble_mode'])
        self.plot()

    def next_ensemble_member(self):
        self.set_ensemble_member(self.layout['ensemble_member'] + 1)

    def prev_ensemble_member(self):
        self.set_ensemble_member(self.layout['ensemble_member'] - 1)

    def set_match(self, match_index):
        self.match_index = match_index
        for plot_settings in self.layout['plot_settings']:
            plot_settings.match_index = match_index
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


def plot_3d_scatter(cyclone_matches, unmatched_cyclones):
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ps = {'hu': {'xs': [], 'ys': [], 'zs': []},
          'ts': {'xs': [], 'ys': [], 'zs': []},
          'no': {'xs': [], 'ys': [], 'zs': []}}

    for match in cyclone_matches:
        best_track = match.best_track
        cyclone = match.cyclone
        plotted_dates = []

        for date, cls in zip(best_track.dates, best_track.cls):
            if date in cyclone.dates and cyclone.pmins[date]:
                plotted_dates.append(date)
                if cls == 'HU':
                    ps['hu']['xs'].append(cyclone.pmins[date])
                    ps['hu']['ys'].append(cyclone.p_ambient_diffs[date])
                    ps['hu']['zs'].append(cyclone.vortmax_track.vortmax_by_date[date].vort)
                else:
                    ps['ts']['xs'].append(cyclone.pmins[date])
                    ps['ts']['ys'].append(cyclone.p_ambient_diffs[date])
                    ps['ts']['zs'].append(cyclone.vortmax_track.vortmax_by_date[date].vort)

        for date in cyclone.dates:
            if date not in plotted_dates and cyclone.pmins[date]:
                ps['no']['xs'].append(cyclone.pmins[date])
                ps['no']['ys'].append(cyclone.p_ambient_diffs[date])
                ps['no']['zs'].append(cyclone.vortmax_track.vortmax_by_date[date].vort)

    ax.scatter(c='r', marker='o', **ps['hu'])
    ax.scatter(c='b', marker='o', **ps['ts'])
    ax.scatter(c='b', marker='x', **ps['no'])


SCATTER_ATTRS = {
    # Normally x.
    'vort': {'range': (0, 0.0012)},
    # 1st row.
    'pmin': {'range': (96000, 104000)},
    'pambdiff': {'range': (-1000, 5000)},
    'mindist': {'range': (0, 1000)},
    'max_windspeed': {'range': (0, 100)},
    # 2nd row.
    't995': {'range': (260, 320)},
    't850': {'range': (250, 310)},
    't_anom': {'range': (-15, 10)},
    'max_windspeed_dist': {'range': (0, 1000)},
    'max_windspeed_dir': {'range': (-np.pi, np.pi)},
    'lon': {'range': (200, 360)},
    'lat': {'range': (0, 70)},
}


def plot_2d_scatter(ps, var1, var2, unmatched_lim=None):
    attr1 = SCATTER_ATTRS[var1]
    attr2 = SCATTER_ATTRS[var2]

    plt.xlim(attr1['range'])
    plt.ylim(attr2['range'])

    plt.xlabel(var1)
    plt.ylabel(var2)

    if not unmatched_lim:
        unmatched_lim = len(ps['unmatched']['xs'])

    plt.plot(ps['unmatched']['xs'][:unmatched_lim],
             ps['unmatched']['ys'][:unmatched_lim], 'g+', zorder=0)

    plt.plot(ps['no']['xs'], ps['no']['ys'], 'bx', zorder=1)
    plt.plot(ps['ts']['xs'], ps['ts']['ys'], 'bo', zorder=2)
    plt.plot(ps['hu']['xs'], ps['hu']['ys'], 'ro', zorder=3)


def plot_2d_error_scatter(ps, var1, var2, unmatched_lim=None):
    attr1 = SCATTER_ATTRS[var1]
    attr2 = SCATTER_ATTRS[var2]

    plt.xlim(attr1['range'])
    plt.ylim(attr2['range'])

    plt.xlabel(var1)
    plt.ylabel(var2)

    if not unmatched_lim:
        unmatched_lim = len(ps['un']['xs'])

    plt.plot(ps['un']['xs'][:unmatched_lim], ps['un']['ys'][:unmatched_lim], 'kx', zorder=0)
    plt.plot(ps['fp']['xs'], ps['fp']['ys'], 'bo', zorder=1)
    plt.plot(ps['fn']['xs'], ps['fn']['ys'], 'r^', zorder=2)
    plt.plot(ps['tp']['xs'], ps['tp']['ys'], 'ro', zorder=3)
    plt.plot(ps['tn']['xs'], ps['tn']['ys'], 'b^', zorder=4)


def plot_ensemble_matches(c20data, matches):
    plt.clf()
    raster_on_earth(c20data.lons, c20data.lats, None, None, None, 'wa')
    for match in matches:
        plot_track(match.av_vort_track)
        if match.store_all_tracks:
            for vort_track in match.vort_tracks:
                plot_track(vort_track, 'b+')

        # if raw_input() == 'q':
            # return


def plot_vortmax_track_with_date(vortmax_track, date=None):
    plot_track(vortmax_track, plt_fmt='b--')
    if date:
        try:
            index = np.where(vortmax_track.dates == date)[0][0]
            plot_point_on_earth(vortmax_track.lons[index], vortmax_track.lats[index], 'ko')
        except:
            pass
            # print("Couldn't plot track at date {0} (start {1}, end {2})".format(
            #     date, vortmax_track.dates[0], vortmax_track.dates[-1]))


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
            pass
            # print("Couldn't plot best_track at date {0} (start {1}, end {2})".format(
            #     date, best_track.dates[0], best_track.dates[-1]))


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
        raster_on_earth(c20data.lons, c20data.lats, c20data.prmsl, vmin=99500, vmax=103000, loc='wa')

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


def cvorticity(u, v, dx, dy):
    '''Calculates the (2nd order) vorticity by calling into a c function'''
    vort = np.zeros_like(u)
    cvort(u, v, u.shape[0], u.shape[1], dx, dy, vort)
    return vort


def plot_grib_vorticity_at_level(c20gribdata, level_key):
    level = c20gribdata.levels[level_key]
    raster_on_earth(level.lons, level.lats, level.vort, loc='wa')


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

        if track_index >= len(best_track.lons) or vortmax_track_index >= len(vortmax_track.lons):
            break


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
        raster_on_earth(c20data.lons, c20data.lats, c20data.prmsl, vmin=99500, vmax=103000)

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


def plot_track(track, plt_fmt=None, zorder=1):
    if plt_fmt:
        plot_path_on_earth(track.lons, track.lats, plt_fmt, zorder=zorder)
    else:
        plot_path_on_earth(track.lons, track.lats, 'r-', zorder=zorder)


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
        # raster_on_earth(c20data.lons, c20data.lats, c20data.prmsl, 97000, 106000, 'wa')
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
    new_lons = lons.copy()
    m = new_lons > 180
    new_lons[m] = new_lons[m] - 360
    return new_lons


def lon_convert(lon):
    return lon if lon <= 180 else lon - 360


def plot_path_on_earth(lons, lats, plot_fmt=None, zorder=1):
    if plot_fmt:
        plt.plot(lons_convert(lons), lats, plot_fmt, zorder=zorder)
    else:
        plt.plot(lons_convert(lons), lats, zorder=zorder)


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

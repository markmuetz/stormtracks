import os
import datetime as dt
from collections import OrderedDict
import shutil

import pylab as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

from c20data import C20Data
from plotting import lon_convert
from results import StormtracksResultsManager
from analysis import StormtracksAnalysis, ClassificationAnalysis
from ibtracsdata import IbtracsData
import plotting
from load_settings import settings


def save_figure(name):
    if not os.path.exists(settings.FIGURE_OUTPUT_DIR):
        os.makedirs(settings.FIGURE_OUTPUT_DIR)
    plt.savefig(os.path.join(settings.FIGURE_OUTPUT_DIR, name))


def plot_classifer_metrics(cla_analysis=None):
    if not cla_analysis:
        cla_analysis = ClassificationAnalysis()

    cal_cd, val_cd, clas = cla_analysis.get_trained_classifiers()

    plt.figure()
    for (cla, settings, fmt, name) in clas:
        cla.predict(cal_cd)
        plt.subplot(2, 2, 1)
        plt.plot(cla.ppv, cla.sensitivity, fmt, label=name)

        plt.subplot(2, 2, 2)
        plt.plot(cla.fpr, cla.sensitivity, fmt, label=name)

        cla.predict(val_cd)
        plt.subplot(2, 2, 3)
        plt.plot(cla.ppv, cla.sensitivity, fmt, label=name)

        plt.subplot(2, 2, 4)
        plt.plot(cla.fpr, cla.sensitivity, fmt, label=name)

    plt.subplot(2, 2, 1)
    # plt.xlabel('PPV')
    plt.ylabel('Calibration\nsensitivity')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.subplot(2, 2, 2)
    # plt.xlabel('FPR')
    # plt.ylabel('sensitivity')
    plt.xlim((0, 0.1))
    plt.ylim((0, 1))
    plt.legend(bbox_to_anchor=(1.1, 1.1), numpoints=1)

    plt.subplot(2, 2, 3)
    plt.xlabel('PPV')
    plt.ylabel('Validation\nsensitivity')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.subplot(2, 2, 4)
    plt.xlabel('FPR')
    # plt.ylabel('sensitivity')
    plt.xlim((0, 0.1))
    plt.ylim((0, 1))

    save_figure('ppv_and_fpr_vs_sens.png')
    return cla_analysis


def plot_wld(stormtracks_analysis=None, years=None):
    if not stormtracks_analysis:
        stormtracks_analysis = StormtracksAnalysis(2000)

    if not years:
        years = range(2000, 2010)

    wlds = {}
    wlds['draws'] = []

    for year in years:
        print(year)
        stormtracks_analysis.set_year(year)
        k0, w0, k1, w1, d = stormtracks_analysis.run_wld_analysis(active_configs={'scale': 3})

        if k0 not in wlds:
            wlds[k0] = []
        wlds[k0].append(w0)

        if k1 not in wlds:
            wlds[k1] = []
        wlds[k1].append(w1)

        wlds['draws'].append(d)

    print(wlds)

    plt.figure()
    plt.title('wld')
    for k in wlds:
        plt.plot(years, wlds[k], label=k)
    plt.legend(loc='best')


def plot_tracking_stats(stormtracks_analysis=None, years=None, sort_col='cumoveroverlap'):
    if not stormtracks_analysis:
        stormtracks_analysis = StormtracksAnalysis(2000)

    k995 = []
    for scale in (1, 2, 3):
        config = {'pressure_level': 995, 'scale': scale, 'tracker': 'nearest_neighbour'}
        k995.append(stormtracks_analysis.good_matches_key(config))

    keys = ('pl995', 'pl850', 'scale3')
    
    config_keys = {}
    for key in keys:
        config_keys[key] = []

    for scale in (1, 2, 3):
        config = {'pressure_level': 995, 'scale': scale, 'tracker': 'nearest_neighbour'}
        config_keys['pl995'].append(stormtracks_analysis.good_matches_key(config))

        config = {'pressure_level': 850, 'scale': scale, 'tracker': 'nearest_neighbour'}
        config_keys['pl850'].append(stormtracks_analysis.good_matches_key(config))

    for pl in (995, 850):
        config = {'pressure_level': pl, 'scale': 3, 'tracker': 'nearest_neighbour'}
        config_keys['scale3'].append(stormtracks_analysis.good_matches_key(config))

    all_wins = {}
    for key in keys:
        all_wins[key] = []

    if not years:
        years = range(2000, 2010)

    for year in years:
        print(year)
        stormtracks_analysis.set_year(year)

        res = {}
        res['pl995'] = stormtracks_analysis.run_position_analysis(sort_on=sort_col, \
            active_configs={'pressure_level': 995})
        res['pl850'] = stormtracks_analysis.run_position_analysis(sort_on=sort_col, \
            active_configs={'pressure_level': 850})
        res['scale3'] = stormtracks_analysis.run_position_analysis(sort_on=sort_col,
                                                                   active_configs={'scale': 3})
        
        for key in keys:
            print(key)
            wins = []
            for config_key in config_keys[key]:
                print('  {0}'.format(config_key))
                try:
                    wins.append(res[key][config_key][0])
                except KeyError:
                    wins.append(0)

            all_wins[key].append(wins)

    for key in keys:
        all_wins[key] = np.array(all_wins[key])

    plt.figure(1)
    plt.clf()
    plt.title('995 hPa Pressure Level')
    plt.plot(years, all_wins['pl995'][:, 0], 'r-') # scale 1
    plt.plot(years, all_wins['pl995'][:, 1], 'g-') # scale 2
    plt.plot(years, all_wins['pl995'][:, 2], 'b-') # scale 3

    plt.figure(2)
    plt.clf()
    plt.title('850 hPa Pressure Level')
    plt.plot(years, all_wins['pl850'][:, 0], 'r--') # scale 1
    plt.plot(years, all_wins['pl850'][:, 1], 'g--') # scale 2
    plt.plot(years, all_wins['pl850'][:, 2], 'b--') # scale 3

    plt.figure(3)
    plt.clf()
    plt.title('Scale 3')
    plt.plot(years, all_wins['scale3'][:, 0], 'b-') # pl 995
    plt.plot(years, all_wins['scale3'][:, 1], 'b--') # pl 850

    return stormtracks_analysis


def plot_data_processing_figures():
    plot_katrina()
    plot_katrina_maxs_mins()


def plot_katrina():
    c20data = C20Data(2005, fields=['u', 'v'])
    c20data.set_date(dt.datetime(2005, 8, 27, 18))

    fig = plt.figure(1)
    plt.clf()

    loc = {'llcrnrlat': 15, 'urcrnrlat': 35, 'llcrnrlon': -100, 'urcrnrlon': -70}
    plt.subplot(311)
    
    vec_plot_on_earth(c20data.lons, c20data.lats, -c20data.u, c20data.v, loc=loc)
    plt.subplot(312)
    raster_on_earth(c20data.lons, c20data.lats, c20data.vort, loc=loc)

    c20data = C20Data(2005, fields=['u', 'v'], upscaling=True, scale_factor=3)
    c20data.set_date(dt.datetime(2005, 8, 27, 18))
    plt.subplot(313)
    raster_on_earth(c20data.up_lons, c20data.up_lats, c20data.up_vort, loc=loc)

    save_figure('katrina_data_proc.png')


def plot_katrina_maxs_mins():
    c20data = C20Data(2005, fields=['psl', 'u', 'v'])
    c20data.set_date(dt.datetime(2005, 8, 27, 18))
    loc = {'llcrnrlat': 0, 'urcrnrlat': 45, 'llcrnrlon': -120, 'urcrnrlon': -60}

    fig = plt.figure(2)
    plt.clf()

    plt.subplot(221)
    raster_on_earth(c20data.lons, c20data.lats, c20data.vort, loc=loc, colorbar=False)

    plt.subplot(222)
    raster_on_earth(c20data.lons, c20data.lats, None, loc=loc)
    points = c20data.vmaxs
    for p_val, p_loc in points:
        plot_point_on_earth(p_loc[0] + 1, p_loc[1] + 1, 'ro')
    points = c20data.vmins
    for p_val, p_loc in points:
        plot_point_on_earth(p_loc[0] + 1, p_loc[1] + 1, 'kx')


    plt.subplot(223)
    raster_on_earth(c20data.lons, c20data.lats, c20data.psl, vmin=99000, vmax=103000, loc=loc, colorbar=False)

    plt.subplot(224)
    raster_on_earth(c20data.lons, c20data.lats, None, loc=loc)
    points = c20data.pmaxs
    for p_val, p_loc in points:
        plot_point_on_earth(p_loc[0] + 1, p_loc[1] + 1, 'ro')
    points = c20data.pmins
    for p_val, p_loc in points:
        plot_point_on_earth(p_loc[0] + 1, p_loc[1] + 1, 'kx')

    save_figure('katrina_max_mins.png')


def plot_matching_figures():
    plot_six_configs_figure()
    plot_individual_katrina_figure()


def plot_individual_katrina_figure(em=7):
    c20data = C20Data(2005, fields=['psl', 'u', 'v'])
    srm = StormtracksResultsManager('pyro_tracking_analysis')
    ta = StormtracksAnalysis(2005)
    ibdata = IbtracsData()
    w, k = ibdata.load_wilma_katrina()
    bt = k
    loc = {'llcrnrlat': 10, 'urcrnrlat': 45, 'llcrnrlon': -100, 'urcrnrlon': -65}

    config = {'pressure_level': 850, 'scale': 1, 'tracker': 'nearest_neighbour'}

    fig = plt.figure()
    plt.clf()

    print(config)
    all_matches = []

    key = ta.good_matches_key(config)
    good_matches = srm.get_result(2005, em, key)

    for good_match in good_matches:
        if good_match.best_track.name == bt.name:
            break

    raster_on_earth(c20data.lons, c20data.lats, None, loc=loc)
    plotting.plot_match_with_date(good_match, None)
    save_figure('katrina_individual_match_em7')


def plot_six_configs_figure():
    c20data = C20Data(2005, fields=['psl', 'u', 'v'])
    srm = StormtracksResultsManager('pyro_tracking_analysis')
    ta = StormtracksAnalysis(2005)
    ibdata = IbtracsData()
    w, k = ibdata.load_wilma_katrina()
    bt = k
    loc = {'llcrnrlat': 10, 'urcrnrlat': 45, 'llcrnrlon': -100, 'urcrnrlon': -65}
    fig3 = plt.figure(3)
    plt.clf()

    for j, config in enumerate(ta.analysis_config_options):
        plt.subplot(3, 2, j + 1)
        print(config)
        all_matches = []
        raster_on_earth(c20data.lons, c20data.lats, None, loc=loc)
        plotting.plot_track(bt)

        for i in range(56):
            key = ta.good_matches_key(config)
            good_matches = srm.get_result(2005, i, key)
            matches = []
            for good_match in good_matches:
                if good_match.best_track.name == bt.name:
                    matches.append(good_match)

            if matches:
                all_matches.append(matches)
                for match in matches:
                    vt = match.vort_track
                    mask = (vt.dates >= bt.dates[0]) & (vt.dates <= bt.dates[-1])
                    plotting.plot_path_on_earth(vt.lons[mask], vt.lats[mask], 'b--')

                    # plotting.plot_track(vt, 'b--')
                    # return vt, bt
            else:
                print('Could not find wilma in {0}-{1}'.format(i, key))

        fig3.set_size_inches(7, 10.5)

        save_figure('katrina_six_tracking_configs')


def plot_point_on_earth(lon, lat, plot_fmt=None):
    if plot_fmt:
        plt.plot(lon_convert(lon), lat, plot_fmt)
    else:
        plt.plot(lon_convert(lon), lat)


def raster_on_earth(lons, lats, data, vmin=None, vmax=None, loc=None, colorbar=True):
    if not loc:
        m = Basemap(projection='cyl', resolution='c', 
                    llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
    else:
        m = Basemap(projection='cyl', resolution='c', **loc)

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

    if colorbar and data is not None:
        m.colorbar(location='right', pad='7%')


def vec_plot_on_earth(lons, lats, x_data, y_data, vmin=-4, vmax=12, loc=None):
    plot_lons, plot_x_data = extend_data(lons, lats, x_data)
    plot_lons, plot_y_data = extend_data(lons, lats, y_data)

    lons, lats = np.meshgrid(plot_lons, lats)

    if not loc:
        m = Basemap(projection='cyl', resolution='c', 
                    llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
    else:
        m = Basemap(projection='cyl', resolution='c', **loc)
    x, y = m(lons, lats)

    mag = np.sqrt(plot_x_data**2 + plot_y_data**2)
    vmin, vmax = mag.min(), mag.max()
    m.contourf(x, y, mag)
    #m.pcolormesh(x, y, mag, vmin=vmin, vmax=vmax)
    #m.quiver(x, y, plot_x_data, plot_y_data)
    skip = 1
    m.quiver(x[::skip, ::skip], y[::skip, ::skip], plot_x_data[::skip, ::skip], plot_y_data[::skip, ::skip], scale=500)

    m.drawcoastlines()
    m.drawparallels(np.arange(-90.,90.,45.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180.,180.,60.), labels=[0, 0, 0, 1], fontsize=10)


    m.colorbar(location='right', pad='7%')
    plt.show()


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

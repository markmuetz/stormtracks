import datetime as dt

import pylab as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

from c20data import C20Data
from plotting import lon_convert
from results import StormtracksResultsManager
from analysis import StormtracksAnalysis
from ibtracsdata import IbtracsData
import plotting


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


def plot_katrina_maxs_mins():
    c20data = C20Data(2005, fields=['psl', 'u', 'v'])
    c20data.set_date(dt.datetime(2005, 8, 27, 18))
    loc = {'llcrnrlat': 0, 'urcrnrlat': 45, 'llcrnrlon': -120, 'urcrnrlon': -60}

    fig = plt.figure(1)
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


def plot_matching_figures():
    c20data = C20Data(2005, fields=['psl', 'u', 'v'])
    srm = StormtracksResultsManager('pyro_tracking_analysis')
    ta = StormtracksAnalysis(2005)
    ibdata = IbtracsData()
    w, k = ibdata.load_wilma_katrina()
    bt = k
    loc = {'llcrnrlat': 10, 'urcrnrlat': 45, 'llcrnrlon': -100, 'urcrnrlon': -65}
    plt.figure(3)
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

        plt.pause(0.1)
        print(len(all_matches))

        if config['scale'] == 3 and config['pressure_level'] == 850:
            plt.figure(4)
            plt.clf()
            # for i, matches in enumerate(all_matches):
            for i, ensemble_member in enumerate([4, 10, 19, 42]):
                matches = all_matches[i]
                plt.subplot(2, 2, i + 1)

                print(ensemble_member)
                raster_on_earth(c20data.lons, c20data.lats, None, loc=loc)
                plotting.plot_track(bt)

                for match in matches:
                    vt = match.vort_track
                    mask = (vt.dates >= bt.dates[0]) & (vt.dates <= bt.dates[-1])
                    plotting.plot_path_on_earth(vt.lons[mask], vt.lats[mask], 'b--')


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
        m.colorbar(location='bottom', pad='7%')


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


    m.colorbar(location='bottom', pad='7%')
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

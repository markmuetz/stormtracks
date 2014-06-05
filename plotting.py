import pylab as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

def plot_ibtrack_with_date(d, i, track):
    plot_ibtrack(track)
    if track.cls[i] == 'HU':
	plt.plot(track.lon[i], track.lat[i], 'ro')
    else:
	plt.plot(track.lon[i], track.lat[i], 'ko')

def time_plot_ibtracks(ncd, track):
    for i, d in enumerate(track.dates):
	plt.clf()
	plt.title(d)

	plt.subplot(2, 1, 1)
	plot_ibtrack_with_date(d, i, track)
	plot_on_earth(ncd.lon, ncd.lat, ncd.get_pressure_from_date(d), vmin=99500, vmax=103000)

	plt.subplot(2, 1, 2)
	plot_ibtrack_with_date(d, i, track)
	plot_on_earth(ncd.lon, ncd.lat, ncd.get_vort_from_date(d), vmin=-5, vmax=15)

	plt.pause(0.1)

def plot_matches(c_set_matches):
    for t, c_sets in c_set_matches.items():
	plt.clf()
	plot_ibtrack(t, offset=360)
	for c_set in c_sets:
	    plot_cyclone_track(c_set)
	raw_input()


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
        plt.plot(isobar.glob_path[:, 0], isobar.glob_path[:, 1])

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

def plot_ibtracks(ss):
    plt.xlim((-180, 180))
    plt.ylim((-90, 90))
    for s in ss:
	plot_ibtrack(s)

def plot_ibtrack(s, offset=0):
    plt.plot(s.lon + offset, s.lat, 'b-')

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

def plot_on_earth(lons, lats, data, vmin=-4, vmax=12, cbar_loc='left', cbar_ticks=None):
    #import ipdb; ipdb.set_trace()
    #if ax == None:
	#ax = plt.gca()
    #plot_lons, plot_data = extend_data(lons, lats, data)
    plot_lons, plot_data = extend_data(lons, lats, data)

    lons, lats = np.meshgrid(plot_lons, lats)

    m = Basemap(projection='cyl', resolution='c', llcrnrlat=0, urcrnrlat=60, llcrnrlon=-120, urcrnrlon=-30)
    #m = Basemap(projection='cyl', resolution='c', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
    x, y = m(lons, lats)

    m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax)
    #m.pcolormesh(x, y, plot_data)

    m.drawcoastlines()
    if cbar_loc == 'left':
	p_labels = [0, 1, 0, 0]
    else:
	p_labels = [1, 0, 0, 0]

    m.drawparallels(np.arange(-90.,90.1,45.), labels=p_labels, fontsize=10)
    m.drawmeridians(np.arange(-180.,180.,60.), labels=[0, 0, 0, 1], fontsize=10)

    #import ipdb; ipdb.set_trace()
    #if cbar_ticks == None:
	#cbar = m.colorbar(location=cbar_loc, pad='7%')
    #else:
	#cbar = m.colorbar(location=cbar_loc, pad='7%', ticks=cbar_ticks)

    #if cbar_loc == 'left':
	#cbar.ax.xaxis.get_offset_text().set_position((10,0))
    #plt.show()

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




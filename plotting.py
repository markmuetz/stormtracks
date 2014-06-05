import pylab as plt
import numpy as np

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
    plt.xlim((0, 360))
    plt.ylim((-90, 90))
    for s in ss:
	plot_ibtrack(s)

def plot_ibtrack(s):
    plt.plot(s.lon + 360, s.lat, 'b-')

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



import datetime as dt
from itertools import tee, izip
from collections import OrderedDict

import pylab as plt
import numpy as np
from scipy.interpolate import RectSphereBivariateSpline

from cyclone import CycloneSet, Cyclone, Isobar
from c20data import C20Data
from utils.fill_raster import fill_raster, path_to_raster

class VortMaxTrack(object):
    def __init__(self, start_vortmax):
        if len(start_vortmax.prev_vortmax):
            raise Exception('start vortmax must have no previous vortmaxes')
        self.start_vortmax = start_vortmax

        self.vortmaxes = []
        self.vortmax_by_date = OrderedDict()

        self._build_track()

    def _build_track(self):
        self.vortmaxes.append(self.start_vortmax)
        if len(self.start_vortmax.next_vortmax):
            vortmax = self.start_vortmax.next_vortmax[0]
            self.vortmax_by_date[vortmax.date] = vortmax

            while len(vortmax.next_vortmax) != 0:
                self.vortmaxes.append(vortmax)
                vortmax = vortmax.next_vortmax[0]
                self.vortmax_by_date[vortmax.date] = vortmax


class VortMax(object):
    def __init__(self, date, pos, vort):
        self.date = date
        self.pos  = pos
        self.vort  = vort
        self.next_vortmax = []
        self.prev_vortmax = []
        self.secondary_vortmax = []

    def add_next(self, vortmax):
        self.next_vortmax.append(vortmax)
        vortmax.prev_vortmax.append(self)


class CycloneTracker(object):
    def __init__(self, glob_cyclones, max_dist=10, min_cyclone_set_duration=1.49):
        self.glob_cyclones = glob_cyclones
        self.max_dist = max_dist
        self.min_cyclone_set_duration = dt.timedelta(min_cyclone_set_duration)

    def track(self):
        cyclone_sets = []

        prev_date = None
        for date in self.glob_cyclones.dates:
            if not date in self.glob_cyclones.cyclones_by_date.keys():
                continue

            if prev_date:
                prev_cyclones = self.glob_cyclones.cyclones_by_date[prev_date]
                cyclones = self.glob_cyclones.cyclones_by_date[date]
                for prev_cyclone in prev_cyclones:
                    if not prev_cyclone.cyclone_set:
                        cyclone_set = CycloneSet()
                        cyclone_set.add_cyclone(prev_cyclone)
                        cyclone_sets.append(cyclone_set)

                    for cyclone in cyclones:
                        cp = cyclone.cell_pos
                        pcp = prev_cyclone.cell_pos

                        if dist((cp[0], cp[1]), (pcp[0], pcp[1])) < self.max_dist:
                            prev_cyclone.cyclone_set.add_cyclone(cyclone)
            prev_date = date

        return [cs for cs in cyclone_sets if cs.end_date - cs.start_date > self.min_cyclone_set_duration]


class GlobalCyclones(object):
    def __init__(self, c20data, ensemble_member=0):
        self.c20data = c20data
        self.dates = c20data.dates
        self.lon = c20data.lon
        self.lat = c20data.lat
        self.f_lon = c20data.f_lon
        self.f_lat = c20data.f_lat

        self.date = None
        self.cyclones_by_date = {}
        self.ensemble_member = ensemble_member
        self.ensemble_mode = 'member'

    def set_year(self, year):
        self.year = year
        self.c20data.set_year(year)
        self.dates = self.c20data.dates

    def _construct_vortmax_tracks_by_date(self):
        self.vort_tracks_by_date = OrderedDict()
        for vortmaxes in self.vortmax_time_series.values():
            for vortmax in vortmaxes:
                if len(vortmax.prev_vortmax) == 0:
                    vortmax_track = VortMaxTrack(vortmax)
                    for date in vortmax_track.vortmax_by_date.keys():
                        if not date in self.vort_tracks_by_date:
                            self.vort_tracks_by_date[date] = []
                        self.vort_tracks_by_date[date].append(vortmax_track)

    def track_vort_maxima(self, start_date, end_date, use_upscaled=False):
        if start_date < self.dates[0]:
            raise Exception('Start date is out of date range, try setting the year appropriately')
        elif end_date > self.dates[-1]:
            raise Exception('End date is out of date range, try setting the year appropriately')

        index = np.where(self.dates == start_date)[0][0]
        end_index = np.where(self.dates == end_date)[0][0]

        self.vortmax_time_series = OrderedDict()

        dist_cutoff = 10
        vort_cutoff = 5e-5
        while index <= end_index:
            date = self.dates[index]
            self.set_date(date, self.ensemble_member)

            vortmaxes = []

	    if use_upscaled:
		vmaxs = self.c20data.up_vmaxs
	    else:
		vmaxs = self.c20data.vmaxs

            for vmax in vmaxs:
		if (220 < vmax[1][0] < 340 and
		    0 < vmax[1][1] < 60):
		    if vmax[0] > vort_cutoff:
			vortmax = VortMax(date, vmax[1], vmax[0])
			vortmaxes.append(vortmax)

	    secondary_vortmaxes = []
	    for i in range(len(vortmaxes)):
		v1 = vortmaxes[i]
		for j in range(i + 1, len(vortmaxes)):
		    v2 = vortmaxes[j]
		    if dist(v1.pos, v2.pos) < dist_cutoff:
			if v1.vort > v2.vort:
			    v1.secondary_vortmax.append(v2)
			    secondary_vortmaxes.append(v2)
			elif v1.vort <= v2.vort:
			    v2.secondary_vortmax.append(v1)
			    secondary_vortmaxes.append(v1)

	    for v in secondary_vortmaxes:
		if v in vortmaxes:
		    vortmaxes.remove(v)

	    for i, v in enumerate(vortmaxes):
		v.index = i

            self.vortmax_time_series[date] = vortmaxes

            index += 1

        index = 0
        for vs1, vs2 in pairwise(self.vortmax_time_series.values()):
            #print('{0}: l1 {1}, l2 {2}'.format(index, len(vs1), len(vs2)))
            index += 1
            for i, v1 in enumerate(vs1):
                min_dist = 8
                v2next = None
                for j, v2 in enumerate(vs2):
                    d = dist(v1.pos, v2.pos)
                    if d < min_dist:
                        min_dist = d
                        v2next = v2

                if v2next:
                    v1.add_next(v2next)
                    if len(v1.next_vortmax) != 1:
                        raise Exception('There should only ever be one next_vormax')


        for vs in self.vortmax_time_series.values():
            for v in vs:
                if len(v.prev_vortmax) > 1:
                    min_dist = 8
                    vprev = None
                    for pv in v.prev_vortmax:
                        d = dist(pv.pos, v.pos)
                        if d < min_dist:
                            min_dist = d
                            vprev = pv

                    for pv in v.prev_vortmax:
                        if pv != vprev:
                            pv.next_vortmax.remove(v)

                    v.prev_vortmax = [vprev]
                #if dist(v1.pos, v2.pos) < dist_cutoff:
                    #v1.add_next(v2)
        self._construct_vortmax_tracks_by_date()


    def find_cyclones_in_date_range(self, start_date, end_date):
        if start_date < self.dates[0]:
            raise Exception('Start date is out of date range, try setting the year appropriately')
        elif end_date > self.dates[-1]:
            raise Exception('End date is out of date range, try setting the year appropriately')

        index = np.where(self.dates == start_date)[0][0]
        date = self.dates[index]

        while date <= end_date:
            print(date)
            self.set_date(date)
            self.cyclones_by_date[date] = []
            self.find_all_cyclones()
            self.find_candidate_cyclones()

            index += 1
            date = self.dates[index]

    def set_date(self, date, ensemble_member):
        if date != self.date:
            self.date = date
            self.c20data.set_date(date, ensemble_member)

    def find_all_cyclones(self):
        self.pressures = np.arange(94000, 103000, 100)
        cn = plt.contour(self.c20data.psl, levels=self.pressures)
        self.grid_coords_contours = get_contour_verts(cn)

        pressures = self.pressures
        grid_coords_contours = self.grid_coords_contours
        pmins = self.c20data.pmins

        all_cyclones = []
        for p, ploc in pmins:
            # Note swapped x, y
            all_cyclones.append(Cyclone(ploc[0], ploc[1], self.date))

        # Create all isobars and add them to any cyclones centres they encompass,
        # but only if it's the only cyclone centre.
        for pressure, contour_set in zip(pressures, grid_coords_contours):
            for grid_coord_contour in contour_set:
                contour = np.zeros_like(grid_coord_contour) 
                contour[:, 0] = self.f_lon(grid_coord_contour[:, 0])
                contour[:, 1] = self.f_lat(grid_coord_contour[:, 1])

                isobar = Isobar(pressure, contour)
                contained_cyclones = []
                for cyclone in all_cyclones:
                    if isobar.contains(cyclone.cell_pos):
                        contained_cyclones.append(cyclone)

                # Only one cyclone contained, simple.
                if len(contained_cyclones) == 1:
                    contained_cyclones[0].isobars.append(isobar)

                # More than one cyclone contained, see if centres are adjacent.
                elif len(contained_cyclones) > 1:
                    is_found = True

                    for i in range(len(contained_cyclones)):
                        for j in range(i + 1, len(contained_cyclones)):
                            cp1 = contained_cyclones[i].cell_pos
                            cp2 = contained_cyclones[j].cell_pos

                            #p1 = self.psl[cp1[0], cp1[1]]
                            #p2 = self.psl[cp2[0], cp2[1]]
                            if abs(cp1[0] - cp2[0]) > 2 or abs(cp1[1] - cp2[1]) > 2:
                                is_found = False

                            #if p1 != p2:
                                #is_found = False
                                #break

                    if is_found:
                        contained_cyclones[0].isobars.append(isobar)
        self.all_cyclones = all_cyclones

    def find_candidate_cyclones(self):
        candidate_cyclones = []

        for cyclone in self.all_cyclones:
            if len(cyclone.isobars) == 0:
                continue
            elif cyclone.isobars[-1].pressure - cyclone.isobars[0].pressure < 300:
                continue
            #else:
                #area = 0
                #bounds_path = cyclone.isobars[-1].co
                #for i in range(len(bounds_path) - 1):
                    #area += bounds_path[i, 0] * bounds_path[(i + 1), 1]
                #area += bounds_path[-1, 0] * bounds_path[0, 1]
                #area /= 2

                #if run_count != 0:
                    #for prev_cyclone in timestep_candidate_cyclones[run_count - 1]:
                        #if dist((cyclone.cell_pos[0], cyclone.cell_pos[1]), (prev_cyclone.cell_pos[0], prev_cyclone.cell_pos[1])) < 10:
                            #prev_cyclone.cyclone_set.add_cyclone(cyclone)

            candidate_cyclones.append(cyclone)
            self.cyclones_by_date[self.date].append(cyclone)

        self.candidate_cyclones = candidate_cyclones

    def mask_clim_fields(self):
        for cyclone in self.candidate_cyclones:
            roci = cyclone.isobars[-1]
            bounded_vort = self.vort[int(roci.ymin):int(roci.ymax) + 1,
                                     int(roci.xmin):int(roci.xmax) + 1]
            bounded_psl = self.psl[int(roci.ymin):int(roci.ymax) + 1,
                                   int(roci.xmin):int(roci.xmax) + 1]
            bounded_u = self.u[int(roci.ymin):int(roci.ymax) + 1,
                               int(roci.xmin):int(roci.xmax) + 1]
            bounded_v = self.v[int(roci.ymin):int(roci.ymax) + 1,
                               int(roci.xmin):int(roci.xmax) + 1]

            raster_path = path_to_raster(roci.contour)
            cyclone_mask = fill_raster(raster_path)[0]
            
            cyclone.vort = np.ma.array(bounded_vort, mask=cyclone_mask == 0)
            cyclone.psl = np.ma.array(bounded_psl, mask=cyclone_mask == 0)
            cyclone.u = np.ma.array(bounded_u, mask=cyclone_mask == 0)
            cyclone.v = np.ma.array(bounded_v, mask=cyclone_mask == 0)


def main(args):
    c20data = C20Data(2005)

    start_date = c20data.dates[int(args.start)]
    end_date = c20data.dates[int(args.end)]

    glob_cyclones = GlobalCyclones(c20data)

    glob_cyclones.find_cyclones_in_date_range(start_date, end_date)

    tracker = CycloneTracker(glob_cyclones)
    cyclone_sets = tracker.track()

    return glob_cyclones, cyclone_sets


# TODO: Add to utils.
def upscale_field(lon, lat, field, x_scale=2, y_scale=2, is_degrees=True):
    if is_degrees:
	lon = lon * np.pi / 180.
	lat = (lat + 90) * np.pi / 180.

    d_lon = lon[1] - lon[0]
    d_lat = lat[1] - lat[0]

    new_lon = np.linspace(lon[0], lon[-1], len(lon) * x_scale)
    new_lat = np.linspace(lat[0], lat[-1], len(lat) * x_scale)

    mesh_new_lat, mesh_new_lon = np.meshgrid(new_lat, new_lon)

    if True:
	lut = RectSphereBivariateSpline(lat[1:-1], lon[1:-1], field[1:-1, 1:-1])

	interp_field = lut.ev(mesh_new_lat[1:-1, 1:-1].ravel(),
			      mesh_new_lon[1:-1, 1:-1].ravel()).reshape(mesh_new_lon.shape[0] - 2, mesh_new_lon.shape[1] - 2).T
    else:
	pass
    if is_degrees:
	new_lon = new_lon * 180. / np.pi
	new_lat = (new_lat * 180. / np.pi) - 90

    return new_lon[1:-1], new_lat[1:-1], interp_field


def find_cyclone(cyclone_sets, date, loc):
    for c_set in cyclone_sets:
        if c_set.start_date == date:
            c = c_set.cyclones[0]
            if c.cell_pos == loc:
                return c_set
    return None


def find_wilma(cyclone_sets):
    return find_cyclone(cyclone_sets, dt.datetime(2005, 10, 18, 12), (278, 16))


def find_katrina(cyclone_sets):
    return find_cyclone(cyclone_sets, dt.datetime(2005, 8, 22, 12), (248, 16))


def load_katrina():
    args = create_args()
    args.start = 932
    args.end = 950
    glob_cyclones, cyclone_sets = main(args)
    k = find_katrina(cyclone_sets)
    return k, glob_cyclones, cyclone_sets


def load_wilma():
    args = create_args()
    args.start = 1162
    args.end = 1200
    glob_cyclones, cyclone_sets = main(args)
    w = find_wilma(cyclone_sets)
    return w, glob_cyclones, cyclone_sets


# TODO: utils.
def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)

    return contours


# TODO: Should live in utils.
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

# TODO: utils.
def geo_dist(p1, p2):
    return np.arcos(np.sin(p1[1]) * np.sin(p2[1]) + np.cos(p1[1]) * np.cos(p2[1]) * (p1[0] - p2[0])) * EARTH_RADIUS

# TODO: utils.
def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# TODO: utils.
def voronoi(extrema, maximums, minimums):
    voronoi_arr = np.zeros_like(extrema)
    for i in range(1, voronoi_arr.shape[0] - 1):
        for j in range(0, voronoi_arr.shape[1]):
            min_dist = 1e9
            for k, extrema_point in enumerate(minimums + maximums):
                test_dist = dist((i, j), extrema_point)
                if test_dist < min_dist:
                    min_dist = test_dist
                    voronoi_arr[i, j] = k

    for k, extrema_point in enumerate(minimums + maximums):
        voronoi[extrema_point[0], extrema_point[1]] = 0
    voronoi[voronoi > len(minimums)] = -1
    return voronoi

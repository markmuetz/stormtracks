from argparse import ArgumentParser
import datetime as dt
import math
import time
from itertools import *

from netCDF4 import Dataset
import pylab as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.ndimage.filters import maximum_filter, minimum_filter

from cyclone import CycloneSet, Cyclone, Isobar
from fill_raster import fill_raster, path_to_raster

from c_wrapper import cvort, cvort4

EARTH_RADIUS = 6371
EARTH_CIRC = EARTH_RADIUS * 2 * math.pi

DATA_DIR = 'data/c20/full'

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

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
    def __init__(self, ncdata):
        self.ncdata = ncdata
        self.dates = ncdata.dates
        self.lon = ncdata.lon
        self.lat = ncdata.lat
        self.f_lon = ncdata.f_lon
        self.f_lat = ncdata.f_lat

        self.current_date = None
        self.cyclones_by_date = {}

    def set_year(self, year):
        self.year = year
        self.ncdata.set_year(year)
        self.dates = self.ncdata.dates

    def track_vort_maxima(self, ensemble_member, start_date, end_date):
        if start_date < self.dates[0]:
            raise Exception('Start date is out of date range, try setting the year appropriately')
        elif end_date > self.dates[-1]:
            raise Exception('End date is out of date range, try setting the year appropriately')

        index = np.where(self.dates == start_date)[0][0]
        end_index = np.where(self.dates == end_date)[0][0]

        self.vortmax_time_series = []

        dist_cutoff = 10
        vort_cutoff = 1e-4
        while index <= end_index:
            date = self.dates[index]
            #print(date)
            self.set_date(date, ensemble_member)

            vortmaxes = []

            for vmax in self.ncdata.vmaxs:
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

            self.vortmax_time_series.append(vortmaxes)

            index += 1

        index = 0
        for vs1, vs2 in pairwise(self.vortmax_time_series):
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
                        import ipdb; ipdb.set_trace()


        for vs in self.vortmax_time_series:
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

    def set_date(self, date, ensemble_member=0):
        if date != self.current_date:
            self.current_date = date
            self.ncdata.set_date(date, ensemble_member, 'member')

    def find_all_cyclones(self):
        self.pressures = np.arange(94000, 103000, 100)
        cn = plt.contour(self.ncdata.psl, levels=self.pressures)
        self.grid_coords_contours = get_contour_verts(cn)

        pressures = self.pressures
        grid_coords_contours = self.grid_coords_contours
        pmins = self.ncdata.pmins

        all_cyclones = []
        for p, ploc in pmins:
            # Note swapped x, y
            all_cyclones.append(Cyclone(ploc[0], ploc[1], self.current_date))

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
            self.cyclones_by_date[self.current_date].append(cyclone)

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


class NCData(object):
    def __init__(self, start_year, ensemble=True, smoothing=False, verbose=True):
        self._year = start_year
        self.dx = None
        self.current_date = None
        self.smoothing = smoothing
        self.ensemble = ensemble
        self.verbose = verbose

        self.debug = False

        self.load_datasets(self._year)

    def __say(self, text):
        if self.verbose:
            print(text)

    def set_year(self, year):
        self._year = year
        self.close_datasets()
        self.load_datasets(self._year)

    def close_datasets(self):
        self.nc_prmsl.close()
        self.nc_u.close()
        self.nc_v.close()

    def load_datasets(self, year):
        y = str(year)
        if not self.ensemble:
            self.nc_prmsl = Dataset('{0}/{0}/prmsl.{0}.nc'.format(y))
            self.nc_u = Dataset('data/c20/{0}/uwnd.sig995.{0}.nc'.format(y))
            self.nc_v = Dataset('data/c20/{0}/vwnd.sig995.{0}.nc'.format(y))
            start_date = dt.datetime(1800, 1, 1)
            hours_since_1800 = self.nc_prmsl.variables['time'][:]
            self.dates = np.array([start_date + dt.timedelta(hs / 24.) for hs in hours_since_1800])
        else:
            self.nc_prmsl = Dataset('{0}/{1}/prmsl_{1}.nc'.format(DATA_DIR, y))
            self.nc_u = Dataset('{0}/{1}/u9950_{1}.nc'.format(DATA_DIR, y))
            self.nc_v = Dataset('{0}/{1}/v9950_{1}.nc'.format(DATA_DIR, y))
            start_date = dt.datetime(1, 1, 1)
            hours_since_JC = self.nc_prmsl.variables['time'][:]
            self.dates = np.array([start_date + dt.timedelta(hs / 24.) - dt.timedelta(2) for hs in hours_since_JC])
            self.number_enseble_members = self.nc_prmsl.variables['prmsl'].shape[1]

        self.lon = self.nc_prmsl.variables['lon'][:]
        self.lat = self.nc_prmsl.variables['lat'][:]

        dlon = self.lon[2] - self.lon[0]

        # N.B. array as dx varies with lat.
        self.dx = (dlon * np.cos(self.lat * math.pi / 180) * EARTH_CIRC)
        self.dy = (self.lat[0] - self.lat[2]) * EARTH_CIRC

        self.f_lon = interp1d(np.arange(0, 180), self.lon)
        self.f_lat = interp1d(np.arange(0, 91), self.lat)

    def first_date(self, ensemble_member=0, ensemble_mode='member'):
        return self.set_date(self.dates[0], ensemble_member, ensemble_mode)

    def next_date(self, ensemble_member=0, ensemble_mode='member'):
        index = np.where(self.dates == self.current_date)[0][0]
        if index < len(self.dates):
            date = self.dates[index + 1]
            return self.set_date(date, ensemble_member, ensemble_mode)
        else:
            return None

    def set_date(self, date, ensemble_member=0, ensemble_mode='member'):
        if date != self.current_date or ensemble_member != self.current_ensemble_member:
            try:
                self.__say("Setting date to {0}".format(date))
                index = np.where(self.dates == date)[0][0]
                self.current_date = date
                self.current_ensemble_member = ensemble_member
                if not self.ensemble:
                    self.__process_data(index)
                else:
                    self.__process_ensemble_data(index, ensemble_member, ensemble_mode)
            except:
                self.current_date = None
                self.current_ensemble_member = None
                raise
        return date

    def cvorticity(self, u, v):
        vort = np.zeros_like(u)
        cvort(u, v, u.shape[0], u.shape[1], self.dx, self.dy, vort)
        return vort

    def cvorticity4(self, u, v):
        '''Taken from Walsh's Algorithm'''
        vort = np.zeros_like(u)
        cvort4(u, v, u.shape[0], u.shape[1], self.dx, self.dy, vort)
        return vort

    def vorticity(self, u, v):
        vort = np.zeros_like(u)

        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                du_dy = (u[i + 1, j] - u[i - 1, j])/ self.dy
                dv_dx = (v[i, j + 1] - v[i, j - 1])/ self.dx[i]

                vort[i, j] = dv_dx - du_dy
        return vort

    def fourth_order_vorticity(self, u, v):
        '''Taken from Walsh's Algorithm'''
        vort = np.zeros_like(u)

        for i in range(2, u.shape[0] - 2):
            for j in range(2, u.shape[1] - 2):
                du_dy1 = 2 * (u[i + 1, j] - u[i - 1, j]) / (3 * self.dy)
                du_dy2 = (u[i + 2, j] - u[i - 2, j]) / (12 * self.dy)
                du_dy = du_dy1 - du_dy2

                dv_dx1 = 2 * (v[i, j + 1] - v[i, j - 1]) / (3 * self.dx[i])
                dv_dx2 = (v[i, j + 2] - v[i, j - 2]) / (12 * self.dx[i])
                dv_dx = dv_dx1 - dv_dx2

                vort[i, j] = dv_dx - du_dy
        return vort

    def __process_data(self, i):
        start = time.time()
        self.psl = self.nc_prmsl.variables['prmsl'][i]
        end = time.time()

        # TODO: Why minus sign?
        self.u = - self.nc_u.variables['uwnd'][i]
        self.v = self.nc_v.variables['vwnd'][i]

        self.__say('  Loaded psl, u, v in {0}'.format(end - start))

        start = time.time()
        self.vort  = self.vorticity(self.u, self.v, self.lon, self.lat)
        self.vort4 = self.fourth_order_vorticity(self.u, self.v, self.lon, self.lat)
        end = time.time()
        self.__say("  Calc'd vorticity in {0}".format(end - start))

        start = time.time()
        e, index_pmaxs, index_pmins = find_extrema2(self.psl)
        self.pmins = [(self.psl[pmin[0], pmin[1]], (self.lon[pmin[1]], self.lat[pmin[0]])) for pmin in index_pmins]
        e, index_vmaxs, index_vmins = find_extrema2(self.vort)
        self.vmaxs = [(self.vort[vmax[0], vmax[1]], (self.lon[vmax[1]], self.lat[vmax[0]])) for vmax in index_vmaxs]

        end = time.time()
        self.__say('  Found maxima/minima in {0}'.format(end - start))

        if self.smoothing:
            start = time.time()
            self.smoothed_vort = ndimage.filters.gaussian_filter(self.vort, 1, mode='nearest')
            e, index_svmaxs, index_svmins = find_extrema2(self.smoothed_vort)
            self.smoothed_vmaxs = [(self.smoothed_vort[svmax[0], svmax[1]], (self.lon[svmax[1]], self.lat[svmax[0]])) for svmax in index_svmaxs]
            end = time.time()
            self.__say('  Smoothed vorticity in {0}'.format(end - start))

    def __process_ensemble_data(self, i, ensemble_member, ensemble_mode):
        if ensemble_mode not in ['member', 'mean', 'full']:
            raise Exception('ensemble_mode should be one of member, mean or full')

        if ensemble_mode == 'member':
            if ensemble_member < 0 or ensemble_member >= self.number_enseble_members:
                raise Exception('Ensemble member must be be between 0 and {0}'.format(self.number_enseble_members))

        start = time.time()
        if ensemble_mode == 'member':
            self.psl = self.nc_prmsl.variables['prmsl'][i, ensemble_member]
        elif ensemble_mode == 'mean':
            self.psl = self.nc_prmsl.variables['prmsl'][i].mean(axis=0)
        elif ensemble_mode == 'full':
            self.psl = self.nc_prmsl.variables['prmsl'][i]

        end = time.time()

        # TODO: Why minus sign?
        if ensemble_mode == 'member':
            self.u = - self.nc_u.variables['u9950'][i, ensemble_member]
            self.v = self.nc_v.variables['v9950'][i, ensemble_member]
        elif ensemble_mode == 'mean':
            self.u = - self.nc_u.variables['u9950'][i].mean(axis=0)
            self.v = self.nc_v.variables['v9950'][i].mean(axis=0)
        elif ensemble_mode == 'full':
            self.u = - self.nc_u.variables['u9950'][i]
            self.v = self.nc_v.variables['v9950'][i]

        self.__say('  Loaded psl, u, v in {0}'.format(end - start))

        start = time.time()
        if ensemble_mode in ['member', 'mean']:
            self.vort  = self.cvorticity(self.u, self.v)
            self.vort4 = self.cvorticity4(self.u, self.v)
        else:
            vort = []
            vort4 = []
            for i in range(self.number_enseble_members):
                vort.append(self.cvorticity(self.u[i], self.v[i]))
                vort4.append(self.cvorticity4(self.u[i], self.v[i]))
            self.vort = np.array(vort)
            self.vort4 = np.array(vort4)

        end = time.time()
        self.__say("  Calc'd c vorticity in {0}".format(end - start))

        if self.debug:
            start = time.time()
            vort  = self.vorticity(self.u, self.v)
            vort4 = self.fourth_order_vorticity(self.u, self.v)
            end = time.time()
            self.__say("  Calc'd vorticity in {0}".format(end - start))

            if abs((self.vort - vort).max()) > 1e-10:
                raise Exception('Difference between python/c vort calc')

            if abs((self.vort4 - vort4).max()) > 1e-10:
                raise Exception('Difference between python/c vort4 calc')

        if ensemble_mode in ['member', 'mean']:
            e, index_pmaxs, index_pmins = find_extrema2(self.psl)
            self.pmins = [(self.psl[pmin[0], pmin[1]], (self.lon[pmin[1]], self.lat[pmin[0]])) for pmin in index_pmins]

            e, index_vmaxs, index_vmins = find_extrema2(self.vort)
            self.vmaxs = [(self.vort[vmax[0], vmax[1]], (self.lon[vmax[1]], self.lat[vmax[0]])) for vmax in index_vmaxs]

            e, index_v4maxs, index_v4mins = find_extrema2(self.vort4)
            self.v4maxs = [(self.vort4[v4max[0], v4max[1]], (self.lon[v4max[1]], self.lat[v4max[0]])) for v4max in index_v4maxs]
        else:
            self.pmins = []
            self.vmaxs = []
            self.v4maxs = []

            for i in range(self.number_enseble_members):
                e, index_pmaxs, index_pmins = find_extrema2(self.psl[i])
                self.pmins.append([(self.psl[i, pmin[0], pmin[1]], (self.lon[pmin[1]], self.lat[pmin[0]])) for pmin in index_pmins])

                e, index_vmaxs, index_vmins = find_extrema2(self.vort[i])
                self.vmaxs.append([(self.vort[i, vmax[0], vmax[1]], (self.lon[vmax[1]], self.lat[vmax[0]])) for vmax in index_vmaxs])

                e, index_v4maxs, index_v4mins = find_extrema2(self.vort4[i])
                self.v4maxs.append([(self.vort4[i, v4max[0], v4max[1]], (self.lon[v4max[1]], self.lat[v4max[0]])) for v4max in index_v4maxs])

        end = time.time()
        self.__say('  Found maxima/minima in {0}'.format(end - start))
        if self.smoothing:
            start = time.time()
            self.smoothed_vort = ndimage.filters.gaussian_filter(self.vort, 1, mode='nearest')
            e, index_svmaxs, index_svmins = find_extrema2(self.smoothed_vort)
            self.smoothed_vmaxs = [(self.smoothed_vort[svmax[0], svmax[1]], (self.lon[svmax[1]], self.lat[svmax[0]])) for svmax in index_svmaxs]
            end = time.time()
            self.__say('  Smoothed vorticity in {0}'.format(end - start))

    def get_pressure_from_date(self, date):
        self.set_date(date)
        return self.psl

    def get_vort_from_date(self, date):
        self.set_date(date)
        return self.vort


def main(args):
    ncdata = NCData(2005)

    start_date = ncdata.dates[int(args.start)]
    end_date = ncdata.dates[int(args.end)]

    glob_cyclones = GlobalCyclones(ncdata)

    glob_cyclones.find_cyclones_in_date_range(start_date, end_date)

    tracker = CycloneTracker(glob_cyclones)
    cyclone_sets = tracker.track()

    return glob_cyclones, cyclone_sets


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


def find_extrema(array):
    extrema = np.zeros_like(array)
    maximums = []
    minimums = []
    for i in range(1, array.shape[0] - 1):
        for j in range(0, array.shape[1]):
            val = array[i, j]

            is_max, is_min = True, True
            for ii in range(i - 1, i + 2):
                for jj in range(j - 1, j + 2):
                    if val < array[ii, jj % array.shape[1]]: 
                        is_max = False
                    elif val > array[ii, jj % array.shape[1]]: 
                        is_min = False
            if is_max:
                extrema[i, j] = 1
                maximums.append((i, j))
            elif is_min:
                extrema[i, j] = -1
                minimums.append((i, j))
    return extrema, maximums, minimums

def find_extrema2(array):
    extrema = np.zeros_like(array)
    maximums = []
    minimums = []

    local_max = maximum_filter(array, size=(3, 3)) == array
    local_min = minimum_filter(array, size=(3, 3)) == array
    extrema += local_max
    extrema -= local_min

    where_max = np.where(local_max)
    where_min = np.where(local_min)

    for max_point in zip(where_max[0], where_max[1]):
        if max_point[0] != 0 and max_point[0] != array.shape[0] - 1:
            maximums.append(max_point)

    for min_point in zip(where_min[0], where_min[1]):
        if min_point[0] != 0 and min_point[0] != array.shape[0] - 1:
            minimums.append(min_point)

    return extrema, maximums, minimums


def geo_dist(p1, p2):
    return np.arcos(np.sin(p1[1]) * np.sin(p2[1]) + np.cos(p1[1]) * np.cos(p2[1]) * (p1[0] - p2[0])) * EARTH_RADIUS

def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


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


def create_args():
    parser = ArgumentParser()
    parser.add_argument('-l', '--sleep', help='Sleep time', default='0.1')
    parser.add_argument('-s', '--start', help='Number of timesteps', default='0')
    parser.add_argument('-e', '--end', help='Number of timesteps', default='10')
    parser.add_argument('-p', '--plot-pressures', help='Plot pressures', action='store_true')
    parser.add_argument('-w', '--plot-winds', help='Plot winds', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = create_args()
    main(args)

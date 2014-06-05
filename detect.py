from argparse import ArgumentParser
import datetime as dt

from netCDF4 import Dataset
import pylab as plt
import numpy as np
from scipy.interpolate import interp1d

from cyclone import CycloneSet, Cyclone, Isobar
from fill_raster import fill_raster, path_to_raster

class NCData(object):
    def __init__(self):
        self.load_datasets()

    def load_datasets(self):
        self.nc_prmsl = Dataset('data/c20/2005/prmsl.2005.nc')
        #nc_prmsl = Dataset('/home/markmuetz/tmp/prmsl_2005.nc')
        self.nc_u = Dataset('data/c20/2005/uwnd.sig995.2005.nc')
        self.nc_v = Dataset('data/c20/2005/vwnd.sig995.2005.nc')

        start_date = dt.datetime(1800, 1, 1)

        #start_date = dt.datetime(1, 1, 1)
        #all_times = np.array([start_date + dt.timedelta(hs / 24.) - dt.timedelta(2) for hs in hours_since_1800])

        hours_since_1800 = self.nc_prmsl.variables['time'][:]
        self.dates = np.array([start_date + dt.timedelta(hs / 24.) for hs in hours_since_1800])
        self.lon = self.nc_prmsl.variables['lon'][:]
        self.lat = self.nc_prmsl.variables['lat'][:]


    def get_pressure_from_date(self, date):
        i = np.where(self.dates == date)[0][0]
        return self.nc_prmsl.variables['prmsl'][i]

    def get_vort_from_date(self, date):
        i = np.where(self.dates == date)[0][0]
        u = - self.nc_u.variables['uwnd'][i]
        v = self.nc_v.variables['vwnd'][i]
        return vorticity(u, v)



def main(args):
    ncdata = NCData()

    nc_prmsl, nc_u, nc_v = ncdata.nc_prmsl, ncdata.nc_u, ncdata.nc_v

    lon = ncdata.lon
    lat = ncdata.lat

    f_lon = interp1d(np.arange(0, 180), lon)
    f_lat = interp1d(np.arange(0, 91), lat)

    timestep_cyclone_sets = []
    timestep_candidate_cyclones = []
    timestep_all_cyclones = []
    run_count = 0
    cyclones_by_date = {}

    all_times = ncdata.dates
    times = []

    for i in range(int(args.start), int(args.end)):
        date = all_times[i]
        times.append(date)
        print(date)
        psl = nc_prmsl.variables['prmsl'][i]

        u = - nc_u.variables['uwnd'][i]
        v = nc_v.variables['vwnd'][i]

        vort = vorticity(u, v)

        e, maxs, mins = find_extrema(psl)

        pressures = np.arange(94000, 103000, 100)
        cn = plt.contour(psl, levels=pressures)

        contours = get_contour_verts(cn)

        cyclones = {}
        cyclones_by_date[date] = []
        for min_point in mins:
            cyclones[min_point] = Cyclone(min_point[1], min_point[0], date, lon, lat, [])

        for pressure, contour_set in zip(pressures, contours):
            for contour in contour_set:
                isobar = Isobar(pressure, contour, lon, lat, f_lon, f_lat)
                contained_points = []
                for min_point in mins:
                    # Note swapped x, y
                    if isobar.contains((min_point[1], min_point[0])):
                        contained_points.append(min_point)
                if len(contained_points) == 1:
                    cyclones[contained_points[0]].isobars.append(isobar)
                if len(contained_points) > 1:
                    prev_pressure = None
                    is_found = True

                    for i in range(len(contained_points)):
                        for j in range(i + 1, len(contained_points)):
                            cp1 = contained_points[i]
                            cp2 = contained_points[j]

                            p1 = psl[cp1[0], cp1[1]]
                            p2 = psl[cp2[0], cp2[1]]
                            if abs(cp1[0] - cp2[0]) > 2 or abs(cp1[1] - cp2[1]) > 2:
                                is_found = False

                            #if p1 != p2:
                                #is_found = False
                                #break

                    if is_found:
                        cyclones[contained_points[0]].isobars.append(isobar)


        all_cyclones = []
        candidate_cyclones = []
        cyclone_sets = []

        for cyclone in cyclones.values():
            all_cyclones.append(cyclone)

            if len(cyclone.isobars) == 0:
                continue
            elif cyclone.isobars[-1].pressure - cyclone.isobars[0].pressure < 500:
                continue
            else:
                cyclone_set = CycloneSet()
                cyclone_set.add_cyclone(cyclone)
                cyclone_sets.append(cyclone_set)

                area = 0
                bounds_path = cyclone.isobars[-1].path
                for i in range(len(bounds_path) - 1):
                    area += bounds_path[i, 0] * bounds_path[(i + 1), 1]
                area += bounds_path[-1, 0] * bounds_path[0, 1]
                area /= 2

                if run_count != 0:
                    for prev_cyclone in timestep_candidate_cyclones[run_count - 1]:
                        if dist((cyclone.cell_pos[0], cyclone.cell_pos[1]), (prev_cyclone.cell_pos[0], prev_cyclone.cell_pos[1])) < 10:
                            prev_cyclone.cyclone_set.add_cyclone(cyclone)

            candidate_cyclones.append(cyclone)
            cyclones_by_date[date].append(cyclone)
            
            roci = cyclone.isobars[-1]
            bounded_vort = vort[int(roci.ymin):int(roci.ymax) + 1,
                                int(roci.xmin):int(roci.xmax) + 1]
            bounded_psl = psl[int(roci.ymin):int(roci.ymax) + 1,
                              int(roci.xmin):int(roci.xmax) + 1]
            bounded_u = u[int(roci.ymin):int(roci.ymax) + 1,
                          int(roci.xmin):int(roci.xmax) + 1]
            bounded_v = v[int(roci.ymin):int(roci.ymax) + 1,
                          int(roci.xmin):int(roci.xmax) + 1]

            raster_path = path_to_raster(roci.path)
            cyclone_mask = fill_raster(raster_path)[0]
            
            cyclone.vort = np.ma.array(bounded_vort, mask=cyclone_mask == 0)
            cyclone.psl = np.ma.array(bounded_psl, mask=cyclone_mask == 0)
            cyclone.u = np.ma.array(bounded_u, mask=cyclone_mask == 0)
            cyclone.v = np.ma.array(bounded_v, mask=cyclone_mask == 0)

        run_count += 1

        timestep_candidate_cyclones.append(candidate_cyclones)
        timestep_cyclone_sets.append(cyclone_sets)
        timestep_all_cyclones.append(all_cyclones)

    return timestep_cyclone_sets, np.array(times), cyclones_by_date

def find_cyclone(tc_sets, times, date, loc):
    i = np.where(times == date)[0][0]
    c_sets = tc_sets[i]
    for c_set in c_sets:
        c = c_set.cyclones[0]
        if c.cell_pos == loc:
            return c_set
    return None

def find_wilma(tc_sets, times):
    return find_cyclone(tc_sets, times, dt.datetime(2005, 10, 18, 12), (278, 16))

def find_katrina(tc_sets, times):
    return find_cyclone(tc_sets, times, dt.datetime(2005, 8, 22, 12), (248, 16))

def load_katrina():
    args = create_args()
    args.start = 932
    args.end = 950
    tc_sets, times = main(args)
    k = find_katrina(tc_sets, times)
    return k

def load_wilma():
    args = create_args()
    args.start = 1162
    args.end = 1200
    tc_sets, times = main(args)
    w = find_wilma(tc_sets, times)
    return w

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

def vorticity(u, v):
    vort = np.zeros_like(u)
    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            du_dy = (u[i + 1, j] - u[i - 1, j])/ 2.
            dv_dx = (v[i, j + 1] - v[i, j - 1])/ 2.
            vort[i, j] = dv_dx - du_dy
    return vort


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

def dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def voronoi(extrema, maximums, minimums):
    voronoi = np.zeros_like(extrema)
    for i in range(1, voronoi.shape[0] - 1):
        for j in range(0, voronoi.shape[1]):
            min_dist = 1e9
            for k, extrema_point in enumerate(minimums + maximums):
                test_dist = dist((i, j), extrema_point)
                if test_dist < min_dist:
                    min_dist = test_dist
                    voronoi[i, j] = k

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

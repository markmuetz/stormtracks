# Code in here is old and has fallen out of use. It may be useful soon though.
import numpy as np
import datetime as dt
import pylab as plt


class CycloneTracker(object):
    def __init__(self, glob_cyclones, max_dist=10, min_cyclone_set_duration=1.49):
        self.glob_cyclones = glob_cyclones
        self.max_dist = max_dist
        self.min_cyclone_set_duration = dt.timedelta(min_cyclone_set_duration)

    def track(self):
        cyclone_sets = []

        prev_date = None
        for date in self.glob_cyclones.dates:
            if date not in self.glob_cyclones.cyclones_by_date.keys():
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

        return [cs for cs in cyclone_sets
                if cs.end_date - cs.start_date > self.min_cyclone_set_duration]


class GlobalCyclones(object):
    def __init__(self, c20data, ensemble_member=0):
        self.c20data = c20data
        self.dates = c20data.dates
        self.lons = c20data.lons
        self.lats = c20data.lats
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

                            # p1 = self.psl[cp1[0], cp1[1]]
                            # p2 = self.psl[cp2[0], cp2[1]]
                            if abs(cp1[0] - cp2[0]) > 2 or abs(cp1[1] - cp2[1]) > 2:
                                is_found = False

                            # if p1 != p2:
                                # is_found = False
                                # break

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
            elif False:
                area = 0
                bounds_path = cyclone.isobars[-1].co
                for i in range(len(bounds_path) - 1):
                    area += bounds_path[i, 0] * bounds_path[(i + 1), 1]
                area += bounds_path[-1, 0] * bounds_path[0, 1]
                area /= 2

                if run_count != 0:
                    for prev_cyclone in timestep_candidate_cyclones[run_count - 1]:
                        if dist(cyclone.cell_pos, prev_cyclone.cell_pos) < 10:
                            prev_cyclone.cyclone_set.add_cyclone(cyclone)

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


class Isobar(object):
    def __init__(self, pressure, contour):
        self.pressure = pressure
        self.contour = contour

        self.xmin, self.xmax = self.contour[:, 0].min(), self.contour[:, 0].max()
        self.ymin, self.ymax = self.contour[:, 1].min(), self.contour[:, 1].max()

        self._is_closed = (self.contour[0][0] == self.contour[-1][0] and
                           self.contour[0][1] == self.contour[-1][1])

    @property
    def is_closed(self):
        return self._is_closed

    def contains(self, point, tol=1e-6):
        if not self.is_closed:
            # print('path not closed')
            return False

        # Should speed up execution a bit.
        if (point[0] < self.xmin or point[0] > self.xmax or
                point[1] < self.ymin or point[1] > self.ymax):
            # print('out of bounds')
            return False

        path = self.contour
        crossing = 0
        px, py = point[0], point[1]

        for i in range(1, len(path)):
            prev_path_point = path[i - 1]
            prev_ppx, prev_ppy = prev_path_point[0], prev_path_point[1]
            pp = path[i]
            ppx, ppy = pp[0], pp[1]
            # print(prev_ppx, prev_ppy)
            # print(ppx, ppy)
            if ppx < px <= prev_ppx or prev_ppx < px <= ppx:
                t = (px - prev_ppx) / (ppx - prev_ppx)
            # print(t)
            # print(ppy, prev_ppy)
            cy = t * (ppy - prev_ppy) + prev_ppy
            # print(px, cy)
            # plt.plot(px, cy, 'ro')
            if abs(cy - py) < tol:
                return True
            elif cy > py:
                crossing += 1
                # print(crossing)

        # print(crossing)
        return crossing % 2 == 1


class CycloneSet(object):
    def __init__(self):
        self._cyclones = []

    def add_cyclone(self, cyclone):
        cyclone.cyclone_set = self
        self._cyclones.append(cyclone)

    def get_cyclone(self, date):
        for cyclone in self._cyclones:
            if cyclone.date == date:
                return cyclone
        return None

    @property
    def cyclones(self):
        return self._cyclones

    @property
    def start_date(self):
        return self._cyclones[0].date

    @property
    def end_date(self):
        return self._cyclones[-1].date


class Cyclone(object):
    def __init__(self, lon, lat, date):
        # self.cat = CAT.uncat
        self.date = date
        self.lon = lon
        self.lat = lat
        self.cell_pos = (lon, lat)

        self.isobars = []
        self.next_cyclone = None
        self.prev_cyclone = None
        self.vort = None
        self.psl = None
        self.u = None
        self.v = None

        self._wind_speed = None
        self._min_psl = None
        self._max_vort = None
        self._max_wind_speed = None
        self.cyclone_set = None

    @property
    def min_psl(self):
        if self._min_psl is None:
            self._min_psl = self.psl.min()
        return self._min_psl

    @property
    def max_vort(self):
        if self._max_vort is None:
            self._max_vort = abs(self.vort.max())
        return self._max_vort

    @property
    def max_wind_speed(self):
        if self._max_wind_speed is None:
            self._max_wind_speed = self.wind_speed.max()
        return self._max_wind_speed

    @property
    def wind_speed(self):
        if self._wind_speed is None:
            self._wind_speed = np.sqrt(self.u ** 2 + self.v ** 2)
        return self._wind_speed


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

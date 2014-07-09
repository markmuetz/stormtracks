import numpy as np
#from enum import Enum
import datetime as dt

# Uses Saffir-Simpson scale
#CAT = Enum('CAT', ['uncat', 'tropdep', 'tropstorm', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5'])


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
            #print(t)
            #print(ppy, prev_ppy)
            cy = t * (ppy - prev_ppy) + prev_ppy
            #print(px, cy)
            #plt.plot(px, cy, 'ro')
            if abs(cy - py) < tol:
                return True
            elif cy > py:
                crossing += 1
                #print(crossing)

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
        #self.cat = CAT.uncat
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

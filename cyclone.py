import numpy as np
from enum import Enum
import pylab as plt

# Uses Saffir-Simpson scale
CAT = Enum('CAT', ['uncat', 'tropdep', 'tropstorm', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5'])

class Pos:
    def __init__(self, x, y):
	self.x = x
	self.y = y

def plot_isobar(isobar, point):
    plt.clf()
    plt.figure(10)
    plt.plot(isobar.path[:, 0], isobar.path[:, 1])
    plt.plot(point.x, point.y, 'kx')
    print(isobar.contains(point))


class Isobar(object):
    def __init__(self, pressure, path, lon, lat, f_lon, f_lat):
	self.pressure = pressure
	self.path = path
	self.lon = lon
	self.lat = lat
	self._glob_path = None

	self.xmin, self.xmax = self.path[:, 0].min(), self.path[:, 0].max()
	self.ymin, self.ymax = self.path[:, 1].min(), self.path[:, 1].max()

	self._is_closed = self.path[0][0] == self.path[-1][0] and\
	                  self.path[0][1] == self.path[-1][1]
	self.f_lon = f_lon
	self.f_lat = f_lat


    @property
    def glob_path(self):
	if self._glob_path == None:
	    self._glob_path = np.zeros_like(self.path) 
	    self._glob_path[:, 0] = self.f_lon(self.path[:, 0])
	    self._glob_path[:, 1] = self.f_lat(self.path[:, 1])
	return self._glob_path
    
    @property
    def is_closed(self):
	return self._is_closed
    
    def contains(self, point, tol=1e-6):
	if not self.is_closed:
	    #print('path not closed')
	    return False

	# Should speed up execution a bit.
	if point.x < self.xmin or point.x > self.xmax or\
	   point.y < self.ymin or point.y > self.ymax:
	    #print('out of bounds')
	    return False

	path = self.path
	crossing = 0
	px, py = point.x, point.y

	for i in range(1, len(path)):
	    prev_path_point = path[i - 1]
	    prev_ppx, prev_ppy = prev_path_point[0], prev_path_point[1]
	    pp = path[i]
	    ppx, ppy = pp[0], pp[1]
	    #print(prev_ppx, prev_ppy)
	    #print(ppx, ppy)
	    if ppx < px <= prev_ppx or prev_ppx < px <= ppx:
		t = (px - prev_ppx) / (ppx - prev_ppx)
		#print(t)
		#print(ppy, prev_ppy)
		cy = t * (ppy - prev_ppy) + prev_ppy
		#print(px, cy)
		#plt.plot(px, cy, 'ro')
		if (abs(cy - py) < tol):
		    return True
		elif cy > py:
		    crossing += 1
		    #print(crossing)


	#print(crossing)
	return crossing % 2 == 1


class Cyclone(object):
    def __init__(self, i, j, lon, lat, isobars):
	self.cat = CAT.uncat
	self.i = i
	self.j = j
	self.lon = lon
	self.lat = lat
	self.cell_pos = Pos(lon[i], lat[j])
	self.isobars = isobars
	self.next_cyclone = None
	self.prev_cyclone = None
	self.vort = None
	self.psl = None

    @property
    def is_head(self):
	return self.prev_cyclone == None

    @property
    def is_tail(self):
	return self.next_cyclone == None

    @property
    def chain_length(self):
	length = 0
	c = self
	while c.next_cyclone:
	    length += 1
	    c = c.next_cyclone
	return length



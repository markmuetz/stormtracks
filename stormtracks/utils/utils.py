import os
from itertools import tee, izip
import tarfile

import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.interpolate import RectSphereBivariateSpline

EARTH_RADIUS = 6371


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def find_extrema_slow(array, print_warning=True):
    '''
    Takes an array and finds its local extrema.

    Returns an array with 0s for not an extrema, 1s for maxs and -1 for mins
    and a list of the indices of all maximums and minimums
    '''
    if print_warning:
        print('Warning, this function is quite slow. Consider using find_extrema(array) instead')
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


def find_extrema(array):
    '''
    Takes an array and finds its local extrema.

    Returns an array with 0s for not an extrema, 1s for maxs and -1 for mins
    and a list of the indices of all maximums and minimums

    N.B. this function is much faster than the above.
    '''
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


def upscale_field(lons, lats, field, x_scale=2, y_scale=2, is_degrees=True):
    '''
    Takes a field defined on a sphere using lons/lats and returns an upscaled
    version, using cubic spline interpolation.
    '''
    if is_degrees:
        lons = lons * np.pi / 180.
        lats = (lats + 90) * np.pi / 180.

    d_lon = lons[1] - lons[0]
    d_lat = lats[1] - lats[0]

    new_lon = np.linspace(lons[0], lons[-1], len(lons) * x_scale)
    new_lat = np.linspace(lats[0], lats[-1], len(lats) * x_scale)

    mesh_new_lat, mesh_new_lon = np.meshgrid(new_lat, new_lon)

    if True:
        lut = RectSphereBivariateSpline(lats[1:-1], lons[1:-1], field[1:-1, 1:-1])

        interp_field = lut.ev(mesh_new_lat[1:-1, 1:-1].ravel(),
                              mesh_new_lon[1:-1, 1:-1].ravel()).reshape(mesh_new_lon.shape[0] - 2,
                                                                        mesh_new_lon.shape[1] - 2).T
    else:
        pass
    if is_degrees:
        new_lon = new_lon * 180. / np.pi
        new_lat = (new_lat * 180. / np.pi) - 90

    return new_lon[1:-1], new_lat[1:-1], interp_field


def geo_dist(p1, p2):
    '''Returns the geodesic distance between two points

    p1, p2 should be of the form (lon, lat) in degrees
    '''
    lon1, lat1 = np.pi * p1[0] / 180., np.pi * p1[1] / 180.
    lon2, lat2 = np.pi * p2[0] / 180., np.pi * p2[1] / 180.
    return np.arccos(
        np.sin(lat1) * np.sin(lat2) +
        np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)) * EARTH_RADIUS


def dist(p1, p2):
    '''Returns the cartesian distance between two points'''
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def raster_voronoi(extrema, maximums, minimums):
    '''
    Takes a 2D array and points of max/mins, and returns a 2D array
    with the voronoi sections filled with different values
    '''
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


def compress_dir(data_dir):
    '''Compresses a given data_dir, taking care to get file names correct'''
    curr_dir = os.getcwd()

    parent_dir = os.path.dirname(data_dir)
    os.chdir(parent_dir)
    compressed_file = data_dir + '.bz2'
    tar = tarfile.open(compressed_file, 'w:bz2')
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            tar.add(os.path.relpath(os.path.join(root, file), start=parent_dir))
    tar.close()

    os.chdir(curr_dir)


def decompress_file(compressed_file):
    '''Decompresses a given tarball, taking care to get file names correct'''
    tar = tarfile.open(compressed_file)
    tar.extractall(os.path.dirname(compressed_file))
    tar.close()

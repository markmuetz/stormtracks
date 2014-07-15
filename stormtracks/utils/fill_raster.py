import math
import numpy as np
import pylab as plt


def contains(array, i, j, val, side):
    if side == 'left':
        if j == 0:
            return False
        jj = j - 1
    if side == 'right':
        if j == array.shape[1] - 1:
            return False
        jj = j + 1

    r = range(max(i - 1, 0), min(i + 2, array.shape[0]))
    if val == 2:
        r = r[::-1]

    for ii in r:
        if array[ii, jj] == val:
            return True
    return False


def left_contains(array, i, j, val):
    return contains(array, i, j, val, 'left')


def right_contains(array, i, j, val):
    return contains(array, i, j, val, 'right')


def fill_raster(array):
    'Takes an outline of a shape (raster), and fills in the central cells'
    # Not working for cases outline is concave in x-dir.
    filled_a = np.copy(array)
    working_a = np.zeros_like(array)

    tops = []
    bottoms = []
    sides = []
    for j in range(array.shape[1]):
        is_top = True
        top = None
        bottom = None
        for i in range(array.shape[0]):
            if array[i, j]:
                if is_top:
                    is_top = False
                    top = [i, j]
                    bottom = [i, j]
                else:
                    bottom = [i, j]
        if top:
            working_a[top[0], top[1]] = 4
            working_a[bottom[0], bottom[1]] = 2
            tops.append(top)
            bottoms.append(bottom)

    for top, bottom in zip(tops, bottoms):
        found_bottom = False
        while top != bottom:
            top[0] += 1
            filled_a[top[0], top[1]] = 1

    return filled_a, working_a


def path_to_raster(path):
    'Takes a path in with path[:, 1] as the x vals and returns a raster outline'
    # Not working for cases when the path stretches over multiple grid cells.
    imin = int(math.floor(path[:, 1].min()))
    imax = int(math.floor(path[:, 1].max()) + 1)
    jmin = int(math.floor(path[:, 0].min()))
    jmax = int(math.floor(path[:, 0].max()) + 1)
    a = np.zeros((imax - imin, jmax - jmin))
    for point in path:
        on_x_boundary = False
        on_y_boundary = False

        if not point[0].is_integer():
            on_x_boundary = True
        elif not point[1].is_integer():
            on_y_boundary = True

        i = int(math.floor(point[1])) - imin
        j = int(math.floor(point[0])) - jmin
        a[i, j] = 1
        if on_x_boundary and i > 0:
            a[i - 1, j] = 1
        if on_y_boundary and j > 0:
            a[i, j - 1] = 1
    return a

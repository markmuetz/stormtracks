import os
from glob import glob
from collections import OrderedDict
import datetime as dt

import numpy as np
import pygrib

from utils.c_wrapper import cvort, cvort4
from load_settings import settings

EARTH_RADIUS = 6371
EARTH_CIRC = EARTH_RADIUS * 2 * np.pi

GRIB_FILE_TPL = 'pgrbanl_{0:04d}{1:02d}{2:02d}{3:02d}_{4:02d}'


class C20GribDataField(object):
    def __init__(self, grb):
        grib_fields = str(grb).split(':')
        self.index = grib_fields[0]
        self.name = grib_fields[1]
        self.units = grib_fields[2]
        self.unknown1 = grib_fields[3]
        self.unknown2 = grib_fields[4]
        self.level_name = grib_fields[5]
        self.unknown3 = grib_fields[6]
        self.unknown4 = grib_fields[7]

        self.data, lats, lons = grb.data()

        # Lons are 2d, make 1d and convert from 0 to 360 to -180 to 180.
        self.lons = lons[0, :]
        self.lons[self.lons >= 180] -= 360

        # Lats are 2d, make 1d
        self.lats = lats[:, 0]


class C20GribLevel(object):
    def __init__(self, u_data_field, v_data_field):
        if not u_data_field.level_name == v_data_field.level_name:
            raise Exception('u/v levels not equal: {0}, {1},'.format(
                u_data_field.level, v_data_field.level))

        self.level_name = u_data_field.level_name
        self.level_key = int(self.level_name.split(' ')[1])

        self.u_data_field = u_data_field
        self.v_data_field = v_data_field

        self.u = self.u_data_field.data.astype(np.float32)
        self.v = self.v_data_field.data.astype(np.float32)

        if not (self.u_data_field.lons == self.v_data_field.lons).all():
            raise Exception('u/v lons not equal')

        if not (self.u_data_field.lats == self.v_data_field.lats).all():
            raise Exception('u/v lats not equal')

        self.lons = self.u_data_field.lons.astype(np.float32)
        self.lats = self.u_data_field.lats.astype(np.float32)

        dlon = self.lons[2] - self.lons[0]

        # N.B. array as dx varies with lat.
        self.dx = (dlon * np.cos(self.lats * np.pi / 180) * EARTH_CIRC)
        self.dy = (self.lats[0] - self.lats[2]) * EARTH_CIRC
        self.vort = self.cvorticity(self.u, self.v)

    def cvorticity(self, u, v):
        '''Calculates the (2nd order) vorticity by calling into a c function'''
        vort = np.zeros_like(u)
        cvort(u, v, u.shape[0], u.shape[1], self.dx, self.dy, vort)
        return vort


class C20GribData(object):
    def __init__(self):
        self.year_dirs = glob(os.path.join(settings.C20_GRIB_DATA_DIR, '*'))
        self.years = [int(os.path.basename(yd)) for yd in self.year_dirs]
        self.year = None

    def next_timestep(self):
        return self.set_date(self.dates[self.dates.index(self.date) + 1], self.ensemble_member)

    def prev_timestep(self):
        return self.set_date(self.dates[self.dates.index(self.date) - 1], self.ensemble_member)

    def set_date(self, date, ensemble_member=23):
        year = date.year
        if year != self.year:
            if year not in self.years:
                raise Exception('No data for year {0}, please download these data first'.format(
                    year))
            self.year_dir = os.path.join(settings.C20_GRIB_DATA_DIR, str(year))

            self.dates = []
            self.ensemble_members = []
            for filename in sorted(glob(os.path.join(self.year_dir, '*'))):
                if os.path.splitext(filename)[-1] == '.tar':
                    continue
                basename = os.path.basename(filename)
                date_str = basename.split('_')[1]
                file_date = dt.datetime.strptime(date_str, '%Y%m%d%H')
                self.dates.append(file_date)
                file_ensemble_member = int(basename.split('_')[2])
                self.ensemble_members.append(file_ensemble_member)

        if date not in self.dates:
            raise Exception('No data for date {0}, please download these data first'.format(date))
        if ensemble_member not in self.ensemble_members:
            raise Exception('No data for em {0}, please download these data first'.format(
                ensemble_member))

        self.date = date
        self.ensemble_member = ensemble_member
        self._load_datasets(date, ensemble_member)

        return self.date

    def _load_datasets(self, date, ensemble_member):
        filename = GRIB_FILE_TPL.format(date.year, date.month, date.day, date.hour, ensemble_member)
        filepath = os.path.join(settings.C20_GRIB_DATA_DIR, str(date.year), filename)
        self.grbs = pygrib.open(filepath)
        self.levels = OrderedDict()

        for u_index, v_index in zip(range(106, 130), range(130, 154)):
            u_grb = self.grbs[u_index]
            v_grb = self.grbs[v_index]
            u_data_field = C20GribDataField(u_grb)
            v_data_field = C20GribDataField(v_grb)

            level = C20GribLevel(u_data_field, v_data_field)
            print('  {0}'.format(level.level_name))
            self.levels[level.level_key] = level

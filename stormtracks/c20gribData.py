import os
from glob import glob
from collections import OrderedDict

import pygrib

from load_settings import settings

GRIB_FILE_TPL = 'pgrbanl_{0:04d}{1:02d}{2:02d}{3:02d}_{4:02d}'


class C20GribDataField(object):
    def __init__(self, grib_fields):
        self.index = grib_fields[0]
        self.name = grib_fields[1]
        self.units = grib_fields[2]
        self.unknown1 = grib_fields[3]
        self.unknown2 = grib_fields[4]
        self.level = grib_fields[5]
        self.unknown3 = grib_fields[6]
        self.unknown4 = grib_fields[7]


class C20GribData(object):
    def __init__(self, date):
        self.date = date
        self.year_dirs = glob(os.path.join(settings.C20_GRIB_DATA_DIR, '*'))
        self.load_datasets(2005, 10, 18, 6, 23)

    def load_datasets(self, year, month, day, hour, ensemble_member):
        filename = GRIB_FILE_TPL.format(year, month, day, hour, ensemble_member)
        filepath = os.path.join(settings.C20_GRIB_DATA_DIR, str(year), filename)
        self.grbs = pygrib.open(filepath)
        self.us = OrderedDict()

        print('u')
        for i in range(106, 130):
            grb = self.grbs[i]
            c20gribdDataField = C20GribDataField(str(grb).split(':'))
            print('  {0}'.format(c20gribdDataField.level))
            self.us[c20gribdDataField.level] = c20gribdDataField

        self.vs = OrderedDict()
        print('v')
        for i in range(130, 154):
            grb = self.grbs[i]
            c20gribdDataField = C20GribDataField(str(grb).split(':'))
            print('  {0}'.format(c20gribdDataField.level))
            self.vs[c20gribdDataField.level] = c20gribdDataField

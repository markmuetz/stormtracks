from __future__ import print_function

import os
from glob import glob
import random
from collections import OrderedDict

import netCDF4 as nc

from utils import find_extrema


def main():
    all_errors = OrderedDict()
    data_dir = '/media/markmuetz/SAMSUNG/DATA/stormtracks'
    for year_dir in sorted(glob('{0}/*'.format(data_dir))):
        print('Checking dir {0}'.format(year_dir))
        errors = check_year_dir_for_error(year_dir)
        if len(errors):
            all_errors[year_dir] = errors

    print('')
    if len(all_errors):
        print('There were problems with the data in {0}'.format(data_dir))
        print_errors(all_errors)
    else:
        print('No errors found with data in {0}'.format(data_dir))


def print_errors(all_errors):
    for error_dir, errors in all_errors.items():
        print('Problem in dir {0}'.format(error_dir))
        for error in errors:
            print('  {0}'.format(error))


def check_year_dir_for_error(year_dir):
    errors = []
    files = glob('{0}/*'.format(year_dir))
    if len(files) != 3:
        error_message = 'Only {0} files in dir {1} (expected 3)'.format(len(files), year_dir)
        errors.append(error_message)

    for f in files:
        try:
            d = nc.Dataset(f)
            var_name = os.path.basename(f).split('_')[0]
            v = d.variables[var_name]
            timestep = random.randrange(0, v.shape[0])
            ensemble_member = random.randrange(0, v.shape[1])
            find_extrema(v[timestep, ensemble_member])
        except Exception as e:
            error_message = 'Problem {0} with file {1}'.format(e.message, f)
            errors.append(error_message)

    return errors


if __name__ == '__main__':
    main()

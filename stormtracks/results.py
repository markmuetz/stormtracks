import os
from glob import glob

import pandas as pd

from load_settings import settings
from utils.utils import compress_file, decompress_file

RESULTS_TPL = '{0}.hdf'


class ResultNotFound(Exception):
    '''Simple exception thrown if result cannot be found in results manager or on disk'''
    pass


class StormtracksResultsManager(object):
    '''Manager class that is responsible for loading and saving all python results

    Simple key/value store.
    Load/saves to settings.OUTPUT_DIR.
    '''
    def __init__(self, name, output_dir=None):
        self.name = name
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = settings.OUTPUT_DIR

    def save_result(self, year, result_key, result):
        '''Saves a given result based on year, user chosen result_key'''
	dirname = os.path.join(self.output_dir, self.name)
	if not os.path.exists(dirname):
	    os.makedirs(dirname)

	filename = RESULTS_TPL.format(year)
	print('saving {0}'.format(filename))
	path = os.path.join(dirname, filename)
	result.to_hdf(path, result_key)

    def get_result(self, year, result_key):
        '''Returns a result from an HDF file.'''
	dirname = os.path.join(self.output_dir, self.name)
	filename = RESULTS_TPL.format(year)
	path = os.path.join(dirname, filename)

	try:
	    result = pd.read_hdf(path, result_key)
	except Exception, e:
	    raise ResultNotFound

        return result


    def delete(self, year, result_key):
        '''Deletes a specific result from disk'''
	raise NotImplementedError('Not sure how to delete one result')

    def compress_year(self, year, delete=False):
        '''Compresses a given year's dir and then optionally deletes that year'''
        year_filename = os.path.join(self.output_dir, self.name, RESULTS_TPL.format(year))
        compressed_filename = compress_file(year_filename)
        if delete:
            self.delete_year(year)
        return compressed_filename

    def delete_year(self, year):
        '''Deletes a year (use with caution!)'''
        year_filename = os.path.join(self.output_dir, self.name, RESULTS_TPL.format(year))
        os.remove(year_filename)

    def decompress_year(self, year):
        '''Decompresses a given year's tarball'''
        filename = os.path.join(self.output_dir, self.name, '{0}.bz2'.format(RESULTS_TPL.format(year)))
        decompress_file(filename)

    def list_years(self):
        '''List all saved years'''
        years = []
        dirname = os.path.join(self.output_dir, self.name)

        for year_dirname in glob(os.path.join(dirname, '*')):
            try:
                year = int(os.path.splitext(os.path.basename(year_dirname))[0])
                years.append(year)
            except:
                pass
        return sorted(years)

    def list_results(self, year):
        '''List all results saved for a particular year'''
        dirname = os.path.join(self.output_dir, self.name)
	print(os.path.join(dirname, RESULTS_TPL.format(year)))
	store = pd.HDFStore(os.path.join(dirname, RESULTS_TPL.format(year)))
	results = [field[0][1:] for field in store.items()]
	store.close()
        return sorted(results)

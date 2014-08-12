import os
from glob import glob
import cPickle
import datetime as dt
from collections import OrderedDict
import shutil

import numpy as np

from load_settings import settings
from utils.utils import compress_dir, decompress_file

RESULTS_TPL = '{0}-{1}.pkl'


class ResultNotFound(Exception):
    '''Simple exception thrown if result cannot be found in results manager or on disk'''
    pass


class StormtracksNumpyResultsManager(object):
    '''Super simple key/value store for numpy arrays.'''
    def __init__(self, name):
        self.name = name

    def save(self, key, result):
        dirname = os.path.join(settings.OUTPUT_DIR, self.name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        filename = os.path.join(dirname, key)
        np.save(filename, result)

    def load(self, key):
        dirname = os.path.join(settings.OUTPUT_DIR, self.name)
        filename = os.path.join(dirname, '{0}.npy'.format(key))
        try:
            result = np.load(filename)
        except:
            raise ResultNotFound
        return result


class StormtracksResultsManager(object):
    '''Manager class that is responsible for loading and saving all python results

    Simple key/value store.
    Load/saves to settings.OUTPUT_DIR.
    Saves each result to e.g.::

        settings.OUTPUT_DIR/<self.name>/<year>/<ensemble_member>-<key>.pkl

    (using that as a directory/filename structure).
    '''
    def __init__(self, name, cache_loaded=False):
        self.name = name
        self.cache_loaded = cache_loaded
        self._results = OrderedDict()
        self._saved = []

    def add_result(self, year, ensemble_member, result_key, result):
        '''Adds a given result based on year, ensemble_member and a user chosen result_key'''
        if year not in self._results:
            self._results[year] = OrderedDict()

        if ensemble_member not in self._results[year]:
            self._results[year][ensemble_member] = OrderedDict()

        if result_key in self._results[year][ensemble_member].keys():
            raise Exception('Result {0} has already been added'.format(result_key))
        self._results[year][ensemble_member][result_key] = result

    def get_result(self, year, ensemble_member, result_key):
        '''Returns a saved/already loaded result if it exists, or result from disk

        throws a ResultNotFound error if it can't find the requested result on disk.
        '''
        try:
            # Check and see if it's already been loaded.
            result = self._results[year][ensemble_member][result_key]
        except KeyError:
            # It's not in the loaded results, try to load it from disk.
            try:
                result = self._load(year, ensemble_member, result_key)
            except:
                print('Could not find entry for {0}, {1}'.format(year, ensemble_member))
                raise

        return result

    def save(self):
        '''Saves all unsaved results that have been added so far'''
        for year in self._results.keys():
            y = str(year)
            dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            for ensemble_member in self._results[year].keys():
                for results_key in self._results[year][ensemble_member].keys():
                    if (year, ensemble_member, results_key) not in self._saved:
                        filename = RESULTS_TPL.format(ensemble_member, results_key)
                        print('saving {0}'.format(filename))
                        f = open(os.path.join(dirname, filename), 'w')
                        cPickle.dump([results_key,
                                      self._results[year][ensemble_member][results_key]], f)
                        self._saved.append((year, ensemble_member, results_key))

    def _load(self, year, ensemble_member, result_key):
        '''Loads results from disk'''
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
        try:
            filename = os.path.join(dirname, RESULTS_TPL.format(ensemble_member,
                                                                result_key))
            result = self._load_filename(year, ensemble_member, filename)
        except ResultNotFound:
            raise
        except Exception, e:
            print('Results {0}, {1} could not be loaded'.format(year, ensemble_member))
            print('{0}'.format(e.message))
            raise

        return result

    def _load_filename(self, year, ensemble_member, filename):
        try:
            f = open(filename, 'r')
        except IOError:
            raise ResultNotFound

        key_and_result = cPickle.load(f)
        key = key_and_result[0]
        result = key_and_result[1]

        if self.cache_loaded:
            self.add_result(year, ensemble_member, key, result)
            self._saved.append((year, ensemble_member, key))

        return result

    def delete(self, year, ensemble_member, result_key):
        '''Deletes a specific result from disk'''
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
        try:
            os.remove(os.path.join(dirname, RESULTS_TPL.format(ensemble_member,
                                                               result_key)))
        except Exception, e:
            print('Results {0}, {1} could not be deleted'.format(year, ensemble_member))
            print('{0}'.format(e.message))
            raise e

    def compress_year(self, year, delete=False):
        '''Compresses a given year's dir and then deletes that year'''
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
        compress_dir(dirname)
        if delete:
            self.delete_year(year)

    def delete_year(self, year):
        '''Deletes a year (use with caution!)'''
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
        shutil.rmtree(dirname)

    def decompress_year(self, year):
        '''Decompresses a given year's tarball'''
        y = str(year)
        filename = os.path.join(settings.OUTPUT_DIR, self.name, '{0}.bz2'.format(y))
        decompress_file(filename)

    def list_years(self):
        '''List all saved years'''
        years = []
        dirname = os.path.join(settings.OUTPUT_DIR, self.name)

        for year_dirname in glob(os.path.join(dirname, '*')):
            try:
                year = int(os.path.basename(year_dirname))
                years.append(year)
            except:
                pass
        return sorted(years)

    def list_ensemble_members(self, year):
        '''List all results saved for a particular year'''
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
        ensemble_members = []
        for fn in glob(os.path.join(dirname, RESULTS_TPL.format('*', '*'))):
            try:
                ensemble_member = int(os.path.basename(fn).split('-')[0])
                ensemble_members.append(ensemble_member)
            except:
                pass
        return sorted(set(ensemble_members))

    def list_results(self, year, ensemble_member):
        '''List all results saved for a particular year/ensemble_member'''
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
        _results_names = []
        for fn in glob(os.path.join(dirname, RESULTS_TPL.format(ensemble_member, '*'))):
            result_list = os.path.basename(fn).split('.')[0].split('-')[1:]
            _results_names.append('-'.join(result_list))
        return sorted(_results_names)

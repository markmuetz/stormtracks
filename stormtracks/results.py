import os
from glob import glob
import cPickle
import datetime as dt
from collections import OrderedDict

from load_settings import settings

RESULTS_TPL = '{0}-{1}-{2}.pkl'


class StormtracksResults(object):
    '''Utility class that is easier to use than a dict

    :param dct: dict used to populate this class' fields
    '''
    def __init__(self, dct):
        for k, v in dct.items():
            self.__dict__[k.replace(';', '_').replace(':', '_')] = v


class StormtracksResultsManager(object):
    '''Manager class that is responsible for loading and saving all results

    Load/saves to settings.OUTPUT_DIR.
    Saves each result based on its year/ensemble_member
    (using that as a directory/filename structure).
    Saves each result as a name in a dictionary that then gets serialized to disk.
    '''
    def __init__(self, name):
        self._results = OrderedDict()
        self.name = name
        self.saved = []

    def add_result(self, year, ensemble_member, name, result):
        '''Adds a given result based on year, ensemble_member and a user chosen name'''
        if year not in self._results:
            self._results[year] = OrderedDict()

        if ensemble_member not in self._results[year]:
            self._results[year][ensemble_member] = OrderedDict()

        if name in self._results[year][ensemble_member].keys():
            raise Exception('Result {0} has already been added'.format(name))
        self._results[year][ensemble_member][name] = result

    def get_results(self, year, ensemble_member):
        return self._results[year][ensemble_member]

    def get_results_object(self, year, ensemble_member):
        '''Gets a set of results based on year, ensemble_member'''
        try:
            results = StormtracksResults(self._results[year][ensemble_member])
            return results
        except KeyError:
            print('Could not find entry for {0}, {1}'.format(year, ensemble_member))

    def save(self):
        '''Saves all unsaved results that have been added so far'''
        for year in self._results.keys():
            y = str(year)
            dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            for ensemble_member in self._results[year].keys():
                for results_key in self._results[year][ensemble_member].keys():
                    if (year, ensemble_member, results_key) not in self.saved:
                        filename = RESULTS_TPL.format(self.name, ensemble_member, results_key)
                        print('saving {0}'.format(filename))
                        f = open(os.path.join(dirname, filename), 'w')
                        cPickle.dump([results_key,
                                      self._results[year][ensemble_member][results_key]], f)
                        self.saved.append((year, ensemble_member, results_key))

    def load(self, year=2005, ensemble_member=0, result_key=None):
        '''Loads results from disk'''
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
        try:
            if not result_key:
                filenames = sorted(glob(os.path.join(dirname, RESULTS_TPL.format(self.name,
                                                                                 ensemble_member,
                                                                                 '*'))))
                for filename in filenames:
                    print(filename)
                    self._load_filename(year, ensemble_member, filename)
            else:
                filename = os.path.join(dirname, RESULTS_TPL.format(self.name,
                                                                    ensemble_member,
                                                                    result_key))
                print(filename)
                self._load_filename(year, ensemble_member, filename)

        except Exception, e:
            print('Results {0}, {1} could not be loaded'.format(year, ensemble_member))
            print('{0}'.format(e.message))
            raise e

    def _load_filename(self, year, ensemble_member, filename):
        f = open(filename, 'r')
        key_and_result = cPickle.load(f)
        key = key_and_result[0]
        result = key_and_result[1]

        self.add_result(year, ensemble_member, key, result)
        self.saved.append((year, ensemble_member, key))

    def delete(self, year=2005, ensemble_member=0):
        '''Deletes a specific result from disk'''
        raise Exception('Needs fixing')
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
        try:
            os.remove(os.path.join(dirname, RESULTS_TPL.format(self.name,
                                                               ensemble_member,
                                                               ensemble_member)))
        except Exception, e:
            print('Results {0}, {1} could not be deleted'.format(year, ensemble_member))
            print('{0}'.format(e.message))
            raise e

    def print_list_years(self):
        '''Print all saved results'''
        for name in self.list_years():
            print(name)

    def list_ensemble_members(self, year):
        '''List all results saved for a particular year'''
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
        _results_names = []
        for fn in glob(os.path.join(dirname, RESULTS_TPL.format(self.name, '*', '*'))):
            _results_names.append(os.path.basename(fn).split('.')[0])
        return sorted(_results_names)

    def list_results(self, year, ensemble_member):
        '''List all results saved for a particular year'''
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)
        _results_names = []
        for fn in glob(os.path.join(dirname, RESULTS_TPL.format(self.name, ensemble_member, '*'))):
            _results_names.append(os.path.basename(fn).split('.')[0].split('-')[-1])
        return sorted(_results_names)

    def list_years(self):
        '''List all saved years'''
        years = []
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, self.name, y)

        for year_dirname in glob(os.path.join(dirname, '*')):
            years.append(os.path.basename(year_dirname))
        return sorted(years)

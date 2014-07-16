import os
from glob import glob
import cPickle
import datetime as dt
from collections import OrderedDict

from load_settings import settings

RESULTS_TPL = 'results_em_{0}.pkl'


class StormtracksResult(object):
    def __init__(self, dct):
        for k, v in dct.items():
            self.__dict__[k] = v


class StormtracksResultsManager(object):
    def __init__(self):
        self._results = OrderedDict()

    def add_result(self, year, ensemble_member, name, result):
        if year not in self._results:
            self._results[year] = OrderedDict()

        if ensemble_member not in self._results[year]:
            self._results[year][ensemble_member] = OrderedDict()

        if name in self._results[year][ensemble_member].keys():
            raise Exception('Result {0} has already been added'.format(name))
        self._results[year][ensemble_member][name] = result

    def get_result(self, year, ensemble_member):
        try:
            result = StormtracksResult(self._results[year][ensemble_member])
            return result
        except KeyError:
            print('Could not find entry for {0}, {1}'.format(year, ensemble_member))

    def save(self):
        for year in self._results.keys():
            y = str(year)
            dirname = os.path.join(settings.OUTPUT_DIR, y)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            for ensemble_member in self._results[year].keys():
                f = open(os.path.join(dirname, RESULTS_TPL.format(ensemble_member)), 'w')
                cPickle.dump(self._results[year][ensemble_member], f)

    def load(self, year=2005, ensemble_member=0):
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, y)
        try:
            f = open(os.path.join(dirname, RESULTS_TPL.format(ensemble_member)), 'r')
            _results = cPickle.load(f)

            for name, result in _results.items():
                self.add_result(year, ensemble_member, name, result)

        except Exception, e:
            print('Results {0}, {1} could not be loaded'.format(year, ensemble_member))
            print('{0}'.format(e.message))
            raise e

    def delete(self, year=2005, ensemble_member=0):
        y = str(year)
        dirname = os.path.join(settings.OUTPUT_DIR, y)
        try:
            os.remove(os.path.join(dirname, RESULTS_TPL.format(ensemble_member)))
        except Exception, e:
            print('Results {0}, {1} could not be deleted'.format(year, ensemble_member))
            print('{0}'.format(e.message))
            raise e

    def print_list_years(self):
        for name in self.list_years():
            print(name)

    def list_ensemble_members(self, year):
        dirname = os.path.join(settings.OUTPUT_DIR, str(year))
        _results_names = []
        for fn in glob(os.path.join(dirname, RESULTS_TPL.format('*'))):
            _results_names.append('_'.join(os.path.basename(fn).split('.')[0].split('_')[1:]))
        return sorted(_results_names)

    def list_years(self):
        years = []
        for dirname in glob(os.path.join(settings.OUTPUT_DIR, '*')):
            years.append(os.path.basename(dirname))
        return sorted(years)

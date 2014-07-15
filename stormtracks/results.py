import os
from glob import glob
import cPickle
import datetime as dt

from load_settings import settings


class StormtracksResultsManager(object):
    def __init__(self):
        self.results = {'version': 0.1}

    def add_result(self, name, result):
        if name in self.results.keys():
            raise Exception('Result {0} has already been added'.format(name))
        self.results[name] = result

    def save(self, name):
        f = open(os.path.join(settings.OUTPUT_DIR, 'results_{0}.pkl'.format(name)), 'w')
        self.results['save_time'] = dt.datetime.now()
        cPickle.dump(self.results, f)

    def load(self, name):
        try:
            f = open(os.path.join(settings.OUTPUT_DIR, 'results_{0}.pkl'.format(name)), 'r')
            results = cPickle.load(f)
            if results['version'] != self.results['version']:
                r = raw_input('version mismatch, may not work! Press c to continue anyway: ')
                if r != 'c':
                    raise Exception('Version mismatch (user cancelled)')

            self.results = results
        except Exception, e:
            print('Results {0} could not be loaded'.format(name))
            print('{0}'.format(e.message))

    def print_list(self):
        for name in self.list():
            print(name)

    def list(self):
        results_names = []
        for fn in glob(os.path.join(settings.OUTPUT_DIR, 'results_*.pkl')):
            results_names.append('_'.join(os.path.basename(fn).split('.')[0].split('_')[1:]))
        return results_names

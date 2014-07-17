from __future__ import print_function
from collections import OrderedDict

STATUSES = {
    'outstanding': 'O',
    'working': 'W',
    'complete': 'C',
    'failure': 'F',
    'timeout': 'T',
    }

TASKS = [
    'vort_track']


class PyroTask(object):
    def __init__(self, year, ensemble_member, task):
        self.year = year
        self.ensemble_member = ensemble_member
        self.task = task
        self.status = 'outstanding'

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        if value not in TASKS:
            raise Exception('Task {0} not recognised'.format(value))
        self._task = value

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        if value not in STATUSES:
            raise Exception('Status {0} not recognised'.format(value))
        self._status = value


class PyroTaskSchedule(object):
    def __init__(self, start_year=2000, end_year=2012, num_ensemble_members=56):
        self.start_year = start_year
        self.end_year = end_year
        self.num_ensemble_members = num_ensemble_members

        self._schedule = OrderedDict()
        for year in range(start_year, end_year + 1):
            self._schedule[year] = []
            for em in range(num_ensemble_members):
                self._schedule[year].append(PyroTask(year, em, 'vort_track'))

    def get_next_outstanding(self):
        years = range(self.start_year, self.end_year + 1)
        for year in years:
            for em in range(self.num_ensemble_members):
                task = self._schedule[year][em]
                if task.status == 'outstanding':
                    return task
        return None

    def update_task_status(self, task):
        self._schedule[task.year][task.ensemble_member].status = task.status

    def print_years(self, years=None):
        if not years:
            years = range(self.start_year, self.end_year + 1)

        for year in years:
            tasks = self._schedule[year]
            print('{0:4d}: '.format(year), end='')
            for task in tasks:
                print(STATUSES[task.status], end='')
            print('')

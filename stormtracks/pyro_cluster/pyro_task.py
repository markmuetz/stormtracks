from __future__ import print_function
from collections import OrderedDict

STATUSES = {
    'outstanding': '-',
    'working': 'W',
    'complete': '+',
    'failure': 'F',
    'timeout': 'T',
    }

TASKS = [
    'vort_tracking',
    'analysis',
    ]


class PyroTask(object):
    '''Represents on task that is to be done

    :param year: year of task
    :param ensemble_member: ensemble_member of task
    :param name: name of the task to be done
    '''
    def __init__(self, year, ensemble_member, name, data=None):
        self.year = year
        self.ensemble_member = ensemble_member
        self.name = name
        self.data = data
        self.status = 'outstanding'

    @property
    def name(self):
        '''What name this is doing'''
        return self._name

    @name.setter
    def name(self, value):
        if value not in TASKS:
            raise Exception('Task {0} not recognised'.format(value))
        self._name = value

    @property
    def status(self):
        '''Current status of task, must be in STATUSES'''
        return self._status

    @status.setter
    def status(self, value):
        if value not in STATUSES:
            raise Exception('Status {0} not recognised'.format(value))
        self._status = value


class PyroTaskSchedule(object):
    '''Keeps track of all tasks to be done, and can issue the next outstanding class

    :param start_year: year from which to start tasks
    :param end_year: year from which to end tasks (inclusive)
    :param num_ensemble_members: how many ensemble members to keep tasks for
    '''
    def __init__(self, start_year=2000, end_year=2012, num_ensemble_members=56):
        self.start_year = start_year
        self.end_year = end_year
        self.num_ensemble_members = num_ensemble_members

        self._schedule = OrderedDict()
        for year in range(start_year, end_year + 1):
            self._schedule[year] = []
            for em in range(num_ensemble_members):
                self._schedule[year].append(PyroTask(year, em, 'vort_tracking'))

    def get_next_outstanding(self):
        '''Returns the next outstanding task, None if there are no more'''
        years = range(self.start_year, self.end_year + 1)
        for year in years:
            for em in range(self.num_ensemble_members):
                task = self._schedule[year][em]
                if task.status == 'outstanding':
                    return task
        return None

    def get_progress_for_year(self, year, include_year=True):
        '''Returns a string representing the progress of the year

        * ``-`` :task to be done
        * ``W`` :task being worked on
        * ``+`` :task complete
        * ``F`` :task failure
        * ``T`` :task timeout
        '''
        progress = []
        tasks = self._schedule[year]
        if include_year:
            progress.append('{0:4d}: '.format(year))
        for task in tasks:
            progress.append(STATUSES[task.status])
        return ''.join(progress)

    def get_progress(self, years=None, include_year=False):
        '''Returns a string representing the progress for all years'''
        progress = []
        if not years:
            years = range(self.start_year, self.end_year + 1)

        for year in years:
            progress.append(self.get_progress_for_year(year, include_year))
            progress.append('\n')
        return ''.join(progress)

    def print_years(self, years=None, include_year=True):
        '''Prints progress'''
        print(self.get_progress(years, include_year), end='')


class PyroResultsAnalysis(object):
    def __init__(self, year):
        scales = [1, 2, 3]
        pressure_levels = [995, 850]
        trackers = ['nearest_neighbour']
        ensemble_members = range(2)
        self.current_ensemble_member = 0
        self._tasks = []

        for ensemble_member in ensemble_members:
            em_tasks = []
            for scale in scales:
                for pressure_level in pressure_levels:
                    for tracker_name in trackers:
                        result_key = 'scale:{0};pl:{1};tracker:{2}'.format(scale,
                                                                           pressure_level,
                                                                           tracker_name)
                        em_tasks.append(PyroTask(year, ensemble_members, 'analysis', result_key))
            self._tasks.append(em_tasks)

    def get_next_outstanding(self):
        '''Returns the next outstanding task, None if there are no more'''
        for em_tasks in self._tasks:
            for task in em_tasks:
                if task.status == 'outstanding':
                    self.current_ensemble_member = task.ensemble_member
                    return task
        return None

    def get_progress(self):
        '''Returns a string representing the progress for current ensemble member'''
        progress = []

        em_tasks = self._tasks[self.current_ensemble_member]
        progress.append('em{0:2d}: '.format(em_tasks[0].ensemble_member))
        for task in em_tasks:
            progress.append(STATUSES[task.status])
        progress.append('\n')
        return ''.join(progress)

    def print_progress(self):
        print(self.get_progress())

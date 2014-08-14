from __future__ import print_function
from collections import OrderedDict

from stormtracks.analysis import StormtracksAnalysis

STATUSES = {
    'outstanding': '-',
    'working': 'W',
    'complete': '+',
    'failure': 'F',
    'timeout': 'T',
    }

TASKS = [
    'vort_tracking',
    'tracking_analysis',
    'field_collection_analysis',
    ]


class PyroTask(object):
    '''Represents task that is to be done

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


class PyroVortTracking(object):
    '''Keeps track of all vort tracking tasks to be done, and can issue the next outstanding class

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


class PyroTrackingAnalysis(object):
    '''Keeps track of all tracking analysis tasks to be done, and can issue the next outstanding class

    :param year: year to analyse
    '''
    def __init__(self, year, num_ensemble_members=56, config=None):
        self.stormtracks_analysis = StormtracksAnalysis(year)
        self.ensemble_members = range(num_ensemble_members)
        self._tasks = []
        self.task_count = 0
        self.config = config

        for ensemble_member in self.ensemble_members:
            em_tasks = []
            if not config:
                for config in self.stormtracks_analysis.analysis_config_options:
                    em_tasks.append(PyroTask(year, ensemble_member, 'tracking_analysis', config))
                    self.task_count += 1
                self._tasks.append(em_tasks)
            else:
                em_tasks.append(PyroTask(year, ensemble_member, 'tracking_analysis', config))
                self.task_count += 1
                self._tasks.append(em_tasks)

    def get_next_outstanding(self):
        '''Returns the next outstanding task, None if there are no more'''
        for em_tasks in self._tasks:
            for task in em_tasks:
                if task.status == 'outstanding':
                    return task
        return None

    def get_progress(self):
        '''Returns a string representing the progress'''
        progress = []

        if self.config:
            for ensemble_member in self.ensemble_members:
                task = self._tasks[ensemble_member][0]
                progress.append(STATUSES[task.status])
            progress.append('\n')
        else:
            for i, config in enumerate(self.stormtracks_analysis.analysis_config_options):
                for ensemble_member in self.ensemble_members:
                    task = self._tasks[ensemble_member][i]
                    progress.append(STATUSES[task.status])
                progress.append('\n')

        return ''.join(progress)

    def print_progress(self):
        '''Prints the current progress'''
        print(self.get_progress(), end='')


class PyroFieldCollectionAnalysis(object):
    '''Keeps track of all field collection analysis tasks to be done,
    and can issue the next outstanding task

    :param year: year to analyse
    '''
    def __init__(self, year, num_ensemble_members=56):
        self.ensemble_members = range(num_ensemble_members)
        self._tasks = []
        self.task_count = 0

        for ensemble_member in self.ensemble_members:
            task = PyroTask(year, ensemble_member, 'field_collection_analysis')
            self._tasks.append(task)
            self.task_count += 1

    def get_next_outstanding(self):
        '''Returns the next outstanding task, None if there are no more'''
        for task in self._tasks:
            if task.status == 'outstanding':
                return task
        return None

    def get_progress(self):
        '''Returns a string representing the progress'''
        progress = []

        for task in self._tasks:
            progress.append(STATUSES[task.status])
        progress.append('\n')
        return ''.join(progress)

    def print_progress(self):
        '''Prints the current progress'''
        print(self.get_progress(), end='')

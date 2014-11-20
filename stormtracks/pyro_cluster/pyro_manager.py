#!/usr/bin/python
from __future__ import print_function

import socket
import time
import copy
from argparse import ArgumentParser

import Pyro4
from Pyro4.errors import ConnectionClosedError

from stormtracks.load_settings import pyro_settings
from stormtracks.pyro_cluster.pyro_task import PyroVortTracking, PyroTrackingAnalysis, \
    PyroFieldCollectionAnalysis
from stormtracks.logger import setup_logging
from stormtracks.results import StormtracksResultsManager

hostname = socket.gethostname()
short_hostname = hostname.split('.')[0]
log = setup_logging('pyro_manager', 'pyro_manager_{0}.log'.format(short_hostname))


def main(task_name, years, num_ensemble_members=56, use_best_config=True, compress=True):
    '''Established communication with all pyro_workers

    N.B. pyro_nameserver must be set up first, and pyro workers must be running
    on each of the servers defined by pyro_settings.worker servers (either by running
    pyro_start.py or by manually starting them).

    Starts off by finding each of the pyro_workers, then generates a task schedule before
    handing over to run_tasks(...).
    '''
    log.info('Calling from {0}'.format(socket.gethostname()))
    log.info('Running task {0}'.format(task_name))

    workers = {}
    free_workers = copy.copy(pyro_settings.worker_servers)

    for server_name in free_workers:
        log.info('Adding server {0}'.format(server_name))

        worker_proxy = Pyro4.Proxy('PYRONAME:stormtracks.worker_{0}'.format(server_name))
        async_worker_proxy = Pyro4.async(worker_proxy)
        workers[server_name] = (worker_proxy, async_worker_proxy)

    if task_name == 'vort_tracking':
        results_manager = StormtracksResultsManager('pyro_vort_tracking')
    elif task_name == 'tracking_analysis':
        results_manager = StormtracksResultsManager('pyro_tracking_analysis')
    elif task_name == 'field_collection_analysis':
        results_manager = StormtracksResultsManager('pyro_field_collection_analysis')
    else:
        raise Exception('Task {0} not known'.format(task_name))

    for year in years:
        log.info('Running for year {0}'.format(year))
        if task_name == 'vort_tracking':
            task_provider = PyroTaskSchedule(year, year, num_ensemble_members=num_ensemble_members)
        elif task_name == 'tracking_analysis':
            if use_best_config:
                config = {'pressure_level': 850, 'scale': 3, 'tracker': 'nearest_neighbour'}
                task_provider = PyroTrackingAnalysis(year,
                                                     num_ensemble_members=num_ensemble_members,
                                                     config=config)
            else:
                task_provider = PyroTrackingAnalysis(year,
                                                     num_ensemble_members=num_ensemble_members)
        elif task_name == 'field_collection_analysis':
            task_provider = PyroFieldCollectionAnalysis(year,
                                                        num_ensemble_members=num_ensemble_members)

        run_tasks(year, task_provider, workers, free_workers, task_name=task_name)
        log.info('Tasks complete')

        if compress:
            log.info('Task complete, compressing year')
            if task_name == 'vort_tracking':
                results_manager.compress_year(year, delete=True)
            elif task_name == 'tracking_analysis':
                results_manager.compress_year(year, delete=True)
            elif task_name == 'field_collection_analysis':
                results_manager.compress_year(year, delete=True)
                # Compress/delete tracking results now I'm finished with them too.
                tracking_results_manager = StormtracksResultsManager('pyro_tracking_analysis')
                tracking_results_manager.compress_year(year, delete=True)

        log.info('Year {0} complete'.format(year))


def run_tasks(year, task_provider, workers, free_workers, task_name):
    '''Takes the given task_provider and list of workers and farms out the various tasks.'''
    start = time.time()

    asyncs = []
    asyncs_to_remove = []
    sleep_count = 0
    max_len_free_workers = len(free_workers)
    all_tasks_complete = False
    task = task_provider.get_next_outstanding()
    tasks_completed = 0

    while not all_tasks_complete:
        while task and free_workers:
            server_name = free_workers.pop()
            log.info('Requesting work from {0} year {1} ensemble {2}'.format(
                server_name, task.year, task.ensemble_member))

            worker_proxy, async_worker_proxy = workers[server_name]

            async_response = async_worker_proxy.do_work(task.year,
                                                        task.ensemble_member,
                                                        task.name,
                                                        task.data)
            async_response.server_name = server_name
            async_response.task = task
            asyncs.append(async_response)

            task.status = 'working'

            task = task_provider.get_next_outstanding()

            if not task:
                log.info('All tasks now being worked on')

        if task_name == 'vort_tracking':
            print('Step {0:4d}: '.format(sleep_count), end='')
            task_provider.print_years([year])
        elif task_name in ('tracking_analysis', 'field_collection_analysis'):
            print('Year {0:4d}, Step {1:4d}: '.format(year, sleep_count))
            task_provider.print_progress()

        sleep_count += 1
        time.sleep(3)

        for async_response in asyncs:
            time.sleep(0.1)
            try:
                if async_response.ready:
                    response = async_response.value

                    log.info('{0:8s}: {1}'.format(async_response.server_name, response['status']))
                    if response['status'] == 'complete':
                        async_response.task.status = response['status']
                        asyncs.remove(async_response)

                        free_workers.append(async_response.server_name)

                        tasks_completed += 1

                        if task_name == 'vort_tracking':
                            log.info(task_provider.get_progress_for_year(year))
                        elif task_name == 'tracking_analysis':
                            log.info('Tasks completed: {0}/{1}'.format(tasks_completed,
                                                                       task_provider.task_count))
                    elif response['status'] == 'failure':
                        log.error('Failure from {0}'.format(async_response.server_name))
                        log.error(response['exception'])
                        asyncs_to_remove.append(async_response)
                    else:
                        raise Exception(response['status'])

                else:
                    # print('{0:8s}: Not ready'.format(async_response.server_name))
                    pass
            except ConnectionClosedError, cce:
                log.error('Connection from {0} closed'.format(async_response.server_name))
                asyncs_to_remove.append(async_response)

        for async_response in asyncs_to_remove:
            log.error('{0} being removed from workers'.format(async_response.server_name))
            asyncs.remove(async_response)
            task = async_response.task
            task.status = 'outstanding'
            max_len_free_workers -= 1
        asyncs_to_remove = []

        if not task and len(free_workers) == max_len_free_workers:
            all_tasks_complete = True

    end = time.time()

    log.info('Completed {0} tasks in {1}s'.format(tasks_completed, end - start))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--regen', action='store_true')
    parser.add_argument('-a', '--analysis', default='field_collection_analysis')
    parser.add_argument('-u', '--use-best-config', action='store_true', default=False)
    parser.add_argument('-s', '--start-year', type=int, default=2000)
    parser.add_argument('-e', '--end-year', type=int, default=2009)
    parser.add_argument('-n', '--num-ensemble-members', type=int, default=56)
    parser.add_argument('-d', '--download', action='store_true', default=False)
    parser.add_argument('--download-start-year', type=int, default=1890)
    parser.add_argument('--download-end-year', type=int, default=1909)
    args = parser.parse_args()

    if args.download:
        import stormtracks.download as dl
        log.info('Downloading range: {0}-{1}'.format(args.download_start_year,
                                                     args.download_end_year))
        dl.download_full_c20_range(args.download_start_year, args.download_end_year)
        log.info('Downloaded range: {0}-{1}'.format(args.download_start_year,
                                                    args.download_end_year))

    years = range(args.start_year, args.end_year + 1)
    if args.analysis == 'field_collection_analysis':
        if args.regen:
            main('tracking_analysis', years, args.num_ensemble_members, compress=False)
        main('field_collection_analysis', years, args.num_ensemble_members)
    else:
        main(args.analysis, years, args.num_ensemble_members, use_best_config=args.use_best_config)

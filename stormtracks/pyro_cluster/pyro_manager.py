#!/usr/bin/python
from __future__ import print_function
import socket
import time
import copy

import Pyro4
from Pyro4.errors import ConnectionClosedError

from stormtracks.load_settings import pyro_settings
from stormtracks.pyro_cluster.pyro_task import PyroTaskSchedule
from stormtracks.logger import Logger

hostname = socket.gethostname()
short_hostname = hostname.split('.')[0]
log = Logger('pyro_manager', 'pyro_manager_{0}.log'.format(short_hostname)).get()


def main():
    '''Established communication with and gives tasks to all pyro_workers

    N.B. pyro_nameserver must be set up first, and pyro workers must be running
    on each of the servers defined by pyro_settings.worker servers (either by running
    pyro_start.py or by manually starting them).


    Starts off by finding each of the pyro_workers, then generates a task schedule and uses
    this to farm out work to each of the workers. Once all work has been done, finished.
    '''
    start = time.time()

    log.info('Calling from {0}'.format(socket.gethostname()))
    year = 2005
    schedule = PyroTaskSchedule(year, year)
    asyncs = []

    workers = {}
    free_workers = copy.copy(pyro_settings.worker_servers)

    for server_name in free_workers:
        log.info('Adding server {0}'.format(server_name))

        worker_proxy = Pyro4.Proxy('PYRONAME:stormtracks.worker_{0}'.format(server_name))
        async_worker_proxy = Pyro4.async(worker_proxy)
        workers[server_name] = (worker_proxy, async_worker_proxy)

    sleep_count = 0
    max_len_free_workers = len(free_workers)
    all_tasks_complete = False
    task = schedule.get_next_outstanding()

    while not all_tasks_complete:
        if task:
            while free_workers:
                server_name = free_workers.pop()
                log.info('Requesting work from {0} year {1} ensemble {2}'.format(
                    server_name, task.year, task.ensemble_member))

                worker_proxy, async_worker_proxy = workers[server_name]

                async_response = async_worker_proxy.do_work(task.year,
                                                            task.ensemble_member,
                                                            task.task)
                async_response.server_name = server_name
                async_response.task = task
                asyncs.append(async_response)

                task.status = 'working'

                task = schedule.get_next_outstanding()

                if not task:
                    log.info('All tasks now being worked on')

        print('Step {0:4d}: '.format(sleep_count), end='')
        schedule.print_years([year])

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

                        log.info(schedule.get_progress_for_year(year))
                    elif response['status'] == 'failure':
                        log.error(response['exception'])
                        task = async_response.task
                        task.status = 'outstanding'

                        free_workers.append(async_response.server_name)
                    else:
                        raise Exception(response['status'])

                else:
                    # print('{0:8s}: Not ready'.format(async_response.server_name))
                    pass
            except ConnectionClosedError, cce:
                log.error('Connection from {0} closed'.format(async_response.server_name))
                asyncs.remove(async_response)
                task = async_response.task
                task.status = 'outstanding'
                max_len_free_workers -= 1

        if not task and len(free_workers) == max_len_free_workers:
            all_tasks_complete = True

    end = time.time()
    tasks_completed = (1 + schedule.end_year - schedule.start_year) * schedule.num_ensemble_members

    log.info('Completed {0} tasks in {1}s'.format(tasks_completed, end - start))


if __name__ == '__main__':
    main()
